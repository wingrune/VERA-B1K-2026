"""Fair planner of task 83 (store_honey) with the native assisted grasp and GPU IK: only the native observation (3 RGB-D cameras + proprio)
comes in, the 23D action goes out.

Plan (the stages of the oracle, on perception): turn in place to map the room and find the cabinet by its top (settle, scan, explore); look at
it from a stop in front (go_view, drawer_view); drive to a stand in front of a pull ledge of the top drawer, pinch the ledge and pull the drawer
out by backing the base (plan_drawer .. drawer_retreat, drawer_done); find the jar standing on the counter (find_jar), side-pinch it from a stand next to the counter (plan_jar .. jar_grasp; the
native assisted grasp holds it), lift it and bring it to the carry posture (jar_lift, jar_retract); carry it to a stand in front of the open
drawer and lower it into the drawer, open the fingers (plan_place .. place_retreat, done). Every step is a named phase with a reason for each
transition, listed by log() for the trace of the harness. IK on the GPU (cuRobo worker, ik.py): reachable_stands for the base poses,
solve_near seeded with the current joints for the arm servo; no CPU fallback with BIMBO_IK_REQUIRE_GPU=1. Interface: Planner.reset(seed),
Planner.act(obs) -> (1, 23) float32, Planner.log() -> dict; entrypoint() builds the planner. Variant (BIMBO_83_VARIANT, VARIANTS): parameter
overrides for parallel one-change jobs.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from . import ik
from . import params as P
from .phases import drawer, explore, jar, place
from .phases.base import Core

EST_EVERY = 30
VARIANTS = {
    'base': {},
    'noview': dict(VIEW_DISTANCES_M=()),       # no view stop before the drawer stand
    'rawodo': dict(ODO_CORRECTED=False),       # translation along the raw odometry heading
    'track': dict(CAB_TRACK=True),             # every complete cabinet view pulls the estimate
}
DEFAULT_VARIANT = 'base'


def step_failed(c, action, q, w) -> None:
    c.to_rest(action, q)


PHASES = {**explore.PHASES, **drawer.PHASES, **jar.PHASES, **place.PHASES, 'failed': step_failed}
HOLD_PHASES = ('drawer_pre', 'drawer_reach', 'drawer_grip', 'drawer_pull', 'drawer_release', 'jar_look', 'jar_pre', 'jar_reach', 'jar_grasp',
               'jar_lift', 'jar_retract', 'place_above', 'place_lower', 'place_release', 'place_retreat')
JAR_SEARCH_PHASES = ('scan', 'find_jar', 'plan_jar', 'go_jar', 'jar_look')


class Planner(Core):
    def __init__(self) -> None:
        super().__init__()
        self.est_cache: Dict[str, Any] = {}

    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        name = os.environ.get('BIMBO_83_VARIANT') or DEFAULT_VARIANT
        self.variant = name if name in VARIANTS else DEFAULT_VARIANT
        for k, v in VARIANTS[self.variant].items():
            setattr(P, k, v)
        ik.start()                       # GPU IK worker: once per process, warms up while the episode loads and settles
        self.reset_state()
        self.est_cache = {}

    def act(self, obs: Dict[str, Any], privileged: Any = None) -> np.ndarray:
        assert privileged is None, 'the fair planner takes the observation only'
        self.perc.freeze_pose = self.phase in HOLD_PHASES
        self.perc.jar_dense = self.phase in JAR_SEARCH_PHASES
        self.perc.observe(obs, self.step)
        world = self.perc.world()
        q = K.q_from_proprio(self.perc.proprio)
        if self.cmd is None:
            self.cmd = np.r_[np.zeros(3), q[0:4], q[4:11], K.GRIPPER_OPEN, q[13:20], K.GRIPPER_OPEN]
        action = self.cmd.copy()
        action[:3] = 0.
        self.step += 1
        self.age += 1
        if self.age > P.TIMEOUT.get(self.phase, 10 ** 9):
            self.on_timeout(q)
        PHASES[self.phase](self, action, q, world)
        action[K.GRIPPER_ACTION['left']], action[K.GRIPPER_ACTION['right']] = self.grip['left'], self.grip['right']
        self.cmd = action.copy()
        assert np.isfinite(action).all()
        return action[None].astype(np.float32)

    def on_timeout(self, q: np.ndarray) -> None:
        ph, why = self.phase, f'{self.phase} timed out after {self.age} controls ({self.reason})'
        if ph == 'scan':
            self.change('explore', why)
        elif ph == 'explore':
            self.scan_total, self.scan_last_yaw = 0., self.perc.pose()[2]
            self.change('scan', why)
        elif ph in ('go_view', 'drawer_view'):
            drawer.view_done(self, why)
        elif ph == 'go_drawer':
            self.counts['nav_stuck'] += 1
            drawer.next_candidate(self, why)
        elif ph == 'drawer_retreat':
            self.change('drawer_done', why)
        elif ph.startswith('drawer_'):
            self.counts['drawer_tries'] += 1
            drawer.next_candidate(self, why)
        elif ph == 'find_jar':
            self.fail(why)
        elif ph == 'jar_retract':
            if self.finger_mean(q, self.side) >= P.JAR_HELD_MIN:
                self.hold = self.joints(q)
                self.change('plan_place', why + '; carrying from here')
            else:
                self.counts['jar_tries'] += 1
                jar.next_jar_candidate(self, why)
        elif ph in ('go_jar', 'jar_look') or ph.startswith('jar_'):
            self.counts['jar_tries'] += 1
            jar.next_jar_candidate(self, why)
        elif ph == 'place_lower':
            self.hold = self.joints(q)
            self.change('place_release', why + '; releasing where the jar is')
        elif ph in ('go_place', 'place_above'):
            self.counts['place_tries'] += 1
            place.next_place_candidate(self, why)
        elif ph == 'place_release':
            self.change('place_retreat', why)
        elif ph == 'place_retreat':
            self.change('done', why)
        else:
            self.age = 0

    def log(self) -> Dict[str, Any]:
        if self.step % EST_EVERY == 1 or not self.est_cache:
            self.est_cache = self.perc.est()
        return dict(variant=getattr(self, 'variant', None), phase=self.phase, reason=self.reason, step=self.step, age=self.age, side=self.side, tip=self.tip,
                    err=[round(float(v), 4) for v in self.err], counts=dict(self.counts), est=self.est_cache, ik=ik.stats(),
                    drawer_open_m=round(self.drawer_open_m, 3), history=self.history[-12:])


def entrypoint() -> Planner:
    return Planner()
