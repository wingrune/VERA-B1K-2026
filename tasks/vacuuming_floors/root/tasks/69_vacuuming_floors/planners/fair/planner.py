"""Fair planner of task 69 (vacuuming_floors): only the native observation (3 RGB-D cameras + proprio, camera calibration) comes in, the 23D
action goes out. Prior: the vacuum asset model (data/models.npz). No simulator state, no instance number.

Plan (STRATEGY.md, stages as the oracle's): settle the arms and turn in place, finding the vacuum by its shape in the depth cloud and the dust
as dark flakes on the floor (settle, scan, survey); choose the pinch of the handle loop from its -x side and a stand that reaches it (GPU
reachable_stands), walk there, refit the vacuum, walk the arm in (joint space), close (the native assisted grasp holds the vacuum), lift it a
centimetre so that its projection cube spans the floor (plan_grasp, go_grasp, look_grasp, pre_grasp, approach, close, lift); switch it on with
the free hand pressing the handle at the marker, read from its colour (press_plan, press_reach, press_in, press_hold, press_out, press_home);
sweep: the base drives the cube over every flake (plan_sweep, adjust, drive, verify, check_flake, explore). Every step is a named phase with a reason,
listed by log() for the trace. IK on the GPU (cuRobo worker, ik.py): reachable_stands as a stand mask, solve_near seeded with the current
joints for every joint target, an arm-only solve for the free hand; model fits batched on the GPU (fitkern). Interface: Planner.reset(seed),
Planner.act(obs) -> (1, 23) float32, Planner.log() -> dict; entrypoint() builds the planner.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from . import fitkern
from . import ik
from . import params as P
from .phases import grasp, reach, recovery, search, sweep, switch_on
from .phases.base import Core

PHASES = {**search.PHASES, **grasp.PHASES, **reach.PHASES, **switch_on.PHASES, **sweep.PHASES, **recovery.PHASES}
EST_EVERY = 30


class Planner(Core):
    def __init__(self) -> None:
        super().__init__()
        self.est_cache: Dict[str, Any] = {}

    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        ik.start()                       # GPU IK worker: once per process, warms up while the episode loads and settles
        fitkern.reset_stats()
        self.reset_state()
        self.est_cache = {}

    def act(self, obs: Dict[str, Any], privileged: Any = None) -> np.ndarray:
        assert privileged is None, 'the fair planner takes the observation only'
        self.perc.observe(obs, self.step)
        q = K.q_from_proprio(self.perc.proprio)
        world = self.perc.world(q)
        if self.cmd is None:
            self.cmd = np.r_[np.zeros(3), q[0:4], q[4:11], K.GRIPPER_CLOSED, q[13:20], K.GRIPPER_CLOSED]
            self.start_q = q.copy()
        action = self.cmd.copy()
        action[:3] = 0.
        self.step += 1
        self.age += 1
        PHASES[self.phase](self, action, q, world)
        if self.arm_cmd is not None:
            action[K.TRUNK_ACTION] = self.arm_cmd[:4]
            action[K.ARM_ACTION[self.side]] = self.arm_cmd[4:]
        if self.other_cmd is not None:
            action[K.ARM_ACTION[self.free_side]] = self.other_cmd
        action[K.GRIPPER_ACTION['left']], action[K.GRIPPER_ACTION['right']] = self.grip['left'], self.grip['right']
        self.cmd = action.copy()
        self.cmd_last = None if self.arm_cmd is None else self.arm_cmd.copy()      # the arm command of this control (phases read the previous one)
        assert np.isfinite(action).all()
        self.track = 0. if self.arm_cmd is None else round(float(np.max(np.abs(q[self.dofs()] - self.arm_cmd))), 3)
        self._q = q
        return action[None].astype(np.float32)

    def log(self) -> Dict[str, Any]:
        q = getattr(self, '_q', None)
        if q is not None and (self.step % EST_EVERY == 1 or not self.est_cache):
            self.est_cache = self.perc.est(q)
        g = self.grasp
        return dict(phase=self.phase, reason=self.reason, step=self.step, age=self.age, side=self.side,
                    err=[round(float(v), 4) for v in self.err], track=getattr(self, 'track', None), counts=dict(self.counts), est=self.est_cache, ik=ik.stats(),
                    fit=fitkern.stats(), pick_stats=self.pick_stats, press_check=getattr(self, 'press_check_log', {}), approach_check=getattr(self, 'approach_check_log', {}),
                    press=self.press_log[-8:], looks=self.look_log[-6:], sweep=self.sweep_log[-24:], hard=sorted(self.hard), history=self.history[-24:],
                    grasp_step=self.grasp_step, approach=getattr(self, 'approach_diag', None),
                    grasp=None if g is None else dict(side=g['side'], pitch=round(float(np.degrees(g['frame']['pitch']))), sign=g['frame']['sign'],
                                                      stand=np.round(g['stand'], 3).tolist(), face=g['press_face']['name'], nopress=bool(g.get('nopress'))))


def entrypoint() -> Planner:
    return Planner()
