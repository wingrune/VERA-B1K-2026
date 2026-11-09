"""Fair planner of task 42 (chop_an_onion): only the native observation (3 RGB-D cameras + proprio, camera calibration) comes in, the 23D
action goes out. Priors: the asset models of the board and the parer, the sink basin and bowl sizes (data/models.npz, models.py). No simulator
state, no instance number.

Plan (STRATEGY.md, the stages of the oracle, on perception): turn in place until the counter is seen (settle, scan, survey), drive to a vantage
point in front of it and look over it (approach, look); the onion from the sink basin with the left hand into the bowl (plan, go, look, reach,
grasp, lift, carry, lower, release, look_bowl), the knife with the right hand, a touch cut of the onion and of a half in the bowl (cut_start,
chop_go, chop, cut_up, look_bowl, wait_slicer, dice), the knife and then the board into the sink basin (sink_go, sink_lower, look_sink).
Every step is a named phase with a reason, listed by log() for the trace. IK on the GPU (cuRobo worker, ik.py): reachable_stands as a stand
mask, solve_near (seeded with the current joints) for the servo; model fits batched on the GPU (fitkern). Interface: Planner.reset(seed),
Planner.act(obs) -> (1, 23) float32, Planner.log() -> dict; entrypoint() builds the planner.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.skills_lib import r1pro_kinematics as K

import ast
import os

from . import fitkern
from . import ik
from . import params as P
from .phases import cut, pick, place, search, sink
from .phases.base import Core

EST_EVERY = 30


def step_done(c, action, q, w) -> None:
    c.to_rest(action, q)


def step_failed(c, action, q, w) -> None:
    c.to_rest(action, q)


PHASES = {**search.PHASES, **pick.PHASES, **place.PHASES, **cut.PHASES, **sink.PHASES, 'done': step_done, 'failed': step_failed}


class Planner(Core):
    def __init__(self) -> None:
        super().__init__()
        self.est_cache: Dict[str, Any] = {}

    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        for kv in filter(None, os.environ.get('BIMBO_42_OVERRIDE', '').split(';')):      # development rounds only: NAME=value;NAME=value
            name, value = kv.split('=', 1)
            setattr(P, name.strip(), ast.literal_eval(value))
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
            self.cmd = np.r_[np.zeros(3), q[0:4], q[4:11], K.GRIPPER_OPEN, q[13:20], K.GRIPPER_OPEN]
        action = self.cmd.copy()
        action[:3] = 0.
        self.step += 1
        self.age += 1
        PHASES[self.phase](self, action, q, world)
        action[K.GRIPPER_ACTION['left']], action[K.GRIPPER_ACTION['right']] = self.grip['left'], self.grip['right']
        self.cmd = action.copy()
        assert np.isfinite(action).all()
        self._q = q
        return action[None].astype(np.float32)

    def log(self) -> Dict[str, Any]:
        q = getattr(self, '_q', None)
        if q is not None and (self.step % EST_EVERY == 1 or not self.est_cache):
            self.est_cache = self.perc.est(q)
        return dict(phase=self.phase, reason=self.reason, step=self.step, age=self.age, side=self.side, role=self.role,
                    err=[round(float(v), 4) for v in self.err], counts=dict(self.counts), stages=dict(self.stages), est=self.est_cache,
                    ik=ik.stats(), ik_backend=ik.backend(), fit=fitkern.stats(), pick_stats=self.pick_stats, history=self.history[-24:], history_all=self.history[-400:], debug=self.debug)


def entrypoint() -> Planner:
    return Planner()
