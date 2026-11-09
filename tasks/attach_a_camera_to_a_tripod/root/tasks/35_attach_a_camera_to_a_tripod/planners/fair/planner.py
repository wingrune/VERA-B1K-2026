"""Fair planner of task 35 (attach_a_camera_to_a_tripod): only the native observation (3 RGB-D cameras + proprio, camera calibration) comes
in, the 23D action goes out. Priors: the camera and tripod asset models (data/models.npz). No simulator state.

Plan (STRATEGY.md, stages as the oracle's): settle the arms and turn in place until the camera is found on the floor (settle, scan, survey);
choose a top pinch across the lens barrel of the fitted camera model and a stand that reaches it, walk there, refit, reach, move in, close
(the native assisted grasp holds the camera; the hold is read from the finger gap), lift (plan_grasp, go_grasp, look_grasp, pre_grasp,
descend, close, back_off, lift); tuck the camera in front of the chest, find the tripod, walk to a stand that reaches the pose over its
mount, refit the tripod (tuck, find_tripod, plan_mount, carry, look_mount); bring the camera's M frame over the tripod's F frame and lower it
until the native attachment joins them, retrying other tripod yaw hypotheses and small dithers (to_pre_mount, insert, retract, release).
Every step is a named phase with a reason, listed by log() for the trace. IK on the GPU (cuRobo worker, ik.py): reachable_stands as a stand
mask, solve_near (seeded with the current joints) for the servo; model fits batched on the GPU (fitkern). Interface: Planner.reset(seed),
Planner.act(obs) -> (1, 23) float32, Planner.log() -> dict; entrypoint() builds the planner.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from . import fitkern
from . import ik
from .phases import carry, grasp, mount, search
from .phases.base import Core

PHASES = {**search.PHASES, **grasp.PHASES, **carry.PHASES, **mount.PHASES}
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
        action = self.cmd.copy()
        action[:3] = 0.
        self.step += 1
        self.age += 1
        PHASES[self.phase](self, action, q, world)
        action[14], action[22] = self.grip['left'], self.grip['right']
        self.cmd = action.copy()
        assert np.isfinite(action).all()
        self._q = q
        return action[None].astype(np.float32)

    def log(self) -> Dict[str, Any]:
        q = getattr(self, '_q', None)
        if q is not None and (self.step % EST_EVERY == 1 or not self.est_cache):
            self.est_cache = self.perc.est(q)
        g = self.grasp
        return dict(phase=self.phase, reason=self.reason, step=self.step, age=self.age, side=self.side, attempt=self.attempt,
                    err=[round(float(v), 4) for v in self.err], counts=dict(self.counts), est=self.est_cache, ik=ik.stats(),
                    fit=fitkern.stats(), pick_stats=self.pick_stats, mount_log=self.mount_log[-8:], history=self.history[-16:],
                    grasp=None if g is None else dict(site=g['opt']['site'], tilt=round(float(g['opt']['tilt']), 3), side=g['side'],
                                                      stand=np.round(g['stand'], 3).tolist()))


def entrypoint() -> Planner:
    return Planner()
