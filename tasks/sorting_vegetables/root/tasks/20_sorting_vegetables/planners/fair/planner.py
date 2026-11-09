"""Fair planner of task 20 (sorting_vegetables): only the native observation (3 RGB-D cameras + proprio, camera calibration) comes in, the
23D action goes out. Priors: the asset sizes of the bowls and the vegetables (params, data/models.npz), the BDDL init (which kinds share a
basket). No simulator state, no instance number.

Plan (STRATEGY.md; the oracle's one-by-one variant, on perception): turn in place to find the bowls on the counter and the vegetables in
the baskets on the floor (settle, scan, survey); then for each vegetable: choose it and a stand (plan_veg), drive there (go_veg), look at it
from the stand (look_veg), pinch it from above with the arm that reaches (veg_pre, veg_descend, veg_close: the native assisted grasp holds
it), lift it, tuck it, drive to a stand by the bowl of its group (plan_drop, go_bowl, look_bowl), reach over the bowl, lower it in and let
go (bowl_pre, bowl_lower, veg_release, veg_retract). Every step is a named phase with a reason, listed by log() for the trace. IK on the GPU
(cuRobo worker, ik.py): reachable_stands as a stand mask, solve_near seeded with the current joints for joint targets.
Interface: Planner.reset(seed), Planner.act(obs) -> (1, 23) float32, Planner.log() -> dict; entrypoint() builds the planner.
"""
from __future__ import annotations

import ast
import os
from typing import Any, Dict

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from . import ik
from . import params as P
from .phases import pick, place, search
from .phases.base import Core

EST_EVERY = 30


def step_done(c, action, q, w) -> None:
    """Nothing left that the planner can do from what it knows: after RETRY_WAIT controls it forgets the barred spots and the baskets it
    looked into and searches again. The challenge evaluator runs every episode to its timeout unless the task succeeds, so giving up only
    idles (301 and 302 ended with half their time left)."""
    c.to_rest(action, q)
    if c.age >= P.RETRY_WAIT:
        c.bad_spots, c.looked = [], set()
        c.counts['scans'], c.counts['plan_fail'] = 0, 0
        c.counts['retries'] = c.counts.get('retries', 0) + 1
        c.change('survey', f"retry {c.counts['retries']}: search again")


def step_failed(c, action, q, w) -> None:
    step_done(c, action, q, w)


PHASES = {**search.PHASES, **pick.PHASES, **place.PHASES, 'done': step_done, 'failed': step_failed}


class Planner(Core):
    def __init__(self) -> None:
        super().__init__()
        self.est_cache: Dict[str, Any] = {}
        self.variant = ''

    def reset(self, seed: int = 0) -> None:
        for kv in filter(None, os.environ.get('BIMBO_20_FAIR_OVERRIDE', '').split(';')):      # development rounds only: NAME=value;NAME=value
            name, value = kv.split('=', 1)
            setattr(P, name.strip(), ast.literal_eval(value))
        self.variant = os.environ.get('BIMBO_20_FAIR_VARIANT', '')                           # development rounds only: 'scan' stops after the scan
        ik.start()                       # GPU IK worker: once per process, warms up while the episode loads and settles
        self.reset_state()
        self.est_cache = {}
        self.scanned: list = []

    def act(self, obs: Dict[str, Any], privileged: Any = None) -> np.ndarray:
        assert privileged is None, 'the fair planner takes the observation only'
        self.perc.observe(obs, self.step, perceive=True if self.phase in ('look_veg', 'veg_look', 'look_bowl', 'look_basket') else None)
        q = K.q_from_proprio(self.perc.proprio)
        world = self.perc.world(q)
        if self.cmd is None:
            self.cmd = np.r_[np.zeros(3), q[0:4], q[4:11], K.GRIPPER_CLOSED, q[13:20], K.GRIPPER_CLOSED]
        action = self.cmd.copy()
        action[:3] = 0.
        self.step += 1
        self.age += 1
        PHASES[self.phase](self, action, q, world)
        action[K.GRIPPER_ACTION['left']], action[K.GRIPPER_ACTION['right']] = self.grip['left'], self.grip['right']
        self.cmd = action.copy()
        assert np.isfinite(action).all()
        return action[None].astype(np.float32)

    def log(self) -> Dict[str, Any]:
        if self.step % EST_EVERY == 1 or not self.est_cache:
            self.est_cache = self.perc.est()
        est = dict(self.est_cache, base=np.round(self.perc.pose(), 4).tolist(),          # the pose of this very control (a cached one lags
                   loc=dict(total=np.round(self.perc.loc_total, 4).tolist(), last=self.perc.last_loc))   # 0.28 rad behind in a turn)
        t = self.task
        return dict(phase=self.phase, reason=self.reason, step=self.step, age=self.age, err=[round(float(v), 4) for v in self.err],
                    counts=dict(self.counts), perc=dict(self.perc.counts), est=est, ik=ik.stats(), ik_backend=ik.STATS.get('backend'),
                    task=None if t is None else {k: (np.round(v, 3).tolist() if isinstance(v, np.ndarray) else v) for k, v in t.items()
                                                 if k in ('kind', 'group', 'side', 'stand', 'bowl', 'veg_xy', 'tries')},
                    bowls=dict(self.bowl_of), history=self.history[-24:], variant=self.variant, debug=self.debug)


def entrypoint() -> Planner:
    return Planner()
