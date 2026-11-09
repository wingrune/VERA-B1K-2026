"""Fair planner of task 55 (re_shelving_library_books) with the native assisted grasp and GPU IK: only the native observation (3 RGB-D cameras +
proprio, camera calibration) comes in, the 23D action goes out. Priors: the offline static scene map built from training episodes (map_data.py,
data/PROVENANCE.json) and the three book models.

Phases of the fair layer around the oracle strategy (logic.py): settle; scan (a full turn in place, then the head points are matched to the map:
competing pose hypotheses, the best kept); look (base still, arms at rest: the pose is refined on the map and on the desk edges, the books and
desk items are measured; every plan starts from a fresh look); goview (drive to a stop that sees the desk when no book is known yet); look_hand
(on arriving at a push, pinch or insert stand: re-measure, re-plan when the target book moved). The logic's phases (plan, go, hand, steady,
retreat, done, failed) run on perception.state(). A book closed in the gripper rides with the hand frame; it counts as inside when the hand opens
with its estimate in a cubby. Interface: reset(seed), act(obs) -> (1, 23) float32, log() -> dict; entrypoint() builds the planner.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .logic import Planner as Logic
from .phases.look import LookMixin, MAX_VIEWS
from .perception import Perception
from .perception import FINGER_HELD_MIN
from .world import BOOKS, State

FINGER_EMPTY = FINGER_HELD_MIN          # a gripper closed on air, below the band of a book between the fingers
TRACK_EVERY = 6
REDIRECT_PLAN = ('settle', 'retreat', 'steady', 'go', 'hand', 'plan', 'failed', 'look_hand')


class Planner(LookMixin, Logic):
    def __init__(self) -> None:
        self.perc = Perception()
        super().__init__()

    def reset(self, seed: int = 0) -> None:
        super().reset(seed)
        self.perc = Perception()
        self.scan_turn, self.scan_prev = 0., None
        self.views = 0
        self.view_goal: Optional[np.ndarray] = None
        self.after_look = 'plan'
        self.look_reason = ''
        self.snapshot: Dict[str, Any] = {}
        self.est_cache: Dict[str, Any] = {}
        self.s: Optional[State] = None
        self.looks = 0

    # ---- interface ---------------------------------------------------------------------------------------------------------------------
    def act(self, obs: Dict[str, Any], privileged: Any = None) -> np.ndarray:
        assert privileged is None, 'the fair planner takes the observation only'
        q = self.perc.observe(obs, self.step)
        self.q_last = q
        if self.perc.T is not None and self.step % TRACK_EVERY == 0 and self.phase not in ('scan',):
            self.perc.track()
        s = self.belief()
        if self.cmd is None:
            self.cmd = np.r_[np.zeros(3), q[0:4], q[4:11], -1., q[13:20], -1.]
        side = (self.act_plan or {}).get('side') or (self.hold_side if self.holding is not None else 'left')
        self.tip = self.finger_mid_world(side, q, s)
        action = self.cmd.copy()
        action[:3] = 0.
        action[K.GRIPPER_ACTION['left']], action[K.GRIPPER_ACTION['right']] = self.grip['left'], self.grip['right']
        self.step += 1
        self.age += 1
        getattr(self, 'phase_' + self.phase)(action, q, s)
        self.cmd = action.copy()
        assert np.isfinite(action).all()
        return action[None].astype(np.float32)

    def belief(self) -> State:
        a = self.perc.attached
        if a is not None:
            closed = self.grip[a['side']] < 0.
            gap = self.perc.finger_gap(a['side'])
            if not closed:
                self.perc.detach(cubby_check=True)
            elif gap < FINGER_EMPTY:
                # fingers finished closing on nothing; while they are still moving the book stays attached
                self.perc.detach(cubby_check=False)
        self.s = self.perc.state(self.step)
        return self.s

    def log(self) -> Dict[str, Any]:
        out = super().log()
        if self.step % 30 == 1 or not self.est_cache:
            self.est_cache = self.perc.est()
        b = self.perc.base()
        out.update(est=dict(self.est_cache, base=None, base_yaw=None), base_map=[round(float(v), 3) for v in b], views=self.views,
                   look=self.look_reason)
        return out

    # ---- phase routing ----------------------------------------------------------------------------------------------------------------
    def change(self, phase: str, reason: str) -> None:
        if phase == 'plan' and self.phase in REDIRECT_PLAN:
            phase, self.after_look, self.look_reason = 'look', 'plan', reason
        if phase == 'look' and self.perc.T is None:
            phase = 'scan'
        if phase == 'hand' and self.phase == 'go' and self.act_plan is not None:
            phase, self.after_look, self.look_reason = 'look_hand', 'hand', reason
        if phase == 'failed' and self.holding is None and self.views < MAX_VIEWS and len(self.perc.books) + sum(self.perc.inside.values()) < len(BOOKS):
            phase, reason = 'goview', f'{reason}; a book is not seen yet'
        super().change(phase, reason)

    def finish_change(self, phase: str, reason: str) -> None:
        """A transition that is not redirected (the end of a look)."""
        Logic.change(self, phase, reason)

    def next_waypoint(self, a: Dict[str, Any], why: str) -> None:
        w = a['waypoints'][self.wi]
        if a['kind'] == 'pinch' and w['name'] == 'close':
            self.perc.attach(a['book'], a['side'])
        super().next_waypoint(a, why)

    def check(self, cond: tuple, s: State) -> bool:
        if cond[0] == 'held':
            a = self.perc.attached
            return a is not None and a['book'] == cond[1] and self.perc.holds(a['side'])
        return super().check(cond, s)

    def build_grid(self, s: State) -> None:
        m = self.perc.map
        g = nav_grid.Grid(m.origin, m.origin + np.array(m.shape) * m.res, m.res)
        n0, n1 = min(g.shape[0], m.shape[0]), min(g.shape[1], m.shape[1])
        g.blocked[:n0, :n1] = m.occ[:n0, :n1]
        self.grid = g
        self.inflated = g.inflated(0.31)


def entrypoint() -> Planner:
    return Planner()
