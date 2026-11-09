"""Strategy of the task 55 fair planner: the plans of the oracle (planners/oracle_ag, copied at the fair start and adapted) run on the State that
perception estimates (perception.py), never on the simulator state.

Plan, per book (nearest first): the book is pushed over a desk edge (a two-fingertip paddle, tool vertical) until it overhangs by OVERHANG_M, then
pinched at the overhang (tool axis horizontal into the book, jaw vertical: one finger above, one below), lifted, carried to a top-row cubby of a
bookcase and slid in, released and the hand pulled back. Every step is a named phase; transitions carry a reason; a phase that stalls or times out
re-plans from the live estimate (never by sleeping).

An action (push, pinch, insert) is a base pose plus a list of hand waypoints: point, gripper orientation, gripper command, how the tip gets there
(go: servo to it; slide: move the goal towards it at a fixed speed, so the contact is slow), dwell, and an optional stop condition read from the
estimate. The follower runs them in order. The books are held by the shared assisted grasp; every IK runs on the GPU through ik.py (stand choice
by a batched reachable_stands scan, the servo by solve_near seeded with the current joints).

Layout: the constants and plan switches are in params.py; the methods of the plan live in mixins under phases/ (base, reach,
navigate, plan, desk, sheet, pinch, push, push_geometry, insert, go, hand). This module assembles them into Planner and keeps
the interface (reset, log, act).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from . import ik
from .params import GRIP_OPEN, VARIANT
from .phases.base import BaseMixin
from .phases.reach import ReachMixin
from .phases.navigate import NavigateMixin
from .phases.plan import PlanMixin
from .phases.desk import DeskMixin
from .phases.sheet import SheetMixin
from .phases.pinch import PinchMixin
from .phases.push import PushMixin
from .phases.push_geometry import PushGeometryMixin
from .phases.insert import InsertMixin
from .phases.go import GoMixin
from .phases.hand import HandMixin


class Planner(
        BaseMixin, ReachMixin, NavigateMixin, PlanMixin, DeskMixin, SheetMixin, PinchMixin, PushMixin, PushGeometryMixin, InsertMixin,
        GoMixin, HandMixin):
    def __init__(self) -> None:
        self.reset(0)

    # ---- interface -------------------------------------------------------------------------------------------------------------------
    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        ik.start()                       # GPU IK worker: once per process, warms up while the episode loads and settles
        self.phase, self.reason, self.age, self.step = 'settle', 'start', 0, 0
        self.counts: Dict[str, int] = dict(replans=0, ik_checks=0, nav_stuck=0, pushes=0, pinches=0, inserts=0, putbacks=0, lift_fail=0, plan_fail=0, ik_none=0)
        self.ik_backend, self.ik_calls, self.ik_ms = 'unset', 0, 0
        self.cmd: Optional[np.ndarray] = None
        self.grid: Optional[nav_grid.Grid] = None
        self.inflated: Optional[nav_grid.Grid] = None
        self.servo: Optional[K.ArmServo] = None
        self.act_plan: Optional[Dict[str, Any]] = None
        self.wi = 0
        self.path: Optional[List[np.ndarray]] = None
        self.path_age = 0
        self.stuck_ref: Optional[Tuple[np.ndarray, int]] = None
        self.grip = {'left': GRIP_OPEN, 'right': GRIP_OPEN}
        self.err = (float('nan'), float('nan'))
        self.history: List[Tuple[int, str, str]] = []
        self.tip: Optional[List[float]] = None
        self.attempts: Dict[Tuple[str, str], int] = {}
        self.dead_books: set = set()
        self.put_back: set = set()
        self.holding: Optional[str] = None
        self.dwell = 0
        self.slide_goal: Optional[np.ndarray] = None
        self.hand0: Optional[np.ndarray] = None
        self.stop_after: List[str] = []
        self.q_last: Optional[np.ndarray] = None
        self.hold_side = 'left'
        self.soft = False
        self.notes: List[str] = []
        self.grasp_off_r: Optional[np.ndarray] = None    # book centre minus finger midpoint, robot frame, captured once the pinch has settled
        self.grasp_tip_r: Optional[np.ndarray] = None
        self.grasp_rot_r: Optional[np.ndarray] = None
        self.slide_extra = 0
        self.blocker = ''
        self.turn_dir, self.turn_extra, self.turn_long = None, 0, False
        self._sheet_step, self._sheet_hit = -1, False
        print(f'FAIR_GRASP {ik.fair_mode() or "base"}', flush=True)

    def log(self) -> Dict[str, Any]:
        a = self.act_plan or {}
        return dict(tip=self.tip, phase=self.phase, reason=self.reason, step=self.step, age=self.age, kind=a.get('kind'), book=a.get('book'),
                    err=[round(float(v), 4) for v in self.err], counts=dict(self.counts), waypoint=self.wi, history=self.history[-14:], notes=self.notes[-40:],
                    ik_backend=self.ik_backend, ik_calls=self.ik_calls, ik_ms=self.ik_ms, ik=ik.stats(), variant=sorted(VARIANT),
                    grasp=ik.fair_mode() or 'base')

    def act(self, obs: Dict[str, Any], privileged: Any = None) -> np.ndarray:
        """The observation planner in planner.py is the only act(). This method never reads the simulator."""
        del obs, privileged
        raise RuntimeError('fair logic does not read simulator state; planner.Planner.act is the entry')


def entrypoint() -> Planner:
    return Planner()
