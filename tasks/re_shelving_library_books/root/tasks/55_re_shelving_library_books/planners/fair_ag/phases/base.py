"""Shared helpers of the phases: phase change, rest posture, the held book in the hand frame, notes.
Mixin of logic.Planner of the task 55 fair planner; the methods were moved verbatim from logic.py."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from ..world import State
from ..params import GRID_RES, REST_STEP, ROBOT_RADIUS, SOFT_CATEGORIES


class BaseMixin:
    def finger_mid_world(self, side: str, q: np.ndarray, s: State) -> List[float]:
        servo = K.ArmServo(side)
        servo.use_fingers_mid()
        p = G.to_world(s.base, s.base_z, servo.pose(q)[:3, 3])
        return [round(float(v), 3) for v in p]

    # ---- helpers ---------------------------------------------------------------------------------------------------------------------
    def change(self, phase: str, reason: str) -> None:
        self.history.append((self.step, phase, reason))
        self.phase, self.reason, self.age = phase, reason, 0
        self.path, self.stuck_ref = None, None
        self.turn_dir, self.turn_extra, self.turn_long = None, 0, False
        if phase in self.stop_after:
            self.phase, self.reason = 'done', f'stopped after {phase} (BIMBO_ORACLE_STOP)'

    def obstacle_boxes(self, s: State, movable: Tuple[str, ...] = ()) -> List[tuple]:
        return [b for b in (s.obstacles or []) if b[2] not in movable]

    def build_grid(self, s: State) -> None:
        boxes = self.obstacle_boxes(s, SOFT_CATEGORIES if self.soft else ())
        lo = np.min([b[0] for b in boxes] + [s.base[:2]], axis=0) - 1.0
        hi = np.max([b[1] for b in boxes] + [s.base[:2]], axis=0) + 1.0
        self.grid = nav_grid.Grid(lo, hi, GRID_RES)
        for a, b, _ in boxes:
            self.grid.add_box(a, b)
        self.inflated = self.grid.inflated(ROBOT_RADIUS)

    def to_rest(self, action: np.ndarray, q: np.ndarray, hold: Optional[str] = None) -> float:
        """Walk the trunk and the arms to the carry posture (the gripper commands are not touched: a held book stays held); the arm `hold` and
        the trunk keep their joint targets while a book is carried. Returns the largest remaining joint error."""
        worst = 0.
        for side in ('left', 'right'):
            if side == hold:
                continue
            target = K.rest(side)
            dofs = list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])
            cur = q[dofs]
            new = cur + np.clip(target - cur, -REST_STEP, REST_STEP)
            worst = max(worst, float(np.max(np.abs(target - cur))))
            action[K.ARM_ACTION[side]] = new[4:]
            if hold is None:
                action[K.TRUNK_ACTION] = new[:4]
        return worst

    def carry_side(self) -> Optional[str]:
        return self.hold_side if self.holding is not None else None

    def freeze_hold_arm(self, action: np.ndarray, q: np.ndarray) -> None:
        """Replay the trunk and carrying-arm command captured when the pinch settled. The grasp is friction only:
        copying the measured joints lets the arm sag, and any other motion of these joints drops the book."""
        del q
        side = self.hold_side
        cmd = getattr(self, 'hold_cmd', None)
        if side is None or cmd is None:
            return
        action[K.TRUNK_ACTION] = np.asarray(cmd[K.TRUNK_ACTION], float)
        action[K.ARM_ACTION[side]] = np.asarray(cmd[K.ARM_ACTION[side]], float)

    def capture_grasp(self, q: np.ndarray, s: State) -> None:
        """Book offset in the robot frame, and, once, the joint command that is holding it. Later captures refresh the
        offset only: replacing the joint command from a new measurement lets the arm sag."""
        if self.holding is None:
            return
        fm, hand_rot = self.hand_frame(self.hold_side, q, s)
        tip_r = G.to_robot(s.base, s.base_z, fm)
        book_r = G.to_robot(s.base, s.base_z, s.books[self.holding].box.centre)
        self.grasp_tip_r = tip_r
        self.grasp_off_r = book_r - tip_r
        self.grasp_rot_r = G.rotation_to_robot(s.base, hand_rot)
        if getattr(self, 'hold_cmd', None) is None and self.cmd is not None:
            self.hold_cmd = np.array(self.cmd, float, copy=True)

    def grasp_drift(self, q: np.ndarray, s: State) -> float:
        if self.grasp_off_r is None or self.holding is None or self.hold_side is None:
            return 0.
        fm, _ = self.hand_frame(self.hold_side, q, s)
        tip_r = G.to_robot(s.base, s.base_z, fm)
        book_r = G.to_robot(s.base, s.base_z, s.books[self.holding].box.centre)
        return float(np.linalg.norm((book_r - tip_r) - self.grasp_off_r))

    @staticmethod
    def base_for_tip(tip_world: np.ndarray, tip_robot: np.ndarray, yaw: float, base_z: float) -> np.ndarray:
        """Base pose that puts a frozen fingertip (robot frame) at `tip_world`."""
        c, s_ = math.cos(yaw), math.sin(yaw)
        delta = np.array([c * tip_robot[0] - s_ * tip_robot[1], s_ * tip_robot[0] + c * tip_robot[1]])
        return np.array([tip_world[0] - delta[0], tip_world[1] - delta[1], yaw])

    def note(self, msg: str) -> None:
        self.notes.append(msg)
        del self.notes[:-40]

    def key(self, book: str, kind: str) -> Tuple[str, str]:
        return (book, kind)
