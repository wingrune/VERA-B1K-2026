"""Shared machinery of the phases of the fair planner of task 35: plan state, transitions, frames, arm helpers (the navigation part is
phases/nav.py). IK on the GPU: cuRobo reachable_stands as a stand mask only, the servo solves solve_near from the current joints. Everything
works on the World (estimates in the odometry frame); nothing reads the simulator."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import ik
from .. import params as P
from ..perception import Perception
from ..world import World
from .nav import NavMixin


class Core(NavMixin):
    """State and helpers shared by the phases. A phase step is called as step(core, action, q, world)."""

    def __init__(self) -> None:
        self.perc = Perception()
        self.reset_state()

    def reset_state(self) -> None:
        self.perc.reset()
        self.phase, self.reason, self.age, self.step = 'settle', 'start', 0, 0
        self.counts: Dict[str, int] = dict(replans=0, nav_stuck=0, grasp_tries=0, grasp_fail=0, drops=0, mount_rounds=0, mount_tries=0,
                                           plans=0, candidates_failed=0, surveys=0, approach_rejects=0, ik_none=0)
        self.cmd: Optional[np.ndarray] = None
        self.inflated: Optional[nav_grid.Grid] = None
        self.side = 'left'
        self.servo: Optional[ik.IKServo] = None
        self.grasp_cands: List[Dict[str, Any]] = []
        self.grasp: Optional[Dict[str, Any]] = None
        self.mount: Optional[Dict[str, Any]] = None
        self.path: Optional[List[np.ndarray]] = None
        self.path_age = 0
        self.stuck_ref: Optional[Tuple[np.ndarray, int]] = None
        self.grip = {'left': K.GRIPPER_CLOSED, 'right': K.GRIPPER_CLOSED}
        self.bad_grasp: set = set()
        self.bad_mount: set = set()
        self.routes: Dict[tuple, float] = {}
        self.hold_q: Optional[np.ndarray] = None
        self.lift0: Optional[np.ndarray] = None
        self.rot_hold: Optional[np.ndarray] = None
        self.line0: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.tuck: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.attempt = 0                   # index into the mount attempts (tripod yaw hypothesis x dither)
        self.held_age = 0
        self.best_err, self.best_age = float('inf'), 0
        self.look_t0: Optional[int] = None
        self.turned = 0.
        self.scan_from: Optional[float] = None
        self.err = (float('nan'), float('nan'))
        self.history: List[Tuple[int, str, str]] = []
        self.mount_log: List[Dict[str, Any]] = []
        self.pick_stats: Dict[str, int] = {}

    # ---- transitions ----------------------------------------------------------------------------------------------------------------
    def change(self, phase: str, reason: str) -> None:
        self.history.append((self.step, phase, reason))
        self.phase, self.reason, self.age = phase, reason, 0
        self.path, self.stuck_ref = None, None
        self.perc.map_hold = phase in P.MAP_HOLD_PHASES      # the arm and the camera in it move: their points must not become obstacles

    # ---- frames -----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def tool_goal(point: np.ndarray, rotation: np.ndarray, w: World):
        return G.to_robot(w.base, 0., point), G.rotation_to_robot(w.base, rotation)

    def mid_world(self, q: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray]:
        T = self.perc.mid_T(q, side)
        return T[:3, 3].copy(), T[:3, :3].copy()

    @staticmethod
    def finger_gap(q: np.ndarray, side: str) -> float:
        return float(sum(q[list(K.FINGER_DOFS[side])]))

    def holds(self, q: np.ndarray, side: str) -> bool:
        """The closed fingers stay apart: something is between them (the assisted grasp holds it). Proprio only."""
        return self.finger_gap(q, side) > P.HELD_GAP_M

    # ---- motion -----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def rest_pose(side: str) -> np.ndarray:
        """Trunk 4 + arm 7 of the travel posture: the rest trunk (the head camera pitched down onto the floor) with the arms either in the
        rest pose (hands about 1 m ahead at shoulder height) or hanging (COMPACT_ARMS: nothing farther than 0.35 m from the base axis, so a
        turn in place sweeps no wider than the base)."""
        target = K.rest(side).copy()
        if P.COMPACT_ARMS:
            target[4:] = 0.
        return target

    def to_rest(self, action: np.ndarray, q: np.ndarray, sides=('left', 'right')) -> float:
        worst = 0.
        for side in sides:
            target = self.rest_pose(side)
            dofs = list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])
            cur = q[dofs]
            new = cur + np.clip(target - cur, -P.REST_STEP, P.REST_STEP)
            worst = max(worst, float(np.max(np.abs(target - cur))))
            action[K.TRUNK_ACTION] = new[:4]
            action[K.ARM_ACTION[side]] = new[4:]
        return worst

    def other_rest(self, action: np.ndarray, q: np.ndarray) -> None:
        other = 'right' if self.side == 'left' else 'left'
        target = self.rest_pose(other)[4:]
        cur = q[list(K.ARM_DOFS[other])]
        action[K.ARM_ACTION[other]] = cur + np.clip(target - cur, -P.REST_STEP, P.REST_STEP)

    def open_grip(self, side: str, spread: float) -> None:
        self.grip[side] = float(np.clip(spread / 0.05 * 2. - 1., -1., 1.))

    def hold_arm(self, action: np.ndarray) -> None:
        action[K.TRUNK_ACTION] = self.hold_q[:4]
        action[K.ARM_ACTION[self.side]] = self.hold_q[4:]

    def line_to(self, action, q, w: World, T_goal: np.ndarray, speed: float) -> Tuple[float, bool]:
        """One control along the straight line (position, slerped rotation) from self.line0 to T_goal. Returns (fraction, done)."""
        from scipy.spatial.transform import Rotation as R
        p0, r0 = self.line0
        span = max(float(np.linalg.norm(T_goal[:3, 3] - p0)), float(R.from_matrix(T_goal[:3, :3] @ r0.T).magnitude()) * 0.25)
        lam = min(1., self.age * speed / max(span, 1e-6))
        rv = R.from_matrix(T_goal[:3, :3] @ r0.T).as_rotvec() * lam
        goal, rot = self.tool_goal(p0 + (T_goal[:3, 3] - p0) * lam, R.from_rotvec(rv).as_matrix() @ r0, w)
        self.err = self.servo.step(action, q, goal, rot, plan=False)
        return lam, lam >= 1.
