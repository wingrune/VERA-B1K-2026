"""Shared machinery of the phases of the fair planner of task 83: state of the plan, navigation over the perceived obstacle grid, arm helpers
(servo of the finger midpoint, postures, grippers, batched IK reachability). Everything works on the World (estimates in the odometry frame);
nothing reads the simulator. IK runs on the GPU worker (ik.py): ik.scan (reachable_stands) for the stand choice, solve_near seeded with the
current joints for the servo (ik.IKServo)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import ik
from .. import params as P
from ..navigation import Navigator, PoseGrid
from ..perception import Perception
from ..world import World

ARM_LINKS = ('arm_link4', 'arm_link5', 'arm_link6', 'arm_link7', 'gripper_link')


def world_T(points: Any, rots: Any, n: int) -> np.ndarray:
    """(N, 4, 4) poses of the finger-midpoint frame at `points` with orientation `rots` (each broadcast to N), odometry frame."""
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = np.broadcast_to(np.asarray(rots, float), (n, 3, 3))
    T[:, :3, 3] = np.broadcast_to(np.asarray(points, float), (n, 3))
    return T


class Core:
    """State and helpers shared by the phases. `step` of a phase module is called as step(core, action, q, world)."""

    def __init__(self) -> None:
        self.perc = Perception()
        self.reset_state()

    def reset_state(self) -> None:
        self.perc.reset()
        self.phase, self.reason, self.age, self.step = 'settle', 'start', 0, 0
        self.counts: Dict[str, int] = dict(replans=0, ik_checks=0, drawer_tries=0, nav_stuck=0, candidates_failed=0, explore=0, jar_tries=0,
                                           place_tries=0)
        self.cmd: Optional[np.ndarray] = None
        self.grip = {'left': K.GRIPPER_OPEN, 'right': K.GRIPPER_OPEN}
        self.nav = Navigator()
        self.servo: Optional[ik.IKServo] = None
        self.side = 'right'
        self.cands: List[Dict[str, Any]] = []
        self.goal: Dict[str, Any] = {}
        self.err = (float('nan'), float('nan'))
        self.history: List[Tuple[int, str, str]] = []
        self.scan_total, self.scan_last_yaw = 0., None
        self.hold: Optional[np.ndarray] = None          # trunk 4 + arm 7 joint targets held (pull, retreat, carry)
        self.mark: Dict[str, Any] = {}                   # per-phase reference values (start pose of a pull, stall counters)
        self.tip: Optional[List[float]] = None
        self.drawer_open_m = 0.0                         # base travel of the successful pull (the drawer travel)
        self.cab_viewed = False                          # the cabinet was looked at from the view stop
        self.jar_side = 'right'
        self.jar_goal: Dict[str, Any] = {}
        self.diag: Dict[str, Any] = {}

    # ---- transitions ----------------------------------------------------------------------------------------------------------------
    def change(self, phase: str, reason: str) -> None:
        self.history.append((self.step, phase, reason))
        self.phase, self.reason, self.age = phase, reason, 0
        self.servo = None
        self.nav.reset()
        self.mark = {}

    def fail(self, why: str) -> None:
        self.change('failed', why)

    # ---- navigation -----------------------------------------------------------------------------------------------------------------
    def pose_grid(self, w: World, goal_xy: np.ndarray, shrink: float = 0.0) -> PoseGrid:
        lo = np.minimum(w.base[:2], goal_xy) - P.NAV_REGION_M
        hi = np.maximum(w.base[:2], goal_xy) + P.NAV_REGION_M
        self.counts['replans'] += 1
        return PoseGrid(self.perc.nav_grid(lo, hi), shrink=shrink)

    def navigate(self, action: np.ndarray, q: np.ndarray, w: World, goal_xy: np.ndarray, goal_yaw: float, carry: bool = False) -> Tuple[float, float]:
        """One control towards the base pose; the arms in the carry posture (or the carrying arm held). Returns (distance m, yaw error rad)."""
        if carry and self.hold is not None:
            self.hold_joints(action, self.hold)
            self.to_rest(action, q, sides=(self.other(self.side),), trunk=False)
        else:
            self.to_rest(action, q)
        cmd, dist, yaw_err = self.nav.step(lambda shrink=0.0: self.pose_grid(w, goal_xy, shrink), w.base, goal_xy, goal_yaw,
                                           vmax_far=P.CARRY_VFAR if carry else P.NAV_VFAR, carry=carry)
        action[:3] = cmd
        self.reason = self.nav.last_reason
        return dist, yaw_err

    # ---- arm --------------------------------------------------------------------------------------------------------------------------
    def to_rest(self, action: np.ndarray, q: np.ndarray, sides=('left', 'right'), trunk: bool = True) -> float:
        """Walk the trunk and the arms to the carry posture; returns the largest remaining joint error."""
        worst = 0.
        for side in sides:
            target = K.rest(side)
            dofs = list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])
            cur = q[dofs]
            new = cur + np.clip(target - cur, -P.REST_STEP, P.REST_STEP)
            worst = max(worst, float(np.max(np.abs(target[4:] - cur[4:]))))
            if trunk:
                worst = max(worst, float(np.max(np.abs(target[:4] - cur[:4]))))
                action[K.TRUNK_ACTION] = new[:4]
            action[K.ARM_ACTION[side]] = new[4:]
        return worst

    def make_servo(self, side: str, q: Optional[np.ndarray] = None, lock_trunk: bool = False, max_step: float = 0.016) -> ik.IKServo:
        return ik.make_servo(side, q, lock_trunk, max_step)

    def servo_to(self, action: np.ndarray, q: np.ndarray, w: World, point: np.ndarray, rotation: np.ndarray, max_step: float = 0.016,
                 plan: bool = True, lock_trunk: bool = False) -> Tuple[float, float]:
        """One control of the finger midpoint of the active hand towards a pose given in the odometry frame (GPU solve_near targets)."""
        if self.servo is None or self.servo.side != self.side or abs(self.servo.max_step - max_step) > 1e-9:
            self.servo = self.make_servo(self.side, q, lock_trunk, max_step)
        self.err = self.servo.step(action, q, G.to_robot(w.base, point), G.rotation_to_robot(w.base, rotation), plan=plan)
        return self.err

    def hold_joints(self, action: np.ndarray, joints: np.ndarray) -> None:
        action[K.TRUNK_ACTION] = joints[:4]
        action[K.ARM_ACTION[self.side]] = joints[4:]

    def joints(self, q: np.ndarray) -> np.ndarray:
        return np.r_[q[list(K.TRUNK_DOFS)], q[list(K.ARM_DOFS[self.side])]]

    @staticmethod
    def finger_mean(q: np.ndarray, side: str) -> float:
        """Mean finger joint position (m): 0.05 open, near 0 closed (a 1.3 cm ledge stops them near 0.0065, the 8 cm jar near 0.04)."""
        return float(np.mean(q[list(K.FINGER_DOFS[side])]))

    def tip_point(self, q: np.ndarray, w: World, side: Optional[str] = None) -> np.ndarray:
        """Finger midpoint of a hand (default the active one) in the odometry frame."""
        return G.to_odom(w.base, ik.make_servo(side or self.side).pose(q)[:3, 3])

    def scan(self, side: str, stands: np.ndarray, targets: Sequence[Tuple[Any, Any]]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Batched GPU reachability: for N base poses (x, y, yaw, odometry frame) and K targets (points (N,3) or (3,), rotations), the mask (N,) of
        stands that reach every target and the joint solutions (K arrays of (N, 11))."""
        stands = np.atleast_2d(np.asarray(stands, float)).reshape(-1, 3)
        n = len(stands)
        masks, sols = [], []
        for pts, rots in targets:
            m, qq = ik.scan(side, world_T(pts, rots, n), stands, 0.0)
            masks.append(m)
            sols.append(qq)
        self.counts['ik_checks'] += n * len(targets)
        return (np.all(masks, 0) if masks else np.zeros(n, bool)), sols

    def arm_clear(self, side: str, x: np.ndarray, q: np.ndarray, base: np.ndarray, skip_xy: Optional[np.ndarray] = None,
                  skip_r: float = 0.0) -> bool:
        """No arm link origin from the elbow on is below the perceived top of the cell under it (plus ARM_CLEAR_M) for the joint vector x
        (trunk + arm) with the base at `base`. Cells within skip_r of skip_xy (the object to grasp) are not obstacles."""
        trial = q.copy()
        trial[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])] = x
        poses = K.kinematics().forward(trial)[0]
        for link in ARM_LINKS:
            p = G.to_odom(base, poses[f'{side}_{link}'][:3, 3])
            if skip_xy is not None and float(np.linalg.norm(p[:2] - skip_xy[:2])) < skip_r:
                continue
            top = self.perc.top_height(p[:2])
            if top is not None and p[2] < top + P.ARM_CLEAR_M:
                return False
        return True

    def goal_ledge(self, cab) -> np.ndarray:
        return cab.ledge(self.goal['k'])

    @staticmethod
    def other(side: str) -> str:
        return 'right' if side == 'left' else 'left'

    def arm_side_of(self, point: np.ndarray, base: np.ndarray) -> str:
        """The hand on the side of the point as seen from the base."""
        return 'left' if float(G.to_robot(base, point)[1]) >= 0 else 'right'

    def yaw_to(self, base_xy: np.ndarray, point_xy: np.ndarray) -> float:
        d = np.asarray(point_xy, float)[:2] - base_xy
        return math.atan2(d[1], d[0])
