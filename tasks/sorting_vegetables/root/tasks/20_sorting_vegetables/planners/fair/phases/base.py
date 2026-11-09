"""Shared machinery of the phases of the fair planner of task 20: plan state, transitions, arm helpers, body checks against the perceived
occupancy. IK on the GPU: cuRobo reachable_stands as a stand mask (ik.scan), solve_near from the current joints for joint targets
(ik.IKServo). Everything works on the World (estimates in the odometry frame); nothing reads the simulator."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.skills_lib import r1pro_kinematics as K
from core.skills_lib import rgbd_geometry as RG

from .. import geometry as G
from .. import ik
from .. import params as P
from ..perception.state import Perception
from ..world import World
from .nav import NavMixin

DETECT_PHASES = ('scan', 'look_veg', 'veg_look', 'look_bowl', 'plan_veg', 'plan_drop', 'look_basket', 'plan_view')
EXPIRE_PHASES = ('scan', 'look_veg', 'look_basket', 'plan_veg', 'plan_view')   # the arms are tucked, the head sees what is in front
HOLD_PHASES = ('veg_pre', 'veg_look', 'veg_descend', 'veg_close', 'veg_lift', 'veg_raise', 'veg_carry', 'bowl_pre', 'bowl_in', 'bowl_lower', 'veg_release',
               'veg_retract', 'veg_stow', 'veg_putdown', 'putdown_wait')
BODY_LINKS = ('torso_link3', 'torso_link4', 'zed_link') + tuple(f'{a}_{l}' for a in ('left', 'right')
                                                            for l in ('arm_link2', 'arm_link4', 'arm_link6', 'gripper_link'))
TORSO_LINKS = ('torso_link1', 'torso_link2', 'torso_link3', 'torso_link4')
ARM_CHAIN = ('arm_link2', 'arm_link4', 'arm_link6', 'gripper_link')


def other_side(side: str) -> str:
    return 'left' if side == 'right' else 'right'


class Core(NavMixin):
    """State and helpers shared by the phases. A phase step is called as step(core, action, q, world)."""

    def __init__(self) -> None:
        self.perc = Perception()
        self.reset_state()

    def reset_state(self) -> None:
        self.perc.reset()
        self.phase, self.reason, self.age, self.step = 'settle', 'start', 0, 0
        self.counts: Dict[str, int] = dict(replans=0, nav_stuck=0, plans=0, grasp_fail=0, placed=0, missed=0)
        self.cmd: Optional[np.ndarray] = None
        self.grip = {'left': K.GRIPPER_CLOSED, 'right': K.GRIPPER_CLOSED}
        self.servo: Optional[ik.IKServo] = None
        self.path, self.path_age, self.stuck_ref = None, 0, None
        self.escape, self.escape_dir = 0, np.zeros(2)
        self.line0: Optional[np.ndarray] = None
        self.err = (float('nan'), float('nan'))
        self.history: List[Tuple[int, str, str]] = []
        self.best = (float('inf'), 0)
        self.walking = False
        self.task: Optional[Dict[str, Any]] = None       # the running vegetable: kind, group, side, stand, pinch, bowl
        self.bowl_of: Dict[int, int] = {}                 # group -> index of its bowl track
        self.bad_spots: List[np.ndarray] = []             # vegetable spots that failed: not chosen again for a while
        self.in_bowl: Dict[int, int] = {}                 # bowl track -> vegetables released over it
        self.no_stand: set = set()                        # bowl tracks without a drop stand (no group is given them)
        self.bad_drop: Dict[Tuple[int, str], int] = {}    # (bowl, arm) -> drop plans that found no stand
        self.hold_q: Optional[np.ndarray] = None          # trunk + arm joints of the carry pose
        self.tuck_best: Tuple[float, int] = (float('inf'), 0)
        self.looked: set = set()                          # basket tracks the head has looked into from close by
        self.view: Optional[Tuple[int, Optional[np.ndarray]]] = None
        self.scan_yaw0: Optional[float] = None
        self.turned = 0.0
        self.debug: Dict[str, Any] = {}

    # ---- transitions ----------------------------------------------------------------------------------------------------------------
    def change(self, phase: str, reason: str) -> None:
        self.history.append((self.step, phase, reason))
        self.phase, self.reason, self.age = phase, reason, 0
        self.path, self.stuck_ref = None, None
        self.best = (float('inf'), 0)
        self.tuck_best = (float('inf'), 0)
        self.perc.detect = phase in DETECT_PHASES
        self.perc.expire_ok = phase in EXPIRE_PHASES
        self.perc.map_hold = phase in HOLD_PHASES

    def stalled(self, steps: int = 120, gain: float = 0.01) -> bool:
        """The running reach has not improved by `gain` for `steps` controls."""
        if self.err[0] < self.best[0] - gain:
            self.best = (self.err[0], self.age)
        return self.age - self.best[1] > steps

    # ---- arms -----------------------------------------------------------------------------------------------------------------------
    def to_rest(self, action: np.ndarray, q: np.ndarray, sides=('left', 'right')) -> float:
        """Trunk to its rest pose (the head looks down at 40 degrees), the arms tucked along it."""
        worst = 0.
        for side in sides:
            target = np.r_[K.rest(side)[:4], P.TUCK_ARM]
            cur = q[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])]
            new = cur + np.clip(target - cur, -P.REST_STEP, P.REST_STEP)
            worst = max(worst, float(np.max(np.abs(target - cur))))
            action[K.TRUNK_ACTION], action[K.ARM_ACTION[side]] = new[:4], new[4:]
        return worst

    def tucked(self, worst: float) -> bool:
        """The arms are tucked: the largest joint error of to_rest under TUCKED_RAD, or no longer shrinking for TUCK_STALL controls (a joint
        held off its rest by a contact kept plan_veg waiting its full 300 controls, t20f19)."""
        if worst < self.tuck_best[0] - 0.005:
            self.tuck_best = (worst, self.age)
        return worst < P.TUCKED_RAD or self.age - self.tuck_best[1] > P.TUCK_STALL or self.age > P.STOW_MAX

    def tuck_other(self, action: np.ndarray, q: np.ndarray, side: str) -> None:
        other = other_side(side)
        cur = q[list(K.ARM_DOFS[other])]
        action[K.ARM_ACTION[other]] = cur + np.clip(np.asarray(P.TUCK_ARM) - cur, -P.REST_STEP, P.REST_STEP)

    def open_grip(self, side: str, spread: float) -> None:
        """Open so the fingertips sit `spread` m either side of the midpoint (command linear, 0.05 m at +1)."""
        self.grip[side] = float(np.clip(spread / 0.05 * 2. - 1., -1., 1.))

    @staticmethod
    def finger_gap(q: np.ndarray, side: str) -> float:
        return float(sum(q[list(K.FINGER_DOFS[side])]))

    def mid_world(self, q: np.ndarray, side: str) -> np.ndarray:
        return self.perc.mid_T(q, side)[:3, 3].copy()

    def line_to(self, action: np.ndarray, q: np.ndarray, w: World, goal: np.ndarray, rotation: np.ndarray, speed: float) -> bool:
        """One control along the straight line from self.line0 to `goal` (odometry finger midpoint), constant rotation. True at its end."""
        p0 = self.line0
        lam = min(1., self.age * speed / max(float(np.linalg.norm(goal - p0)), 1e-6))
        self.err = self.servo.step(action, q, G.to_robot(w.base, p0 + (goal - p0) * lam), G.rotation_to_robot(w.base, rotation), plan=False)
        return lam >= 1.

    def line_done(self, action: np.ndarray, q: np.ndarray, w: World, goal: np.ndarray, rotation: np.ndarray, speed: float) -> bool:
        """line_to that ends at the line's end (within 2 cm) or a timeout from its length (the stall check cut such lines short, oracle)."""
        done = self.line_to(action, q, w, goal, rotation, speed)
        return done and self.err[0] < 0.02 or self.age > np.linalg.norm(goal - self.line0) / speed + 60

    def walk_done(self, action: np.ndarray, q: np.ndarray, w: World, goal: np.ndarray, rotation: np.ndarray, speed: float,
                  tol: float = 0.015, max_age: int = 150) -> bool:
        """To an odometry finger-midpoint goal by a solve_near joint target (seeded with the joints now) and a synchronous walk; the straight
        Jacobian line from self.line0 only when there is no solution. True when within tol or after max_age controls."""
        goal_r, rot_r = G.to_robot(w.base, goal), G.rotation_to_robot(w.base, rotation)
        if self.age == 1:
            x = self.servo.solve(q, goal_r, rot_r)
            self.walking = x is not None
            if x is not None:
                self.servo.target, self.servo.sync, self.servo.cmd = x, True, None
        if not self.walking:
            return self.line_done(action, q, w, goal, rotation, speed)
        self.err = self.servo.step(action, q, goal_r, rot_r)
        if self.err[0] < tol or self.age > max_age:
            self.servo.sync, self.servo.target, self.servo.cmd = False, None, None
            return True
        return False

    def hold_pose(self, action: np.ndarray, q: np.ndarray) -> None:
        self.servo.hold(action, q)

    def sees(self, stand: np.ndarray, target: np.ndarray, near_m: float = 0.25) -> bool:
        """The head at the base pose `stand` (rest trunk: HEAD_REST_XZ in the robot frame) has a free line to the odometry point `target`
        through the perceived occupancy, but for the last near_m (the basket's own walls)."""
        head = G.to_odom(stand, np.array([P.HEAD_REST_XZ[0], 0., P.HEAD_REST_XZ[1]]))
        d = np.asarray(target, float) - head
        n = float(np.linalg.norm(d))
        fs = np.arange(0.15, max(0.15, n - near_m), P.VOX_M) / n
        return not len(fs) or not self.perc.vox.points_hit(head + fs[:, None] * d).any()

    # ---- body checks ----------------------------------------------------------------------------------------------------------------
    def body_points(self, x: np.ndarray, side: str, stand: np.ndarray, start: Optional[np.ndarray] = None,
                    fs: Tuple[float, ...] = (1., .5)) -> Tuple[np.ndarray, List[str]]:
        """Odometry points of the torso, the head and both arms (link origins and points between them) for trunk + `side` arm joints x at
        the base pose `stand`, the other arm tucked, at fractions `fs` of the joint walk from `start` (the tucked rest by default)."""
        start = np.r_[K.rest(side)[:4], P.TUCK_ARM] if start is None else start
        pts, names = [], []
        for f in fs:
            qq = np.zeros(22)
            qq[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])] = start + f * (np.asarray(x, float) - start)
            qq[list(K.ARM_DOFS[other_side(side)])] = P.TUCK_ARM
            poses = K.kinematics().forward(qq)[0]
            where = '' if f == 1. else f' at {f:.2f}'
            for n in BODY_LINKS + TORSO_LINKS:
                pts.append(poses[n][:3, 3])
                names.append(n + where)
            for a in ('left', 'right'):
                chain = [poses[f'{a}_{n}'][:3, 3] for n in ARM_CHAIN]
                for k in range(len(ARM_CHAIN) - 1):
                    for u in (1 / 3, 2 / 3):
                        pts.append(chain[k] + u * (chain[k + 1] - chain[k]))
                        names.append(f'{a}_{ARM_CHAIN[k]}-{ARM_CHAIN[k + 1]}{where}')
        return RG.robot_to_odom(np.array(pts), stand), names

    def body_hits(self, x: np.ndarray, side: str, stand: np.ndarray, ignore: Optional[list] = None, start: Optional[np.ndarray] = None,
                  fs: Tuple[float, ...] = (1., .5)) -> Optional[str]:
        """The first body point in a perceived occupied voxel (grown by one voxel), or None. `ignore` lists odometry boxes (lo, hi) of what
        the hand goes to. The IK has no collision model: without this check stands by the counter swept bowls off it (oracle, native 301)."""
        pts, names = self.body_points(x, side, stand, start, fs)
        hit = self.perc.vox.points_hit(pts, P.BODY_MARGIN_CELLS, ignore)
        if hit.any():
            k = int(np.argmax(hit))
            return f'{names[k]}@({pts[k][0]:.2f},{pts[k][1]:.2f},{pts[k][2]:.2f})'
        return None
