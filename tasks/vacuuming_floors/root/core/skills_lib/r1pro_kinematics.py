"""R1Pro kinematics for planners: forward kinematics, geometric Jacobians and a bounded servo of the trunk and one arm.

Pure numpy and scipy, no simulator. The joint table is r1pro.urdf of the pinned robot (base_link root, metres, radians).
Joint vector q (22): trunk 0-3, left arm 4-10, left fingers 11-12, right arm 13-19, right fingers 20-21. `q_from_proprio` builds it from
the native 61D proprio vector. Poses are in the robot frame (base_link). ArmServo moves trunk + one arm towards a gripper pose with bounded
joint steps and writes the absolute position targets into the native 23D action (trunk 3:7, left arm 7:14, right arm 15:22).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

JOINTS = (
    (0, 'base_link', 'torso_link1', 'revolute', (-0.079032, 0.0, 0.34265), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (1, 'torso_link1', 'torso_link2', 'revolute', (0.0, 0.0, 0.4), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (2, 'torso_link2', 'torso_link3', 'revolute', (0.0, 0.0001005, 0.3), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)), (3, 'torso_link3', 'torso_link4', 'revolute', (0.0, -0.00010015, 0.09962), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (-1, 'torso_link4', 'zed_link', 'fixed', (0.065511, 0.060003, 0.47598), (-1.9199, 0.0, -1.5708), (0.0, 0.0, 0.0)), (-1, 'torso_link4', 'left_arm_base_link', 'fixed', (-0.00048618, 0.097234, 0.30302), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (4, 'left_arm_base_link', 'left_arm_link1', 'revolute', (0.0, 0.0735, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (5, 'left_arm_link1', 'left_arm_link2', 'revolute', (0.025012, 0.081265, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    (6, 'left_arm_link2', 'left_arm_link3', 'revolute', (-0.025012, 0.0, -0.1155), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), (7, 'left_arm_link3', 'left_arm_link4', 'revolute', (0.0, 0.035065, -0.1945), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (8, 'left_arm_link4', 'left_arm_link5', 'revolute', (0.0, -0.035065, -0.095), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), (9, 'left_arm_link5', 'left_arm_link6', 'revolute', (0.0, -0.028, -0.16305), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (10, 'left_arm_link6', 'left_arm_link7', 'revolute', (0.0295, 0.028, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (-1, 'left_arm_link7', 'left_gripper_link', 'fixed', (-0.0295, 0.0, -0.16065), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (11, 'left_gripper_link', 'left_gripper_finger_link1', 'prismatic', (-8.8709e-05, 0.013453, -0.03689), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (12, 'left_gripper_link', 'left_gripper_finger_link2', 'prismatic', (8.9046e-05, -0.013453, -0.03689), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    (-1, 'torso_link4', 'right_arm_base_link', 'fixed', (-0.00048706, -0.097236, 0.30302), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), (13, 'right_arm_base_link', 'right_arm_link1', 'revolute', (0.0, -0.0735, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (14, 'right_arm_link1', 'right_arm_link2', 'revolute', (0.023988, -0.081265, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (15, 'right_arm_link2', 'right_arm_link3', 'revolute', (-0.023988, 0.0, -0.1155), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (16, 'right_arm_link3', 'right_arm_link4', 'revolute', (0.0, 0.035058, -0.1945), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (17, 'right_arm_link4', 'right_arm_link5', 'revolute', (0.0, -0.035058, -0.095), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (18, 'right_arm_link5', 'right_arm_link6', 'revolute', (0.0, 0.028001, -0.16305), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (19, 'right_arm_link6', 'right_arm_link7', 'revolute', (0.0295, -0.028001, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    (-1, 'right_arm_link7', 'right_gripper_link', 'fixed', (-0.0295, 0.0, -0.16065), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), (20, 'right_gripper_link', 'right_gripper_finger_link1', 'prismatic', (-8.8709e-05, 0.013453, -0.03689), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (21, 'right_gripper_link', 'right_gripper_finger_link2', 'prismatic', (8.9046e-05, -0.013453, -0.03689), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
)
LIMITS_LEFT = ((-1.1344999074935913, -2.7924997806549072, -1.8325998783111572, -3.054299831390381, -4.4506001472473145, -0.1745000034570694, -2.3561956882476807, -2.0943996906280518, -2.3561956882476807, -1.0471980571746826, -1.5707999467849731), (1.8325998783111572, 2.5306997299194336, 1.5707999467849731, 3.054299831390381, 1.309000015258789, 3.1415998935699463, 2.3561956882476807, 0.3490999639034271, 2.3561956882476807, 1.0471980571746826, 1.5707999467849731))


TRUNK_DOFS = (0, 1, 2, 3)
ARM_DOFS = {'left': tuple(range(4, 11)), 'right': tuple(range(13, 20))}
FINGER_DOFS = {'left': (11, 12), 'right': (20, 21)}
GRIPPER_LINK = {'left': 'left_gripper_link', 'right': 'right_gripper_link'}
TIP_LINK = {'left': 'left_gripper_finger_link2', 'right': 'right_gripper_finger_link2'}
TIP_POINT = (-8.915719809010625e-05, 0.015747327357530594, -0.03796372190117836)  # fingertip of finger link 2 in its own frame
PALM_POINT = (0., 0., -.060)  # grasp centre in the gripper link frame
FINGER_MID_POINT = (0., 0., -.075)  # midpoint between the two fingertips (fingers at their tip depth), gripper link frame
GRIPPER_OPEN, GRIPPER_CLOSED = 1., -1.  # native gripper command: +1 fingers 0.05 m apart each, -1 closed; linear in between
GRIPPER_ACTION = {'left': 14, 'right': 22}
ARM_ACTION = {'left': slice(7, 14), 'right': slice(15, 22)}
TRUNK_ACTION = slice(3, 7)
TRUNK_YAW_LIMIT = .75  # torso_joint4 is kept inside this band by the servo (full range is +-3.05)
REST_LEFT = (.5, -.65, -.5, 0., -1.4, .35, .2, -.7, 0., .3, 0.)  # trunk 4 + arm 7
REST_RIGHT = (.5, -.65, -.5, 0., -1.4, -.35, -.2, -.7, 0., .3, 0.)  # mirrored in the arm joints 2, 3, 5, 7


def rest(side: str) -> np.ndarray:
    return np.asarray(REST_LEFT if side == 'left' else REST_RIGHT, float)


def arm_limits(side: str) -> np.ndarray:
    """(2, 11) lower and upper limits of trunk + the arm of `side` (the right arm mirrors joint 2 of the left)."""
    lim = np.asarray(LIMITS_LEFT, float).copy()
    if side == 'right':
        lim[:, 5] = [-3.1415998935699463, 0.1745000034570694]
    lim[:, 3] = [-TRUNK_YAW_LIMIT, TRUNK_YAW_LIMIT]
    return lim


def q_from_proprio(s: np.ndarray) -> np.ndarray:
    """q (22) from the native 61D proprio: trunk 53:57, left arm 3:10, left fingers 24:26, right arm 28:35, right fingers 49:51."""
    return np.r_[s[53:57], s[3:10], s[24:26], s[28:35], s[49:51]].astype(float)


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


class Kinematics:
    """FK and geometric Jacobians over the JOINTS table (base_link root)."""

    def __init__(self) -> None:
        self.joints = []
        for dof, parent, child, kind, xyz, rpy, axis in JOINTS:
            t = np.eye(4)
            t[:3, 3] = xyz
            t[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
            self.joints.append((dof, parent, child, kind, t, np.asarray(axis, float)))

    def forward(self, q: np.ndarray):
        poses, axes, anc = {'base_link': np.eye(4)}, {}, {'base_link': []}
        for dof, parent, child, kind, origin, axis in self.joints:
            t = poses[parent] @ origin
            motion = np.eye(4)
            v = q[dof] if dof >= 0 else 0.
            if kind == 'revolute':
                k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0.]])
                motion[:3, :3] = np.eye(3) + math.sin(v) * k + (1 - math.cos(v)) * (k @ k)
            elif kind == 'prismatic':
                motion[:3, 3] = axis * v
            axes[dof] = (t[:3, 3], t[:3, :3] @ axis, kind)
            poses[child] = t @ motion
            anc[child] = anc[parent] + ([dof] if dof >= 0 else [])
        return poses, axes, anc

    def jacobian(self, link: str, q: np.ndarray, point: Optional[np.ndarray] = None, cache=None) -> np.ndarray:
        poses, axes, anc = cache or self.forward(q)
        t = poses[link]
        xyz = t[:3, 3] if point is None else t[:3, 3] + t[:3, :3] @ point
        jac = np.zeros((6, len(q)))
        for dof in anc[link]:
            pos, axis, kind = axes[dof]
            if kind == 'prismatic':
                jac[:3, dof] = axis
            else:
                jac[:3, dof] = np.cross(axis, xyz - pos)
                jac[3:, dof] = axis
        return jac


def solve_lsq(fun, x: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_nfev: int = 90) -> np.ndarray:
    """Levenberg-Marquardt on x = mid + half * sin(u): the bounds hold by construction, so a start on a joint limit can still move.
    Forward-difference Jacobian."""
    mid, half = (hi + lo) / 2, (hi - lo) / 2
    u = np.arcsin(np.clip((x - mid) / half, -.98, .98))
    g = lambda v: fun(mid + half * np.sin(v))  # noqa: E731
    r = g(u)
    cost, lam = r @ r, 1e-2
    for _ in range(max_nfev):
        J = np.empty((len(r), len(u)))
        for k in range(len(u)):
            uk = u.copy()
            uk[k] += 1e-6
            J[:, k] = (g(uk) - r) / 1e-6
        A, grad = J.T @ J, J.T @ r
        while lam < 1e12:
            un = u - np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-3 * np.diag(A).max() + 1e-9), grad)
            rn = g(un)
            if rn @ rn < cost:
                break
            lam *= 4
        else:
            break
        done = cost - rn @ rn < 1e-12 * cost
        u, r, cost, lam = un, rn, rn @ rn, max(lam / 3, 1e-9)
        if done:
            break
    return mid + half * np.sin(u)


_KIN: Optional[Kinematics] = None


def kinematics() -> Kinematics:
    global _KIN
    if _KIN is None:
        _KIN = Kinematics()
    return _KIN


class ArmServo:
    """Servo of the trunk (4) and one arm (7) towards a pose of `link` (+ `point` in its frame) in the robot frame, with bounded joint steps.

    `solve` finds a joint target by damped least squares from several starts and keeps the one closest to the current joints; `step` then
    walks to it (at most `max_step` rad per control per joint) and writes the absolute targets into the native action; after the target is
    reached it falls back to resolved-rate control, which tracks a moving goal.
    """

    def __init__(self, side: str = 'left', link: Optional[str] = None, point: Optional[np.ndarray] = None, max_step: float = .016) -> None:
        self.side, self.kin, self.max_step = side, kinematics(), max_step
        self.dofs = np.array(TRUNK_DOFS + ARM_DOFS[side])
        self.limits = arm_limits(side)
        self.link = link or GRIPPER_LINK[side]
        self.point = np.asarray(PALM_POINT if point is None else point, float)
        self.target: Optional[np.ndarray] = None
        self.plan_failed = False   # the last plan=True call found no joint target and fell back to resolved-rate control

    def use_tip(self) -> None:
        self.link, self.point, self.target = TIP_LINK[self.side], np.asarray(TIP_POINT, float), None

    def use_fingers_mid(self) -> None:
        """Servo the midpoint between the fingertips of the gripper link (for a two-finger contact)."""
        self.link, self.point, self.target = GRIPPER_LINK[self.side], np.asarray(FINGER_MID_POINT, float), None

    def use_palm(self) -> None:
        self.link, self.point, self.target = GRIPPER_LINK[self.side], np.asarray(PALM_POINT, float), None

    def pose(self, q: np.ndarray) -> np.ndarray:
        t = self.kin.forward(q)[0][self.link].copy()
        t[:3, 3] += t[:3, :3] @ self.point
        return t

    def solve(self, q: np.ndarray, goal: np.ndarray, rotation: np.ndarray, pos_tol: float = .015, ori_tol: float = .12,
              ori_weight: float = 1.0) -> Optional[np.ndarray]:
        lo, hi = self.limits
        rst = rest(self.side)
        x0 = q[self.dofs]
        sign = 1. if self.side == 'left' else -1.
        starts = [x0, rst, rst + np.array([0, 0, 0, 0, 0, .5 * sign, -.8 * sign, 0, 0, 0, 0])]
        goal = np.asarray(goal, float)

        def residual(x):
            trial = q.copy()
            trial[self.dofs] = x
            t = self.pose(trial)
            return np.r_[5 * (t[:3, 3] - goal), ori_weight * R.from_matrix(rotation @ t[:3, :3].T).as_rotvec(), .008 * (x - rst)]

        best = None
        for start in starts:
            x = solve_lsq(residual, np.clip(start, lo + 1e-5, hi - 1e-5), lo, hi, 90)
            v = residual(x)
            if np.linalg.norm(v[:3]) / 5 < pos_tol and np.linalg.norm(v[3:6]) / max(ori_weight, 1e-9) < ori_tol:
                cost = float(np.linalg.norm(x - x0))
                if best is None or cost < best[0]:
                    best = (cost, x)
        return None if best is None else best[1]

    def errors(self, q: np.ndarray, goal: np.ndarray, rotation: np.ndarray) -> Tuple[float, float]:
        t = self.pose(q)
        return float(np.linalg.norm(goal - t[:3, 3])), float(np.linalg.norm(R.from_matrix(rotation @ t[:3, :3].T).as_rotvec()))

    def step(self, action: np.ndarray, q: np.ndarray, goal: np.ndarray, rotation: np.ndarray, plan: bool = False) -> Tuple[float, float]:
        """One control towards (goal, rotation). plan=True solves a joint target first when none is active. Returns (position error m,
        rotation error rad) of the current pose."""
        goal = np.asarray(goal, float)
        cache = self.kin.forward(q)
        t = cache[0][self.link]
        point = t[:3, 3] + t[:3, :3] @ self.point
        err = goal - point
        ori = R.from_matrix(rotation @ t[:3, :3].T).as_rotvec()
        if plan and self.target is None:
            self.target = self.solve(q, goal, rotation)
            self.plan_failed = self.target is None
        qa = q[self.dofs]
        if self.target is not None:
            delta = self.target - qa
            dq = np.clip(delta, -self.max_step, self.max_step)
            if np.max(np.abs(delta)) < .035:
                self.target = None
        else:
            J = self.kin.jacobian(self.link, q, self.point, cache)[:, self.dofs]
            d = np.r_[np.clip(err * .25, -.012, .012), np.clip(ori * .2, -.03, .03)]
            dq = np.clip(J.T @ np.linalg.solve(J @ J.T + .002 * np.eye(6), d), -self.max_step, self.max_step)
        command = np.clip(qa + dq, self.limits[0] + 1e-5, self.limits[1] - 1e-5)
        action[TRUNK_ACTION] = command[:4]
        action[ARM_ACTION[self.side]] = command[4:]
        return float(np.linalg.norm(err)), float(np.linalg.norm(ori))

    def hold(self, action: np.ndarray, q: np.ndarray) -> None:
        """Keep the trunk and this arm where they are."""
        action[TRUNK_ACTION] = q[list(TRUNK_DOFS)]
        action[ARM_ACTION[self.side]] = q[list(ARM_DOFS[self.side])]
