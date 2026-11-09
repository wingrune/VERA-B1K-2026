"""Torso + both arms in one damped least-squares step: move the right gripper to a goal while the left gripper holds its
pose (it holds the basket on the lawn). One shared torso would otherwise drag the held basket (egg2_301/302)."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from kinematics import Kinematics, from_proprio


class DualServo:
    def __init__(self, assets):
        spec = json.loads((Path(assets) / "robot_model.json").read_text())
        self.names = spec["dof_names"]
        self.kin = Kinematics(Path(assets) / "r1pro.urdf", self.names)
        group = lambda name: [self.names.index(n) for c in spec["controllers"] if c["name"] == name for n in c["dof_names"]]
        self.torso, self.left, self.right = group("trunk"), group("arm_left"), group("arm_right")
        self.indices = self.torso + self.left + self.right                                    # 18
        self.limits = np.asarray(spec["limits"])[:, self.indices]
        self.limits[:, 3] = [-0.75, 0.75]
        self.point = np.array([0.0, 0.0, -0.060])

    def q(self, proprio):
        return from_proprio(proprio, self.names)

    def pose(self, q, arm):
        t = self.kin.forward(q)[0][arm + "_gripper_link"].copy()
        t[:3, 3] += t[:3, :3] @ self.point
        return t

    def solve(self, q, goal, rotation, hold_pos, hold_rot):                                   # joint configuration [18] reaching both, or None
        from scipy.optimize import least_squares
        x0 = q[self.indices]
        right_seeds = [x0[11:], np.r_[-1.4, -0.35, -0.2, -0.7, 0.0, -0.3, 0.0], np.r_[-1.4, -0.35, -0.2, -0.7, 0.0, 0.2, -0.8]]

        def residual(x):
            trial = q.copy()
            trial[self.indices] = x
            r, l = self.pose(trial, "right"), self.pose(trial, "left")
            return np.r_[5 * (r[:3, 3] - goal), R.from_matrix(rotation @ r[:3, :3].T).as_rotvec(), 10 * (l[:3, 3] - hold_pos), 2 * R.from_matrix(hold_rot @ l[:3, :3].T).as_rotvec(), 0.008 * (x - x0)]
        best = None
        for seed in right_seeds:
            x = np.clip(np.r_[x0[:11], seed], self.limits[0] + 1e-5, self.limits[1] - 1e-5)
            res = least_squares(residual, x, bounds=(self.limits[0], self.limits[1]), max_nfev=120)
            v = residual(res.x)
            ok = np.linalg.norm(v[:3]) / 5 < 0.015 and np.linalg.norm(v[3:6]) < 0.12 and np.linalg.norm(v[6:9]) / 10 < 0.01 and np.linalg.norm(v[9:12]) / 2 < 0.06
            if ok and (best is None or np.linalg.norm(res.x - x0) < np.linalg.norm(best - x0)):
                best = res.x
        return best

    def move_to(self, action, q, target, rate=0.016):                                         # synchronized joint-space step toward target [18]; True when there
        delta = target - q[self.indices]
        dq = delta * min(1.0, rate / max(np.max(np.abs(delta)), 1e-9))
        command = np.clip(q[self.indices] + dq, self.limits[0] + 1e-5, self.limits[1] - 1e-5)
        action[3:7], action[7:14], action[15:22] = command[:4], command[4:11], command[11:18]
        return bool(np.max(np.abs(delta)) < 0.035)

    def step(self, action, q, goal, rotation, hold_pos, hold_rot):                            # right gripper -> (goal, rotation); left gripper kept at (hold_pos, hold_rot)
        cache = self.kin.forward(q)
        rows, deltas, err = [], [], None
        for arm, target, rot, gain in (("right", goal, rotation, 1.0), ("left", hold_pos, hold_rot, 2.0)):
            t = cache[0][arm + "_gripper_link"]
            pos = t[:3, 3] + t[:3, :3] @ self.point
            e = target - pos
            o = R.from_matrix(rot @ t[:3, :3].T).as_rotvec()
            rows.append(self.kin.jacobian(arm + "_gripper_link", q, self.point, cache)[:, self.indices])
            deltas.append(np.r_[np.clip(e * 0.25 * gain, -0.012, 0.012), np.clip(o * 0.2 * gain, -0.03, 0.03)])
            if arm == "right":
                err = (float(np.linalg.norm(e)), float(np.linalg.norm(o)))
        J, delta = np.vstack(rows), np.concatenate(deltas)                                     # [12, 18], [12]
        dq = np.clip(J.T @ np.linalg.solve(J @ J.T + 0.002 * np.eye(12), delta), -0.016, 0.016)
        command = np.clip(q[self.indices] + dq, self.limits[0] + 1e-5, self.limits[1] - 1e-5)
        action[3:7], action[7:14], action[15:22] = command[:4], command[4:11], command[11:18]
        return err
