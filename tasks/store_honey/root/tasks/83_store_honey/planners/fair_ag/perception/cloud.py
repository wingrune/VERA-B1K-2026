"""Point clouds of the fair planner of task 83: the three cameras back-projected in the robot frame, the robot's own body removed, the base
tilt and lift taken out of the heights."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from core.skills_lib import r1pro_kinematics as K
from core.skills_lib import rgbd_geometry as G

from .. import params as P

CAMS = ('head', 'left_wrist', 'right_wrist')


def rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def self_keep(pts: np.ndarray, q: np.ndarray, radius: float = P.SELF_RADIUS_M) -> np.ndarray:
    """Mask of the points that are not the robot's own arms (distance to the arm link chains, from the joint state) or torso (horizontal radius)."""
    poses = K.kinematics().forward(q)[0]
    keep = np.hypot(pts[:, 0], pts[:, 1]) > P.BODY_RADIUS_M
    for side in ('left', 'right'):
        chain = [poses[f'{side}_arm_link{i}'][:3, 3] for i in range(1, 8)]
        g = poses[f'{side}_gripper_link']
        chain += [g[:3, 3], g[:3, 3] + g[:3, :3] @ np.array([0., 0., -0.14])]
        for a, b in zip(chain[:-1], chain[1:]):
            for t in np.linspace(0., 1., 5):
                keep &= np.linalg.norm(pts - (a + (b - a) * t), axis=1) > radius
    return keep


class CloudMixin:
    def _clouds(self, obs: Dict[str, Any], q: np.ndarray, cam_rel: np.ndarray) -> Dict[str, np.ndarray]:
        """Self-filtered points of the three cameras in the robot frame."""
        out = {}
        for cam in CAMS:
            pts, _, _ = G.backproject(G.as_numpy(obs[G.depth_key(cam)]), cam, cam_rel, stride=P.CLOUD_STRIDE)
            keep = self_keep(pts, q) & (np.linalg.norm(pts[:, :2], axis=1) < P.CLOUD_RANGE_M)
            out[cam] = pts[keep]
        return out

    def _lift(self, head_robot: np.ndarray) -> None:
        """The floor in the head cloud sits at z = -lift when the base stands on something it drove onto: the low end of the point heights in a
        ring 1.2-3.5 m around the base."""
        r = np.hypot(head_robot[:, 0], head_robot[:, 1])
        z = head_robot[(r > 1.2) & (r < 3.5) & (head_robot[:, 2] > P.LIFT_Z_MIN_M) & (head_robot[:, 2] < 0.1), 2]
        if len(z) > 400:
            self.lift = 0.6 * self.lift + 0.4 * max(0., -(float(np.percentile(z, 8)) + 0.005))

    def _level(self, clouds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Remove the tilt of the base against the floor from the heights of all points (the floor plane is fitted in the head cloud)."""
        fit = G.floor_tilt(clouds['head'])
        if fit is not None:
            a, b, _ = fit
            self.tilt = (0.7 * self.tilt[0] + 0.3 * a, 0.7 * self.tilt[1] + 0.3 * b)
        a, b = self.tilt
        out = {}
        for cam, pts in clouds.items():
            pts = pts.copy()
            pts[:, 2] -= a * pts[:, 0] + b * pts[:, 1]
            out[cam] = pts
        return out
