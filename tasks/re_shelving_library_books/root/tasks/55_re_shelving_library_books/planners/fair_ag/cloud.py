"""Point clouds of the fair planner of task 55: the three cameras back-projected into the robot frame (camera calibration from cam_rel_poses),
the robot's own arms and torso removed, and the map-frame transform. Pure numpy, shared by the planner and the offline map builder."""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

from core.skills_lib import r1pro_kinematics as K
from core.skills_lib import rgbd_geometry as RG

CAMS = ('head', 'left_wrist', 'right_wrist')
SELF_RADIUS_M = 0.09         # points this near an arm link chain are the robot's own
BODY_RADIUS_M = 0.36         # points this near the base axis (horizontal) and below the head are the robot's own


def self_keep(pts: np.ndarray, q: np.ndarray, radius: float = SELF_RADIUS_M) -> np.ndarray:
    """Mask of the points that are not the robot's own arms (distance to the link chains from the joint state) or body."""
    poses = K.kinematics().forward(q)[0]
    keep = (np.hypot(pts[:, 0], pts[:, 1]) > BODY_RADIUS_M) | (pts[:, 2] > 1.75)
    for side in ('left', 'right'):
        chain = [poses[f'{side}_arm_link{i}'][:3, 3] for i in range(1, 8)]
        g = poses[f'{side}_gripper_link']
        chain += [g[:3, 3], g[:3, 3] + g[:3, :3] @ np.array([0., 0., -0.14])]
        for a, b in zip(chain[:-1], chain[1:]):
            for t in np.linspace(0., 1., 5):
                keep &= np.linalg.norm(pts - (a + (b - a) * t), axis=1) > radius
    return keep


def camera_points(depth: np.ndarray, cam: str, cam_rel: np.ndarray, q: np.ndarray, stride: int, max_range: float = 4.5) -> np.ndarray:
    """Self-filtered robot-frame points of one camera."""
    pts, _, _ = RG.backproject(depth, cam, cam_rel, stride=stride)
    pts = pts[np.linalg.norm(pts[:, :2], axis=1) < max_range]
    return pts[self_keep(pts, q)] if len(pts) else pts


def to_map(pts: np.ndarray, base: Sequence[float], base_z: float) -> np.ndarray:
    c, s = math.cos(base[2]), math.sin(base[2])
    p = np.asarray(pts, float)
    return np.stack([base[0] + c * p[:, 0] - s * p[:, 1], base[1] + s * p[:, 0] + c * p[:, 1], p[:, 2] + base_z], 1)


def obs_arrays(obs: Dict) -> Dict[str, np.ndarray]:
    """The native observation dict to plain arrays: proprio, cam_rel and the depth image of every camera."""
    out = dict(proprio=np.asarray(RG.as_numpy(obs[RG.PROPRIO_KEY]), float).reshape(-1),
               cam_rel=np.asarray(RG.as_numpy(obs[RG.CAM_POSES_KEY]), float).reshape(-1))
    for cam in CAMS:
        out[cam] = np.asarray(RG.as_numpy(obs[RG.depth_key(cam)]), np.float32)
    return out
