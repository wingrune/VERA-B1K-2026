"""Coloured point clouds of the fair planner of task 20: the three cameras back-projected into the robot frame, the robot's own body cut out
by its kinematics (colour cannot tell the hand from the scene: the cut is geometric). Adapted from the fair planner of task 42."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.skills_lib import r1pro_kinematics as K
from core.skills_lib import rgbd_geometry as G

from .. import params as P

CAMS = ('head', 'left_wrist', 'right_wrist')
FINGER_LINKS = ('gripper_finger_link1', 'gripper_finger_link2')


def body_segments(q: np.ndarray):
    """(A, B, radius) segments of the torso, both arms (shoulder to the fingertips) in the robot frame, from the joint state."""
    poses = K.kinematics().forward(q)[0]
    chain = [poses[f'torso_link{i}'][:3, 3] for i in range(1, 5)] + [poses['zed_link'][:3, 3]]
    segs = [(a, b, P.TORSO_RADIUS_M) for a, b in zip(chain[:-1], chain[1:])]
    for side in ('left', 'right'):
        chain = [poses[f'{side}_arm_link{i}'][:3, 3] for i in range(1, 8)]
        g = poses[f'{side}_gripper_link']
        chain += [g[:3, 3], g[:3, 3] + g[:3, :3] @ np.array([0., 0., -0.05])]
        segs += [(a, b, P.SELF_RADIUS_M) for a, b in zip(chain[:-1], chain[1:])]
        for f in FINGER_LINKS:
            t = poses[f'{side}_{f}']
            tip = t[:3, 3] + t[:3, :3] @ np.asarray(K.TIP_POINT, float)
            segs.append((g[:3, 3] + g[:3, :3] @ np.array([0., 0., -0.05]), tip, P.FINGER_RADIUS_M))
    return segs


def self_keep(pts: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Mask of the points that are not the robot (torso, arm and finger segments, the base as a low cylinder)."""
    r = np.hypot(pts[:, 0], pts[:, 1])
    keep = (r > P.BASE_RADIUS_M) | (pts[:, 2] > P.BASE_TOP_M) | (pts[:, 2] < 0.01)
    for a, b, rad in body_segments(q):
        ab = b - a
        t = np.clip((pts - a) @ ab / max(float(ab @ ab), 1e-12), 0., 1.)
        keep &= np.linalg.norm(pts - (a + t[:, None] * ab), axis=1) > rad
    return keep


def camera(obs: Dict[str, Any], cam: str, q: np.ndarray, cam_rel: np.ndarray, stride: int, held: Optional[Tuple[np.ndarray, float]] = None,
           max_range: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points (N, 3) robot frame, pixels (N, 2) row/col, colours (N, 3) uint8) of one camera without the robot's body. `held` (centre,
    radius, robot frame) also cuts the object in a hand."""
    depth = G.as_numpy(obs[G.depth_key(cam)])
    rgb = G.as_numpy(obs[G.rgb_key(cam)])[..., :3]
    pts, pix, col = G.backproject(depth, cam, cam_rel, stride=stride, rgb=rgb)
    keep = self_keep(pts, q) & (np.linalg.norm(pts[:, :2], axis=1) < max_range)
    if held is not None:
        keep &= np.linalg.norm(pts - held[0], axis=1) > held[1]
    return pts[keep], pix[keep], col[keep]
