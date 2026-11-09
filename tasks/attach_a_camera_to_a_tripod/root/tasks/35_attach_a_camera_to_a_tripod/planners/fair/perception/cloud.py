"""Point clouds of the fair planner of task 35: the three cameras back-projected into the robot frame, the robot's own body cut out by its
kinematics (the native renderer draws the hand dark, so colour cannot tell the hand from the black camera: the cut is geometric), and the held
camera cut out by its predicted pose."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.skills_lib import r1pro_kinematics as K
from core.skills_lib import rgbd_geometry as G

from .. import params as P

CAMS = ('head', 'left_wrist', 'right_wrist')
FINGER_LINKS = ('gripper_finger_link1', 'gripper_finger_link2')


def body_segments(q: np.ndarray):
    """(A, B, radius) segments of both arms (shoulder to the fingertips) and of the torso in the robot frame, from the joint state."""
    poses = K.kinematics().forward(q)[0]
    segs = []
    if P.TORSO_CUT:      # the torso leans forward in the rest trunk pose: the head camera sees its front beyond BODY_RADIUS_M
        chain = [poses[f'torso_link{i}'][:3, 3] for i in range(1, 5)] + [poses['zed_link'][:3, 3]]
        segs += [(a, b, P.TORSO_RADIUS_M) for a, b in zip(chain[:-1], chain[1:])]
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


def self_keep(pts: np.ndarray, q: np.ndarray, floor_z: float) -> np.ndarray:
    """Mask of the points that are not the robot (arm and finger segments, and the base and torso as a vertical cylinder)."""
    keep = (np.hypot(pts[:, 0], pts[:, 1]) > P.BODY_RADIUS_M) | (pts[:, 2] < floor_z + 0.01)
    for a, b, r in body_segments(q):
        ab = b - a
        t = np.clip((pts - a) @ ab / max(float(ab @ ab), 1e-12), 0., 1.)
        keep &= np.linalg.norm(pts - (a + t[:, None] * ab), axis=1) > r
    return keep


def clouds(obs: Dict[str, Any], q: np.ndarray, cam_rel: np.ndarray, floor_z: float, held_centre: Optional[np.ndarray] = None
           ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """{camera: (points (N, 3) robot frame, pixels (N, 2) row/col)} without the robot's own body and without the held camera."""
    out = {}
    for cam in CAMS:
        depth = G.as_numpy(obs[G.depth_key(cam)])
        stride = P.HEAD_STRIDE if cam == 'head' else P.WRIST_STRIDE
        pts, pix, _ = G.backproject(depth, cam, cam_rel, stride=stride)
        keep = self_keep(pts, q, floor_z) & (np.hypot(pts[:, 0], pts[:, 1]) < P.CLOUD_RANGE_M)
        if held_centre is not None:
            keep &= np.linalg.norm(pts - held_centre, axis=1) > P.HELD_RADIUS_M
        out[cam] = (pts[keep], pix[keep])
    return out


def views(obs: Dict[str, Any], cam_rel: np.ndarray, base: np.ndarray) -> list:
    """Depth images with their camera poses in the odometry frame (for the see-through test of the fits)."""
    out = []
    c, s = np.cos(base[2]), np.sin(base[2])
    T_or = np.array([[c, -s, 0., base[0]], [s, c, 0., base[1]], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    for cam in CAMS:
        size, fx = G.CAM_SIZE_FX[cam]
        out.append(dict(depth=np.asarray(G.as_numpy(obs[G.depth_key(cam)]), float), T=T_or @ G.cam_pose(cam_rel, cam), fx=fx, size=size,
                        cam=cam))
    return out


def floor_height(head_pts: np.ndarray, previous: float) -> float:
    """Floor height in the robot frame: the low end of the head points 0.8-3.5 m away (the robot stands on the floor)."""
    r = np.hypot(head_pts[:, 0], head_pts[:, 1])
    z = head_pts[(r > 0.8) & (r < 3.5) & (head_pts[:, 2] > -0.3) & (head_pts[:, 2] < 0.2), 2]
    if len(z) < 300:
        return previous
    return 0.7 * previous + 0.3 * float(np.percentile(z, 5))
