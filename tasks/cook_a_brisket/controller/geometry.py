"""Cameras from the allowed observation dict, and depth -> points in the robot base frame.
Intrinsics and the OpenGL -> CV flip follow the colleagues' radio controller (perception.make_cameras)."""
import numpy as np
from scipy.spatial.transform import Rotation as R

ROLES = [("left_wrist", "left_realsense_link", 388.6639 / 480), ("right_wrist", "right_realsense_link", 388.6639 / 480), ("head", "zed_link", 306 / 720)]


def cameras(obs):                                                                             # role -> rgb [H, W, 3], depth [H, W], K [3, 3], R [3, 3], t [3] (base frame)
    rel = np.asarray(obs["robot_r1::cam_rel_poses"]).reshape(3, 7)
    cams = {}
    for i, (role, link, focal) in enumerate(ROLES):
        prefix = f"robot_r1::robot_r1:{link}:Camera:0::"
        rgb = obs[prefix + "rgb"][..., :3]
        h, w = rgb.shape[:2]
        K = np.array([[focal * w, 0, w / 2], [0, focal * w, h / 2], [0, 0, 1.0]])
        cams[role] = {"rgb": rgb, "depth": obs.get(prefix + "depth_linear"), "K": K, "R": R.from_quat(rel[i, 3:]).as_matrix() @ np.diag([1, -1, -1]), "t": rel[i, :3]}
    return cams


def pointcloud(cam, mask=None, stride=1, max_range=5.0):                                     # [n, 3] base-frame points, [n, 3] colours
    d = cam["depth"]
    valid = np.isfinite(d) & (d > 0.06) & (d < max_range)
    if mask is not None:
        valid &= mask
    y, x = np.where(valid)
    y, x = y[::stride], x[::stride]
    z = d[y, x]
    p = np.c_[(x + 0.5 - cam["K"][0, 2]) / cam["K"][0, 0] * z, (y + 0.5 - cam["K"][1, 2]) / cam["K"][1, 1] * z, z]
    return p @ cam["R"].T + cam["t"], cam["rgb"][y, x]


def project(cam, points):                                                                     # base-frame [n, 3] -> pixel [n, 2], depth [n]
    c = (np.asarray(points) - cam["t"]) @ cam["R"]
    uv = c @ cam["K"].T
    return uv[:, :2] / np.maximum(uv[:, 2:], 1e-6), c[:, 2]


def resized(cam, size):                                                                       # same camera at size x size pixels (nearest depth)
    import cv2
    h = cam["rgb"].shape[0]
    out = dict(cam)
    out["rgb"] = cv2.resize(cam["rgb"], (size, size), interpolation=cv2.INTER_AREA)
    out["depth"] = cv2.resize(cam["depth"], (size, size), interpolation=cv2.INTER_NEAREST)
    out["K"] = cam["K"].copy()
    out["K"][:2] *= size / h
    return out


def ground_tilt(cam, rng=np.random.default_rng(0)):                                           # degrees between the base z axis and the visible ground normal (None if no ground plane)
    p, _ = pointcloud(cam, stride=6, max_range=4.0)
    p = p[(np.linalg.norm(p[:, :2], axis=1) > 0.6) & (p[:, 2] > -0.4) & (p[:, 2] < 0.4)]
    if len(p) < 200:
        return None
    p = p[rng.choice(len(p), min(len(p), 1500), replace=False)]
    best, normal = 0, None
    for _ in range(60):                                                                       # RANSAC plane
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n = n / np.linalg.norm(n)
        if abs(n[2]) < 0.85:                                                                  # walls and fences are not the ground (full_v18_313 logged 90 deg tilts)
            continue
        inliers = int(np.sum(np.abs((p - a) @ n) < 0.02))
        if inliers > best:
            best, normal = inliers, n
    if normal is None or best < 0.3 * len(p):
        return None
    return float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
