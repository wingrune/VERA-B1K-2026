"""RGB-D geometry for planners that see only the native observation: back-projection of the three cameras, base odometry from the proprio
velocity, planar fits. Pure numpy and scipy, no simulator.

Observation layout (native contract): `robot_r1::cam_rel_poses` is 21D, the poses [x, y, z, qx, qy, qz, qw] of left wrist, right wrist and head
camera in the robot frame (base_link), in the USD camera convention (-Z forward, +Y up). Depth is planar (distance to the image plane), +inf for no
hit. Intrinsics are square pinholes with the principal point at the image centre. The robot pose in the world is not observed:
`Odometry` integrates the measured base velocity (proprio 0:3, robot frame) into a pose in the frame of the first control.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

CAM_ORDER = ('left_wrist', 'right_wrist', 'head')
CAM_SIZE_FX: Dict[str, Tuple[int, float]] = {'left_wrist': (480, 388.6639), 'right_wrist': (480, 388.6639), 'head': (720, 306.0)}
CAM_LINK = {'left_wrist': 'left_realsense_link', 'right_wrist': 'right_realsense_link', 'head': 'zed_link'}
PROPRIO_KEY = 'robot_r1::proprio'
CAM_POSES_KEY = 'robot_r1::cam_rel_poses'
DEPTH_VALID_M = (0.06, 5.0)


def as_numpy(x) -> np.ndarray:
    """Observation entry (torch tensor with batch 1, or array) -> numpy without the batch dim."""
    a = x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
    return a[0] if a.shape[:1] == (1,) and a.ndim > 1 else a


def rgb_key(cam: str) -> str:
    return f'robot_r1::robot_r1:{CAM_LINK[cam]}:Camera:0::rgb'


def depth_key(cam: str) -> str:
    return f'robot_r1::robot_r1:{CAM_LINK[cam]}:Camera:0::depth_linear'


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def cam_pose(cam_rel: np.ndarray, cam: str) -> np.ndarray:
    """4x4 camera (OpenGL axes) to robot frame transform from the 21D cam_rel_poses."""
    p = np.asarray(cam_rel, float).reshape(3, 7)[CAM_ORDER.index(cam)]
    t = np.eye(4)
    t[:3, :3] = quat_to_matrix(p[3:])
    t[:3, 3] = p[:3]
    return t


def backproject(depth: np.ndarray, cam: str, cam_rel: np.ndarray, stride: int = 4, valid: Tuple[float, float] = DEPTH_VALID_M,
                rgb: Optional[np.ndarray] = None, region: Optional[Tuple[int, int, int, int]] = None):
    """Points of one camera in the robot frame. Returns (points [N,3], pixels [N,2] as (row, col), colours [N,3] uint8 or None).
    `region` is (row0, row1, col0, col1) in pixels; `stride` subsamples the grid."""
    size, fx = CAM_SIZE_FX[cam]
    r0, r1, c0, c1 = region or (0, size, 0, size)
    rows = np.arange(r0, r1, stride)
    cols = np.arange(c0, c1, stride)
    d = np.asarray(depth, np.float64)[np.ix_(rows, cols)]
    ok = np.isfinite(d) & (d >= valid[0]) & (d <= valid[1])
    vv, uu = np.meshgrid(rows, cols, indexing='ij')
    d, vv, uu = d[ok], vv[ok], uu[ok]
    x = (uu + .5 - size / 2) / fx * d
    y = -(vv + .5 - size / 2) / fx * d
    cam_pts = np.stack([x, y, -d], 1)
    t = cam_pose(cam_rel, cam)
    pts = cam_pts @ t[:3, :3].T + t[:3, 3]
    colours = None if rgb is None else np.asarray(rgb)[vv, uu][:, :3]
    return pts, np.stack([vv, uu], 1), colours


def pixels_to_robot(depth: np.ndarray, cam: str, cam_rel: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                    valid: Tuple[float, float] = DEPTH_VALID_M) -> np.ndarray:
    """Robot-frame points of given pixels (row, col arrays); pixels without a valid depth are dropped."""
    size, fx = CAM_SIZE_FX[cam]
    d = np.asarray(depth, np.float64)[rows, cols]
    ok = np.isfinite(d) & (d >= valid[0]) & (d <= valid[1])
    d, vv, uu = d[ok], np.asarray(rows)[ok], np.asarray(cols)[ok]
    cam_pts = np.stack([(uu + .5 - size / 2) / fx * d, -(vv + .5 - size / 2) / fx * d, -d], 1)
    t = cam_pose(cam_rel, cam)
    return cam_pts @ t[:3, :3].T + t[:3, 3]


def project(points_robot: np.ndarray, cam: str, cam_rel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Robot-frame points to (row, col) pixels and planar depth of a camera (depth <= 0 is behind it)."""
    size, fx = CAM_SIZE_FX[cam]
    t = cam_pose(cam_rel, cam)
    c = (np.asarray(points_robot, float) - t[:3, 3]) @ t[:3, :3]
    d = -c[:, 2]
    safe = np.where(d > 1e-6, d, 1e-6)
    col = c[:, 0] / safe * fx + size / 2 - .5
    row = -c[:, 1] / safe * fx + size / 2 - .5
    return np.stack([row, col], 1), d


class Odometry:
    """Planar base pose (x, y, yaw) in the frame of the base at the first update, from the measured base velocity (robot frame) at 30 Hz."""

    def __init__(self, dt: float = 1 / 30) -> None:
        self.dt = dt
        self.pose = np.zeros(3)
        self.scale = 1.0          # measured velocities are multiplied by this (calibrated against the map by the caller)
        self.raw_turn = 0.0       # integrated unscaled heading change, for the calibration

    def update(self, base_qvel: np.ndarray) -> np.ndarray:
        self.raw_turn += float(base_qvel[2]) * self.dt
        vx, vy, wz = (self.scale * float(v) for v in base_qvel[:3])
        yaw = self.pose[2] + 0.5 * wz * self.dt
        c, s = math.cos(yaw), math.sin(yaw)
        self.pose = self.pose + np.array([(c * vx - s * vy) * self.dt, (s * vx + c * vy) * self.dt, wz * self.dt])
        return self.pose.copy()


def robot_to_odom(points: np.ndarray, base: np.ndarray, base_z: float = 0.) -> np.ndarray:
    """Robot-frame points to the odometry frame given the base pose (x, y, yaw); z stays relative to the robot frame origin height `base_z`."""
    c, s = math.cos(base[2]), math.sin(base[2])
    p = np.asarray(points, float)
    return np.stack([base[0] + c * p[:, 0] - s * p[:, 1], base[1] + s * p[:, 0] + c * p[:, 1], p[:, 2] + base_z], 1)


def odom_to_robot(points: np.ndarray, base: np.ndarray, base_z: float = 0.) -> np.ndarray:
    c, s = math.cos(base[2]), math.sin(base[2])
    p = np.asarray(points, float)
    d = p[:, :2] - base[:2]
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], p[:, 2] - base_z], 1)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def height_peak(z: np.ndarray, lo: float, hi: float, bin_m: float = 0.01, min_count: int = 50) -> Optional[float]:
    """Height of the most populated bin of z inside [lo, hi] (a horizontal surface), or None."""
    sel = z[(z >= lo) & (z <= hi)]
    if len(sel) < min_count:
        return None
    n = max(1, int(round((hi - lo) / bin_m)))
    hist, edges = np.histogram(sel, bins=n, range=(lo, hi))
    smooth = hist + np.r_[hist[1:], 0] + np.r_[0, hist[:-1]]      # mass of a bin and its neighbours
    k = int(np.argmax(smooth))
    if smooth[k] < min_count / 4:
        return None
    lo_k = max(0, k - 1)
    k = lo_k + int(np.argmax(hist[lo_k:k + 2]))                    # the fullest of the three
    near = sel[np.abs(sel - (edges[k] + edges[k + 1]) / 2) < bin_m * 1.5]
    return float(np.median(near))


def min_area_rect(xy: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Minimum-area oriented rectangle of 2D points: (centre, yaw of the first axis in [0, pi/2), extents along the (first, second) axis)."""
    xy = np.asarray(xy, float)
    best = None
    for deg in range(0, 90, 2):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        u = xy @ np.array([c, s])
        v = xy @ np.array([-s, c])
        ext = np.array([u.max() - u.min(), v.max() - v.min()])
        area = ext[0] * ext[1]
        if best is None or area < best[0]:
            best = (area, a, (u.max() + u.min()) / 2, (v.max() + v.min()) / 2, ext)
    _, a, cu, cv, ext = best
    c, s = math.cos(a), math.sin(a)
    centre = cu * np.array([c, s]) + cv * np.array([-s, c])
    return centre, a, ext


def clusters(mask: np.ndarray, min_cells: int = 4):
    """Connected components (8-neighbourhood) of a boolean grid: list of (label, cell indices [N,2]) with at least min_cells cells."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    out = []
    for k in range(1, n + 1):
        idx = np.argwhere(lab == k)
        if len(idx) >= min_cells:
            out.append((k, idx))
    return out


def floor_tilt(points: np.ndarray, near: Tuple[float, float] = (0.6, 3.5), band: float = 0.15, min_points: int = 300,
               iterations: int = 3) -> Optional[Tuple[float, float, float]]:
    """Plane z = a x + b y + c of the floor in the robot frame from the points close to ground level (|z| < band) at a horizontal range of `near`
    metres, by least squares with outlier rejection. The slope (a, b) is the tilt of the base against the floor (a robot that pitches or rolls on its
    suspension shifts the far floor by centimetres in the head cloud). None when too few floor points are in view."""
    r = np.hypot(points[:, 0], points[:, 1])
    sel = (np.abs(points[:, 2]) < band) & (r > near[0]) & (r < near[1])
    if sel.sum() < min_points:
        return None
    p = points[sel]
    a = np.c_[p[:, 0], p[:, 1], np.ones(len(p))]
    keep = np.ones(len(p), bool)
    coef = np.zeros(3)
    for _ in range(iterations):
        if keep.sum() < min_points:
            return None
        coef, *_ = np.linalg.lstsq(a[keep], p[keep, 2], rcond=None)
        res = np.abs(a @ coef - p[:, 2])
        keep = res < max(0.02, 2.5 * float(np.median(res[keep])))
    return float(coef[0]), float(coef[1]), float(coef[2])


def backproject_grid(depth: np.ndarray, cam: str, cam_rel: np.ndarray, stride: int = 4, valid: Tuple[float, float] = DEPTH_VALID_M) -> np.ndarray:
    """Robot-frame points of a subsampled depth image kept in image layout, [H, W, 3] with NaN where the depth is missing or out of range."""
    size, fx = CAM_SIZE_FX[cam]
    rows = np.arange(0, size, stride)
    cols = np.arange(0, size, stride)
    d = np.asarray(depth, np.float64)[np.ix_(rows, cols)]
    ok = np.isfinite(d) & (d >= valid[0]) & (d <= valid[1])
    d = np.where(ok, d, 0.)
    vv, uu = np.meshgrid(rows, cols, indexing='ij')
    x = (uu + .5 - size / 2) / fx * d
    y = -(vv + .5 - size / 2) / fx * d
    t = cam_pose(cam_rel, cam)
    pts = np.stack([x, y, -d], -1) @ t[:3, :3].T + t[:3, 3]
    pts[~ok] = np.nan
    return pts


def wall_heading(grid: np.ndarray, max_jump: float = 0.08, max_nz: float = 0.25, min_count: int = 400) -> Optional[Tuple[float, float]]:
    """Dominant direction of the vertical surfaces seen in a point grid [H, W, 3] (robot frame, NaN for no data), modulo a quarter turn.
    Normals come from the central differences of neighbouring points (no jump over `max_jump` metres); those of nearly vertical surfaces
    (|n_z| < max_nz) vote with the angle of their horizontal part, times 4 (rectangular rooms have walls and furniture along two perpendicular
    directions). Returns (angle in [-pi/4, pi/4) of a wall normal against the robot x axis, confidence in [0, 1]) or None."""
    dx = grid[1:-1, 2:] - grid[1:-1, :-2]
    dy = grid[2:, 1:-1] - grid[:-2, 1:-1]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1)
    ok = np.isfinite(norm) & (norm > 1e-9)
    ok &= (np.linalg.norm(dx, axis=-1) < 2 * max_jump) & (np.linalg.norm(dy, axis=-1) < 2 * max_jump)
    n = np.where(ok[..., None], n / np.where(ok, norm, 1.)[..., None], 0.)
    vertical = ok & (np.abs(n[..., 2]) < max_nz)
    if int(vertical.sum()) < min_count:
        return None
    theta = np.arctan2(n[vertical, 1], n[vertical, 0])
    z = np.exp(4j * theta).mean()
    return float(np.angle(z) / 4), float(abs(z))


def vertical_surface_samples(grid: np.ndarray, max_jump: float = 0.08, max_nz: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Points of nearly vertical surfaces in a point grid [H, W, 3] (robot frame, NaN for no data) with the unit horizontal direction of their
    normal: (points [N, 3], normals [N, 2]). The normal sign is arbitrary (it follows the image axes)."""
    dx = grid[1:-1, 2:] - grid[1:-1, :-2]
    dy = grid[2:, 1:-1] - grid[:-2, 1:-1]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1)
    ok = np.isfinite(norm) & (norm > 1e-9) & (np.linalg.norm(dx, axis=-1) < 2 * max_jump) & (np.linalg.norm(dy, axis=-1) < 2 * max_jump)
    n = n / np.where(ok, norm, 1.)[..., None]
    vertical = ok & (np.abs(n[..., 2]) < max_nz)
    h = np.linalg.norm(n[vertical][:, :2], axis=1)
    return grid[1:-1, 1:-1][vertical], n[vertical][:, :2] / np.maximum(h, 1e-9)[:, None]
