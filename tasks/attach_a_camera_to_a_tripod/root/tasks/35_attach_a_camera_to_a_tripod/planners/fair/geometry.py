"""Geometry of the task 35 fair planner: camera pinch options from its model (collision vertices) in the estimated pose, the mount goal from
the attachment frames of the models, frame helpers. Frames: the odometry frame of the planner plays the role of the world.

Conventions: world frame, z up. A tool axis `a` is the unit direction from the wrist to the fingertips; the gripper link z axis is -a and
its y axis is the jaw (finger separation), as in the task 98 oracle (core.skills_lib.r1pro_kinematics: the finger midpoint is
FINGER_MID_POINT on the gripper link). The camera is pinched from above across a narrow part (the lens barrel first, as the colleagues'
controller and the human demos); the assisted grasp then holds it by a joint at the contact.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

UP = np.array([0., 0., 1.])
END_FRAC = 0.28          # the end regions of the hull along an axis (the colleagues' barrel: x above 72% of the extent)
MAX_WIDTH_M = 0.085      # pinch only where the slab is narrower than this (the open gripper spans about 0.10 m)
MIN_WIDTH_M = 0.02
FLOOR_CLEAR_M = 0.022    # finger midpoint at least this far above the floor
GRASP_BACK_M = 0.12      # pre-grasp point: backed off along the tool axis
SITE_COST = {'barrel': 0.0, 'centre': 0.6, 'rear': 1.2}
TILT_COST = 0.4          # per rad of tool tilt from straight down


def unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def horiz(v: Sequence[float]) -> np.ndarray:
    v = np.array(v, float)
    v[2] = 0.
    return unit(v)


def rot_yaw(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def jaw_rotation(axis: Sequence[float], jaw: Sequence[float]) -> np.ndarray:
    """Gripper rotation for a tool axis and a jaw.
    Gripper link orientation (columns x, y, z in the world) whose fingers point along `axis` and whose jaw is `jaw` (made orthogonal)."""
    z = -unit(axis)
    j = np.asarray(jaw, float)
    y = j - float(j @ z) * z
    if np.linalg.norm(y) < 1e-6:
        y = np.cross(z, np.array([1., 0., 0.]))
    y = unit(y)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def to_robot(base: Sequence[float], base_z: float, p_world: Sequence[float]) -> np.ndarray:
    """World point to the base_link frame."""
    c, s = math.cos(base[2]), math.sin(base[2])
    d = np.asarray(p_world, float)
    return np.array([c * (d[0] - base[0]) + s * (d[1] - base[1]), -s * (d[0] - base[0]) + c * (d[1] - base[1]), d[2] - base_z])


def rotation_to_robot(base: Sequence[float], rotation_world: np.ndarray) -> np.ndarray:
    return rot_yaw(-base[2]) @ rotation_world


def pose_T(rotation: np.ndarray, point: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rotation, point
    return T


# ---- camera pinch -------------------------------------------------------------------------------------------------------------------------
def pinch_sites(hull: np.ndarray) -> List[Dict[str, object]]:
    """Narrow places of the camera hull for a top pinch.
    Narrow places of the camera hull (root frame) for a top pinch: for each local horizontal axis `u` (0 = x, 1 = y) a slab across it at
    the +u end (barrel when u is the long axis), the centre and the -u end; the fingers close along the other horizontal axis. Each site:
    centre (root frame, the slab mid-width and mid-height), width, closing axis index, along-u index, name."""
    hull = np.asarray(hull, float)
    lo, hi = hull.min(0), hull.max(0)
    out = []
    for u in (0, 1):
        a = 1 - u
        ext = hi[u] - lo[u]
        regions = (('barrel', hi[u] - END_FRAC * ext, hi[u]), ('centre', lo[u] + 0.4 * ext, lo[u] + 0.6 * ext), ('rear', lo[u], lo[u] + END_FRAC * ext))
        for name, r0, r1 in regions:      # vertex regions, not thin slabs: collision meshes can be sparse
            slab = hull[(hull[:, u] >= r0 - 1e-9) & (hull[:, u] <= r1 + 1e-9)]
            if len(slab) < 4:
                continue
            uc = 0.5 * (r0 + r1)
            w = float(slab[:, a].max() - slab[:, a].min())
            if not MIN_WIDTH_M <= w <= MAX_WIDTH_M:
                continue
            c = np.zeros(3)
            c[u] = uc
            c[a] = 0.5 * (slab[:, a].max() + slab[:, a].min())
            c[2] = 0.5 * (slab[:, 2].max() + slab[:, 2].min())
            out.append(dict(name=name, centre=c, width=w, close_axis=a, along_axis=u, top=float(slab[:, 2].max()),
                            cost=SITE_COST[name] + 0.5 * u + 2.0 * max(0., w - 0.06)))   # the lens barrel lies along the root x axis
    return out


def grasp_options(cam_T: np.ndarray, hull: np.ndarray, tilts: Sequence[float]) -> List[Dict[str, object]]:
    """World pinch poses of the sites.
    World pinch poses: each site x jaw sign x tilt (rad from straight down, towards -u or +u). Returns mid (finger midpoint), pre
    (backed off along the tool axis), rotation (gripper link), axis, cost and the site."""
    Rc, pc = cam_T[:3, :3], cam_T[:3, 3]
    out = []
    for site in pinch_sites(hull):
        mid = Rc @ site['centre'] + pc
        mid[2] = max(mid[2], FLOOR_CLEAR_M)
        jaw = horiz(Rc[:, site['close_axis']])
        along = horiz(Rc[:, site['along_axis']])
        for sign in (1., -1.):
            for tilt in tilts:
                for lean in ((0.,) if tilt == 0. else (1., -1.)):
                    axis = unit(-math.cos(tilt) * UP + math.sin(tilt) * lean * along)
                    rot = jaw_rotation(axis, sign * jaw)
                    out.append(dict(site=site['name'], width=site['width'], sign=sign, tilt=tilt, lean=lean, axis=axis, mid=mid.copy(),
                                    pre=mid - axis * GRASP_BACK_M, rotation=rot, cost=float(site['cost']) + TILT_COST * tilt))
    return out


# ---- mount -----------------------------------------------------------------------------------------------------------------------------------
def mount_goal(f_T: np.ndarray, m_in_cam: np.ndarray, mid_in_cam: np.ndarray) -> np.ndarray:
    """Hand pose that puts M on F.
    Finger-midpoint pose (4x4, gripper rotation) that puts the camera's M frame on the tripod's F frame, given the M frame and the held
    finger-midpoint frame in the camera root frame (both constant while held)."""
    cam_goal = f_T @ np.linalg.inv(m_in_cam)
    return cam_goal @ mid_in_cam


def raised(T: np.ndarray, dz: float) -> np.ndarray:
    out = T.copy()
    out[2, 3] += dz
    return out


def yaw_of(T: np.ndarray) -> float:
    return math.atan2(T[1, 0], T[0, 0])


def planar_T(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    return pose_T(rot_yaw(yaw), np.array([x, y, z]))
