"""Geometry of the task 42 fair planner (numpy only): frames, gripper rotations, grasp and place points, stand rings. Frames: the odometry frame
of the planner plays the role of the world (z up). A gripper's fingers point along its tool axis; the finger midpoint is K.FINGER_MID_POINT on
the gripper link (core.skills_lib.r1pro_kinematics). Nothing here knows an instance number or reads the simulator."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

UP = np.array([0., 0., 1.])
PRE_GAP_M = 0.10


def unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1., 0., 0.])


def horiz(v: Sequence[float]) -> np.ndarray:
    v = np.array(v, float)
    v[2] = 0.
    return unit(v)


def rot_yaw(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def tool_rotation(axis: Sequence[float]) -> np.ndarray:
    """Gripper rotation, fingers along axis."""
    z = -unit(axis)
    x = np.cross(UP, z)
    if float(np.linalg.norm(x)) < 1e-6:
        x = np.array([1., 0., 0.])
    x = unit(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def top_rotation(yaw: float) -> np.ndarray:
    """Gripper pointing down, turned by yaw.
    tool_rotation(-UP) (fingers straight down) turned by `yaw` about the vertical: the gripper x axis points along (cos, sin, 0)."""
    return rot_yaw(yaw) @ tool_rotation(-UP)


def tilt_rotation(toward: np.ndarray, radians: float) -> np.ndarray:
    direction = -UP * math.cos(radians) + horiz(toward) * math.sin(radians)
    return tool_rotation(direction)


def to_robot(base: Sequence[float], base_z: float, p_world: Sequence[float]) -> np.ndarray:
    c, s = math.cos(base[2]), math.sin(base[2])
    d = np.asarray(p_world, float)
    return np.array([c * (d[0] - base[0]) + s * (d[1] - base[1]), -s * (d[0] - base[0]) + c * (d[1] - base[1]), d[2] - base_z])


def rotation_to_robot(base: Sequence[float], rotation_world: np.ndarray) -> np.ndarray:
    return rot_yaw(-float(base[2])) @ rotation_world


def base_candidates(target_xy, radii, bearings) -> List[Tuple[np.ndarray, float]]:
    """Base poses on rings."""
    t = np.asarray(target_xy, float)[:2]
    return [(t + r * np.array([math.cos(b), math.sin(b)]), b + math.pi) for r in radii for b in bearings]


def grasp_points(centre, size, mode, approach) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pregrasp, grasp, rotation for a pinch.
    (pregrasp, grasp, gripper rotation in the world). Top pinches from above; side pinches along `approach`."""
    approach = horiz(approach)
    if mode == 'top':
        grasp = np.array([centre[0], centre[1], centre[2] + 0.15 * float(size[2])])
        return grasp + np.array([0., 0., PRE_GAP_M]), grasp, tool_rotation(-UP)
    grasp = np.asarray(centre, float).copy()
    pre = grasp - approach * (0.45 * float(max(size[0], size[1])) + PRE_GAP_M)
    return pre, grasp, tool_rotation(approach)


def xy_close(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    return float(np.linalg.norm(np.asarray(a, float)[:2] - np.asarray(b, float)[:2])) <= tol
