"""Frames of the task 20 fair planner: the odometry frame (the base at the first control is the origin, z up from the floor) and the robot
frame (base_link). A tool axis `a` is the unit direction from the wrist to the fingertips; the gripper link z axis is -a and its y axis is the
jaw (finger separation), as core.skills_lib.r1pro_kinematics (FINGER_MID_POINT is the midpoint between the fingertips)."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

UP = np.array([0., 0., 1.])
DOWN = -UP


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
    """Gripper link orientation (columns x, y, z in the odometry frame) whose fingers point along `axis` and whose jaw is `jaw` (made
    orthogonal)."""
    z = -unit(axis)
    j = np.asarray(jaw, float)
    y = j - float(j @ z) * z
    if np.linalg.norm(y) < 1e-6:
        y = np.cross(z, np.array([1., 0., 0.]))
    y = unit(y)
    return np.column_stack([np.cross(y, z), y, z])


def to_robot(base: Sequence[float], p: Sequence[float]) -> np.ndarray:
    c, s = math.cos(base[2]), math.sin(base[2])
    d = np.asarray(p, float)
    return np.array([c * (d[0] - base[0]) + s * (d[1] - base[1]), -s * (d[0] - base[0]) + c * (d[1] - base[1]), d[2]])


def to_odom(base: Sequence[float], p: Sequence[float]) -> np.ndarray:
    c, s = math.cos(base[2]), math.sin(base[2])
    p = np.asarray(p, float)
    return np.array([base[0] + c * p[0] - s * p[1], base[1] + s * p[0] + c * p[1], p[2]])


def rotation_to_robot(base: Sequence[float], rotation: np.ndarray) -> np.ndarray:
    return rot_yaw(-base[2]) @ rotation


def pose_T(rotation: np.ndarray, point: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rotation, point
    return T


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
