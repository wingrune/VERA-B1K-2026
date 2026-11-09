"""Pure geometry of the fair planner of task 83 (numpy only): frames and tool orientations.

Conventions: odometry frame, z up from the robot frame origin. A tool axis `a` is the unit direction from the wrist to the fingertips; the gripper
link z axis is -a, its y axis is the jaw (finger separation) and its x axis completes the frame."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

UP = np.array([0., 0., 1.])


def unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def rot_yaw(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def jaw_rotation(axis: Sequence[float], jaw: Sequence[float]) -> np.ndarray:
    """Gripper link orientation (columns x, y, z in the odometry frame) whose fingers point along `axis` and whose jaw is `jaw` (made orthogonal)."""
    z = -unit(axis)
    y = np.asarray(jaw, float) - float(np.asarray(jaw, float) @ z) * z
    if np.linalg.norm(y) < 1e-6:
        y = np.cross(z, np.array([1., 0., 0.]))
    y = unit(y)
    return np.column_stack([np.cross(y, z), y, z])


def to_robot(base: Sequence[float], p: Sequence[float]) -> np.ndarray:
    """Odometry-frame point to the base_link frame (heights are already robot-frame heights)."""
    c, s = math.cos(base[2]), math.sin(base[2])
    d = np.asarray(p, float)
    return np.array([c * (d[0] - base[0]) + s * (d[1] - base[1]), -s * (d[0] - base[0]) + c * (d[1] - base[1]), d[2]])


def to_odom(base: Sequence[float], p: Sequence[float]) -> np.ndarray:
    c, s = math.cos(base[2]), math.sin(base[2])
    d = np.asarray(p, float)
    return np.array([base[0] + c * d[0] - s * d[1], base[1] + s * d[0] + c * d[1], d[2]])


def rotation_to_robot(base: Sequence[float], rotation: np.ndarray) -> np.ndarray:
    return rot_yaw(-base[2]) @ rotation
