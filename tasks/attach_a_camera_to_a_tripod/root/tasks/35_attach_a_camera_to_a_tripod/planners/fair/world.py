"""World model of the fair planner of task 35: what perception and odometry believe, in the odometry frame (the base at the first control is
the origin, x forward, z up from the robot frame origin). Plain numpy values; nothing here comes from the simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ObjectEst:
    """A fitted object: T maps the model frame (data/models.npz) to the odometry frame. yaws lists the yaw hypotheses of the fit, best
    first (the tripod legs are three-fold symmetric; the pan handle breaks the tie when it is seen)."""
    T: np.ndarray
    cost: float
    inlier: float
    n_cloud: int
    seen_step: int
    yaws: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)

    @property
    def xy(self) -> np.ndarray:
        return self.T[:2, 3].copy()


@dataclass
class World:
    step: int
    base: np.ndarray                         # (x, y, yaw) odometry
    floor_z: float                           # floor height in the robot frame
    camera: Optional[ObjectEst] = None       # the camera on the floor (None while held or not seen)
    tripod: Optional[ObjectEst] = None
    held_T: Optional[np.ndarray] = None      # predicted pose of the held camera (odometry frame), from the hand and the grasp transform
