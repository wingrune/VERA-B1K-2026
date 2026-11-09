"""World model of the fair planner of task 69: what perception and odometry believe, in the odometry frame (the base at the first control is
the origin, x forward, z up from the robot frame origin). Plain numpy values; nothing here comes from the simulator. Frozen."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import models as MD


@dataclass
class VacuumEst:
    """The vacuum standing on the floor, fitted from the cloud.

    The vacuum standing on the floor, fitted from the cloud: T maps the model frame (data/models.npz) to the odometry frame."""
    T: np.ndarray
    cost: float
    inlier: float
    n_cloud: int
    seen_step: int

    @property
    def xy(self) -> np.ndarray:
        return self.T[:2, 3].copy()


@dataclass
class Dust:
    """A dark spot on the floor (odometry frame xy), seen `n` times.

    A dark spot on the floor (odometry frame xy), seen `n` times; `visits` counts sweeps over it, `gone` marks it as removed."""
    xy: np.ndarray
    n: int = 1
    seen_step: int = 0
    visits: int = 0
    gone: bool = False
    pid: int = 0
    misses: int = 0
    visit_step: int = -1                     # the step the vacuum head last stood on it (-1: never)
    checked: bool = False                    # looked at from a vantage point after its pass
    reseen: int = 0                          # sightings since that visit (it was not removed)


@dataclass
class World:
    step: int
    base: np.ndarray                         # (x, y, yaw) odometry
    floor_z: float                           # floor height in the robot frame
    vacuum: Optional[VacuumEst] = None       # the standing vacuum (None while held or not seen)
    held_T: Optional[np.ndarray] = None      # predicted pose of the held vacuum (odometry frame), from the hand and the grasp transform
    dust: List[Dust] = field(default_factory=list)   # the particles believed to be on the floor (not gone)
    toggled: Optional[bool] = None           # the marker colour: True green, False red, None not seen
    tilted: bool = False                     # the marker was seen far from where the hand holds it (the vacuum caught on something)

    @property
    def vac_T(self) -> Optional[np.ndarray]:
        """The vacuum pose in use: held.

        The vacuum pose in use: held (predicted from the hand) or standing (fitted)."""
        if self.held_T is not None:
            return self.held_T
        return None if self.vacuum is None else self.vacuum.T

    def cube(self) -> Optional[tuple]:
        """Projection cube (centre, rotation matrix.

        Projection cube (centre, rotation matrix, full extents) in the odometry frame."""
        T = self.vac_T
        if T is None:
            return None
        return T[:3, :3] @ MD.CUBE_CENTRE + T[:3, 3], T[:3, :3] @ MD.CUBE_ROT, MD.CUBE_SCALE.copy()
