"""World model of the fair planner of task 83: what perception and odometry believe, in the odometry frame (the base at the first control is the
origin, x forward, z up from the robot frame origin, which stands on the floor). Plain numpy values. Nothing here comes from the simulator."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import params as P


@dataclass
class CabinetEst:
    """Estimated cabinet: the rectangle of its top and the direction f out of its front (the drawers are pulled along f)."""
    centre: np.ndarray            # xy of the top rectangle centre
    f: np.ndarray                 # unit xy, out of the front
    length: float                 # along the front
    depth: float                  # across the front
    top_z: float
    face: float                   # position of the drawer face along f, from the centre
    face_measured: bool = False
    complete: bool = False        # both ends of the front seen (else the centre along the front comes from the prior length)
    seen_step: int = -1

    @property
    def lateral(self) -> np.ndarray:
        """Unit xy along the front: f turned by +90 degrees."""
        return np.array([-self.f[1], self.f[0]])

    @property
    def yaw_facing(self) -> float:
        """Base yaw that faces the cabinet front."""
        return math.atan2(-self.f[1], -self.f[0])

    def ledge(self, k: int) -> np.ndarray:
        """Pinch point (xyz) of the pull ledge k in {+1, -1} of the top drawer: k = +1 lies along +lateral from the centre."""
        xy = self.centre + self.f * (self.face + P.LEDGE_OUT_OF_FACE_M) + self.lateral * k * P.LEDGE_LATERAL_M
        return np.array([xy[0], xy[1], self.top_z - P.LEDGE_BELOW_TOP_M])

    def corners(self) -> np.ndarray:
        h, d = self.length / 2, self.depth / 2
        return np.array([self.centre + self.f * sf * d + self.lateral * sl * h for sf, sl in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


@dataclass
class JarEst:
    """Estimated jar: xy of its axis, the height of the surface it stands on and of its top."""
    xy: np.ndarray
    support: float
    top: float
    sightings: int = 0
    seen_step: int = -1

    @property
    def grasp_z(self) -> float:
        return self.support + P.JAR_GRASP_ABOVE_M

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.xy[0], self.xy[1], self.support + P.JAR_HEIGHT_M / 2])


@dataclass
class World:
    step: int
    base: np.ndarray                         # (x, y, yaw) odometry, heading corrected by the walls
    cabinet: Optional[CabinetEst] = None
    jar: Optional[JarEst] = None
    base_z: float = 0.0
