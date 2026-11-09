"""World model of the fair planner of task 98: what perception and odometry believe, in the odometry frame (the base at the first control is the
origin, x forward, z up measured from the robot frame origin). Plain numpy values. Nothing here comes from the simulator."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Box:
    """Axis-aligned box in the odometry frame."""
    lo: np.ndarray
    hi: np.ndarray

    @property
    def centre(self) -> np.ndarray:
        return (self.lo + self.hi) / 2


@dataclass
class ScannerEst:
    """Estimated scanner: yaw is the direction of the button face normal (the +x axis of the scanner root frame)."""
    centre: np.ndarray          # xy
    yaw: float
    half: Tuple[float, float]   # half extents along (front axis, side axis)
    z_lo: float                 # desk top
    z_hi: float                 # top face of the scanner
    seen_step: int = -1         # control at which it was last measured
    front_known: bool = False   # the button side is decided (marker seen), not only the long axis

    def corners(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        ex, ey = np.array([c, s]), np.array([-s, c])
        a, b = self.half
        return np.array([self.centre + ex * sx * a + ey * sy * b for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


@dataclass
class World:
    step: int
    base: np.ndarray                         # (x, y, yaw) odometry
    scanner: Optional[ScannerEst] = None
    marker: Optional[np.ndarray] = None      # button centre, from the scanner pose and the template
    laptop_box: Optional[Box] = None
    table_box: Optional[Box] = None
    obstacles: List[tuple] = field(default_factory=list)   # (lo_xy, hi_xy, kind) footprints for the base, kind in {'wall', 'desk', 'clutter'}
    base_z: float = 0.0

    @property
    def scanner_box(self) -> Box:
        pts = self.scanner.corners()
        return Box(np.array([pts[:, 0].min(), pts[:, 1].min(), self.scanner.z_lo]), np.array([pts[:, 0].max(), pts[:, 1].max(), self.scanner.z_hi]))

    @property
    def ready(self) -> bool:
        return self.scanner is not None and self.laptop_box is not None and self.table_box is not None and self.marker is not None
