"""World model of the fair planner of task 42: what perception and odometry believe, in the odometry frame (the base at the first control is the
origin, x forward; heights are metres above the floor seen by the head camera). Plain numpy values; nothing here comes from the simulator."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CounterEst:
    """Counter top estimate.
    The counter top (a horizontal slab at the height `top`): an oriented rectangle. `u` is the unit xy direction along its long side,
    `n` the unit xy normal that points out of the counter towards the side the robot saw it from (its front edge)."""
    top: float
    centre: np.ndarray            # xy
    u: np.ndarray
    n: np.ndarray
    length: float                 # along u
    depth: float                  # along n
    seen_step: int = -1

    def to_local(self, xy: np.ndarray) -> np.ndarray:
        d = np.asarray(xy, float)[:2] - self.centre
        return np.array([d @ self.u, d @ self.n])

    def from_local(self, a: float, b: float) -> np.ndarray:
        return self.centre + a * self.u + b * self.n

    @property
    def front(self) -> np.ndarray:
        """xy of the middle of the front edge."""
        return self.centre + 0.5 * self.depth * self.n

    @property
    def yaw_facing(self) -> float:
        """Base yaw that looks straight at the counter."""
        return math.atan2(-self.n[1], -self.n[0])


@dataclass
class BasinEst:
    """Sink basin estimate.
    The sink basin (the box the goal atom `inside` tests): oriented rectangle sunk below the counter top."""
    centre: np.ndarray            # xy
    yaw: float                    # of the first axis
    size: np.ndarray              # extents along (first, second) axis
    floor: float                  # height of the basin floor
    seen_step: int = -1

    @property
    def long_axis(self) -> np.ndarray:
        a = self.yaw + (0. if self.size[0] >= self.size[1] else math.pi / 2)
        return np.array([math.cos(a), math.sin(a)])


@dataclass
class ItemEst:
    """Task item estimate.
    A task item (onion, parer, bowl, board): box centre (xyz, height above the floor), yaw of the model frame (None: not needed or unknown),
    footprint size (xy extents of its bounding box), top height, and for the board and the parer the grip point and the closing axis."""
    role: str
    centre: np.ndarray
    top: float
    size: np.ndarray
    yaw: Optional[float] = None
    n_points: int = 0
    fit: float = 0.0              # inlier share of the model fit (shape roles)
    seen_step: int = -1
    grip: Optional[np.ndarray] = None
    axis: Optional[np.ndarray] = None
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def xy(self) -> np.ndarray:
        return self.centre[:2].copy()


@dataclass
class World:
    step: int
    base: np.ndarray                         # (x, y, yaw) odometry
    floor_z: float                           # floor height in the robot frame
    counter: Optional[CounterEst] = None
    basin: Optional[BasinEst] = None
    items: Dict[str, ItemEst] = field(default_factory=dict)
    halves: List[np.ndarray] = field(default_factory=list)   # xyz of the pieces of the cut onion in the bowl
