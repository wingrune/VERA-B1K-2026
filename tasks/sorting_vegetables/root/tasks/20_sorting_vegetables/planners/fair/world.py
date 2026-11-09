"""World model of the fair planner of task 20: what perception and odometry believe, in the odometry frame (the base at the first control is
the origin, x forward, z the height above the floor). Plain numpy values; nothing here comes from the simulator.

Groups (the three `exists` of the goal, any bowl may serve any group): 0 bok choy and onions, 1 leeks and broccoli, 2 corn."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

GROUPS = 3
KINDS = ('corn', 'onion', 'green')     # what the colours tell apart; green is bok choy (group 0) or a leek or broccoli (group 1)


@dataclass
class BowlEst:
    """A mixing bowl on the counter: centre of its rim circle (xy) and the rim height."""
    centre: np.ndarray
    rim: float
    n: int = 1                     # sightings merged
    seen_step: int = -1
    contents: Dict[str, int] = field(default_factory=dict)   # colour class -> points inside the rim seen at the last sighting
    last: Optional[np.ndarray] = None   # centre of the newest sighting (the running mean lags a drifting odometry or a pushed bowl)

    @property
    def xy(self) -> np.ndarray:
        return self.centre[:2].copy()


@dataclass
class VegEst:
    """A vegetable seen on the floor or in a basket: the blob of its colour class. `axis` is the unit xy of the long side of its footprint."""
    kind: str
    centre: np.ndarray             # xyz of the blob's box centre
    top: float
    axis: np.ndarray
    length: float
    width: float
    n_points: int = 0
    seen_step: int = -1
    group: Optional[int] = None
    basket: Optional[int] = None   # index of the basket it lies in
    rim: Optional[float] = None    # rim height of that basket
    pts: Optional[np.ndarray] = None   # the blob's points (odometry) of the last sighting
    misses: int = 0                # standing head views that had its spot in sight and did not find it

    @property
    def xy(self) -> np.ndarray:
        return self.centre[:2].copy()


@dataclass
class BasketEst:
    """A wicker basket on the floor: oriented footprint and rim height; `kinds` counts the vegetable classes seen in it."""
    centre: np.ndarray             # xy
    yaw: float
    size: np.ndarray               # extents along (yaw, yaw + pi/2)
    rim: float
    n: int = 1
    seen_step: int = -1
    kinds: Dict[str, int] = field(default_factory=dict)


@dataclass
class World:
    step: int
    base: np.ndarray                         # (x, y, yaw) odometry
    bowls: List[BowlEst] = field(default_factory=list)
    vegs: List[VegEst] = field(default_factory=list)
    baskets: List[BasketEst] = field(default_factory=list)
    counter_top: Optional[float] = None
    cloud: Optional[np.ndarray] = None       # points (odometry) low over the floor of the latest standing views (head, and the wrist when looking)
