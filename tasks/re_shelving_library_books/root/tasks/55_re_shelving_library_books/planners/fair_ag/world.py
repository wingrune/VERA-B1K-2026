"""What the fair planner of task 55 believes, in the map frame: plain numpy values built by perception (perception.py) from the observation and
the offline scene map (scene_map.py). The fields mirror the State the oracle logic (logic.py) plans on, so the same plans run on estimates.
Nothing here comes from the simulator."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

BOOKS = ('book1', 'book2', 'book3')


@dataclass
class Box:
    """Axis-aligned box in the map frame."""
    lo: np.ndarray
    hi: np.ndarray

    @property
    def centre(self) -> np.ndarray:
        return (self.lo + self.hi) / 2

    @property
    def size(self) -> np.ndarray:
        return self.hi - self.lo


@dataclass
class Book:
    """Estimated book: root pose from the model fitted to the points, AABB of the model at that pose, model extents in its root frame."""
    pos: np.ndarray
    quat: np.ndarray                 # xyzw, yaw only
    box: Box
    local: Box

    @property
    def yaw(self) -> float:
        x, y, z, w = (float(v) for v in self.quat)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@dataclass
class Case:
    """A bookcase of the map: its box and its cubbies (the container volumes a book must end up in)."""
    box: Box
    cubbies: List[Box]


@dataclass
class State:
    step: int
    base: np.ndarray                 # (x, y, yaw) map frame, from localisation and odometry
    base_z: float
    books: Dict[str, Book]
    table_box: Box
    cases: Dict[str, Case]
    atoms: Dict[str, bool]           # inside_book<i>: the planner's belief (book placed in a cubby and seen there)
    eef: Dict[str, np.ndarray] = field(default_factory=dict)
    obstacles: Optional[Any] = None  # (lo_xy, hi_xy, category) footprints: the map plus what perception sees on the desk
    obstacle_source: str = 'map'
    contact: Dict[str, int] = field(default_factory=dict)

    def inside(self, book: str) -> bool:
        return bool(self.atoms.get('inside_' + book, False))

    @property
    def n_inside(self) -> int:
        return sum(self.inside(b) for b in BOOKS if b in self.books or self.inside(b))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([0., 0., math.sin(yaw / 2), math.cos(yaw / 2)])
