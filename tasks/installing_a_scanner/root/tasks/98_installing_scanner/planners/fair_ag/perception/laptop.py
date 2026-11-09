"""The laptop of the fair planner of task 98: a flat box of known footprint class that does not move.
"""
from __future__ import annotations

import numpy as np

from .. import params as P
from ..world import Box

class LaptopMixin:
    def _fit_laptop(self, cp: np.ndarray) -> None:
        """The laptop is a flat box of known footprint class that does not move: take a flat cluster of the right size, refine the stored box only
        with clusters that agree with it (an arm or hand hovering over the desk is not the laptop)."""
        lo, hi = cp[:, :2].min(0), cp[:, :2].max(0)
        ext = hi - lo
        if np.any(ext < P.LAPTOP_EXTENT_M[0]) or np.any(ext > P.LAPTOP_EXTENT_M[1]):
            return
        zmax = float(np.percentile(cp[:, 2], 97))
        prev = self.laptop_box
        if prev is not None:
            if np.linalg.norm((lo + hi) / 2 - prev.centre[:2]) > P.LAPTOP_AGREE_M:
                return
            lo, hi = np.minimum(lo, prev.lo[:2]), np.maximum(hi, prev.hi[:2])
            if np.any(hi - lo > P.LAPTOP_EXTENT_M[1]):
                return
        self.laptop_box = Box(np.array([*lo, self.top_z]), np.array([*hi, zmax]))
        self.counts['laptop_frames'] += 1
