"""Position of the base from the walls: in the desk frame the walls and furniture run along the axes, so the planes of the vertical surfaces are
lines x = const and y = const. Their offsets are recorded during the first scan turn (the odometry is accurate there: the base only turns) and
every later view of them gives the position of the base along the axis they are perpendicular to, free of drift.

`WallMap.observe` takes the vertical-surface points of one view in the desk frame with their normals and returns, per axis, the shift of the
observed planes against the recorded ones (None when no recorded plane is seen)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from . import params as P


class WallMap:
    def __init__(self) -> None:
        n = int(2 * P.OBST_HALF_M / P.WALL_BIN_M)
        self.hist = [np.zeros(n) for _ in range(2)]          # votes per axis and offset bin (recording)
        self.planes: List[np.ndarray] = [np.zeros(0), np.zeros(0)]   # frozen plane offsets (m) per axis
        self.frozen = False

    def _split(self, pts: np.ndarray, normals: np.ndarray) -> List[np.ndarray]:
        """Offsets (m) of the points with normals along the desk x axis, and along y."""
        c = np.cos(np.radians(P.WALL_AXIS_TOL_DEG))
        along = [np.abs(normals[:, 0]) > c, np.abs(normals[:, 1]) > c]
        return [pts[along[0], 0], pts[along[1], 1]]

    def record(self, pts: np.ndarray, normals: np.ndarray) -> None:
        for k, off in enumerate(self._split(pts, normals)):
            idx = np.floor((off + P.OBST_HALF_M) / P.WALL_BIN_M).astype(int)
            idx = idx[(idx >= 0) & (idx < len(self.hist[k]))]
            np.add.at(self.hist[k], idx, 1)

    def freeze(self) -> Dict[str, int]:
        """Peaks of the recorded votes become the planes."""
        for k in (0, 1):
            h = ndimage.gaussian_filter1d(self.hist[k], 1.5)
            peaks = [i for i in range(1, len(h) - 1) if h[i] >= h[i - 1] and h[i] > h[i + 1] and self.hist[k][max(i - 2, 0):i + 3].sum() >= P.WALL_MIN_VOTES]
            self.planes[k] = np.array([(i + 0.5) * P.WALL_BIN_M - P.OBST_HALF_M for i in peaks])
        self.frozen = True
        return dict(x=len(self.planes[0]), y=len(self.planes[1]))

    def observe(self, pts: np.ndarray, normals: np.ndarray) -> List[Optional[float]]:
        """Shift (m) of the observed planes against the recorded ones along x and y, or None."""
        out: List[Optional[float]] = [None, None]
        for k, off in enumerate(self._split(pts, normals)):
            if len(self.planes[k]) == 0 or len(off) < P.WALL_MIN_VIEW_POINTS:
                continue
            shifts, weights = [], []
            for plane in self.planes[k]:
                near = off[np.abs(off - plane) < P.WALL_MATCH_M]
                if len(near) >= P.WALL_MIN_VIEW_POINTS:
                    shifts.append(float(np.median(near)) - plane)
                    weights.append(len(near))
            if shifts:
                out[k] = float(np.average(shifts, weights=weights))
        return out
