"""Objects on the desk for the fair planner of task 98: the hits above the desk top are clustered; the tall cluster is the scanner, the flat one
the laptop.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .. import params as P

class ObjectsMixin:
    def _update_objects(self, pts: np.ndarray) -> None:
        lo, hi = self.desk_box.lo[:2] - 0.04, self.desk_box.hi[:2] + 0.04
        sel = (pts[:, 2] > self.top_z + 0.012) & (pts[:, 2] < self.top_z + 0.14) & np.all(pts[:, :2] > lo, 1) & np.all(pts[:, :2] < hi, 1)
        pts = pts[sel]
        if len(pts) < P.OBJ_MIN_CELLS:
            return
        shape = tuple(np.ceil((hi - lo) / P.OBJ_RES_M).astype(int) + 1)
        ij = np.floor((pts[:, :2] - lo) / P.OBJ_RES_M).astype(int)
        grid = np.zeros(shape, bool)
        grid[ij[:, 0], ij[:, 1]] = True
        lab, n = ndimage.label(ndimage.binary_dilation(grid, iterations=2), structure=np.ones((3, 3), int))
        point_lab = lab[ij[:, 0], ij[:, 1]]
        best_scanner, best_laptop = None, None
        self.last_clusters = []
        for k in range(1, n + 1):
            cp = pts[point_lab == k]
            if len(cp) < P.OBJ_MIN_CELLS:
                continue
            h = float(np.percentile(cp[:, 2], 97)) - self.top_z
            self.last_clusters.append((round(h, 3), len(cp), np.round(cp[:, :2].mean(0), 2).tolist()))
            if h >= P.SCANNER_HEIGHT_MIN_M:
                if self._scanner_key(cp) < (float('inf') if best_scanner is None else self._scanner_key(best_scanner[0])):
                    best_scanner = (cp, h)
            elif P.LAPTOP_HEIGHT_RANGE_M[0] <= h <= P.LAPTOP_HEIGHT_RANGE_M[1]:
                if best_laptop is None or len(cp) > len(best_laptop):
                    best_laptop = cp
        if best_scanner is not None:
            self._fit_scanner(*best_scanner)
        if best_laptop is not None:
            self._fit_laptop(best_laptop)
