"""Obstacle layers and the desk top of the fair planner of task 98: height-banded hit grids around the base, the desk top as the biggest
horizontal surface, its frame D and box.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from core.skills_lib import rgbd_geometry as G

from .. import params as P
from ..world import Box
from .cloud import rot2

class TableMixin:
    def _cell(self, xy: np.ndarray, res: float) -> np.ndarray:
        return np.floor((xy + P.OBST_HALF_M) / res).astype(int)

    def _update_layers(self, pts_odom: np.ndarray) -> None:
        if not len(pts_odom):
            return
        z = pts_odom[:, 2]
        c0 = self._cell(self.pose()[:2], P.OBST_RES_M)
        r = int(P.VISITED_RADIUS_M / P.OBST_RES_M)
        n = self.visited.shape[0]
        self.visited[max(c0[0] - r, 0):min(c0[0] + r + 1, n), max(c0[1] - r, 0):min(c0[1] + r + 1, n)] = True
        for arr, sel in ((self.obst, (z > P.OBST_BAND_M[0]) & (z < P.OBST_BAND_M[1])), (self.tall, z >= P.OBST_BAND_M[1]),
                         (self.floor_seen, z < P.FLOOR_BAND_M)):
            if sel.any():
                c = self._cell(pts_odom[sel, :2], P.OBST_RES_M)
                c = c[(c >= 0).all(1) & (c < arr.shape[0]).all(1)]
                np.add.at(arr, (c[:, 0], c[:, 1]), 1)
                np.minimum(arr, 250, out=arr)
        self.cache_version += 1

    def _update_desk(self, pts_odom: np.ndarray) -> None:
        near = np.linalg.norm(pts_odom[:, :2] - self.pose()[:2], axis=1) < 3.5
        lo, hi = P.DESK_TOP_RANGE_M
        n_bins = int(round((hi - lo) / 0.01))
        hist, _ = np.histogram(pts_odom[near, 2], bins=n_bins, range=(lo, hi))
        self.z_hist += hist                                         # votes of every frame: the desk top is the biggest horizontal surface overall
        smooth = self.z_hist + np.r_[self.z_hist[1:], 0] + np.r_[0, self.z_hist[:-1]]
        k = int(np.argmax(smooth))
        if smooth[k] >= P.TOP_MIN_VOTES:
            window = self.z_hist[max(k - 1, 0):k + 2].astype(float)
            centres = lo + (np.arange(max(k - 1, 0), min(k + 2, n_bins)) + 0.5) * 0.01
            peak = float((window * centres).sum() / window.sum())
            if self.top_z is not None and abs(peak - self.top_z) > P.TOP_RESET_M:
                self.top_hits[:] = 0                                # the surface layer was built at the wrong height
            self.top_z = peak
        if self.top_z is None:
            return
        sel = np.abs(pts_odom[:, 2] - self.top_z) < P.TOP_BAND_M
        if sel.any():
            c = self._cell(pts_odom[sel, :2], P.TOP_RES_M)
            c = c[(c >= 0).all(1) & (c < self.top_hits.shape[0]).all(1)]
            np.add.at(self.top_hits, (c[:, 0], c[:, 1]), 1)
            np.minimum(self.top_hits, 250, out=self.top_hits)
        mask = self.top_hits >= 2
        if not mask.any():
            return
        lab, n = ndimage.label(ndimage.binary_closing(mask, iterations=2))
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        cells = np.argwhere((lab == k) & mask)
        self.desk_area = len(cells) * P.TOP_RES_M ** 2
        if self.desk_area < P.DESK_MIN_AREA_M2:
            return
        self.counts['desk_frames'] += 1
        xy_odom = (cells + 0.5) * P.TOP_RES_M - P.OBST_HALF_M
        if not self.locked:
            sub = xy_odom[:: max(1, len(xy_odom) // 4000)]
            _, ang, _ = G.min_area_rect(sub)
            self.psi = G.wrap(((ang + math.pi / 4) % (math.pi / 2)) - math.pi / 4)
            self.locked = self.desk_area >= P.DESK_LOCK_AREA_M2
        xy = xy_odom @ rot2(-self.psi).T
        self.desk_box = Box(np.array([*xy.min(0) - P.TOP_RES_M / 2, self.top_z - 0.8]), np.array([*xy.max(0) + P.TOP_RES_M / 2, self.top_z]))
