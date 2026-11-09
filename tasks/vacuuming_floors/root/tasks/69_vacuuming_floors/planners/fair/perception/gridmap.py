"""Floor map of the fair planner of task 69: obstacle hits of the depth clouds on a 5 cm grid (cells blocked by hits in two height layers, or many in
one), the floor seen, the floor looked at for flakes, the route grid for the planner. Mixed into perception.state.Perception."""
from __future__ import annotations

import numpy as np

from core.skills_lib import nav_grid

from .. import params as P


class MapMixin:
    def _cells(self, xy: np.ndarray) -> np.ndarray:
        c = np.floor((xy + P.OBST_HALF_M) / P.OBST_RES_M).astype(int)
        return np.clip(c, 0, self.obst.shape[0] - 1)

    def _layers(self, pts: np.ndarray, head: np.ndarray) -> None:
        if self.map_hold or (P.MAP_FREEZE_HELD and self.held is not None):
            return
        h = pts[:, 2] - self.floor_z
        near = np.hypot(pts[:, 0] - self.pose()[0], pts[:, 1] - self.pose()[1]) < P.OBST_SELF_R_M
        ob = pts[(h > P.OBST_BAND_M[0]) & (h < P.OBST_BAND_M[1]) & ~near]
        if len(ob):
            cells = self._cells(ob[:, :2])
            bits = (1 << np.digitize(ob[:, 2] - self.floor_z, P.OBST_LAYERS_M)).astype(np.uint8)
            np.bitwise_or.at(self.obst_layers, (cells[:, 0], cells[:, 1]), bits)
            c = np.unique(cells, axis=0)
            self.obst[c[:, 0], c[:, 1]] = np.minimum(self.obst[c[:, 0], c[:, 1]].astype(int) + 1, 250).astype(np.uint8)
        fl = head[np.abs(head[:, 2] - self.floor_z) < P.FLOOR_BAND_M]
        if len(fl):
            c = self._cells(fl[:, :2])
            self.floor_seen[c[:, 0], c[:, 1]] = True

    def blocked_cells(self) -> np.ndarray:
        """Cells blocked by obstacle hits (layers or count).

        Cells with obstacle hits: a few hits in two height layers (a wall, furniture), or many in one (a thin leg). A lone silhouette pixel is neither."""
        layers = (self.obst_layers & 1) + ((self.obst_layers >> 1) & 1) + ((self.obst_layers >> 2) & 1)
        return ((self.obst >= P.OBST_MIN_HITS) & (layers >= 2)) | (self.obst >= P.OBST_SINGLE_HITS)

    def _covered(self, head: np.ndarray, base: np.ndarray) -> None:
        fl = head[np.abs(head[:, 2] - self.floor_z) < P.FLOOR_BAND_M]
        if len(fl):
            r = np.hypot(fl[:, 0] - base[0], fl[:, 1] - base[1])
            c = self._cells(fl[(r > P.COVER_RANGE_M[0]) & (r < P.COVER_RANGE_M[1]), :2])
            self.dust_cov[c[:, 0], c[:, 1]] = True

    def uncovered_floor(self, centre: np.ndarray, radius: float) -> np.ndarray:
        """Floor cells seen but not looked at for flakes near `centre`.

        Centres (N, 2) of floor cells that were seen but never looked at for flakes at a useful range, within `radius` of `centre`, every 4th cell."""
        ij = np.argwhere(self.floor_seen & ~self.dust_cov)[::4]
        xy = ij * P.OBST_RES_M - P.OBST_HALF_M + 0.5 * P.OBST_RES_M
        return xy[np.hypot(xy[:, 0] - centre[0], xy[:, 1] - centre[1]) < radius]

    def nav_grid(self, lo: np.ndarray, hi: np.ndarray) -> nav_grid.Grid:
        """Obstacle grid (GRID_RES) over [lo, hi] from the obstacle hits.

        Obstacle grid (GRID_RES) over [lo, hi] from the obstacle hits; cells never seen as floor cost UNKNOWN_COST more."""
        grid = nav_grid.Grid(lo, hi, P.GRID_RES)
        ix, iy = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing='ij')
        centres = np.stack([lo[0] + (ix + .5) * P.GRID_RES, lo[1] + (iy + .5) * P.GRID_RES], -1)
        cc = self._cells(centres.reshape(-1, 2)).reshape(centres.shape)
        grid.blocked = self.blocked_cells()[cc[..., 0], cc[..., 1]]
        near = np.linalg.norm(centres - self.pose()[:2], axis=-1) < 0.8
        seen = self.floor_seen[cc[..., 0], cc[..., 1]] | near
        grid.extra_cost = np.where(seen, 0., P.UNKNOWN_COST)
        return grid

