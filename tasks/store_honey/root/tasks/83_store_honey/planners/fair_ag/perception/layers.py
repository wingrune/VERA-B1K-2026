"""Obstacle layers of the fair planner of task 83 in the odometry frame (cells of MAP_RES_M): per frame, a cell with points in the obstacle band
gains OBST_HIT, a cell with only floor points loses OBST_FREE (a wall seen a few centimetres off after the odometry drifted is cleared by the
next view of the floor next to it). Cells the base stood in are free."""
from __future__ import annotations

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .. import params as P


class LayersMixin:
    def _cell(self, xy: np.ndarray, res: float = P.MAP_RES_M) -> np.ndarray:
        return np.floor((np.asarray(xy, float) + P.MAP_HALF_M) / res).astype(int)

    def _cells_of(self, pts_xy: np.ndarray, shape: int) -> np.ndarray:
        c = self._cell(pts_xy)
        c = c[(c >= 0).all(1) & (c < shape).all(1)]
        return np.unique(c, axis=0) if len(c) else c

    def _update_layers(self, pts_odom: np.ndarray) -> None:
        n = self.obst.shape[0]
        c0 = self._cell(self.pose()[:2])
        r = int(P.VISITED_RADIUS_M / P.MAP_RES_M)
        self.visited[max(c0[0] - r, 0):min(c0[0] + r + 1, n), max(c0[1] - r, 0):min(c0[1] + r + 1, n)] = True
        if not len(pts_odom):
            return
        z = pts_odom[:, 2]
        band = (z > P.OBST_BAND_M[0]) & (z < P.OBST_BAND_M[1])
        if self._jar_in_hand():
            pose = self.pose()
            dx = pts_odom[:, 0] - pose[0]
            dy = pts_odom[:, 1] - pose[1]
            cy, sy = np.cos(pose[2]), np.sin(pose[2])
            xr = cy * dx + sy * dy
            yr = -sy * dx + cy * dy
            carry = (xr > 0.05) & (xr < 0.7) & (np.abs(yr) < 0.55) & (z > 0.7) & (z < 1.4)
            band &= ~carry
        hit = np.zeros((n, n), bool)
        c = self._cells_of(pts_odom[band, :2], n)
        if len(c):
            hit[c[:, 0], c[:, 1]] = True
        floor = np.zeros((n, n), bool)
        c = self._cells_of(pts_odom[z < P.FLOOR_BAND_M, :2], n)
        if len(c):
            floor[c[:, 0], c[:, 1]] = True
        self.floor_seen |= floor
        cnt = self.obst.astype(np.int16)
        cnt[hit] += P.OBST_HIT
        cnt[floor & ~hit] -= P.OBST_FREE
        self.obst = np.clip(cnt, 0, P.OBST_MAX).astype(np.uint8)

    def _jar_in_hand(self) -> bool:
        """A hand whose closed fingers stay apart at a jar's width holds the jar (proprio only); its points are the robot's own load."""
        try:
            proprio = getattr(self, 'proprio', None)
            if proprio is None:
                return False
            q = np.asarray(K.q_from_proprio(proprio), float)
            for dofs in K.FINGER_DOFS.values():
                m = float(np.mean(q[list(dofs)]))
                if 0.02 <= m <= 0.046:
                    return True
        except Exception:
            return False
        return False

    def blocked_cells(self) -> np.ndarray:
        return (self.obst >= P.OBST_MIN) & ~self.visited

    def nav_grid(self, lo: np.ndarray, hi: np.ndarray) -> nav_grid.Grid:
        """Obstacle grid (cells NAV_RES_M) over the box lo..hi of the odometry frame, from the blocked cells of the layers."""
        grid = nav_grid.Grid(lo, hi, P.NAV_RES_M)
        ix, iy = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing='ij')
        centres = np.stack([lo[0] + (ix + .5) * grid.res, lo[1] + (iy + .5) * grid.res], -1)
        cell = self._cell(centres)
        n = self.obst.shape[0]
        inside = (cell >= 0).all(-1) & (cell < n).all(-1)
        cc = np.clip(cell, 0, n - 1)
        blocked = self.blocked_cells()
        grid.blocked = np.where(inside, blocked[cc[..., 0], cc[..., 1]], True)
        if P.UNKNOWN_BLOCKS:
            grid.blocked |= ~self.floor_seen[cc[..., 0], cc[..., 1]]
        return grid
