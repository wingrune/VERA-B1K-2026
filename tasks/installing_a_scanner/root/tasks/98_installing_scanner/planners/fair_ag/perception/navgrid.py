"""Navigation grid of the fair planner of task 98 from the perceived obstacle layers, in the desk frame.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from core.skills_lib import nav_grid

from .. import params as P
from .cloud import rot2

class NavGridMixin:
    def nav_grid(self, soft_clutter: bool, extra_lo: np.ndarray, extra_hi: np.ndarray) -> nav_grid.Grid:
        """Obstacle grid in the desk frame (res GRID_RES). Cells with an obstacle (and the desk) are blocked; cells never seen as floor, not visited
        by the base and outside its blind zone cost UNKNOWN_COST more to cross. With soft_clutter the compact low obstacles (chairs, which the base
        may push aside) are not blocked but cost SOFT_COST more."""
        res = P.OBST_RES_M
        occ = self.obst >= P.OBST_MIN_HITS
        if self.desk_box is not None:       # the desk is one hard block of its own: cut it out so that chairs touching it stay separate objects
            n = occ.shape[0]
            ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            d_xy = (np.stack([ii, jj], -1) + 0.5) * res - P.OBST_HALF_M
            d_xy = d_xy @ rot2(-self.psi).T
            in_desk = np.all(d_xy > self.desk_box.lo[:2] - P.DESK_CUT_MARGIN_M, -1) & np.all(d_xy < self.desk_box.hi[:2] + P.DESK_CUT_MARGIN_M, -1)
            occ = occ & ~in_desk
        soft_cells = np.zeros_like(occ)
        if soft_clutter and occ.any():
            labels, n_lab = ndimage.label(ndimage.binary_dilation(occ, iterations=1))
            tall_lab = set(np.unique(labels[(self.tall >= 1) & (labels > 0)]).tolist())
            for k, sl in enumerate(ndimage.find_objects(labels), start=1):
                ext = np.array([sl[0].stop - sl[0].start, sl[1].stop - sl[1].start]) * res
                if k not in tall_lab and ext.max() < P.SOFT_CLUTTER_MAX_M:
                    soft_cells[sl] |= labels[sl] == k
        soft_cells &= ~self.bumps
        hard = ndimage.binary_dilation((occ & ~soft_cells) | self.bumps, iterations=1)
        known = ndimage.binary_closing(self.floor_seen >= 1, iterations=P.FLOOR_CLOSING) | ndimage.binary_dilation(self.visited, iterations=1) | soft_cells
        xy_obst = np.argwhere(occ)
        xy_d = ((xy_obst + 0.5) * res - P.OBST_HALF_M) @ rot2(-self.psi).T
        lo = np.minimum(extra_lo, xy_d.min(0) if len(xy_d) else extra_lo) - 1.0
        hi = np.maximum(extra_hi, xy_d.max(0) if len(xy_d) else extra_hi) + 1.0
        grid = nav_grid.Grid(lo, hi, P.GRID_RES)
        ix, iy = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing='ij')
        centres = np.stack([lo[0] + (ix + .5) * P.GRID_RES, lo[1] + (iy + .5) * P.GRID_RES], -1)
        odom = centres @ rot2(self.psi).T
        cell = np.floor((odom + P.OBST_HALF_M) / res).astype(int)
        inside = (cell >= 0).all(-1) & (cell < self.obst.shape[0]).all(-1)
        cc = np.clip(cell, 0, self.obst.shape[0] - 1)
        base_odom = self.pose()[:2]
        blind = np.linalg.norm(odom - base_odom, axis=-1) < P.BLIND_ZONE_M
        seen = (known[cc[..., 0], cc[..., 1]] & inside) | blind
        # a chair is felt by the robot from its radius away: the soft zone is the chair grown by the robot radius, the cost grows with the depth in it
        zone = ndimage.binary_dilation(soft_cells, iterations=int(math.ceil(P.ROBOT_TIGHT_RADIUS / res))) if soft_cells.any() else soft_cells
        soft = zone[cc[..., 0], cc[..., 1]] & inside
        depth = ndimage.distance_transform_edt(zone) * res if zone.any() else np.zeros(zone.shape)
        soft_depth = np.where(soft, depth[cc[..., 0], cc[..., 1]], 0.)
        grid.blocked = hard[cc[..., 0], cc[..., 1]] & inside
        grid.soft = soft & ~grid.blocked                 # where the route crosses a chair (the base slows down there)
        desk_dist = np.zeros(soft.shape)
        if self.desk_box is not None:     # a chair is pushed away from the desk it stands at: cross it on the desk side, where it can roll out of the way
            gap = np.maximum(np.maximum(self.desk_box.lo[:2] - centres, centres - self.desk_box.hi[:2]), 0.)
            desk_dist = np.hypot(gap[..., 0], gap[..., 1])
        grid.extra_cost = np.where(seen, 0., P.UNKNOWN_COST) + np.where(soft, P.SOFT_COST + P.SOFT_DEPTH_COST * soft_depth + P.SOFT_DESK_COST * desk_dist, 0.)
        if self.desk_box is not None:
            grid.add_box(self.desk_box.lo[:2], self.desk_box.hi[:2])
        return grid
