"""Navigation part of the shared phase machinery of the fair planner of task 42: the route grid from the perceived obstacles, one control of
driving along the route (replans, tightens the footprint when the base is stuck). Mixed into phases.base.Core."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from core.skills_lib import nav_grid

from .. import params as P
from ..world import World


class NavMixin:
    def build_grid(self, w: World, around: List[np.ndarray], radius: float = P.ROBOT_RADIUS) -> None:
        pts = [w.base[:2]] + [np.asarray(a, float)[:2] for a in around]
        lo, hi = np.min(pts, axis=0) - 2.5, np.max(pts, axis=0) + 2.5
        self.grid_box = (lo, hi)
        self.raw_grid = self.perc.nav_grid(lo, hi)
        self.counter_to_grid(w, self.raw_grid)
        self.radius = radius
        self.inflated = self.raw_grid.inflated(radius)
        # the robot stands here: the cells under and around it are free whatever the inflation says (a base that stands 0.3 m from the counter edge
        # sits inside the inflated counter, and every route from it would be refused)
        ix, iy = self.inflated.cell(w.base[:2])
        rr = int(math.ceil(P.START_FREE_M / self.inflated.res))
        for dx in range(-rr, rr + 1):
            for dy in range(-rr, rr + 1):
                if dx * dx + dy * dy <= rr * rr and 0 <= ix + dx < self.inflated.shape[0] and 0 <= iy + dy < self.inflated.shape[1]:
                    self.inflated.blocked[ix + dx, iy + dy] = False
        self.routes = {}

    @staticmethod
    def counter_to_grid(w: World, grid: nav_grid.Grid) -> None:
        """The counter as one clean rectangle: the obstacle hits in and just in front of its footprint (the slab, the cabinet fronts, the edge) are
        replaced by the slab rectangle, so that a base inflated by its own radius may stand at that distance from the edge, and the long run of the
        counter beyond the seen part stays blocked as the hits say."""
        ce = w.counter
        if ce is None:
            return
        ix, iy = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing='ij')
        xy = np.stack([grid.lo[0] + (ix + .5) * grid.res, grid.lo[1] + (iy + .5) * grid.res], -1).reshape(-1, 2)
        d = xy - ce.centre
        a, b = d @ ce.u, d @ ce.n
        clear = (np.abs(a) < 0.5 * ce.length + 0.3) & (b > -0.5 * ce.depth - 0.6) & (b < 0.5 * ce.depth + P.COUNTER_CLEAR_FRONT_M)
        slab = (np.abs(a) < 0.5 * ce.length) & (b > -0.5 * ce.depth - 0.6) & (b < 0.5 * ce.depth)
        grid.blocked.reshape(-1)[clear] = False
        grid.blocked.reshape(-1)[slab] = True

    def retighten(self, w: World, radius: float) -> None:
        if self.raw_grid is None:
            return
        self.radius = radius
        self.inflated = self.raw_grid.inflated(radius)
        # the robot stands here: the cells under and around it are free whatever the inflation says (a base that stands 0.3 m from the counter edge
        # sits inside the inflated counter, and every route from it would be refused)
        ix, iy = self.inflated.cell(w.base[:2])
        rr = int(math.ceil(P.START_FREE_M / self.inflated.res))
        for dx in range(-rr, rr + 1):
            for dy in range(-rr, rr + 1):
                if dx * dx + dy * dy <= rr * rr and 0 <= ix + dx < self.inflated.shape[0] and 0 <= iy + dy < self.inflated.shape[1]:
                    self.inflated.blocked[ix + dx, iy + dy] = False
        self.routes = {}
        self.path = None

    def ascii_grid(self, w: World, goal: Optional[np.ndarray] = None, half: float = 2.6, res: float = 0.1, marks: Optional[List[np.ndarray]] = None) -> List[str]:
        """Debug picture of the route grid around the base: # obstacle, + only in the inflated grid, . free, ? never seen, R base, G goal, C counter front."""
        rows = []
        g, gi = self.raw_grid, self.inflated
        if g is None:
            return rows
        for y in np.arange(w.base[1] + half, w.base[1] - half, -res):
            row = ''
            for x in np.arange(w.base[0] - half, w.base[0] + half, res):
                p = np.array([x, y])
                if np.linalg.norm(p - w.base[:2]) < res:
                    row += 'R'
                elif goal is not None and np.linalg.norm(p - goal[:2]) < res:
                    row += 'G'
                elif marks and any(np.linalg.norm(p - m[:2]) < res * 0.7 for m in marks):
                    row += 'F'
                elif g.blocked[g.cell(p)]:
                    row += '#'
                elif gi.blocked[gi.cell(p)]:
                    row += '+'
                else:
                    row += '.' if (g.extra_cost is None or g.extra_cost[g.cell(p)] == 0) else '?'
            rows.append(row)
        return rows

    def free_pose(self, xy: np.ndarray) -> bool:
        return not self.inflated.blocked[self.inflated.cell(xy)]

    def route_len(self, w: World, xy: np.ndarray) -> float:
        if float(np.linalg.norm(xy - w.base[:2])) < 0.03:
            return 0.
        key = (tuple(np.round(w.base[:2], 2)), tuple(np.round(xy, 2)))
        if key not in self.routes:
            path = self.inflated.path(w.base[:2], xy)
            self.routes[key] = float('inf') if path is None else nav_grid.path_length(path)
        return self.routes[key]

    def navigate(self, action: np.ndarray, q: np.ndarray, w: World, goal_xy: np.ndarray, goal_yaw: float, rest: bool = True,
                 vmax: float = P.NAV_VFAR, wmax: float = P.NAV_WMAX) -> Tuple[float, float]:
        if rest:
            self.to_rest(action, q)
        dist = float(np.linalg.norm(goal_xy - w.base[:2]))
        yaw_err = nav_grid.wrap(goal_yaw - w.base[2])
        if self.path is None or self.path_age >= P.NAV_REPLAN_EVERY:
            self.path = self.inflated.path(w.base[:2], goal_xy)
            self.path_age = 0
            self.counts['replans'] += 1
        self.path_age += 1
        if self.path is None:
            self.reason = 'no route to the base pose'
            return dist, yaw_err
        near = int(np.argmin([np.linalg.norm(p - w.base[:2]) for p in self.path]))
        waypoint = self.path[min(near + 1, len(self.path) - 1)]
        for p in self.path[near:]:
            waypoint = p
            if np.linalg.norm(p - w.base[:2]) > 0.4:
                break
        if dist < 0.4:
            waypoint = goal_xy
        v = min(vmax, P.NAV_VNEAR) if dist < 0.45 else vmax
        action[:3] = nav_grid.drive_command(w.base, waypoint, goal_yaw, P.NAV_GAIN, v, P.NAV_YAW_GAIN, wmax)
        self.reason = f'driving, {dist:.2f} m, yaw err {yaw_err:.2f}'
        if self.stuck_ref is None or np.linalg.norm(self.stuck_ref[0] - w.base[:2]) > 0.05:
            self.stuck_ref = (w.base[:2].copy(), self.age)
        elif self.age - self.stuck_ref[1] > P.NAV_STUCK_STEPS and dist > 0.2:
            self.counts['nav_stuck'] += 1
            self.stuck_ref = (w.base[:2].copy(), self.age)
            self.retighten(w, P.TIGHT_RADIUS)
        return dist, yaw_err
