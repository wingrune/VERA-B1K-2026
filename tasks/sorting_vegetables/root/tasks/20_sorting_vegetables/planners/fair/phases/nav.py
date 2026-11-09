"""Driving of the fair planner of task 20 (as the task 20 oracle drives, on the perceived occupancy): route grids from the head's voxel map,
geodesic distances, path following with replanning, a stuck base backs away from what it presses on."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from core.skills_lib import nav_grid

from .. import params as P
from ..geometry import rot_yaw
from ..world import World


def geodesic(grid: nav_grid.Grid, start_xy: np.ndarray) -> np.ndarray:
    """Driving distance (m) from start_xy to every cell over the free cells of grid (8-connected Dijkstra); inf where unreachable. A start
    inside an inflated obstacle starts from the nearest free cell."""
    free = ~grid.blocked
    nx, ny = free.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, wts = [], [], []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        a = free[max(0, -dx):nx - max(0, dx), max(0, -dy):ny - max(0, dy)]
        b = free[max(0, dx):, max(0, dy):ny + min(0, dy) or None][:a.shape[0], :a.shape[1]]
        ia = idx[max(0, -dx):nx - max(0, dx), max(0, -dy):ny - max(0, dy)]
        ib = idx[max(0, dx):, max(0, dy):ny + min(0, dy) or None][:a.shape[0], :a.shape[1]]
        m = a & b
        rows.append(ia[m]), cols.append(ib[m]), wts.append(np.full(int(m.sum()), grid.res * math.hypot(dx, dy)))
    g = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny, nx * ny)).tocsr()
    start = grid.nearest_free(start_xy, max_r=1.0)
    if start is None:
        return np.full(free.shape, np.inf)
    return dijkstra(g, directed=False, indices=idx[grid.cell(start)]).reshape(nx, ny)


class NavMixin:
    """Needs self.perc (Perception), self.age, self.counts, self.reason, self.path, self.path_age, self.stuck_ref, self.escape."""

    def build_grid(self, w: World, around: Optional[list] = None) -> None:
        """Route grids from the occupancy now, cropped to the box round the base and `around` (xy points) plus 2.5 m."""
        pts = [w.base[:2]] + [np.asarray(p, float)[:2] for p in (around or [])]
        lo = np.maximum(np.min(pts, axis=0) - 2.5, -P.MAP_HALF_M + 0.05)
        hi = np.minimum(np.max(pts, axis=0) + 2.5, P.MAP_HALF_M - 0.05)
        vox = self.perc.vox
        a, b = vox.cell2(lo), vox.cell2(hi)
        lo = vox.centre2(np.array(a)) - 0.5 * P.VOX_M
        g = nav_grid.Grid(lo, lo + (np.array(b) - np.array(a) + 1) * P.VOX_M, P.VOX_M)
        g.blocked = vox.blocked2d()[a[0]:a[0] + g.shape[0], a[1]:a[1] + g.shape[1]].copy()
        self.grid_raw = g
        self.inflated, self.inflated_wide = g.inflated(P.ROBOT_RADIUS), g.inflated(P.ROBOT_RADIUS_WIDE)
        self.geo = geodesic(self.inflated, w.base[:2])

    def route(self, xy: np.ndarray) -> float:
        """Driving distance (m) from the base pose of the last build_grid to xy; inf when no route or outside the grid."""
        c = np.floor((np.asarray(xy, float)[:2] - self.inflated.lo) / self.inflated.res).astype(int)
        if np.any(c < 0) or np.any(c >= np.array(self.inflated.shape)):
            return float('inf')
        return float(self.geo[c[0], c[1]])

    def free_pose(self, xy: np.ndarray) -> bool:
        c = np.floor((np.asarray(xy, float)[:2] - self.inflated.lo) / self.inflated.res).astype(int)
        return bool(np.all(c >= 0) and np.all(c < np.array(self.inflated.shape)) and not self.inflated.blocked[c[0], c[1]])

    def nav_budget(self, goal_xy: np.ndarray) -> float:
        """Controls allowed for the drive to goal_xy, fixed when the drive starts (from the route then; recomputed from the remaining
        route at every control, it shrank as the drive went on, and fell to NAV_SLACK whenever the route read as blocked)."""
        key = tuple(np.round(np.asarray(goal_xy, float)[:2], 3))
        if getattr(self, '_budget', None) is None or self._budget[0] != key or self.age <= 1:
            r = self.route(goal_xy)
            if not np.isfinite(r):
                r = 1.5 * float(np.linalg.norm(np.asarray(goal_xy, float)[:2] - self.perc.pose()[:2]))
            self._budget = (key, min(P.NAV_TIMEOUT, r / P.NAV_MPS_STEP + P.NAV_SLACK))
        return self._budget[1]

    def navigate(self, action: np.ndarray, w: World, goal_xy: np.ndarray, goal_yaw: float) -> Tuple[float, float]:
        """One control of the drive to a base pose (the arms are set by the caller). Replans on the occupancy every NAV_REPLAN_EVERY."""
        base = w.base
        dist, yaw_err = float(np.linalg.norm(goal_xy - base[:2])), nav_grid.wrap(goal_yaw - base[2])
        if self.escape > 0:
            self.escape -= 1
            v = rot_yaw(-base[2])[:2, :2] @ self.escape_dir * P.ESCAPE_SPEED
            action[:3] = [v[0], v[1], 0.]
            self.reason = f'escaping, {dist:.2f} m'
            return dist, yaw_err
        if self.path is None or self.path_age >= P.NAV_REPLAN_EVERY:
            self.build_grid(w, [goal_xy])
            wide, narrow = self.inflated_wide.path(base[:2], goal_xy), self.inflated.path(base[:2], goal_xy)
            long_way = wide is not None and narrow is not None and nav_grid.path_length(wide) > nav_grid.path_length(narrow) + 1.5
            self.path, self.path_age = narrow if wide is None or long_way else wide, 0
            self.counts['replans'] += 1
        self.path_age += 1
        if self.path is None:
            self.reason = 'no route to the base pose'
            if dist < 0.6:                                   # the goal cell is blocked by a fresh hit: creep straight there
                action[:3] = nav_grid.drive_command(base, goal_xy, goal_yaw, P.NAV_GAIN, P.NAV_VNEAR, P.NAV_YAW_GAIN, P.NAV_WMAX)
            return dist, yaw_err
        near = int(np.argmin([np.linalg.norm(p - base[:2]) for p in self.path]))
        if near < len(self.path) - 1 and np.dot(base[:2] - self.path[near], self.path[near + 1] - self.path[near]) > 0:
            near += 1
        waypoint, end = self.path[near], True
        for p in self.path[near:]:
            waypoint = p
            if np.linalg.norm(p - base[:2]) > P.NAV_LOOKAHEAD_M:
                end = False
                break
        if dist < 0.4 or end:
            waypoint = goal_xy
        if dist > 0.6:
            ahead = waypoint - base[:2]
            waypoint = base[:2] + ahead / max(float(np.linalg.norm(ahead)), 1e-9) * 0.6
        action[:3] = nav_grid.drive_command(base, waypoint, goal_yaw, P.NAV_GAIN, P.NAV_VNEAR if dist < 0.5 else P.NAV_VFAR, P.NAV_YAW_GAIN,
                                            P.NAV_WMAX)
        self.reason = f'driving, {dist:.2f} m, yaw err {yaw_err:.2f}'
        if self.stuck_ref is None or np.linalg.norm(self.stuck_ref[0][:2] - base[:2]) > 0.03 or abs(nav_grid.wrap(self.stuck_ref[0][2] - base[2])) > 0.05:
            self.stuck_ref = (base[:3].copy(), self.age)
        elif self.age - self.stuck_ref[1] > P.NAV_STUCK_STEPS and dist > P.NAV_POS_TOL * 2:
            self.counts['nav_stuck'] += 1
            self.path, self.stuck_ref = None, None
            self.escape, self.escape_dir = P.ESCAPE_STEPS, self.free_dir(w)
        return dist, yaw_err

    def free_dir(self, w: World) -> np.ndarray:
        """Unit direction of growing clearance at the base (away from what it is pressed on); backwards on a flat field."""
        self.build_grid(w)
        g = self.grid_raw
        g.clearance(w.base[:2])
        i, j = g.cell(w.base[:2])
        d = g._dist
        v = np.array([d[min(i + 2, d.shape[0] - 1), j] - d[max(i - 2, 0), j], d[i, min(j + 2, d.shape[1] - 1)] - d[i, max(j - 2, 0)]])
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else -np.array([math.cos(w.base[2]), math.sin(w.base[2])])

    def arrived(self, dist: float, yaw_err: float) -> bool:
        if dist < P.NAV_POS_TOL and abs(yaw_err) < P.NAV_YAW_TOL:
            return True
        stalled = self.stuck_ref is not None and self.age - self.stuck_ref[1] > P.NEAR_STALL_STEPS
        return stalled and dist < P.NEAR_POS_TOL and abs(yaw_err) < P.NEAR_YAW_TOL
