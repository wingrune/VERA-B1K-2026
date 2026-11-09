"""Navigation part of the shared phase machinery of the fair planner of task 35: the route grid from the perceived obstacles (plus the fitted
tripod's footprint), stand candidates around a target with their straight-line costs, the cheapest routed candidate, one control of
driving along the route. Mixed into phases.base.Core."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.skills_lib import nav_grid

from .. import models as MD
from .. import params as P
from ..world import World


class NavMixin:
    # ---- grid and stands -----------------------------------------------------------------------------------------------------------------
    def build_grid(self, w: World, around: List[np.ndarray]) -> None:
        pts = [w.base[:2]] + [np.asarray(a, float)[:2] for a in around]
        lo, hi = np.min(pts, axis=0) - 2.5, np.max(pts, axis=0) + 2.5
        grid = self.perc.nav_grid(lo, hi)
        if P.TRIPOD_IN_GRID and w.tripod is not None:     # thin legs leave few depth hits: block the fitted tripod's low footprint
            low = MD.TRIPOD_VERTS[MD.TRIPOD_VERTS[:, 2] < MD.TRIPOD_VERTS[:, 2].min() + P.TRIPOD_GRID_H_M]
            xy = low[:, :2] @ w.tripod.T[:2, :2].T + w.tripod.T[:2, 3]
            self.fill_hull(grid, xy)
        self.raw_grid = grid
        self.inflated = grid.inflated(P.ROBOT_RADIUS)
        self.routes = {}

    @staticmethod
    def fill_hull(grid: nav_grid.Grid, pts: np.ndarray) -> None:
        """Block the cells inside the convex hull of the points (xy)."""
        from scipy.spatial import ConvexHull
        pts = np.asarray(pts, float)
        try:
            eq = ConvexHull(pts).equations
        except Exception:  # noqa: BLE001 - degenerate (collinear) points: their box
            grid.add_box(pts.min(0), pts.max(0))
            return
        a, b = grid.cell(pts.min(0)), grid.cell(pts.max(0))
        for i in range(a[0], b[0] + 1):
            for j in range(a[1], b[1] + 1):
                if np.all(eq[:, :2] @ grid.world((i, j)) + eq[:, 2] <= grid.res * 0.71):
                    grid.blocked[i, j] = True

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

    def stands_around(self, w: World, centre: np.ndarray, rings) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Base poses (N, 3) on rings around `centre`, facing it (with yaw offsets), free in the inflated grid, cheapest first by
        straight-line cost; the current base pose first (it reaches far targets only with the trunk stretched, hence its penalty).
        Returns stands (N, 3), straight-line costs (N,) and penalties (N,)."""
        c = np.asarray(centre, float)[:2]
        far = max(0., float(np.linalg.norm(w.base[:2] - c)) - max(rings) - 0.05)
        rows = [(0., w.base[0], w.base[1], w.base[2])]
        for r in rings:
            for bearing in np.arange(P.N_BEARINGS) * 2 * math.pi / P.N_BEARINGS:
                xy = c + r * np.array([math.cos(bearing), math.sin(bearing)])
                if not self.free_pose(xy):
                    continue
                for dy in P.YAW_OFFSETS:
                    yaw = nav_grid.wrap(bearing + math.pi + dy)
                    rows.append((float(np.linalg.norm(xy - w.base[:2])) + 0.15 * abs(nav_grid.wrap(yaw - w.base[2])), xy[0], xy[1], yaw))
        rows = [rows[0]] + sorted(rows[1:])[:P.MAX_STANDS - 1]
        arr = np.array(rows, float)
        penalty = np.zeros(len(arr))
        penalty[0] = P.STAND_FAR_COST * far          # staying costs more the farther the target lies outside the rings
        return arr[:, 1:], arr[:, 0], penalty

    def pick(self, w: World, cands: List[Dict[str, Any]], check=None) -> Optional[Dict[str, Any]]:
        """Cheapest candidate by route length plus its own cost; `check(cand)` (optional) rejects a candidate before its route is planned."""
        cands.sort(key=lambda c: c['cost'])
        best, routed = None, 0
        st = self.pick_stats = dict(candidates=len(cands), rejected=0, routed=0, no_route=0)
        for cand in cands:
            if routed >= P.ROUTE_CHECKS:
                break
            if check is not None and not check(cand):
                st['rejected'] += 1
                continue
            routed += 1
            st['routed'] += 1
            d = self.route_len(w, cand['stand'][:2])
            if not math.isfinite(d):
                st['no_route'] += 1
                continue
            total = cand['cost'] - cand['ecost'] + d
            if best is None or total < best[0]:
                best = (total, cand)
        st['start_clear'] = round(self.inflated.clearance(w.base[:2]), 3)
        if best is None and st['no_route'] == st['routed'] and st['routed'] and P.RELAX_RADIUS and not getattr(self, '_relaxed', False):
            # every route failed: the start is boxed in by the grid (dilated hits next to the robot); retry once with the tight footprint
            self._relaxed = True
            self.inflated = self.raw_grid.inflated(P.RELAX_RADIUS)
            self.routes = {}
            out = self.pick(w, cands, check)
            self._relaxed = False
            if out is not None:
                self.pick_stats['relaxed'] = True
            return out
        if best is None:
            return None
        cands[:] = [c for c in cands if c is not best[1]]
        best[1]['total'] = best[0]
        return best[1]

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
        v = min(vmax, P.NAV_VNEAR) if dist < 0.5 else vmax
        action[:3] = nav_grid.drive_command(w.base, waypoint, goal_yaw, P.NAV_GAIN, v, P.NAV_YAW_GAIN, wmax)
        self.reason = f'driving, {dist:.2f} m, yaw err {yaw_err:.2f}'
        if self.stuck_ref is None or np.linalg.norm(self.stuck_ref[0] - w.base[:2]) > 0.03:
            self.stuck_ref = (w.base[:2].copy(), self.age)
        elif self.age - self.stuck_ref[1] > P.NAV_STUCK_STEPS and dist > P.NAV_POS_TOL * 2:
            self.counts['nav_stuck'] += 1
            self.path, self.stuck_ref = None, None
        return dist, yaw_err

    def arrived(self, dist: float, yaw_err: float) -> bool:
        if dist < P.NAV_POS_TOL and abs(yaw_err) < P.NAV_YAW_TOL:
            return True
        stalled = self.stuck_ref is not None and self.age - self.stuck_ref[1] > P.NEAR_STALL_STEPS
        return stalled and dist < P.NEAR_POS_TOL and abs(yaw_err) < P.NEAR_YAW_TOL
