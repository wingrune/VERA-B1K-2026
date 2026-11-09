"""Navigation part of the shared phase machinery of the fair planner of task 69: the route grid from the perceived obstacles (plus the
standing vacuum's footprint), the head footprint test, route lengths, one control of driving along the route. Mixed into phases.base.Core."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.skills_lib import nav_grid

from .. import geometry as G
from .. import params as P
from ..world import World


class NavMixin:
    # ---- grid ------------------------------------------------------------------------------------------------------------------------------
    def build_grid(self, w: World, around: List[np.ndarray], keep_vacuum: bool = False) -> None:
        pts = [w.base[:2]] + [np.asarray(a, float)[:2] for a in around]
        lo, hi = np.min(pts, axis=0) - 2.5, np.max(pts, axis=0) + 2.5
        grid = self.perc.nav_grid(lo, hi)
        if keep_vacuum and w.vacuum is not None:        # the standing vacuum: a box around it that the base keeps out of
            grid.add_box(w.vacuum.xy - P.VAC_KEEP_M, w.vacuum.xy + P.VAC_KEEP_M)
        self.raw_grid = grid
        self.inflated = grid.inflated(P.ROBOT_RADIUS)
        self.inflated_far = None
        if keep_vacuum and w.vacuum is not None:        # the long drives keep farther away: the route cuts its corners
            far = self.perc.nav_grid(lo, hi)
            far.add_box(w.vacuum.xy - P.VAC_KEEP_FAR_M, w.vacuum.xy + P.VAC_KEEP_FAR_M)
            self.inflated_far = far.inflated(P.ROBOT_RADIUS)
        self.inflated_tight = grid.inflated(P.ROBOT_RADIUS_TIGHT)
        self.routes = {}

    def free_pose(self, xy: np.ndarray, tight: bool = False) -> bool:
        g = self.inflated_tight if tight else self.inflated
        c = g.cell(xy)
        return not g.blocked[c]

    def head_free(self, centre_xy: np.ndarray, cube_yaw: float) -> bool:
        """The vacuum head rectangle.

        The vacuum head rectangle (plus margin) around the cube centre stays out of every blocked cell of the raw grid."""
        a, b = P.HEAD_HALF[0] + P.HEAD_MARGIN_M, P.HEAD_HALF[1] + P.HEAD_MARGIN_M
        c, s = math.cos(cube_yaw), math.sin(cube_yaw)
        u, v = np.array([c, s]), np.array([-s, c])
        for iu in np.linspace(-a, a, 7):
            for iv in np.linspace(-b, b, 6):
                if self.raw_grid.blocked[self.raw_grid.cell(np.asarray(centre_xy, float) + iu * u + iv * v)]:
                    return False
        return True

    def waypoint_for(self, w: World, stand_xy: np.ndarray) -> Optional[np.ndarray]:
        """The point WAYPOINT_M outside a grasp stand.

        The point WAYPOINT_M outside a grasp stand (radially from the vacuum, turned about the stand when that is not free) that is free on the long-drive grid."""
        u0 = G.unit(stand_xy - w.vacuum.xy)
        far = self.inflated_far if self.inflated_far is not None else self.inflated
        for ang in P.WAYPOINT_TURNS:
            wp = stand_xy + (G.rot_yaw(ang)[:2, :2] @ u0) * P.WAYPOINT_M
            if not far.blocked[far.cell(wp)] and float(np.linalg.norm(wp - w.vacuum.xy)) > float(np.linalg.norm(stand_xy - w.vacuum.xy)) - 0.05:
                return wp
        return None

    def route_len(self, w: World, xy: np.ndarray, tight: bool = False) -> float:
        if float(np.linalg.norm(xy - w.base[:2])) < 0.03:
            return 0.
        key = (tuple(np.round(w.base[:2], 2)), tuple(np.round(xy, 2)), tight)
        if key not in self.routes:
            path = (self.inflated_tight if tight else self.inflated).path(w.base[:2], xy)
            self.routes[key] = float('inf') if path is None else nav_grid.path_length(path)
        return self.routes[key]

    def pick(self, w: World, cands: List[Dict[str, Any]], check=None) -> Optional[Dict[str, Any]]:
        """Cheapest candidate by route length plus its own cost.

        Cheapest candidate by route length plus its own cost; `check(cand)` (optional) rejects a candidate before its route is planned."""
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
            if P.FAR_ROUTE and self.cand_relax < 2 and w.vacuum is not None and math.isfinite(d) and self.inflated_far is not None:
                wp = self.waypoint_for(w, cand['stand'][:2])
                if wp is not None:        # the drive to the stand goes through its waypoint on the long-drive grid: no route there, no candidate
                    far = self.routes.get(('far', tuple(np.round(w.base[:2], 2)), tuple(np.round(wp, 2))))
                    if far is None:
                        path = self.inflated_far.path(w.base[:2], wp)
                        far = self.routes[('far', tuple(np.round(w.base[:2], 2)), tuple(np.round(wp, 2)))] = float('inf') if path is None else nav_grid.path_length(path)
                    d = far + P.WAYPOINT_M
            if not math.isfinite(d):
                st['no_route'] += 1
                continue
            total = cand['cost'] - cand['ecost'] + d
            if best is None or total < best[0]:
                best = (total, cand)
        if best is None and st['no_route'] == st['routed'] and st['routed'] and P.RELAX_RADIUS and not getattr(self, '_relaxed', False):
            self._relaxed = True                       # every route failed: the start is boxed in; retry once with the tight footprint
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

    def navigate(self, action, w, goal_xy, goal_yaw, vfar=P.NAV_VFAR, vnear=P.NAV_VNEAR, wmax=P.NAV_WMAX, tight=False, far=False, gain=P.NAV_GAIN) -> Tuple[float, float]:
        """One control of driving to a base pose along the route.

        One control of driving to a base pose along the route (the arms are the caller's business). Returns (distance, yaw error)."""
        dist = float(np.linalg.norm(goal_xy - w.base[:2]))
        yaw_err = G.wrap(goal_yaw - w.base[2])
        grid = self.inflated_tight if tight else (self.inflated_far if far and self.inflated_far is not None else self.inflated)
        if self.path is None or self.path_age >= P.NAV_REPLAN_EVERY:
            self.path = grid.path(w.base[:2], goal_xy) if dist > 0.05 else [goal_xy]
            if self.path is None and far:                     # no route on the long-drive grid: the standard one, then the tight one
                self.path = self.inflated.path(w.base[:2], goal_xy) or self.inflated_tight.path(w.base[:2], goal_xy)
            self.path_age = 0
            self.counts['replans'] += 1
        self.path_age += 1
        if self.path is None:
            self.path = [goal_xy]                      # no grid route (the start cell may be inside the inflation): straight line
        near = int(np.argmin([np.linalg.norm(p - w.base[:2]) for p in self.path]))
        waypoint = self.path[min(near + 1, len(self.path) - 1)]
        for p in self.path[near:]:
            waypoint = p
            if np.linalg.norm(p - w.base[:2]) > 0.4:
                break
        if dist < 0.4:
            waypoint = goal_xy
        v = vnear if dist < 0.3 else vfar
        action[:3] = nav_grid.drive_command(w.base, waypoint, goal_yaw, gain, v, P.NAV_YAW_GAIN, wmax)
        if self.perc.held is not None and self.cmd is not None:     # the held vacuum swings on the arm: the base speeds up and slows down gently
            prev = self.cmd[:3]
            lim = np.array([P.ACCEL_LIN, P.ACCEL_LIN, P.ACCEL_ANG])
            action[:3] = prev + np.clip(action[:3] - prev, -lim, lim)
        if (self.stuck_ref is None or np.linalg.norm(self.stuck_ref[0] - w.base[:2]) > 0.03 or abs(G.wrap(self.stuck_ref[2] - w.base[2])) > 0.1):
            self.stuck_ref = (w.base[:2].copy(), self.age, float(w.base[2]))
        elif self.age - self.stuck_ref[1] > P.NAV_STUCK_STEPS and (dist > 0.06 or abs(yaw_err) > 0.1):
            self.counts['nav_stuck'] += 1
            self.path, self.stuck_ref = None, None
        return dist, yaw_err

    def arrived(self, dist, yaw_err, pos_tol=P.NAV_POS_TOL, yaw_tol=P.NAV_YAW_TOL) -> bool:
        """Within tolerance, or stalled for NEAR_STALL_STEPS...

        Within tolerance, or stalled for NEAR_STALL_STEPS within the looser near tolerances."""
        if dist < pos_tol and abs(yaw_err) < yaw_tol:
            return True
        stalled = self.stuck_ref is not None and self.age - self.stuck_ref[1] > P.NEAR_STALL_STEPS
        return stalled and dist < P.NEAR_POS_TOL and abs(yaw_err) < P.NEAR_YAW_TOL
