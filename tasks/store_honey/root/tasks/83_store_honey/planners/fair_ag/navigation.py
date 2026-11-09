"""Navigation of the fair planner of task 83 over the perceived obstacle grid: a footprint-aware planner over (x, y, yaw) and a holonomic base
command that follows the planned poses. numpy and scipy only.

The base of the R1Pro is a rectangle about 0.72 m wide and 0.66 m long (base_link BASE_FRONT_M from the front, BASE_REAR_M from the back); the
doorway to the cabinet is about 0.87 m wide, so a circle would either not fit or leave the base free to hit the wall: the planner searches poses
(yaw in YAW_BINS bins) whose rectangle is clear of the grid, and turns only where there is room."""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from core.skills_lib import nav_grid

from . import params as P


def footprint_kernel(yaw: float, res: float = P.NAV_RES_M, margin: float = P.BASE_MARGIN_M) -> np.ndarray:
    """Boolean mask of the grid cells the base rectangle covers at yaw, centred on the base_link cell (odd size)."""
    r = max(P.BASE_FRONT_M, P.BASE_REAR_M, P.BASE_HALF_WIDTH_M) + margin + 0.1
    n = int(math.ceil(r / res))
    xs = np.arange(-n, n + 1) * res
    gx, gy = np.meshgrid(xs, xs, indexing='ij')
    c, s = math.cos(yaw), math.sin(yaw)
    bx, by = c * gx + s * gy, -s * gx + c * gy
    return (bx >= -P.BASE_REAR_M - margin) & (bx <= P.BASE_FRONT_M + margin) & (np.abs(by) <= P.BASE_HALF_WIDTH_M + margin)


class PoseGrid:
    """Obstacle grid plus one blocked map per yaw bin: blocked[k][i, j] says the base rectangle at cell (i, j) with yaw k hits an obstacle.
    `shrink` narrows the rectangle (the retry when no route exists at the full size); near[k] is the clearance penalty of a cell (0 at
    CLEAR_M or more from the blocked poses of its yaw bin, 1 next to them), which keeps the path in the middle of a doorway."""

    def __init__(self, grid: nav_grid.Grid, shrink: float = 0.0) -> None:
        self.grid = grid
        self.shrink = shrink
        self.yaws = np.arange(P.YAW_BINS) * 2 * math.pi / P.YAW_BINS - math.pi
        self.blocked = [binary_dilation(grid.blocked, structure=footprint_kernel(yaw, grid.res, P.BASE_MARGIN_M - shrink)[::-1, ::-1])
                        for yaw in self.yaws]
        self.near = [np.clip(1.0 - distance_transform_edt(~b) * grid.res / P.CLEAR_M, 0.0, 1.0) for b in self.blocked]

    def bin_of(self, yaw: float) -> int:
        return int(round((nav_grid.wrap(yaw) + math.pi) / (2 * math.pi / P.YAW_BINS))) % P.YAW_BINS

    def free(self, xy: np.ndarray, yaw: float) -> bool:
        i, j = self.grid.cell(xy)
        return not self.blocked[self.bin_of(yaw)][i, j]

    def plan(self, start_xy: np.ndarray, start_yaw: float, goal_xy: np.ndarray, goal_yaw: float) -> Optional[List[Tuple[np.ndarray, float]]]:
        """Weighted A* over (cell, yaw bin). Returns poses [(xy, yaw)] from the start to the goal, None when there is no route. The start state is
        allowed even when its rectangle touches an obstacle (the robot is already there)."""
        g = self.grid
        s_cell, g_cell = g.cell(start_xy), g.cell(goal_xy)
        sk, gk = self.bin_of(start_yaw), self.bin_of(goal_yaw)
        if self.blocked[gk][g_cell]:
            return None
        start, goal = (s_cell[0], s_cell[1], sk), (g_cell[0], g_cell[1], gk)
        shape = g.shape
        escape = int(P.START_ESCAPE_M / g.res) if self.blocked[sk][s_cell] else -1   # a start pose inside the grown obstacles may leave it
        r2 = math.sqrt(2)
        moves = [(1, 0, 1.), (-1, 0, 1.), (0, 1, 1.), (0, -1, 1.), (1, 1, r2), (1, -1, r2), (-1, 1, r2), (-1, -1, r2)]
        dist: Dict[Tuple[int, int, int], float] = {start: 0.}
        parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        heap = [(0., start)]
        closed = set()
        while heap:
            _, c = heapq.heappop(heap)
            if c in closed:
                continue
            closed.add(c)
            if c == goal:
                break
            i, j, k = c
            nbrs = [((i + dx, j + dy, k), w * g.res) for dx, dy, w in moves]
            nbrs += [((i, j, (k + 1) % P.YAW_BINS), P.TURN_COST), ((i, j, (k - 1) % P.YAW_BINS), P.TURN_COST)]
            for n, cost in nbrs:
                if not (0 <= n[0] < shape[0] and 0 <= n[1] < shape[1]) or n in closed:
                    continue
                if self.blocked[n[2]][n[0], n[1]] and max(abs(n[0] - start[0]), abs(n[1] - start[1])) > escape:
                    continue
                d = dist[c] + cost + P.CLEAR_COST * cost * float(self.near[n[2]][n[0], n[1]])
                if d < dist.get(n, 1e18):
                    dist[n], parent[n] = d, c
                    heapq.heappush(heap, (d + P.HEURISTIC_WEIGHT * math.hypot(goal[0] - n[0], goal[1] - n[1]) * g.res, n))
        if goal not in dist:
            return None
        cells = [goal]
        while cells[-1] != start:
            cells.append(parent[cells[-1]])
        cells.reverse()
        poses = [(g.world((c[0], c[1])), float(self.yaws[c[2]])) for c in cells]
        poses[0] = (np.asarray(start_xy, float), float(start_yaw))
        poses[-1] = (np.asarray(goal_xy, float), float(goal_yaw))
        return thin(poses)


def thin(poses: List[Tuple[np.ndarray, float]]) -> List[Tuple[np.ndarray, float]]:
    """Drop the poses in the middle of straight runs with constant yaw."""
    out = [poses[0]]
    for a, b, c in zip(poses, poses[1:], poses[2:]):
        d1, d2 = b[0] - a[0], c[0] - b[0]
        if abs(b[1] - a[1]) > 1e-9 or abs(c[1] - b[1]) > 1e-9 or np.linalg.norm(d1 / max(np.linalg.norm(d1), 1e-9) - d2 / max(np.linalg.norm(d2), 1e-9)) > 1e-6:
            out.append(b)
    out.append(poses[-1])
    return out


class Navigator:
    """Follows a planned pose path; replans when it stalls, every REPLAN_EVERY controls, or on request."""

    def __init__(self) -> None:
        self.path: Optional[List[Tuple[np.ndarray, float]]] = None
        self.age = 0
        self.ticks = 0
        self.stuck_ref: Optional[Tuple[np.ndarray, int, float]] = None
        self.stalls = 0
        self.no_route = 0
        self.last_reason = ''
        self.shrink = 0.0

    def reset(self) -> None:
        self.path, self.stuck_ref, self.age, self.no_route = None, None, 0, 0

    def lookahead(self, base: np.ndarray) -> Tuple[np.ndarray, float]:
        """The pose LOOKAHEAD_M of path ahead of the point of the path closest to the base (projection onto the segments)."""
        pts = self.path
        best, k0, f0 = 1e9, 0, 0.0
        for k in range(len(pts) - 1):
            a, b = pts[k][0], pts[k + 1][0]
            d = b - a
            L2 = float(d @ d)
            f = 0.0 if L2 < 1e-12 else float(np.clip((base[:2] - a) @ d / L2, 0.0, 1.0))
            dist = float(np.linalg.norm(a + d * f - base[:2])) + 0.02 * abs(nav_grid.wrap(pts[k][1] - base[2]))
            if dist < best - 1e-9:
                best, k0, f0 = dist, k, f
        travelled = 0.0
        pos = pts[k0][0] + (pts[k0 + 1][0] - pts[k0][0]) * f0 if len(pts) > 1 else pts[0][0]
        for k in range(k0, len(pts) - 1):
            a = pos if k == k0 else pts[k][0]
            seg = float(np.linalg.norm(pts[k + 1][0] - a))
            turn = nav_grid.wrap(pts[k + 1][1] - pts[k][1])
            if seg < 1e-9 and abs(turn) > 1e-9:
                return pts[k + 1][0], pts[k + 1][1]    # a turn on the spot: finish it before moving on
            if travelled + seg >= P.LOOKAHEAD_M:
                f = (P.LOOKAHEAD_M - travelled) / max(seg, 1e-9)
                return a + (pts[k + 1][0] - a) * f, pts[k][1]
            travelled += seg
        return pts[-1]

    def step(self, pg_fn, base: np.ndarray, goal_xy: np.ndarray, goal_yaw: float, vmax_far: float = P.NAV_VFAR,
             carry: bool = False) -> Tuple[np.ndarray, float, float]:
        """Base command (vx, vy, wz) towards the pose; pg_fn(shrink) builds a fresh PoseGrid when a plan is needed (a narrower footprint is
        tried when the full one has no route). Returns (command, distance, yaw error)."""
        dist = float(np.linalg.norm(goal_xy - base[:2]))
        yaw_err = nav_grid.wrap(goal_yaw - base[2])
        if self.path is None or self.age >= P.REPLAN_EVERY:
            self.path, self.shrink = None, 0.0
            shrinks = tuple(P.NAV_SHRINKS)
            if self.no_route >= 5:      # repeated failures at the normal sizes: a slimmer footprint as the last resort (slow drive)
                shrinks = shrinks + (0.06, P.BASE_MARGIN_M + 0.02)
            for shrink in shrinks:
                self.path = pg_fn(shrink).plan(base[:2], base[2], goal_xy, goal_yaw)
                if self.path is not None:
                    self.shrink = shrink
                    break
            self.age = 0
        self.age += 1
        self.ticks += 1
        if self.path is None:
            self.no_route += 1
            self.last_reason = 'no route to the base pose'
            return np.zeros(3), dist, yaw_err
        self.no_route = 0
        way_xy, way_yaw = self.lookahead(base)
        if dist < 0.4:
            way_xy, way_yaw = goal_xy, goal_yaw
        vmax = (P.CARRY_VNEAR if carry else P.NAV_VNEAR) if dist < 0.5 else vmax_far
        if self.shrink > 0:
            vmax = min(vmax, P.NAV_VNEAR)
        wmax = P.CARRY_WMAX if carry else P.NAV_WMAX
        cmd = nav_grid.drive_command(base, way_xy, way_yaw, P.NAV_GAIN, vmax, P.NAV_YAW_GAIN, wmax)
        if 1e-3 < abs(cmd[2]) < P.NAV_WMIN:            # the base does not turn under about 0.12 rad/s
            cmd = cmd.copy()
            cmd[2] = math.copysign(P.NAV_WMIN, cmd[2]) if abs(nav_grid.wrap(way_yaw - base[2])) > P.NAV_YAW_TOL / 2 else 0.
        self.last_reason = f'driving, {dist:.2f} m, yaw err {yaw_err:.2f}, {len(self.path)} poses, shrink {self.shrink:.2f}, cmd {np.round(cmd, 2).tolist()}'
        if self.stuck_ref is None or np.linalg.norm(self.stuck_ref[0] - base[:2]) > 0.03 or abs(nav_grid.wrap(self.stuck_ref[2] - base[2])) > 0.05:
            self.stuck_ref = (base[:2].copy(), self.ticks, float(base[2]))
        elif self.ticks - self.stuck_ref[1] > P.STUCK_STEPS and (dist > P.NAV_POS_TOL * 2 or abs(yaw_err) > P.NAV_YAW_TOL * 2):
            self.stalls += 1
            self.path, self.stuck_ref = None, None
        return cmd, dist, yaw_err

    @staticmethod
    def arrived(dist: float, yaw_err: float) -> bool:
        return dist < P.NAV_POS_TOL and abs(yaw_err) < P.NAV_YAW_TOL
