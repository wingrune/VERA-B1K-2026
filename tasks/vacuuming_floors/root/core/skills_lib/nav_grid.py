"""Occupancy grid, A* and base velocity commands for a holonomic base. Pure numpy and scipy, no simulator.

Grid: world rectangle [lo, hi] at `res` metres per cell. Boxes (axis-aligned, from object AABBs) are blocked, `inflated(radius)` grows the
blocked set by the robot radius. `extra_cost` (optional float array of the grid shape, >= 0) makes cells dearer for `path` (a step into a cell
costs its length times 1 + extra_cost), e.g. cells never seen by the sensors; shortcuts of the smoothing never cost more than the A* route. `path` is 8-connected A* with line-of-sight smoothing; a start or goal inside the inflated set is moved to
the nearest free cell first. `drive_command` turns a waypoint and a heading into the native base action (vx, vy, wz) in [-1, 1].
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

BASE_SPEED = .75  # m/s at action 1.0 (native base output limit)
BASE_TURN = 1.0  # rad/s at action 1.0


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Grid:
    def __init__(self, lo: Sequence[float], hi: Sequence[float], res: float = .05) -> None:
        self.lo, self.hi, self.res = np.asarray(lo, float), np.asarray(hi, float), float(res)
        self.shape = tuple(int(math.ceil(v)) for v in (self.hi - self.lo) / self.res)
        self.blocked = np.zeros(self.shape, bool)
        self.extra_cost: Optional[np.ndarray] = None

    def cell(self, xy: Sequence[float]) -> Tuple[int, int]:
        c = np.floor((np.asarray(xy, float) - self.lo) / self.res).astype(int)
        return int(np.clip(c[0], 0, self.shape[0] - 1)), int(np.clip(c[1], 0, self.shape[1] - 1))

    def world(self, cell: Sequence[int]) -> np.ndarray:
        return self.lo + (np.asarray(cell, float) + .5) * self.res

    def add_box(self, lo_xy: Sequence[float], hi_xy: Sequence[float]) -> None:
        a, b = self.cell(lo_xy), self.cell(hi_xy)
        self.blocked[a[0]:b[0] + 1, a[1]:b[1] + 1] = True

    def inflated(self, radius: float) -> 'Grid':
        out = Grid(self.lo, self.hi, self.res)
        out.blocked = distance_transform_edt(~self.blocked) * self.res < radius
        out.extra_cost = self.extra_cost
        return out

    def clearance(self, xy: Sequence[float]) -> float:
        """Distance (m) from xy to the nearest blocked cell of this grid (0 inside)."""
        if not hasattr(self, '_dist'):
            self._dist = distance_transform_edt(~self.blocked) * self.res
        return float(self._dist[self.cell(xy)])

    def nearest_free(self, xy: Sequence[float], max_r: float = .6) -> Optional[np.ndarray]:
        c = self.cell(xy)
        if not self.blocked[c]:
            return np.asarray(xy, float)
        _, idx = distance_transform_edt(self.blocked, return_indices=True)
        n = (int(idx[0][c]), int(idx[1][c]))
        w = self.world(n)
        return w if np.linalg.norm(w - np.asarray(xy, float)) <= max_r else None

    def line_clear(self, a: Sequence[float], b: Sequence[float]) -> bool:
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.linalg.norm(b - a) / (self.res * .5)) + 1)
        return not any(self.blocked[self.cell(a + (b - a) * k / (n - 1))] for k in range(n))

    def path(self, start: Sequence[float], goal: Sequence[float]) -> Optional[List[np.ndarray]]:
        """Smoothed waypoint list from start to goal (world xy) over free cells, None if there is no route."""
        s, g = self.nearest_free(start), self.nearest_free(goal)
        if s is None or g is None:
            return None
        a, b = self.cell(s), self.cell(g)
        dist = {a: 0.}
        parent = {}
        heap = [(0., a)]
        steps = [(1, 0, 1.), (-1, 0, 1.), (0, 1, 1.), (0, -1, 1.), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
        while heap:
            _, c = heapq.heappop(heap)
            if c == b:
                break
            for dx, dy, w in steps:
                n = (c[0] + dx, c[1] + dy)
                if not (0 <= n[0] < self.shape[0] and 0 <= n[1] < self.shape[1]) or self.blocked[n]:
                    continue
                if dx and dy and (self.blocked[c[0] + dx, c[1]] or self.blocked[c[0], c[1] + dy]):
                    continue
                d = dist[c] + w * (1. if self.extra_cost is None else 1. + float(self.extra_cost[n]))
                if d < dist.get(n, 1e18):
                    dist[n], parent[n] = d, c
                    h = math.hypot(b[0] - n[0], b[1] - n[1])
                    heapq.heappush(heap, (d + h, n))
        if b not in dist:
            return None
        cells = [b]
        while cells[-1] != a:
            cells.append(parent[cells[-1]])
        pts = [s] + [self.world(c) for c in reversed(cells[1:-1])] + [g]
        return self.smooth(pts)

    def segment_cost(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Cost of the straight segment a-b: its length times 1 + extra_cost of the cells it crosses."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        length = float(np.linalg.norm(b - a))
        if self.extra_cost is None or length < 1e-9:
            return length
        n = max(2, int(length / (self.res * .5)) + 1)
        return length * (1. + float(np.mean([self.extra_cost[self.cell(a + (b - a) * k / (n - 1))] for k in range(n)])))

    def smooth(self, pts: List[np.ndarray]) -> List[np.ndarray]:
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not (self.line_clear(pts[i], pts[j]) and (
                    self.extra_cost is None or self.segment_cost(pts[i], pts[j]) <= 1.02 * sum(self.segment_cost(pts[k], pts[k + 1]) for k in range(i, j)))):
                j -= 1
            out.append(pts[j])
            i = j
        return out


def path_length(path: Sequence[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in zip(path, path[1:])))


def drive_command(pose: Sequence[float], waypoint: Sequence[float], heading: float, gain: float = .9, vmax: float = .25,
                  yaw_gain: float = .8, wmax: float = .4) -> np.ndarray:
    """Base action (vx, vy, wz) in the robot frame: go to `waypoint` (world xy) and turn to `heading` (world yaw). pose = (x, y, yaw)."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    d = np.asarray(waypoint, float) - np.asarray(pose[:2], float)
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    n = float(np.linalg.norm(local))
    v = local * gain
    if n > 1e-9 and np.linalg.norm(v) > vmax:
        v = local / n * vmax
    return np.array([v[0], v[1], float(np.clip(wrap(heading - pose[2]) * yaw_gain, -wmax, wmax))])
