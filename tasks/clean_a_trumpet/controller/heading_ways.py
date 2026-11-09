"""Ways for the holonomic base at one fixed heading, through places too narrow for the route planner's disc.

The route planner (scene_map.path) takes the base for a disc 0.40 m round its middle: it does not fit into the nook by
the toilet (0.93 m wide), along the strip between the bath and the basins' counter (0.77 m), nor by the shower. The
chassis itself, a rectangle 0.64 m long and 0.69 m wide, fits into all of them at the right heading, and the base
moves sideways as well as ahead: it can turn where it has room to, and then slide in without turning.

For a heading the cells where the chassis keeps clear of what the map has at chassis height are found at once (the
footprint, turned to the heading, correlated with the cells too near something); a breadth-first search over them from
the goal gives the way in, and its start is a cell where the chassis can turn round (clear at every heading) and the
route planner can stand.
"""
from collections import deque
import numpy as np
from scipy.signal import fftconvolve

TURNS = 16                          # headings checked for a cell to turn round in


class HeadingWays:
    def __init__(self, scene, footprint, lo, hi, margin=.02):
        self.scene = scene; self.footprint = footprint; self.margin = margin; self.res = scene.res
        i0, j0 = scene.cell(np.asarray(lo, float)); i1, j1 = scene.cell(np.asarray(hi, float))
        self.i0, self.j0 = int(i0), int(j0); self.shape = (int(i1) - self.i0 + 1, int(j1) - self.j0 + 1)
        rear, front, half = footprint
        self.reach = int(np.ceil(np.hypot(max(-rear, front), half) / self.res)) + 1
        near = scene.chassis_distance[self.i0:self.i0 + self.shape[0], self.j0:self.j0 + self.shape[1]] <= margin
        self.blocked = np.pad(near, self.reach, constant_values=True).astype(np.float32)   # (outside the region: blocked)
        self.cache = {}; self._turnable = None

    def free(self, yaw):
        """Cells of the region where the chassis at @yaw keeps @margin from everything at chassis height."""
        key = int(round(float(np.degrees(yaw)))) % 360
        if key not in self.cache:
            rear, front, half = self.footprint; r = self.reach
            u, v = np.meshgrid(np.arange(rear, front + 1e-9, self.res / 2), np.arange(-half, half + 1e-9, self.res / 2), indexing='ij')
            c, s = np.cos(yaw), np.sin(yaw)
            di = np.round((c * u - s * v) / self.res).astype(int); dj = np.round((s * u + c * v) / self.res).astype(int)
            kernel = np.zeros((2 * r + 1, 2 * r + 1), np.float32); kernel[di.ravel() + r, dj.ravel() + r] = 1.
            hit = fftconvolve(self.blocked, kernel[::-1, ::-1], mode='same')[r:-r, r:-r]
            self.cache[key] = hit < .5
        return self.cache[key]

    def turnable(self):
        """Cells where the chassis can turn round on the spot and the route planner's disc stands."""
        if self._turnable is None:
            ok = np.ones(self.shape, bool)
            for a in np.arange(TURNS) * 2 * np.pi / TURNS:
                ok &= self.free(a)
            i = np.arange(self.shape[0])[:, None] + self.i0; j = np.arange(self.shape[1])[None, :] + self.j0
            ok &= self.scene.distance[i, j] > self.scene.base_radius + .02
            self._turnable = ok
        return self._turnable

    def xy(self, cell):
        return np.array([self.scene.x0 + (cell[0] + self.i0 + .5) * self.res, self.scene.y0 + (cell[1] + self.j0 + .5) * self.res])

    def cell(self, xy):
        i, j = self.scene.cell(np.asarray(xy, float)); return int(i) - self.i0, int(j) - self.j0

    def plan(self, here, goal, yaw, starts=6):
        """Ways to @goal (xy) at heading @yaw: [(start xy where the base turns to @yaw, [waypoints to the goal])], the
        shortest overall first (at most @starts), or [] if the chassis does not get there at that heading."""
        free = self.free(yaw); g = self.cell(goal)
        if not (0 <= g[0] < self.shape[0] and 0 <= g[1] < self.shape[1]) or not free[g]:
            return []
        dist = np.full(self.shape, -1, np.int32); dist[g] = 0; queue = deque([g])
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        while queue:
            i, j = queue.popleft()
            for di, dj in steps:
                a, b = i + di, j + dj
                if 0 <= a < self.shape[0] and 0 <= b < self.shape[1] and dist[a, b] < 0 and free[a, b]:
                    dist[a, b] = dist[i, j] + 1; queue.append((a, b))
        cand = np.argwhere((dist >= 0) & self.turnable())
        if not len(cand):
            return []
        here = np.asarray(here, float)[:2]
        cost = dist[cand[:, 0], cand[:, 1]] * self.res + np.hypot(*(np.array([self.xy(c) for c in cand]) - here).T)
        out = []; taken = []
        for k in np.argsort(cost):
            start = tuple(cand[k])
            if any(abs(start[0] - t[0]) + abs(start[1] - t[1]) < 10 for t in taken):
                continue                # (starts at least 0.2 m apart)
            taken.append(start)
            out.append((self.xy(start), self.simplify(self.descend(dist, start), free)))
            if len(out) >= starts:
                break
        return out

    def escape(self, here, yaw, accept=None, start_free=None):
        """From @here at heading @yaw, the shortest slide (keeping the heading) to a cell where the chassis can turn
        round and the route planner's disc stands (and @accept(xy), if given, holds): [waypoints], [] if already
        there, None if there is none. @start_free: cells taken as free too, near the start only (where the chassis
        stands it fits, though with less room than this planner keeps)."""
        free = self.free(yaw); h = self.cell(here); turn = self.turnable()
        if start_free is not None:
            near = np.zeros(self.shape, bool); near[max(0, h[0] - 4):h[0] + 5, max(0, h[1] - 4):h[1] + 5] = True
            free = free | (start_free & near)
        if accept is not None:
            turn = turn.copy(); checked = {}
            base_turn = turn

            class Gate:
                def __getitem__(self_, ij):
                    if not base_turn[ij]:
                        return False
                    if ij not in checked:
                        checked[ij] = bool(accept(self.xy(ij)))
                    return checked[ij]
            turn = Gate()
        if not (0 <= h[0] < self.shape[0] and 0 <= h[1] < self.shape[1]):
            return None
        if turn[h]:
            return []
        dist = np.full(self.shape, -1, np.int32); dist[h] = 0; queue = deque([h]); found = None
        while queue and found is None:
            i, j = queue.popleft()
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                a, b = i + di, j + dj
                if 0 <= a < self.shape[0] and 0 <= b < self.shape[1] and dist[a, b] < 0 and (free[a, b] or (a, b) == h):
                    dist[a, b] = dist[i, j] + 1; queue.append((a, b))
                    if turn[a, b]:
                        found = (a, b); break
        if found is None:
            return None
        # (the way back from the found cell to here, then reversed)
        back = self.descend(dist, found)[::-1]
        return self.simplify(back, free | (np.arange(self.shape[0])[:, None] == h[0]) & (np.arange(self.shape[1])[None, :] == h[1]))

    def descend(self, dist, start):
        path = [start]; i, j = start
        while dist[i, j] > 0:
            best = None
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                a, b = i + di, j + dj
                if 0 <= a < self.shape[0] and 0 <= b < self.shape[1] and 0 <= dist[a, b] < dist[i, j] and (best is None or dist[a, b] < dist[best]):
                    best = (a, b)
            i, j = best; path.append(best)
        return path

    def simplify(self, path, free):
        """The cells of @path as few straight legs (each clear all along at the heading): their ends' xy."""
        out = []; k = 0
        while k < len(path) - 1:
            far = k + 1
            for m in range(len(path) - 1, k, -1):
                a, b = np.array(path[k], float), np.array(path[m], float); n = int(np.ceil(np.abs(b - a).max())) + 1
                cells = np.round(a + np.linspace(0., 1., n)[:, None] * (b - a)).astype(int)
                if free[cells[:, 0], cells[:, 1]].all():
                    far = m; break
            out.append(self.xy(path[far])); k = far
        return out
