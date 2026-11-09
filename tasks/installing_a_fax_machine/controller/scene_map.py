"""Static prior of the fax office: free space for the base, a height field for the body.

The map is built offline from the scene's collision geometry (dev/build_scene_map.py) and is the
same for every task instance. Grid frame "M" is the scene frame of the map file. It contains glass
partitions and windows, which the depth cameras cannot see.
"""
import heapq
import json
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation


class SceneMap:
    def __init__(self, path, base_radius=.40):
        data = np.load(path); meta = json.loads(str(data['meta']))
        self.x0, self.y0, self.res = meta['x0'], meta['y0'], meta['res']
        self.shape = tuple(meta['shape']); self.meta = meta; self.base_radius = base_radius
        self.top = data['top'].astype(np.float32); self.desk = data['desk']; self.glass = data['glass']; self.upper = data['upper']
        self.top_static = data['top_static'].astype(np.float32)
        movable = data['movable']
        block = data['low'] | data['mid']
        self.block = block
        # Chairs can be pushed; planning may ignore them when asked to.
        self.block_static = block & ~binary_dilation(movable, iterations=2)
        self.distance = distance_transform_edt(~block) * self.res
        self.distance_static = distance_transform_edt(~self.block_static) * self.res
        self.movable = movable
        self._stencils = {}
        self._di, self._dj, self._d2 = self._stencil(.22)
        # Coarse grid for route planning.
        self.coarse = 2

    def _stencil(self, radius):
        """Cell offsets that cover a disc of @radius around a cell, and their squared distances (m^2)."""
        reach = int(np.ceil(radius / self.res))
        if reach not in self._stencils:
            k = np.arange(-reach, reach + 1); di, dj = np.meshgrid(k, k, indexing='ij')
            self._stencils[reach] = (di, dj, (di**2 + dj**2) * self.res**2)
        return self._stencils[reach]

    def cell(self, xy):
        xy = np.asarray(xy, float)
        return np.floor((xy[..., 0] - self.x0) / self.res).astype(int), np.floor((xy[..., 1] - self.y0) / self.res).astype(int)

    def inside(self, i, j):
        return (i >= 0) & (i < self.shape[0]) & (j >= 0) & (j < self.shape[1])

    def clearance(self, xy, static_only=False):
        """Distance from @xy to the nearest geometry that blocks the base (metres); 0 outside the map."""
        i, j = self.cell(xy); ok = self.inside(i, j)
        field = self.distance_static if static_only else self.distance
        return np.where(ok, field[np.clip(i, 0, self.shape[0] - 1), np.clip(j, 0, self.shape[1] - 1)], 0.)

    def base_free(self, xy, margin=0., static_only=False):
        return bool(self.clearance(xy, static_only) > self.base_radius + margin)

    def sphere_clearance(self, centers, radii, ignore_below=None, static_only=False):
        """Per-sphere gap (m) between each sphere and the height field; negative means penetration.

        static_only: leave out objects that can be pushed (chairs), for places the base has cleared.
        ignore_below: optional per-sphere height; geometry lower than it is disregarded for that
        sphere (used for the fingers, which are meant to work at floor level).
        """
        centers = np.asarray(centers, float); radii = np.asarray(radii, float)
        # The stencil must cover the largest sphere. The robot's own spheres fit the default one;
        # a larger probe (the lean-room test of stance planning) gets a stencil of its own size.
        # Before m05 every sphere used the default: geometry further than 0.22 m from the centre
        # of a large probe was not seen at all.
        di, dj, d2 = (self._di, self._dj, self._d2) if radii.max(initial=0.) <= .22 else self._stencil(float(radii.max()))
        i, j = self.cell(centers[:, :2])
        ii = i[:, None, None] + di[None]; jj = j[:, None, None] + dj[None]
        ok = self.inside(ii, jj)
        field = self.top_static if static_only else self.top
        top = np.where(ok, field[np.clip(ii, 0, self.shape[0] - 1), np.clip(jj, 0, self.shape[1] - 1)], 2.)
        if ignore_below is not None:
            top = np.where(top < np.asarray(ignore_below, float)[:, None, None], -1., top)
        r2 = radii[:, None, None]**2; within = d2[None] < r2
        drop = np.sqrt(np.maximum(r2 - d2[None], 0.))
        gap = centers[:, 2, None, None] - drop - top
        return np.where(within, gap, np.inf).min(axis=(1, 2))

    def path(self, start, goal, margin=.02, static_only=False, max_expansions=60000, keepout=()):
        """A* over base positions; returns a list of xy waypoints (simplified) or None.

        keepout: (xy, radius) discs the base centre must stay out of (objects that are not in the map).
        """
        c = self.coarse; res = self.res * c
        field = (self.distance_static if static_only else self.distance)[::c, ::c]
        if keepout:
            field = field.copy(); gx = self.x0 + (np.arange(field.shape[0]) + .5) * res; gy = self.y0 + (np.arange(field.shape[1]) + .5) * res
            for centre, radius in keepout:
                away = np.hypot(gx[:, None] - centre[0], gy[None, :] - centre[1]) - radius
                # Express the disc as "distance to an obstacle" so the usual clearance test applies.
                field = np.minimum(field, away + self.base_radius + margin)
        free = field > self.base_radius + margin
        def to_cell(xy):
            return (int((xy[0] - self.x0) / res), int((xy[1] - self.y0) / res))
        def to_xy(cell):
            return np.array([self.x0 + (cell[0] + .5) * res, self.y0 + (cell[1] + .5) * res])
        a, b = to_cell(start), to_cell(goal); n, m = free.shape
        if not (0 <= b[0] < n and 0 <= b[1] < m) or not (0 <= a[0] < n and 0 <= a[1] < m):
            return None
        free = free.copy()
        # The robot may already stand closer to something than the planning margin: let it leave.
        for cell in (a, b):
            if not free[cell]:
                xs, ys = np.ogrid[:n, :m]; near = (xs - cell[0])**2 + (ys - cell[1])**2 <= (.12 / res)**2
                free |= near & (field > self.base_radius - .10)
        if not free[b] or not free[a]:
            return None
        queue = [(0., 0., a)]; cost = {a: 0.}; previous = {}; expanded = 0
        while queue and expanded < max_expansions:
            expanded += 1
            _, g, u = heapq.heappop(queue)
            if g > cost[u] + 1e-9:
                continue
            if u == b:
                cells = [u]
                while u != a:
                    u = previous[u]; cells.append(u)
                points = [np.asarray(start, float)[:2]] + [to_xy(x) for x in cells[::-1][1:-1]] + [np.asarray(goal, float)[:2]]
                return self.simplify(points, margin, static_only, keepout)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                v = (u[0] + dx, u[1] + dy)
                if not (0 <= v[0] < n and 0 <= v[1] < m) or not free[v]:
                    continue
                if dx and dy and not (free[u[0] + dx, u[1]] and free[u[0], u[1] + dy]):
                    continue
                # Prefer the middle of corridors: small penalty when close to obstacles.
                step = np.hypot(dx, dy) * res * (1. + .6 * max(0., .25 - (field[v] - self.base_radius)) / .25)
                ng = g + step
                if ng < cost.get(v, np.inf):
                    cost[v] = ng; previous[v] = u
                    heapq.heappush(queue, (ng + np.hypot(v[0] - b[0], v[1] - b[1]) * res, ng, v))
        return None

    def segment_free(self, a, b, margin=.02, static_only=False, keepout=()):
        a = np.asarray(a, float); b = np.asarray(b, float)
        count = max(2, int(np.linalg.norm(b - a) / (self.res * 1.5)) + 1)
        points = a[None] + np.linspace(0, 1, count)[:, None] * (b - a)[None]
        if not np.all(self.clearance(points, static_only) > self.base_radius + margin - .03):
            return False
        return all(np.all(np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1]) > radius - .03) for centre, radius in keepout)

    def simplify(self, points, margin=.02, static_only=False, keepout=()):
        out = [points[0]]; i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self.segment_free(points[i], points[j], margin, static_only, keepout):
                j -= 1
            out.append(points[j]); i = j
        return out
