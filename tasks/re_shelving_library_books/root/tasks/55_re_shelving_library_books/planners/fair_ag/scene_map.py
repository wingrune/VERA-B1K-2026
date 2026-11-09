"""The offline static scene map of the fair planner (map_data.py, built from training episodes by tools/build_map.py) as numpy grids and boxes:
occupancy for localisation and navigation, the desk and bookcase boxes with their cubbies, the three book models, the robot frame height."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage

from .map_data import MAP
from .world import Box, Case

LIKELIHOOD_SIGMA_M = 0.07
FREE_PENALTY, FREE_PENALTY_FROM_M = 0.5, 0.15   # a matched point on floor seen free in the map costs this much
UNSEEN_PENALTY = 0.3                            # and one where the training views never reached (behind the walls)


def unrle(runs: List[int], shape: Tuple[int, int]) -> np.ndarray:
    vals = np.zeros(len(runs), bool)
    vals[1::2] = True
    return np.repeat(vals, runs).reshape(shape)


class SceneMap:
    def __init__(self) -> None:
        m = MAP
        self.res = float(m['res'])
        self.origin = np.asarray(m['origin'], float)
        self.shape = tuple(int(v) for v in m['shape'])
        self.occ = unrle(m['occupancy_rle'], self.shape)
        self.seen = unrle(m['seen_rle'], self.shape)
        box = lambda b: Box(np.asarray(b[0], float), np.asarray(b[1], float))  # noqa: E731
        self.table = box(m['table'])
        self.cases: Dict[str, Case] = {k: Case(box(v['box']), [box(c) for c in v['cubbies']]) for k, v in m['cases'].items()}
        self.books: Dict[str, Box] = {k: box(v) for k, v in m['books'].items()}
        self.base_z = float(m['base_z'])
        self.desk_top = float(m['desk_top_visual_z'] or self.table.hi[2])
        e = m.get('desk_visual_edges') or dict(x=[self.table.lo[0], self.table.hi[0]], y=[self.table.lo[1], self.table.hi[1]])
        self.desk_edges = np.array([e['x'], e['y']], float)          # [axis][lo, hi] of the desk top as the head camera sees it
        dist = ndimage.distance_transform_edt(~self.occ) * self.res
        free_far = self.seen & (dist > FREE_PENALTY_FROM_M)
        self.likelihood = np.exp(-0.5 * (dist / LIKELIHOOD_SIGMA_M) ** 2) - FREE_PENALTY * free_far - UNSEEN_PENALTY * ~ndimage.binary_dilation(self.seen, iterations=2)
        self.clearance = dist

    def cell(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(xy, float) - self.origin) / self.res).astype(int)

    def inside(self, cells: np.ndarray) -> np.ndarray:
        return (cells[..., 0] >= 0) & (cells[..., 1] >= 0) & (cells[..., 0] < self.shape[0]) & (cells[..., 1] < self.shape[1])

    def sample(self, grid: np.ndarray, xy: np.ndarray, fill: float = 0.) -> np.ndarray:
        """Bilinear sample of a cell-centred grid at map points (N, 2)."""
        f = (np.asarray(xy, float) - self.origin) / self.res - 0.5
        return ndimage.map_coordinates(grid.astype(float), [f[:, 0], f[:, 1]], order=1, mode='constant', cval=fill)

    def clutter_boxes(self, around: Box, margin: float, skip: List[Box]) -> List[tuple]:
        """Occupied cells within `margin` of a box (outside the `skip` boxes) merged along x into runs: (lo_xy, hi_xy, 'clutter')."""
        lo = self.cell(around.lo[:2] - margin)
        hi = self.cell(around.hi[:2] + margin)
        lo, hi = np.maximum(lo, 0), np.minimum(hi, np.array(self.shape) - 1)
        ii, jj = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing='ij')
        centres = self.origin + (np.stack([ii, jj], -1) + 0.5) * self.res
        mask = self.occ[ii, jj].copy()
        for s in skip:
            mask &= ~(np.all(centres >= s.lo[:2], -1) & np.all(centres <= s.hi[:2], -1))
        out = []
        for jl in range(mask.shape[1]):
            col = mask[:, jl]
            il = 0
            while il < len(col):
                if not col[il]:
                    il += 1
                    continue
                k = il
                while k + 1 < len(col) and col[k + 1]:
                    k += 1
                a = self.origin + np.array([lo[0] + il, lo[1] + jl]) * self.res
                b = self.origin + np.array([lo[0] + k + 1, lo[1] + jl + 1]) * self.res
                out.append((a.tolist(), b.tolist(), 'clutter'))
                il = k + 1
        return out


@lru_cache(maxsize=1)
def scene_map() -> SceneMap:
    return SceneMap()


def relax_obstacles(obstacles, res: float = 0.0, min_cells: int = 3, inset: float = 0.01):
    """Obstacle list (lo_xy, hi_xy, category) with the 'clutter' boxes relaxed: groups of touching clutter boxes covering fewer than
    `min_cells` map cells (depth speckle of the training cloud) are dropped, the others shrink by `inset` per side.
    Returns (new list, changed)."""
    items = list(obstacles or [])
    idx = [i for i, o in enumerate(items) if len(o) >= 3 and o[2] == 'clutter']
    if not idx:
        return items, False
    res = float(res) or float(scene_map().res)
    lo = np.array([np.asarray(items[i][0], float)[:2] for i in idx])
    hi = np.array([np.asarray(items[i][1], float)[:2] for i in idx])
    n = len(idx)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):
        touch = np.all(lo[a] <= hi + 1e-6, -1) & np.all(lo <= hi[a] + 1e-6, -1)
        for b in np.nonzero(touch)[0]:
            ra, rb = find(a), find(int(b))
            if ra != rb:
                parent[rb] = ra
    cells = np.prod(np.maximum(hi - lo, 0.) / res, -1)
    total: Dict[int, float] = {}
    for k in range(n):
        g = find(k)
        total[g] = total.get(g, 0.) + float(cells[k])
    pos = {i: k for k, i in enumerate(idx)}
    out, changed = [], False
    for i, o in enumerate(items):
        if i not in pos:
            out.append(o)
            continue
        k = pos[i]
        if total[find(k)] < min_cells - 0.5:
            changed = True
            continue
        a, b = lo[k] + inset, hi[k] - inset
        if np.any(b <= a):
            a, b = lo[k], hi[k]
        else:
            changed = True
        out.append((a.tolist(), b.tolist()) + tuple(o[2:]))
    return out, changed
