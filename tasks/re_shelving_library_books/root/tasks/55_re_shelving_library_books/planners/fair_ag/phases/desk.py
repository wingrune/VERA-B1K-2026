"""The desk: items and obstacles on it, free envelopes for a hand, clutter by the depth cloud, plan features.
Mixin of logic.Planner of the task 55 fair planner; the methods were moved verbatim from logic.py."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .. import geometry as G
from .. import ik
from ..world import Box, State
from ..params import MARGIN_TOL, VISIBLE_DEPTH


def _run_far(occ: np.ndarray, i: int, j: int, di: int, dj: int, face: float, axis: int, origin: np.ndarray, res: float, shape: Tuple[int, int]) -> float:
    """Distance from the collision face to the far edge of the occupied run that starts at (i, j) and steps (di, dj)."""
    ci, cj = i, j
    while True:
        ni, nj = ci + di, cj + dj
        if not (0 <= ni < shape[0] and 0 <= nj < shape[1]) or not occ[ni, nj]:
            break
        ci, cj = ni, nj
    if axis == 0:
        return float((origin[0] + (ci + 1) * res) - face) if di > 0 else float(face - (origin[0] + ci * res))
    return float((origin[1] + (cj + 1) * res) - face) if dj > 0 else float(face - (origin[1] + cj * res))


def _desk_margin(occ: np.ndarray, i: int, j: int, table: Box, origin: np.ndarray, res: float, shape: Tuple[int, int], tol: float) -> bool:
    """True when cell (i, j) is the desk's own collision margin: every occupied run outward from the collision box ends within `tol`."""
    cx = float(origin[0] + (i + 0.5) * res)
    cy = float(origin[1] + (j + 0.5) * res)
    lo, hi = table.lo, table.hi
    outs = []
    if cx < float(lo[0]):
        outs.append((0, -1, 0, float(lo[0])))
    elif cx > float(hi[0]):
        outs.append((0, 1, 0, float(hi[0])))
    if cy < float(lo[1]):
        outs.append((1, 0, -1, float(lo[1])))
    elif cy > float(hi[1]):
        outs.append((1, 0, 1, float(hi[1])))
    if not outs:
        return False
    for axis, di, dj, face in outs:
        if _run_far(occ, i, j, di, dj, face, axis, origin, res, shape) > tol + 1e-6:
            return False
    return True


class DeskMixin:
    def desk_items(self, s: State) -> List[Tuple[np.ndarray, np.ndarray]]:
        """xy boxes of what stands on the desk top (lamp), from the exported footprints."""
        t = s.table_box
        out = []
        for lo, hi, cat in (s.obstacles or []):
            c = (np.asarray(lo) + np.asarray(hi)) / 2
            if cat not in ('desk', 'bookcase', 'walls') and np.all(c >= t.lo[:2]) and np.all(c <= t.hi[:2]):
                out.append((np.asarray(lo) - 0.02, np.asarray(hi) + 0.02))
        return out

    def desk_obstacles(self, s: State, exclude: str) -> List[G.Rect]:
        """Footprints (oriented rectangles) of the other books and of what stands on the desk, for the sweep of a pushed book."""
        out = [G.rect_of(G.BookFrame(b)) for n, b in s.books.items() if n != exclude and float(b.box.hi[2]) > s.table_box.hi[2] - 0.01]
        return out + [G.rect_of_box(lo, hi) for lo, hi in self.desk_items(s)]

    def envelope_free(self, s: State, env: Box, margin: Optional[float] = None) -> bool:
        """The hand box next to the desk does not overlap the footprint of anything but the desk, the walls and the bookcases.

        With `margin`, an occupied cell is not an obstacle when it is the desk's own collision margin: walking outward from the
        collision face, the occupied run ends within `margin` metres. A run that continues is a real obstacle and still rejects the box."""
        skip_clutter = margin is not None and getattr(self, 'perc', None) is not None
        if skip_clutter:
            hit = self._real_clutter(s, env, float(margin))
            if hit is not None:
                self.blocker = hit
                return False
        for lo, hi, cat in (s.obstacles or []):
            if cat in ('walls', 'desk', 'bookcase'):
                continue
            if cat == 'clutter' and skip_clutter:
                continue
            c = (np.asarray(lo) + np.asarray(hi)) / 2
            if np.all(c >= s.table_box.lo[:2]) and np.all(c <= s.table_box.hi[:2]):
                continue            # stands on the desk
            if G.boxes_overlap_xy(env.lo, env.hi, np.asarray(lo), np.asarray(hi)):
                self.blocker = f'{cat} {np.round(lo, 3).tolist()}-{np.round(hi, 3).tolist()}'
                return False
        return True

    def _real_clutter(self, s: State, env: Box, tol: float) -> Optional[str]:
        """The first occupied cell in `env` that is not inside the desk or a bookcase and not the desk's own margin, as a blocker string."""
        m = self.perc.map
        origin, res, occ, shape = m.origin, float(m.res), m.occ, m.shape
        lo = np.maximum(m.cell(env.lo[:2]), 0)
        hi = np.minimum(m.cell(np.asarray(env.hi[:2], float) - 1e-9), np.array(shape) - 1)
        table = s.table_box
        for j in range(int(lo[1]), int(hi[1]) + 1):
            for i in range(int(lo[0]), int(hi[0]) + 1):
                if not occ[i, j]:
                    continue
                a = origin + np.array([i, j]) * res
                b = a + res
                if min(b[0], env.hi[0]) - max(a[0], env.lo[0]) <= 0 or min(b[1], env.hi[1]) - max(a[1], env.lo[1]) <= 0:
                    continue
                cx, cy = float(a[0] + 0.5 * res), float(a[1] + 0.5 * res)
                if self._cell_skipped(s, cx, cy):
                    continue
                if _desk_margin(occ, i, j, table, origin, res, shape, tol):
                    continue
                return f'clutter {np.round(a, 3).tolist()}-{np.round(b, 3).tolist()}'
        return None

    def _cell_skipped(self, s: State, cx: float, cy: float) -> bool:
        """Cells the offline clutter boxes leave out: inside the desk or a bookcase, grown by one map cell."""
        grow = 0.03
        t = s.table_box
        if t.lo[0] - grow <= cx <= t.hi[0] + grow and t.lo[1] - grow <= cy <= t.hi[1] + grow:
            return True
        for case in s.cases.values():
            b = case.box
            if b.lo[0] - grow <= cx <= b.hi[0] + grow and b.lo[1] - grow <= cy <= b.hi[1] + grow:
                return True
        return False

    def _feat(self) -> frozenset:
        """Which pieces of the grasp fix this job runs. Empty is the copied oracle pinch."""
        mode = ik.fair_mode()
        if mode == 'halt':
            return frozenset(('halt',))
        if mode == 'under':
            return frozenset(('under',))
        if mode == 'jaws':
            return frozenset(('jaws',))
        if mode == 'nest':
            return frozenset(('halt', 'under', 'jaws', 'sforce'))
        if mode == 'tuck':
            return frozenset(('halt', 'under', 'jaws', 'sforce', 'tuck'))
        if mode == 'onbook':
            return frozenset(('halt', 'onbook', 'jaws'))
        if mode == 'on28':
            return frozenset(('halt', 'onbook', 'jaws', 'wide'))
        if mode in MARGIN_TOL:
            return frozenset(('halt', 'onbook', 'jaws', 'margin'))
        return frozenset()

    def _margin_tol(self) -> float:
        return float(MARGIN_TOL.get(ik.fair_mode(), 0.))

    def _visible_depth(self) -> float:
        """Minimum centre-to-fingertip distance of a pinch measured from the visible lip. 0 keeps every bite the model allows."""
        return float(VISIBLE_DEPTH.get(ik.fair_mode(), 0.))
