"""The book-top cloud at a desk edge: overhang, the visible lip, where the pinch closes, sides to reach from.
Mixin of logic.Planner of the task 55 fair planner; the methods were moved verbatim from logic.py."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import geometry as G
from ..world import State


class SheetMixin:
    def _sheet(self, s: State, book: str, reach: float = 0.0) -> np.ndarray:
        """Book-top points in a strip across the book's width, long enough to cover a push of `reach` metres."""
        perc = getattr(self, 'perc', None)
        if perc is None or book not in s.books:
            return np.zeros((0, 3))
        f = G.BookFrame(s.books[book])
        return perc.sheet_points(f.centre, 0.20 + float(f.span(np.array([1., 0., 0.]))) + float(f.span(np.array([0., 1., 0.]))) + reach)

    def _strip(self, s: State, book: str, normal: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Points of `pts` that sit in the book's strip across `normal` (a neighbour beside the book is not the overhang)."""
        if len(pts) == 0 or book not in s.books:
            return pts
        f = G.BookFrame(s.books[book])
        n = G.horiz(normal)
        across = np.array([-n[1], n[0]])
        lat = pts[:, :2] @ across
        centre = float(f.centre[:2] @ across)
        half = f.span(np.array([-n[1], n[0], 0.])) + 0.04
        return pts[np.abs(lat - centre) < half]

    def _cloud_overhang(self, s: State, book: str, normal: np.ndarray, reach: float = 0.0) -> Optional[Tuple[float, float, np.ndarray]]:
        """(overhang of the outer book-top points, projection of the body, the strip) or None when the cloud is too thin."""
        n = G.horiz(normal)
        pts = self._strip(s, book, n, self._sheet(s, book, reach))
        edge = G.edge_coordinate(s.table_box, n)
        pts = pts[(pts[:, :2] @ n[:2] > edge - 0.28) & (pts[:, :2] @ n[:2] < edge + 0.12)] if len(pts) else pts
        if len(pts) < 25:
            return None
        proj = pts[:, :2] @ n[:2]
        # p98, not the max: one stray point must not look like the book edge. On a 20 cm book p98 sits about 4 mm inside the true edge.
        return float(np.percentile(proj, 98) - edge), float(np.percentile(proj, 35)), pts

    def _sheet_ready(self, s: State, book: str, edge: str, dist: float) -> bool:
        """The visible book overhangs by the oracle amount and its body is still on the desk. Cached for three controls."""
        if self.step - self._sheet_step < 3 and self._sheet_step >= 0:
            return self._sheet_hit
        n = G.EDGES[edge]
        seen = self._cloud_overhang(s, book, n, reach=max(float(dist), 0.05))
        hit = False
        if seen is not None:
            over, body, pts = seen
            edge_at = G.edge_coordinate(s.table_box, G.horiz(n))
            past = int(np.sum((pts[:, :2] @ G.horiz(n)[:2]) > edge_at + 0.02))
            hit = over >= G.OVERHANG_M - 0.003 and body <= edge_at - 0.025 and past >= 8
        self._sheet_step, self._sheet_hit = self.step, hit
        return hit

    def _cloud_pinch(self, s: State, book: str, normal: np.ndarray) -> Optional[Dict[str, Any]]:
        """Oracle pinch on the visible overhang: midpoint 35 mm in from the outer points, at least 18 mm past the lip, at mid height.

        The lower finger then hangs in the air under the book (it is 47 mm below the midpoint; the book is about 30 mm thick) instead of
        closing on the lip. None when the cloud does not show a deep enough overhang."""
        seen = self._cloud_overhang(s, book, normal)
        if seen is None:
            return None
        over, _, pts = seen
        if over < 0.048:
            return None
        n = G.horiz(normal)
        edge = G.edge_coordinate(s.table_box, n)
        outer = edge + over
        coord = outer - G.PINCH_DEPTH_M
        lo, hi = edge + 0.018, outer - 0.012
        if lo > hi - 0.004:
            return None
        coord = float(min(max(coord, lo), hi))
        across = np.array([-n[1], n[0], 0.])
        lip = pts[pts[:, :2] @ n[:2] > outer - 0.03]
        if len(lip) < 8:
            lip = pts
        lat = float(np.median(lip[:, :2] @ across[:2]))
        thick = float(np.clip(s.books[book].box.hi[2] - s.books[book].box.lo[2], 0.020, 0.040))
        top = float(np.percentile(lip[:, 2], 70))
        grasp = n * coord + across * lat
        grasp[2] = top - 0.5 * thick
        bite = float(outer - coord)
        pre = grasp + n * 0.10
        return dict(face='cloud', normal=n, axis=-n, rotation=G.jaw_rotation(-n, G.UP), grasp=grasp, pre=pre,
                    half_width=float(G.BookFrame(s.books[book]).span(across)), skew=0., depth=bite, along=0.)

    def _visible_edge(self, normal: np.ndarray) -> Optional[float]:
        """Edge coordinate of the visible desk top, when that wood sits more than 25 mm inside the collision box.

        The south collision face is 42 mm outside the top the head camera sees. None on edges where the two faces agree."""
        perc = getattr(self, 'perc', None)
        if perc is None:
            return None
        n = G.horiz(normal)
        edge = G.edge_coordinate(perc.map.table, n)
        axes = perc.map.desk_edges
        if abs(float(n[0])) > 0.5:
            vis = float(axes[0][1] if n[0] > 0 else -axes[0][0])
        else:
            vis = float(axes[1][1] if n[1] > 0 else -axes[1][0])
        if edge - vis > 0.025:
            return vis
        return None

    def _sheet_past(self, s: State, book: str, normal: np.ndarray, edge: float) -> bool:
        """The book-top cloud actually crosses the lip. A model that hangs past the wood with no points there would close on air."""
        n = G.horiz(normal)
        pts = self._strip(s, book, n, self._sheet(s, book))
        if len(pts) < 5:
            return False
        return int(np.sum(pts[:, :2] @ n[:2] > float(edge))) >= 5

    def _lip_pinch(self, s: State, book: str, normal: np.ndarray, clear: float = 0.022,
                   edge: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Finger midpoint on the centre line, `clear` metres past the lip, at mid height.

        A bite measured from the outer face lands off the book when that face is a corner or a swapped yaw (301: the
        fingertips closed in the air 7 cm from the centre). The centre has to stay at least 4.5 cm inward of the
        midpoint, and the book has to extend past the fingertips, so the sheet is between the jaws and the lower
        finger hangs past the lip."""
        f = G.BookFrame(s.books[book])
        n = G.horiz(normal)
        edge = G.edge_coordinate(s.table_box, n) if edge is None else float(edge)
        along_g = edge + float(clear)
        along_c = float(f.centre[:2] @ n[:2])
        if along_g - along_c < 0.045:
            return None
        if f.extent(n) < along_g + 0.012:
            return None
        across = np.array([-n[1], n[0], 0.])
        grasp = n * along_g + across * float(f.centre[:2] @ across[:2])
        grasp[2] = f.z_mid
        return dict(face='lip', normal=n, axis=-n, rotation=G.jaw_rotation(-n, G.UP), grasp=grasp, pre=grasp + n * 0.10,
                    half_width=float(f.span(across)), skew=0., depth=float(along_g - along_c), along=0., lip_at=edge)

    def _between(self, s: State, book: str, edge: str, q: np.ndarray, side: str) -> bool:
        """The fingertips are under the overhang, not on the lip: the midpoint has reached the planned pose and is past the desk edge.

        A close on the lip (the stands failure) never gets here. The book-top cloud cannot be used once the fingers are on it: the
        arm filter removes points within 9 cm of the gripper."""
        if edge not in G.EDGES or self.act_plan is None:
            return False
        tip = np.asarray(self.finger_mid_world(side, q, s), float)
        n = G.horiz(G.EDGES[edge])
        planned = None if self.act_plan is None else self.act_plan.get('lip_at')
        edge_at = G.edge_coordinate(s.table_box, n) if planned is None else float(planned)
        if float(tip[:2] @ n[:2]) < edge_at + 0.015:
            return False
        target = np.asarray(self.act_plan['waypoints'][self.wi]['p'], float)
        if float(np.linalg.norm(tip - target)) > 0.022:
            return False
        if book not in s.books:
            return True
        mid = G.BookFrame(s.books[book]).z_mid
        return abs(float(tip[2]) - mid) < 0.025 or abs(float(tip[2]) - float(target[2])) < 0.012

    def bearings_around(self, normal: np.ndarray, book: str = '') -> np.ndarray:
        centre = math.atan2(normal[1], normal[0])
        spread = 40 if book in self.put_back else 80
        return np.array([centre + math.radians(a) for a in range(-spread, spread + 1, 20)])

    def sides_for(self, point: np.ndarray, cand: Dict[str, Any]) -> List[str]:
        d = point[:2] - cand['xy']
        lateral = -math.sin(cand['yaw']) * d[0] + math.cos(cand['yaw']) * d[1]
        return ['left', 'right'] if lateral >= 0 else ['right', 'left']
