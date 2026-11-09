"""Push geometry: a landing spot free of other books, pushes along the face normals of a skewed book.
Mixin of logic.Planner of the task 55 fair planner; the methods were moved verbatim from logic.py."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import geometry as G
from ..world import State
from ..params import ALL_EDGES, NORMAL_SKEW, PUSH_SPREAD_M


class PushGeometryMixin:
    @staticmethod
    def free_landing(f: G.BookFrame, d: np.ndarray, pp: Dict[str, Any], rects: List[G.Rect], pad: float = 0.012) -> Optional[Dict[str, Any]]:
        """The pusher pose with its fingertips (start and the two finger points) off every other footprint: the face centre, else the nearest
        lateral shift along the rear face that is free; None when the whole face is blocked (a neighbour pressed against it)."""
        def free(q: Dict[str, Any]) -> bool:
            for k in (-1., 0., 1.):
                p = q['start'][:2] + q['jaw'][:2] * k * PUSH_SPREAD_M / 2
                for c, ex, ey, hx, hy in rects:
                    if abs(float((p - c) @ ex)) <= hx + pad and abs(float((p - c) @ ey)) <= hy + pad:
                        return False
            return True
        if free(pp):
            return pp
        lim = max(float(pp['half_width']) - PUSH_SPREAD_M / 2, 0.)
        for k in range(1, 9):
            for sg in (1., -1.):
                lat = sg * k * 0.01
                if abs(lat) > lim:
                    continue
                q = G.pusher_pose(f, d, float(pp['start'][2]), lateral=lat)
                if free(q):
                    return q
        return None

    def normal_segments(self, s: State, f: G.BookFrame, book: str, over_edges: List[str],
                        segments: List[Tuple[str, np.ndarray, np.ndarray, float]]) -> List[Tuple[str, np.ndarray, np.ndarray, float]]:
        """For every edge where the book is skewed more than NORMAL_SKEW, replace the pushes by pushes along a face normal u with u.n >= 0.45:
        the pusher on the opposite face centre pushes through the book centre, so the book slides along u without turning. The travel brings
        the overhang over n to push_travel, cut where the footprint would come within DESK_MARGIN_M of another desk edge (a cut push is a step,
        the next one is planned from the live pose). A full push needs a free hand box at its end."""
        t = s.table_box
        out = []
        for edge, n in G.EDGES.items():
            if over_edges and edge not in over_edges:
                continue
            if G.tilt_between(f.face_towards(n)[1], n) <= NORMAL_SKEW:
                out += [sg for sg in segments if sg[0] == edge]
                continue
            over = f.extent(n) - G.edge_coordinate(t, n)
            need = min(G.push_travel(f, t, n), 0.8 * f.span(n) - over)      # the centre stays well on the desk
            if need < 0.01:
                continue
            m = np.array([-n[1], n[0], 0.])
            corner = max(f.corners(), key=lambda c: float(c @ n[:2]))
            lateral = float((corner - f.centre[:2]) @ m[:2])                # the pinch bites near the corner that hangs out most
            for name, (u, _, _) in f.faces().items():
                un = float(u[:2] @ n[:2])
                if un < 0.45:
                    continue
                dist, full = need / un, True
                for e, ne in ALL_EDGES.items():
                    ue = float(u[:2] @ ne[:2])
                    if e == edge or ue <= 1e-6:
                        continue
                    room = (G.edge_coordinate(t, ne) - G.DESK_MARGIN_M - f.extent(ne)) / ue
                    if room < dist:
                        dist, full = max(room, 0.), False
                if dist < 0.02:
                    self.note(f'push {book} {edge} normal {name}: cut to {dist:.3f} m by the other desk edges')
                    continue
                if full:
                    g = G.BookFrame(s.books[book])
                    g.centre = g.centre + u * dist
                    if not self.envelope_free(s, G.book_hand_envelope(g, t, n, 0., lateral=lateral)):
                        self.note(f'push {book} {edge} normal {name} {dist:.2f}: hand box at the end blocked by {self.blocker}')
                        continue
                out.append((edge, n, np.array([u[0], u[1], 0.]), float(dist)))
        return out

    def push_feasible(self, s: State, book: str) -> float:
        """Can `book` still be pushed to its pinch overhang from the live pose? 1.0: a straight push on some edge has a free
        pusher landing, a clear swept footprint to the full travel and a free hand box at the end (or the book already overhangs
        an edge far enough to pinch). 0.5: the same but only after a lateral slide inside the edge room (a nudge chain).
        0.0: every edge is blocked — the rear face is pressed against a neighbour, the sweep hits the lamp or a row book, no free
        end box — so this book can only offer partial strokes and must not starve the queue in front of a pushable one."""
        f = G.BookFrame(s.books[book])
        t = s.table_box
        others = self.desk_obstacles(s, book)
        min_bite = float(getattr(G, 'MIN_BITE_M', 0.012))
        mt = self._margin_tol()
        for edge, n in G.EDGES.items():
            over = f.extent(n) - G.edge_coordinate(t, n)
            if min_bite + 0.003 <= over <= 0.95 * f.span(n):
                return 1.0                       # already pinchable: keep it first
        best = 0.0
        for edge, n in G.EDGES.items():
            travel = G.push_travel(f, t, n)
            if travel < 0.01:
                continue
            if G.sweep_hits(f, n, travel, others) is not None:
                continue
            if self.free_landing(f, n, G.pusher_pose(f, n, float(f.z_lo) + 0.012), others) is not None \
                    and self._env_ok(s, G.book_hand_envelope(f, t, n, travel), mt):
                return 1.0
            neg, pos = G.lateral_room(f, t, n)
            m = np.array([-n[1], n[0], 0.])
            for k in range(1, 25):
                for sg in (1., -1.):
                    shift = sg * k * 0.025
                    if (shift > 0 and shift > pos) or (shift < 0 and -shift > neg):
                        continue
                    g = G.BookFrame(s.books[book])
                    g.centre = g.centre + m * shift
                    pp = G.pusher_pose(g, n, float(f.z_lo) + 0.012)
                    if self.free_landing(g, n, pp, others) is not None \
                            and self._env_ok(s, G.book_hand_envelope(g, t, n, travel), mt):
                        best = 0.5
                        break
                if best > 0.:
                    break
            if best > 0.:
                break
        return best

    def _env_ok(self, s: State, env, mt: float) -> bool:
        """envelope_free with the clutter margin only when the grasp variant grants one (mirrors plan_push)."""
        return self.envelope_free(s, env) if mt <= 0. else self.envelope_free(s, env, mt)
