"""The push of a book over a desk edge: stand, pusher pose, travel limits, waypoints.
Mixin of logic.Planner of the task 55 fair planner; the methods were moved verbatim from logic.py."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import geometry as G
from ..world import State
from ..scene_map import relax_obstacles
from ..params import ALL_EDGES, MAX_ATTEMPTS, PUSH_SPREAD_M, RINGS, SLIDE, VARIANT


class PushMixin:
    def plan_push(self, s: State, q: np.ndarray, book: str) -> Optional[Dict[str, Any]]:
        """Strict plan first (the obstacles as they are); only when it finds nothing, one retry with speckle clutter dropped and the
        remaining clutter boxes inset by 1 cm. The obstacle list is restored afterwards. Every returned plan conditions the next one:
        a stroke that moved the book less than a third of its commanded distance is remembered as a miss, and the next plan changes
        its input (the other lateral sign first, the nudge folded with the push it enables) instead of repeating it."""
        self.push_last = getattr(self, 'push_last', {})
        self.push_miss = getattr(self, 'push_miss', {})
        self._push_feedback(s, book)
        best = self._plan_push_strict(s, q, book)
        if best is not None:
            self._remember_push(s, book, best)
            return best
        obs = getattr(s, 'obstacles', None)
        if not obs:
            return None
        relaxed, changed = relax_obstacles(obs)
        if not changed:
            return None
        try:
            s.obstacles = relaxed
        except Exception:
            return None
        try:
            best = self._plan_push_strict(s, q, book)
        finally:
            s.obstacles = obs
        if best is not None:
            self.note(f'push {book}: planned with relaxed clutter (speckle dropped, boxes inset 0.01 m)')
            self._remember_push(s, book, best)
        return best

    def _push_feedback(self, s: State, book: str) -> None:
        """Measured motion since the last push plan of this book (the fresh look between plans): a stroke whose book moved
        less than 0.35 of its commanded distance is a miss; misses steer the sign order of the lateral strokes and wake the
        folded clearance push. The planner's own record only - no privileged state."""
        last = self.push_last.get(book)
        c = np.asarray(G.BookFrame(s.books[book]).centre[:2], float)
        self.push_last[book] = (c.copy(), None, 0.)
        if last is None or last[1] is None:
            return
        c0, d, dist = last
        moved = float(np.linalg.norm(c - np.asarray(c0, float)))
        if moved >= 0.35 * float(dist):
            return
        key = (book, round(float(d[0]), 2), round(float(d[1]), 2))
        self.push_miss[key] = self.push_miss.get(key, 0) + 1
        self.note(f'push {book}: the stroke along {np.asarray(d, float)[:2].round(2).tolist()} {float(dist):.2f} m '
                  f'moved the book only {moved:.3f} m (miss {self.push_miss[key]})')

    def _missed(self, book: str, d: np.ndarray) -> int:
        return self.push_miss.get((book, round(float(d[0]), 2), round(float(d[1]), 2)), 0)

    def _book_misses(self, book: str) -> int:
        return sum(v for (b, _, _), v in self.push_miss.items() if b == book)

    def _remember_push(self, s: State, book: str, plan: Dict[str, Any]) -> None:
        strokes = plan.get('strokes') or []
        if not strokes:
            return
        c = np.asarray(G.BookFrame(s.books[book]).centre[:2], float)
        self.push_last[book] = (c.copy(), np.asarray(strokes[0]['d'], float), float(strokes[0]['dist']))

    def _plan_push_strict(self, s: State, q: np.ndarray, book: str) -> Optional[Dict[str, Any]]:
        if self.attempts.get(self.key(book, 'push'), 0) >= MAX_ATTEMPTS * 2:
            return None
        f = G.BookFrame(s.books[book])
        if float(f.z_lo) > float(s.table_box.hi[2]) + 0.02:
            return None
        best = None
        inside = float(getattr(s, 'n_inside', 0)) >= 1      # the row model below wakes only after the first insert
        others = self.desk_obstacles(s, book)
        others = [o for o in others if not G.rects_overlap(G.rect_of(f), o, 0.004)]   # already touching (a knocked lamp): not a sweep hit
        over_edges = [e for e, n in G.EDGES.items() if f.extent(n) - G.edge_coordinate(s.table_box, n) > 0.02]
        segments: List[Tuple[str, np.ndarray, np.ndarray, float]] = []
        blocked = set()
        self.blocker = ''
        for edge, n in G.EDGES.items():
            if over_edges and edge not in over_edges:
                continue            # already hanging over an edge: only that edge is pushed towards
            travel = G.push_travel(f, s.table_box, n)
            if travel < 0.01:
                self.note(f'push {book} {edge}: travel {travel:.3f} m (already over)')
                continue
            m = np.array([-n[1], n[0], 0.])
            # straight to the overhang when the hand box there is free, otherwise slide along the edge to the nearest free spot first
            segment = None
            neg, pos = G.lateral_room(f, s.table_box, n)
            # the lateral signs are ordered by the measured motion of the earlier strokes: the sign whose strokes missed
            # is tried last, and after two misses only when the other sign has no room left; with no miss recorded the
            # order is exactly the old one, so fresh paths stay untouched
            signs = (1, -1) if self._missed(book, m) <= self._missed(book, -m) else (-1, 1)
            skip = {sg for sg in (1, -1) if self._missed(book, sg * m) >= 2}
            order = [0.] + [sg * k * 0.025 for k in range(1, 25) for sg in signs]
            if len(skip) < 2:
                order = [0.] + [sg * k * 0.025 for k in range(1, 25) for sg in signs if sg not in skip]
                order += [sg * k * 0.025 for k in range(1, 25) for sg in signs if sg in skip]
            for shift in order:
                if (shift > 0 and shift > pos) or (shift < 0 and -shift > neg):
                    continue
                g = G.BookFrame(s.books[book])
                g.centre = g.centre + m * shift
                if self.envelope_free(s, G.book_hand_envelope(g, s.table_box, n, travel)):
                    segment = (n, travel) if shift == 0. else (m * math.copysign(1., shift), abs(shift))
                    break
            if segment is None:
                self.note(f'push {book} {edge}: no free hand box along the edge (room {neg:.2f}/{pos:.2f}, {self.blocker})')
                continue
            if segment[0] is not n:
                blocked.add(edge)
            d, dist = segment
            if 'step10' in VARIANT:
                dist = min(dist, 0.10)
            segments.append((edge, n, d, dist))
        if 'normal' in VARIANT:
            segments = self.normal_segments(s, f, book, over_edges, segments)
        for edge, n, d, dist_full in segments:
            steps = (0.30, 0.20, 0.14, 0.09, 0.05) if getattr(self, 'desperate', False) else (0.30, 0.20)
            if d is not n and self._book_misses(book) >= 1:
                # a clearance stroke of a book that already missed: fold the nudge and the push to the overhang into one
                # action cycle instead of paying a full go/look_hand/look/re-plan round for the nudge alone; infeasible
                # (no stand reaches both strokes, a sweep hits) falls through to the old single stroke below
                folded = self._folded_push(s, q, book, edge, n, d, dist_full, others)
                if folded is not None:
                    self.note(f'push {book} {edge}: folded the clearance stroke with the push on to the overhang')
                    if best is None or folded['cost'] < best['cost']:
                        best = folded
                    continue
            if inside and not self.descent_clear(s, book, f, d):
                self.note(f'push {book} {edge} {d[:2].round(2).tolist()}: a book stands behind the pushed face, the pusher would come down on it')
                continue
            row = self.row_cap(s, book, f, n, others) if inside and bool(np.allclose(np.asarray(d, float)[:2], np.asarray(n, float)[:2])) else None
            for dist in [dist_full] + [x for x in steps if x < dist_full - 0.04]:
                shove = row['stroke'] if row is not None and row['stroke'] < dist - 0.005 else None
                hit = G.sweep_hits(f, d, dist if shove is None else shove, others if shove is None else row['hard'])
                if hit is not None:
                    self.note(f'push {book} {edge} {d[:2].round(2).tolist()} {dist:.2f}: swept footprint hits the object at {np.round(hit[0], 2).tolist()}')
                    continue
                if shove is not None:
                    self.note(f'push {book} {edge}: row shove, {row["front"]} ahead walks to its overhang, stroke {shove:.2f} m')
                    dist = shove
                pp = G.pusher_pose(f, d, f.z_lo + 0.012)      # the finger midpoint is the end of the fingers: it must be below the book top to push it
                if 'land' in VARIANT:
                    books_on_desk = [G.rect_of(G.BookFrame(b)) for n2, b in s.books.items()
                                     if n2 != book and float(b.box.hi[2]) > s.table_box.hi[2] - 0.01 and float(b.box.lo[2]) < s.table_box.hi[2] + 0.02]
                    pp = self.free_landing(f, d, pp, books_on_desk)
                    if pp is None:
                        self.note(f'push {book} {edge} {d[:2].round(2).tolist()}: the pusher would land on another book')
                        continue
                if 'centre' in VARIANT:
                    r0, jaw = pp['start'][:2] - f.centre[:2], pp['jaw'][:2]
                    cross = float(jaw[0] * d[1] - jaw[1] * d[0])
                    if abs(cross) > 1e-6:
                        lim = max(float(pp['half_width']) - PUSH_SPREAD_M / 2, 0.)
                        lat = float(np.clip(-(r0[0] * d[1] - r0[1] * d[0]) / cross, -lim, lim))
                        pp = G.pusher_pose(f, d, f.z_lo + 0.012, lateral=lat)
                rot = G.jaw_rotation(np.array([0., 0., -1.]), pp['jaw'])
                start = pp['start']
                end = start + d * (dist + 0.02)
                targets = [(start + G.UP * 0.12, rot), (start, rot), (end, rot)]
                picked = self.reach_pick(s, q, self.rank_candidates(s, start[:2], radii=RINGS), targets, self.static_clearance_boxes(s))
                extra = 0.
                for w in (1, 2, 4):
                    if f'sp{w}' in VARIANT:
                        extra += w * (1. + float(n[1])) / 2.
                if ('sforce' in VARIANT or 'sforce' in self._feat()) and 'W' in blocked and edge != 'S':
                    extra += 100.
                found = None if picked is None else self.push_waypoints(s, book, edge, d, dist, start, rot, picked[0], picked[1],
                                                                        picked[2]['cost'] + 4 * dist + extra)
                if found is None:
                    self.note(f'push {book} {edge}: no base pose with IK for {d[:2].round(2).tolist()} {dist:.2f} m')
                    continue
                if dist < dist_full:
                    found['cost'] += 1.0
                    self.note(f'push {book} {edge}: shortened from {dist_full:.2f} to {dist:.2f} m')
                if shove is not None:
                    found['cost'] += 0.5      # a clear direct push elsewhere still wins, an inert nudge does not
                if best is None or found['cost'] < best['cost']:
                    best = found
                break
        return best

    def descent_clear(self, s: State, book: str, f: Any, d: np.ndarray) -> bool:
        """Row model, active after the first insert: no other on-desk book within 0.07 m behind the pushed face. The pusher
        comes down there (measured: the `down` stall of 0.017-0.029 m is the neighbour book's top), and such a stroke shoves
        the neighbour or air, never the target: the repeated inert pushes of the post-insert tails."""
        top = float(s.table_box.hi[2])
        m = np.array([-d[1], d[0], 0.])
        f_front, f_rear = float(f.extent(d)), -float(f.extent(-d))
        f_hi, f_lo = float(f.extent(m)), -float(f.extent(-m))
        cfm, hfm = (f_hi + f_lo) / 2., (f_hi - f_lo) / 2.
        for name, b in s.books.items():
            if name == book or not (float(b.box.hi[2]) > top - 0.01 and float(b.box.lo[2]) < top + 0.02):
                continue
            g = G.BookFrame(b)
            g_hi, g_lo = float(g.extent(m)), -float(g.extent(-m))
            if abs((g_hi + g_lo) / 2. - cfm) >= hfm + (g_hi - g_lo) / 2. - 0.02:
                continue                                   # not beside the pushed face
            g_front, g_rear = float(g.extent(d)), -float(g.extent(-d))
            if g_rear > f_front:
                continue                                   # ahead in the line: not under the pusher
            if f_rear - g_front < 0.07:
                return False
        return True

    def row_cap(self, s: State, book: str, f: Any, n: np.ndarray, others: List[Any]) -> Optional[Dict[str, Any]]:
        """Row model, active after the first insert: when books stand in the pushed line ahead, the stroke is inherited from
        the row - gap to the next book plus push_travel of the front-most book - so the front book reaches exactly its
        overhang and never passes it (falls). The row's own rects leave the obstacle list for the sweep; the lamp and the
        clutter still veto (the sweep of the front book and of the pushed book against the rest). None when no book is in
        the line, the front book already overhangs (pinch it instead), or it would be shoved into a hard obstacle."""
        top = float(s.table_box.hi[2])
        m = np.array([-n[1], n[0], 0.])
        f_front, f_rear = float(f.extent(n)), -float(f.extent(-n))
        cfn = (f_front + f_rear) / 2.
        f_hi, f_lo = float(f.extent(m)), -float(f.extent(-m))
        cfm, hfm = (f_hi + f_lo) / 2., (f_hi - f_lo) / 2.
        row: List[Tuple[str, Any, float, float]] = []
        for name, b in s.books.items():
            if name == book or not (float(b.box.hi[2]) > top - 0.01 and float(b.box.lo[2]) < top + 0.02):
                continue
            g = G.BookFrame(b)
            g_hi, g_lo = float(g.extent(m)), -float(g.extent(-m))
            if abs((g_hi + g_lo) / 2. - cfm) >= hfm + (g_hi - g_lo) / 2. - 0.01:
                continue                                   # out of the pushed line: the shove would not transmit
            g_front, g_rear = float(g.extent(n)), -float(g.extent(-n))
            if (g_front + g_rear) / 2. < cfn:
                continue                                   # behind the pushed book: the descent gate owns it
            if g_rear < f_front - 0.004:
                continue                                   # beside it, not ahead of it in the line
            row.append((name, g, g_front, g_rear))
        if not row:
            return None
        front = max(row, key=lambda t: t[2])
        travel = float(G.push_travel(front[1], s.table_box, n))
        if travel < 0.03:
            return None                                    # the front book already overhangs: pinch it, do not shove it further
        hard = [o for o in others if not any(G.rects_overlap(o, G.rect_of(g), 0.005) for _, g, _, _ in row)]
        if G.sweep_hits(front[1], n, travel, hard) is not None:
            return None                                     # the front book would be shoved into the lamp or the clutter
        stroke = max(min(r[3] for r in row) - f_front, 0.) + travel
        if stroke < 0.05:
            return None
        return dict(stroke=stroke, front=front[0], hard=hard)

    def push_waypoints(self, s: State, book: str, edge: str, d: np.ndarray, dist: float, start: np.ndarray, rot: np.ndarray, base: np.ndarray,
                       side: str, cost: float) -> Dict[str, Any]:
        grip = float(np.clip(PUSH_SPREAD_M / 0.047 * 2. - 1., -1., 1.))
        c0 = G.BookFrame(s.books[book]).centre.copy()
        run = float(dist)
        if 'halt' not in self._feat():
            run = float(dist) + 0.02
        stop = ('moved', book, c0, d, dist)
        if 'halt' in self._feat():
            # the estimate is frozen while the hand hides the book, so the moved check never fires and the paddle shoves it off
            stop = ('sheet', book, edge, np.asarray(d, float), float(dist), np.asarray(start, float))
        end = np.asarray(start, float) + np.asarray(d, float) * run
        wps = [
            dict(name='pre', p=start + G.UP * 0.12, R=rot, grip=grip, mode='go', tol=0.03),
            dict(name='down', p=start, R=rot, grip=grip, mode='slide', speed=SLIDE['down'], tol=0.012),
            dict(name='push', p=end, R=rot, grip=grip, mode='slide', speed=SLIDE['push'], tol=0.012,
                 stop=stop, guard=self.edge_limits(s, book, edge) if 'guard' in VARIANT else
                 (self.safe_limits(s, book, edge) if 'safe' in VARIANT else None)),
            dict(name='clear', p=end + G.UP * 0.10, R=rot, grip=grip, mode='slide', speed=SLIDE['retract'], tol=0.03),
        ]
        if 'safe' in VARIANT:
            wps[2]['backoff'] = -0.03 * np.asarray(d, float)
        return dict(kind='push', book=book, edge=edge, base=base, side=side, waypoints=wps, cost=cost,
                    strokes=[dict(d=np.asarray(d, float), dist=float(dist))],
                    why=f'edge {edge}, along {np.round(d[:2], 2).tolist()}, {dist:.3f} m')

    def _folded_push(self, s: State, q: np.ndarray, book: str, edge: str, n: np.ndarray, d: np.ndarray,
                     shift: float, others: List[Any]) -> Optional[Dict[str, Any]]:
        """A clearance stroke folded with the push it enables: slide the book along the edge to the free hand-box spot, then
        push it over the edge, both strokes from one stand in one action cycle. The separate single nudge costs a full
        go/look_hand/look/re-plan round and its estimate often stays at the planning pose, so the same nudge re-issues
        forever. None when a sweep hits or no stand reaches both strokes; the caller falls back to the single stroke."""
        f = G.BookFrame(s.books[book])
        d1 = np.asarray(d, float)
        n1 = np.asarray(n, float)
        if G.sweep_hits(f, d1, shift, others) is not None:
            return None
        g = G.BookFrame(s.books[book])
        g.centre = g.centre + d1 * shift
        travel = float(G.push_travel(g, s.table_box, n1))
        if travel < 0.01:
            return None
        if G.sweep_hits(g, n1, travel, others) is not None:
            return None
        if float(getattr(s, 'n_inside', 0)) >= 1 and not self.descent_clear(s, book, g, n1):
            return None
        grip = float(np.clip(PUSH_SPREAD_M / 0.047 * 2. - 1., -1., 1.))
        halt = 'halt' in self._feat()
        pp1 = G.pusher_pose(f, d1, f.z_lo + 0.012)
        pp2 = G.pusher_pose(g, n1, g.z_lo + 0.012)
        rot1 = G.jaw_rotation(np.array([0., 0., -1.]), pp1['jaw'])
        rot2 = G.jaw_rotation(np.array([0., 0., -1.]), pp2['jaw'])
        run1 = shift if halt else shift + 0.02
        run2 = travel if halt else travel + 0.02
        s1 = np.asarray(pp1['start'], float)
        e1 = s1 + d1 * run1
        s2 = np.asarray(pp2['start'], float)
        e2 = s2 + n1 * run2
        targets = [(s1 + G.UP * 0.12, rot1), (s1, rot1), (e1, rot1), (s2 + G.UP * 0.12, rot2), (s2, rot2), (e2, rot2)]
        picked = self.reach_pick(s, q, self.rank_candidates(s, 0.5 * (s1[:2] + s2[:2]), radii=RINGS),
                                 targets, self.static_clearance_boxes(s))
        if picked is None:
            return None
        stop1 = ('sheet', book, edge, d1, float(shift), s1) if halt else ('moved', book, f.centre.copy(), d1, float(shift))
        stop2 = ('sheet', book, edge, n1, travel, s2) if halt else ('moved', book, g.centre.copy(), n1, travel)
        guard = self.edge_limits(s, book, edge) if 'guard' in VARIANT else \
            (self.safe_limits(s, book, edge) if 'safe' in VARIANT else None)
        wps = [
            dict(name='pre', p=s1 + G.UP * 0.12, R=rot1, grip=grip, mode='go', tol=0.03),
            dict(name='down', p=s1, R=rot1, grip=grip, mode='slide', speed=SLIDE['down'], tol=0.012),
            dict(name='push', p=e1, R=rot1, grip=grip, mode='slide', speed=SLIDE['push'], tol=0.012,
                 stop=stop1, guard=guard),
            dict(name='clear', p=e1 + G.UP * 0.10, R=rot1, grip=grip, mode='slide', speed=SLIDE['retract'], tol=0.03),
            dict(name='pre', p=s2 + G.UP * 0.12, R=rot2, grip=grip, mode='go', tol=0.03),
            dict(name='down', p=s2, R=rot2, grip=grip, mode='slide', speed=SLIDE['down'], tol=0.012),
            dict(name='push', p=e2, R=rot2, grip=grip, mode='slide', speed=SLIDE['push'], tol=0.012,
                 stop=stop2, guard=guard),
            dict(name='clear', p=e2 + G.UP * 0.10, R=rot2, grip=grip, mode='slide', speed=SLIDE['retract'], tol=0.03),
        ]
        if 'safe' in VARIANT:
            wps[2]['backoff'] = -0.03 * d1
            wps[6]['backoff'] = -0.03 * n1
        cost = picked[2]['cost'] + 4. * (shift + travel) + 0.5      # a clear direct push elsewhere still wins
        return dict(kind='push', book=book, edge=edge, base=picked[0], side=picked[1], waypoints=wps, cost=cost,
                    strokes=[dict(d=d1, dist=float(shift)), dict(d=n1, dist=travel)],
                    why=f'folded clearance {np.round(d1[:2], 2).tolist()} {shift:.2f} m + push '
                        f'{np.round(n1[:2], 2).tolist()} {travel:.2f} m')

    def edge_limits(self, s: State, book: str, target: str) -> Dict[str, float]:
        f = G.BookFrame(s.books[book])
        return {e: max(f.extent(n) - G.edge_coordinate(s.table_box, n) + 0.02, -0.03) for e, n in G.EDGES.items() if e != target}

    def safe_limits(self, s: State, book: str, target: str) -> Dict[str, float]:
        """Overhang limits of the other desk edges for a push: 0.005 m more than now, never closer than DESK_MARGIN_M where the book is inside."""
        f = G.BookFrame(s.books[book])
        out = {}
        for e, n in ALL_EDGES.items():
            if e != target:
                over = f.extent(n) - G.edge_coordinate(s.table_box, n)
                out[e] = over + 0.005 if over > -G.DESK_MARGIN_M else -G.DESK_MARGIN_M
        return out

    def strays(self, s: State, book: str, limits: Dict[str, float]) -> Optional[str]:
        f = G.BookFrame(s.books[book])
        for e, lim in limits.items():
            if f.extent(ALL_EDGES[e]) - G.edge_coordinate(s.table_box, ALL_EDGES[e]) > lim:
                return e
        return None
