"""The pinch of an overhanging book (stand, waypoints, lift) and the put-back of a book with no insert pose.
Mixin of logic.Planner of the task 55 fair planner; the methods were moved verbatim from logic.py."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from .. import geometry as G
from ..world import State
from ..params import (
    CLOSE_DWELL, GRIP_CLOSED, GRIP_OPEN, INSERT_REACH_M, LIFT_CHECK_M, LIFT_M, MAX_ATTEMPTS, RELEASE_DWELL, RINGS, SLIDE, VARIANT)


class PinchMixin:
    def plan_pinch(self, s: State, q: np.ndarray, book: str) -> Optional[Dict[str, Any]]:
        if self.attempts.get(self.key(book, 'pinch'), 0) >= MAX_ATTEMPTS:
            return None
        f = G.BookFrame(s.books[book])
        if float(f.z_lo) > float(s.table_box.hi[2]) + 0.02:
            return None
        best = None
        want_under = 'under' in self._feat()
        for edge, n in G.EDGES.items():
            edge_at = G.edge_coordinate(s.table_box, n)
            over = f.extent(n) - edge_at
            if 'onbook' in self._feat():
                clear = 0.028 if 'wide' in self._feat() else 0.022
                lip = self._lip_pinch(s, book, n, clear)
                where = 'lip'
                if lip is None and 'margin' in self._feat():
                    vis = self._visible_edge(n)
                    if vis is not None and self._sheet_past(s, book, n, vis):
                        lip = self._lip_pinch(s, book, n, clear, edge=vis)
                        need = self._visible_depth()
                        if lip is not None and float(lip['depth']) < need:
                            self.note(f'pinch {book} {edge}: visible bite {float(lip["depth"]):.3f} m is shorter than {need:.2f} m')
                            continue
                        if lip is not None:
                            where = 'visible top'
                    elif vis is not None:
                        self.note(f'pinch {book} {edge}: the sheet is not past the visible top')
                        continue
                if lip is None:
                    self.note(f'pinch {book} {edge}: the centre is not far enough in from the lip')
                    continue
                cands = [lip]
                self.note(f'pinch {book} {edge}: midpoint {float(lip["depth"]):.3f} m out from the centre, past the {where}')
                cloud = None
            else:
                cloud = self._cloud_pinch(s, book, n) if want_under else None
            if cloud is not None:
                cands = [cloud]
                self.note(f'pinch {book} {edge}: cloud overhang, bite {float(cloud["depth"]):.3f} m')
            elif 'onbook' not in self._feat():
                # a short overhang puts the bite on the lip; under refuses it and lets the push go further
                usable = over >= (0.048 if want_under else G.MIN_BITE_M + 0.003) and over <= 0.95 * f.span(n)
                if not usable:
                    self.note(f'pinch {book} {edge}: overhang {over:.3f} m not usable')
                    continue
                cands = G.pinch_overhang(f, s.table_box, n)
                if want_under:
                    kept = []
                    for pp in cands:
                        clear = float(np.asarray(pp['grasp'], float)[:2] @ G.horiz(n)[:2]) - edge_at
                        if clear >= 0.018 and float(pp.get('depth', G.PINCH_DEPTH_M)) >= 0.030:
                            kept.append(pp)
                    cands = kept
                if not cands:
                    skew = math.degrees(float(G.pinch_plan(f, n)['skew']))
                    self.note(f'pinch {book} {edge}: the fingertips would stay over the desk (skew {skew:.0f} deg)')
                    continue
            # the hand box follows the grasp, which may sit out at the overhanging corner rather than at the face centre
            across = np.array([-n[1], n[0]])
            chosen = None
            tried = 0
            for pp in cands:
                lateral = float((np.asarray(pp['grasp'])[:2] - f.centre[:2]) @ across)
                hand = G.book_hand_envelope(f, s.table_box, n, 0., lateral=lateral)
                tol = None
                if 'margin' in self._feat():
                    vis = self._visible_edge(n)
                    along = float(np.asarray(pp['grasp'], float)[:2] @ G.horiz(n)[:2])
                    if vis is not None and along >= vis + 0.015:
                        tol = self._margin_tol()
                if tol is None:
                    if not self.envelope_free(s, hand):
                        continue
                elif not self.envelope_free(s, hand, margin=tol):
                    continue
                elif not self.envelope_free(s, hand):
                    self.note(f'pinch {book} {edge}: desk margin within {tol:.2f} m is not an obstacle')
                tried += 1
                if tried > 4:
                    break
                targets = [(pp['pre'], pp['rotation']), (pp['grasp'], pp['rotation'])]
                bearings = self.bearings_around(np.asarray(pp['normal'], float), book)
                ranked = self.rank_candidates(s, np.asarray(pp['grasp'])[:2], radii=RINGS, bearings=bearings)
                if 'aware' in VARIANT:
                    g = np.asarray(pp['grasp'], float)[:2]
                    ahead = f.centre[:2] - g
                    ahead = ahead / max(float(np.linalg.norm(ahead)), 1e-6)
                    ranked = [c for c in ranked if float((g - np.asarray(c['xy'], float)[:2]) @ ahead) >= INSERT_REACH_M + 0.04]
                picked = self.reach_pick(s, q, ranked, targets, self.static_clearance_boxes(s))
                found = None if picked is None else self.pinch_waypoints(s, book, edge, pp, picked[0], picked[1], picked[2]['cost'])
                if found is not None:
                    if best is None or found['cost'] < best['cost']:
                        best = found
                    chosen = found
                    break
            if chosen is None:
                why = f'hand box blocked by {self.blocker}' if tried == 0 else 'no reachable pose off the desk'
                self.note(f'pinch {book} {edge}: {why}')
        return best

    def pinch_waypoints(self, s: State, book: str, edge: str, pp: Dict[str, Any], base: np.ndarray, side: str, cost: float) -> Dict[str, Any]:
        u, rot = pp['normal'], pp['rotation']
        grasp, pre = pp['grasp'], pp['pre']
        bite = float(pp.get('depth', G.PINCH_DEPTH_M))
        if bite < 0.030 and 'under' not in self._feat():
            pre = np.asarray(grasp, float) + G.UP * 0.08
        if 'tuck' in self._feat():
            # start below the lip, in the air outside the edge, so the lower finger threads under the overhang on the way in
            pre = np.asarray(pre, float).copy()
            pre[2] = min(float(pre[2]), float(np.asarray(grasp, float)[2]) - 0.025)
        z0 = float(s.books[book].box.lo[2])
        wps = [
            dict(name='pre', p=pre, R=rot, grip=GRIP_OPEN, mode='go', tol=0.025),
            dict(name='advance', p=grasp, R=rot, grip=GRIP_OPEN, mode='slide', speed=SLIDE['advance'], tol=0.025),
            dict(name='close', p=grasp, R=rot, grip=GRIP_CLOSED, mode='slide', speed=SLIDE['advance'], dwell=CLOSE_DWELL, tol=0.02),
            dict(name='lift', p=grasp + G.UP * LIFT_M, R=rot, grip=GRIP_CLOSED, mode='slide', speed=SLIDE['lift'], tol=0.012,
                 check=('lifted', book, z0 + LIFT_CHECK_M)),
            dict(name='retract', p=grasp + G.UP * LIFT_M + u * 0.10, R=rot, grip=GRIP_CLOSED, mode='slide', speed=SLIDE['retract'], tol=0.02,
                 dwell=30, check=('held', book, side)),
        ]
        return dict(kind='pinch', book=book, edge=edge, base=base, side=side, waypoints=wps, cost=cost, lip_at=pp.get('lip_at'),
                    why=f'edge {edge}, face {pp["face"]}, skew {math.degrees(pp["skew"]):.0f} deg, bite {bite:.3f} m')

    def plan_putback(self, s: State, q: np.ndarray) -> Dict[str, Any]:
        """The held book back onto the desk: slide the hand 0.15 m towards the desk centre, lower the book to just above the top, open, lift."""
        book, side = self.holding, self.hold_side
        tip, rot = self.hand_frame(side, q, s)
        t = s.table_box
        d = (t.lo[:2] + t.hi[:2]) / 2 - tip[:2]
        inward = np.r_[d / max(float(np.linalg.norm(d)), 1e-6) * 0.15, 0.]
        drop = max(float(s.books[book].box.lo[2]) - float(t.hi[2]) - 0.01, 0.)
        p_in = tip + inward
        p_down = p_in - G.UP * drop
        wps = [
            dict(name='in', p=p_in, R=rot, grip=GRIP_CLOSED, mode='slide', speed=SLIDE['retract'], tol=0.02),
            dict(name='down', p=p_down, R=rot, grip=GRIP_CLOSED, mode='slide', speed=SLIDE['lower'], tol=0.02),
            dict(name='release', p=p_down, R=rot, grip=GRIP_OPEN, mode='slide', speed=0., dwell=RELEASE_DWELL, tol=0.03),
            dict(name='up', p=p_down + G.UP * 0.10, R=rot, grip=GRIP_OPEN, mode='slide', speed=SLIDE['retract'], tol=0.04),
        ]
        self.put_back.add(book)
        return dict(kind='putback', book=book, base=s.base.copy(), side=side, waypoints=wps, cost=0., why='no insert pose for this grasp')
