"""The insert of a held book into a top cubby: cubby and stand, slide waypoints, refresh on the live estimate.
Mixin of logic.Planner of the task 55 fair planner; the methods were moved verbatim from logic.py."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from ..world import Box, State
from ..params import GRIP_CLOSED, GRIP_OPEN, INSERT_REACH_M, MAX_ATTEMPTS, RELEASE_DWELL, SHELF_CLEAR_M, SLIDE


class InsertMixin:
    def hand_frame(self, side: str, q: np.ndarray, s: State) -> Tuple[np.ndarray, np.ndarray]:
        """(finger midpoint in the world, gripper link rotation in the world) from the joint state."""
        servo = K.ArmServo(side)
        servo.use_fingers_mid()
        pose = servo.pose(q)
        return G.to_world(s.base, s.base_z, pose[:3, 3]), G.rot_yaw(s.base[2]) @ pose[:3, :3]

    def plan_insert(self, s: State, q: np.ndarray, book: str) -> Optional[Dict[str, Any]]:
        """Carry the frozen pinch to a base pose in front of a top cubby. The book stays south of the opening until it is
        centred, then slides straight in. The gripper orientation never changes: a wrist twist drops the book at the case.

        After one book is inside (s.n_inside >= 1, the planner's own estimate) a failed insert is never re-issued: each executed
        insert is remembered by (case, cubby, frac, reach, side, stand, yaw) and the retry plans the best candidate it has NOT
        already run. 304 ran the same blocked slide three times; 309 jammed at the cubby opening, the wrist twisted and the
        putback dropped the book to the floor. A retry relaxes the grip-alignment, reach, opening-width and carry-height gates,
        aims shallower and releases closer to the estimated shelf. When no plan exists at all after inside_1, one relaxed pass
        runs before the None that becomes a putback at the case (the book is then lost off the front). Before the first insert
        the behaviour is the original one."""
        if self.attempts.get(self.key(book, 'insert'), 0) >= MAX_ATTEMPTS:
            return None
        step_now = int(getattr(self, 'step', 0))
        prev_step = getattr(self, 'insert_step', None)
        if prev_step is None or step_now < prev_step:   # new episode: clear the per-book memory
            self.insert_failed, self.insert_last = {}, {}
        self.insert_step = step_now
        active = int(getattr(s, 'n_inside', 0)) >= 1
        att = int(self.attempts.get(self.key(book, 'insert'), 0))
        if active:
            last = self.insert_last.get(book)
            if last is not None and last[1] < att:
                self.insert_failed.setdefault(book, set()).add(last[0])
            tried = self.insert_failed.setdefault(book, set())
            tried_cubs = {(t[0], t[1]) for t in tried}
        else:
            tried, tried_cubs = frozenset(), frozenset()
        relax = active and att >= 1
        self.capture_grasp(q, s)
        if self.grasp_off_r is None or self.grasp_tip_r is None or self.grasp_rot_r is None:
            return None
        side = self.hold_side
        bz = float(s.books[book].box.centre[2])
        thick = float(s.books[book].box.hi[2] - s.books[book].box.lo[2])
        yaw0 = float(s.base[2])
        # The slide is along +y. Fingers sit south of the book centre, and the arm reaches north, so the base stays
        # south of the bookcase. A sideways reach drives the body into the case; an off-centre slide meets an upright
        # and the wrist twists the book out.
        ox, oy = float(self.grasp_off_r[0]), float(self.grasp_off_r[1])
        align = math.atan2(ox, oy)
        yaws = [yaw0] + [nav_grid.wrap(align + d) for d in (0., 0.12, 0.24, 0.36, 0.48, 0.60, -0.12, 0.72)]
        frame = G.BookFrame(s.books[book])
        rel = frame.corners() - frame.centre[:2]
        c0, s0 = math.cos(yaw0), math.sin(yaw0)
        rel_r = np.column_stack([c0 * rel[:, 0] + s0 * rel[:, 1], -s0 * rel[:, 0] + c0 * rel[:, 1]])
        best_all = None
        best_new = None
        saw_aligned = False
        saw_clear = False
        rej: Dict[str, int] = {}
        grow = np.array([0.02, 0.02, 0.0])
        boxes = [Box(s.table_box.lo - grow, s.table_box.hi + grow)]

        def north_of_centre(yaw: float) -> float:
            c, s_ = math.cos(yaw), math.sin(yaw)
            return float(np.max(s_ * rel_r[:, 0] + c * rel_r[:, 1]))

        def half_across(yaw: float) -> float:
            c, s_ = math.cos(yaw), math.sin(yaw)
            return float(np.max(np.abs(c * rel_r[:, 0] - s_ * rel_r[:, 1])))

        def shifted(xy: np.ndarray, yaw: float) -> Tuple[np.ndarray, np.ndarray]:
            c, s_ = math.cos(yaw), math.sin(yaw)
            reach = np.array([c * self.grasp_tip_r[0] - s_ * self.grasp_tip_r[1], s_ * self.grasp_tip_r[0] + c * self.grasp_tip_r[1]])
            off = np.array([c * self.grasp_off_r[0] - s_ * self.grasp_off_r[1], s_ * self.grasp_off_r[0] + c * self.grasp_off_r[1]])
            return np.asarray(xy[:2], float) + reach, np.asarray(xy[:2], float) + reach + off

        def base_in_case(xy: np.ndarray) -> bool:
            x, y = float(xy[0]), float(xy[1])
            for case in s.cases.values():
                if case.box.lo[0] - 0.05 <= x <= case.box.hi[0] + 0.05 and y + 0.42 > float(case.box.lo[1]) and y < float(case.box.hi[1]) + 0.05:
                    return True
            return False

        def book_in_case(xy: np.ndarray, yaw: float, north: float) -> bool:
            _, book_xy = shifted(xy, yaw)
            edge = float(book_xy[1] + north)
            for case in s.cases.values():
                if case.box.lo[0] - 0.02 < book_xy[0] < case.box.hi[0] + 0.02 and edge > float(case.box.lo[1]) and book_xy[1] < float(case.box.hi[1]):
                    return True
            return False

        for relax_pass in ((False, True) if relax or (active and not relax) else (False,)):
            # pass 1 is the original scan; the relaxed pass exists only after the first insert, where a None plan
            # means a putback over ground the desk does not cover.
            if relax_pass and not active:
                break
            if best_all is not None:
                break
            band = 0.0 if relax_pass else 0.01
            off_min = 0.0 if relax_pass else 0.025
            off_max = 0.08 if relax_pass else 0.05
            reach_min = INSERT_REACH_M - (0.03 if relax_pass else 0.0)
            width_pad = 0.004 if relax_pass else 0.012
            settle_clear = 0.03 if relax_pass else 0.05
            fracs = (0.30, 0.42, 0.55, 0.22, 0.62) if relax_pass else (0.42, 0.30, 0.55)
            reaches = (0.34, 0.46, 0.22, 0.28, 0.40) if relax_pass else (0.22, 0.34, 0.46)
            for name, idx, cubby in G.top_cubbies(s.cases):
                if any(np.all(b.box.centre >= cubby.lo) and np.all(b.box.centre <= cubby.hi) for n2, b in s.books.items() if n2 != book):
                    continue
                if not (float(cubby.lo[2]) + band < bz < float(cubby.hi[2]) - band):
                    self.note(f'insert {book} {name} cubby {idx}: carry height {bz:.2f} outside the cubby')
                    continue
                cx = float((cubby.lo[0] + cubby.hi[0]) / 2)
                front, depth = float(cubby.lo[1]), float(cubby.hi[1] - cubby.lo[1])
                half_cubby = 0.5 * float(cubby.hi[0] - cubby.lo[0])
                rest_z = float(cubby.lo[2]) + thick / 2 + SHELF_CLEAR_M
                for frac in fracs:
                    centre = np.array([cx, front + frac * depth, bz])
                    for reach in reaches:
                        pre_c = centre - np.array([0., reach, 0.])
                        for yaw in yaws:
                            rot_w = G.rot_yaw(yaw) @ self.grasp_rot_r
                            off_w = G.rot_yaw(yaw) @ self.grasp_off_r
                            reach_w = G.rot_yaw(yaw) @ self.grasp_tip_r
                            if float(off_w[1]) < off_min or abs(float(off_w[0])) > off_max:
                                continue
                            saw_aligned = True
                            if float(reach_w[1]) < reach_min:
                                rej['reach'] = rej.get('reach', 0) + 1
                                continue
                            north = north_of_centre(yaw)
                            if half_across(yaw) + width_pad > half_cubby:
                                self.note(f'insert {book} {name} cubby {idx}: the book is too wide for the opening at yaw {yaw:.2f}')
                                continue
                            if float(pre_c[1]) + north > front - 0.02:
                                rej['north'] = rej.get('north', 0) + 1
                                continue
                            saw_clear = True
                            tip_pre = pre_c - off_w
                            tip = centre - off_w
                            base = self.base_for_tip(tip_pre, self.grasp_tip_r, yaw, s.base_z)
                            sig = (name, idx, round(frac, 2), reach, side, int(round(float(base[0]) / 0.05)),
                                   int(round(float(base[1]) / 0.05)), round(float(yaw), 1))
                            fresh = sig not in tried and (sig[0], sig[1]) not in tried_cubs
                            if base_in_case(base) or book_in_case(base, yaw, north) or book_in_case(s.base, yaw, north):
                                continue
                            if not self.free_pose(base[:2]):
                                continue
                            path = self.route(s.base[:2], base[:2])
                            if path is None or any(base_in_case(p) or book_in_case(p, yaw, north) for p in path):
                                continue
                            cost = nav_grid.path_length(path) + 1.2 * abs(nav_grid.wrap(yaw - yaw0)) + 0.4 * reach
                            if best_all is not None and cost >= best_all['cost'] and (fresh is False or (best_new is not None and cost >= best_new['cost'])):
                                continue
                            if not self.ik_feasible(side, base, s.base_z, q, [(tip, rot_w)], boxes):
                                continue
                            align = np.array([tip[0], tip_pre[1], tip[2]])
                            cand = self.insert_waypoints(book, name, idx, tip, rot_w, base, side, cost, cubby, centre, rest_z,
                                                         align=align, settle_clear=settle_clear)
                            cand['sig'] = sig
                            if best_all is None or cost < best_all['cost']:
                                best_all = cand
                            if fresh and (best_new is None or cost < best_new['cost']):
                                best_new = cand
        best = best_new if best_new is not None else best_all
        if best is not None:
            self.insert_last[book] = (best['sig'], att)
            if best_new is None:
                self.note(f'insert {book}: every slot repeats a failed attempt, retrying the least bad one')
        else:
            why = 'pinch does not face the opening' if not saw_aligned else ('the book would meet the case before it is centred' if not saw_clear else 'no frozen-arm pose in front of a top cubby')
            self.note(f'insert {book}: {why} (rejected {rej})')
        return best

    def insert_waypoints(self, book: str, case: str, idx: int, tip: np.ndarray, rot: np.ndarray, base: np.ndarray, side: str, cost: float,
                         cubby: Box, centre: np.ndarray, rest_z: float, align: Optional[np.ndarray] = None,
                         settle_clear: float = 0.05) -> Dict[str, Any]:
        """Centre the book on the cubby while it is still south of the opening, then slide straight in along +y.
        Every waypoint uses the same gripper orientation: a wrist yaw is what twisted the book out at the case."""
        # The shelf-height estimate sits 2-4 cm low: the held book rests on the shelf before the lower goal is reached,
        # the descent then pushes a friction pinch into the shelf, the wrist twists (ori err > 0.25) and the putback
        # drops the book at the case. Stop the descent 0.05 m above the estimated rest so the lower always completes,
        # open the fingers, and let the book fall the last bit onto the shelf inside the cubby.
        drop = max(float(centre[2]) - float(rest_z), 0.0)
        settle = max(drop - settle_clear, 0.0)
        down = np.array([tip[0], tip[1], tip[2] - settle])
        back = np.array([tip[0], tip[1] - 0.18, down[2]])
        wps = []
        if align is not None and float(np.linalg.norm(np.asarray(align, float)[:2] - np.asarray(tip, float)[:2])) > 0.02:
            wps.append(dict(name='align', p=np.asarray(align, float), R=rot, grip=GRIP_CLOSED, mode='slide', speed=SLIDE['insert'], tol=0.02))
        wps += [
            dict(name='insert', p=tip, R=rot, grip=GRIP_CLOSED, mode='slide', speed=SLIDE['insert'], tol=0.025, stop=('inside', book)),
            dict(name='lower', p=down, R=rot, grip=GRIP_CLOSED, mode='slide', speed=SLIDE['lower'], tol=0.02, check=('inside', book)),
            dict(name='release', p=down, R=rot, grip=GRIP_OPEN, mode='slide', speed=0., dwell=RELEASE_DWELL, tol=0.03, check=('inside', book)),
            dict(name='back', p=back, R=rot, grip=GRIP_OPEN, mode='slide', speed=SLIDE['retract'], tol=0.04),
        ]
        return dict(kind='insert', book=book, case=case, cubby=idx, base=base, side=side, waypoints=wps, cost=cost, cubby_box=cubby,
                    book_target=np.asarray(centre, float), rest_z=float(rest_z), lock_ori=True,
                    why=f'{case} cubby {idx}, frozen pinch, yaw {base[2]:.2f}')

    def refresh_insert(self, a: Dict[str, Any], s: State, q: np.ndarray) -> None:
        """Rebuild the slide from the live grasp. Centre in x while the book is still south of the opening, then go
        straight in. The gripper orientation stays the live one: do not yaw the wrist."""
        fm, hand_rot = self.hand_frame(a['side'], q, s)
        world_off = s.books[a['book']].box.centre - fm
        centre = np.asarray(a['book_target'], float).copy()
        centre[2] = float(s.books[a['book']].box.centre[2])
        cubby = a['cubby_box']
        front = float(cubby.lo[1])
        north = float(G.BookFrame(s.books[a['book']]).span(np.array([0., 1., 0.])))
        safe_y = front - north - 0.02
        align_book_y = min(float(s.books[a['book']].box.centre[1]), safe_y)
        align_book = np.array([float(centre[0]), align_book_y, float(centre[2])])
        align_tip = align_book - world_off
        tip = np.asarray(centre, float) - world_off
        align_tip[2] = float(fm[2])
        tip[2] = float(fm[2])
        new = self.insert_waypoints(a['book'], a['case'], a['cubby'], tip, hand_rot, a['base'], a['side'], a['cost'], cubby, centre, a['rest_z'], align=align_tip)
        a['waypoints'] = new['waypoints']
        a['book_target'] = centre
        self.slide_extra = 0
