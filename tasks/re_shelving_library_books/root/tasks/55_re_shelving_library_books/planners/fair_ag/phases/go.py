"""Phase go (drive to the action stand, the carry with a book) and the waypoint follower bookkeeping.
Mixin of logic.Planner of the task 55 fair planner; the methods were moved verbatim from logic.py."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .. import geometry as G
from .. import ik
from ..world import State
from ..params import (
    CARRY_TIMEOUT, DRIFT_BRAKE_M, DRIFT_FAIL_M, GRIP_CLOSED, GRIP_OPEN, HELD_MAX_M, HOLD_JOINT_STEP, NAV_MAX_STUCK, NAV_TIMEOUT, VARIANT)


class GoMixin:
    # ---- executing an action -------------------------------------------------------------------------------------------------------
    def phase_go(self, action, q, s: State) -> None:
        a = self.act_plan
        dist, yaw_err = self.navigate(action, q, s, a['base'][:2], a['base'][2])
        self.err = (dist, yaw_err)
        if self.holding is not None and a['kind'] != 'pinch':
            self.freeze_hold_arm(action, q)
            fallen = float(s.books[self.holding].box.lo[2]) < float(s.table_box.hi[2]) + 0.03
            if fallen or not self.check(('held', self.holding, self.hold_side), s):
                self.holding = None
                self.grasp_off_r = None
                self.hold_cmd = None
                self.fail_action('the book slipped out of the fingers while driving')
                return
            drift = self.grasp_drift(q, s)
            if drift > DRIFT_FAIL_M:
                self.holding = None
                self.grasp_off_r = None
                self.hold_cmd = None
                self.fail_action('the book shifted in the fingers while driving')
                return
            if drift > DRIFT_BRAKE_M:
                action[:3] = 0.
                self.reason += f', braked: the book shifted {drift:.3f} m in the hand'
                self.braked = getattr(self, 'braked', 0) + 1
                if 'unbrake' in VARIANT and self.braked > 90:
                    self.capture_grasp(q, s)          # still held after the base stood still: the new offset is the grasp now
                    self.braked = 0
                    self.note(f'carry: accepted a {drift:.3f} m shift of the book in the hand')
            else:
                self.braked = 0
            if self.grasp_tip_r is not None:
                tip_r = G.to_robot(s.base, s.base_z, self.hand_frame(self.hold_side, q, s)[0])
                if float(self.grasp_tip_r[2] - tip_r[2]) > 0.02:
                    action[:3] = 0.
                    self.fail_action('the arm sagged on the way to the shelf')
                    return
            x, y = float(s.base[0]), float(s.base[1])
            for case in s.cases.values():
                if case.box.lo[0] - 0.05 <= x <= case.box.hi[0] + 0.05 and y + 0.42 > float(case.box.lo[1]) and y < float(case.box.hi[1]) + 0.05:
                    action[:3] = 0.
                    self.fail_action('the base was about to meet the bookcase with the book still in the hand')
                    return
        if self.arrived(dist, yaw_err):
            self.servo = ik.make_servo(a['side'], max_step=HOLD_JOINT_STEP if self.holding is not None else 0.016)
            self.wi, self.slide_goal, self.dwell = 0, None, 0
            if a['kind'] == 'insert' and self.holding is not None:
                fm, _ = self.hand_frame(self.hold_side, q, s)
                if float(s.books[self.holding].box.centre[1] - fm[1]) < 0.025:
                    self.fail_action('the fingers are not behind the book')
                    return
                self.refresh_insert(a, s, q)
            self.change('hand', 'at the base pose')
        elif self.counts['nav_stuck'] >= NAV_MAX_STUCK * (1 + sum(self.attempts.values())) or \
                self.age > (CARRY_TIMEOUT + self.turn_extra if self.holding is not None else NAV_TIMEOUT):
            self.fail_action('base pose not reached')

    def check(self, cond: tuple, s: State) -> bool:
        kind = cond[0]
        if kind == 'held':
            tip = np.asarray(self.finger_mid_world(cond[2], self.q_last, s), float)
            return float(np.linalg.norm(s.books[cond[1]].box.centre - tip)) < HELD_MAX_M
        if kind == 'inside':
            return s.inside(cond[1])
        if kind == 'placed':
            return s.inside(cond[1]) and float(np.linalg.norm(s.books[cond[1]].box.centre[:2] - np.asarray(cond[2])[:2])) < 0.025
        if kind == 'lifted':
            return float(s.books[cond[1]].box.lo[2]) >= cond[2]
        if kind == 'moved':
            _, book, c0, d, dist = cond
            return float((G.BookFrame(s.books[book]).centre - c0) @ d) >= dist - 0.004
        if kind == 'sheet':
            _, book, edge, d, dist, start = cond
            if self.q_last is None or self.act_plan is None:
                return False
            tip = np.asarray(self.finger_mid_world(self.act_plan['side'], self.q_last, s), float)
            travelled = float((tip - np.asarray(start, float)) @ np.asarray(d, float))
            if travelled >= float(dist) - 0.008:
                return True
            return self._sheet_ready(s, book, edge, float(dist))
        return False

    def next_waypoint(self, a: Dict[str, Any], why: str) -> None:
        self.wi += 1
        self.slide_goal, self.dwell, self.age = None, 0, 0
        if self.wi >= len(a['waypoints']):
            self.finish_action(a)
        else:
            self.history.append((self.step, 'hand', why))

    def finish_action(self, a: Dict[str, Any]) -> None:
        kind = a['kind']
        self.servo = None
        self.act_plan = None
        if kind == 'pinch':
            self.holding, self.hold_side = a['book'], a['side']
            self.grip[a['side']] = GRIP_CLOSED
            self.grasp_off_r = None
            self.change('steady', 'pinch finished, settling the grasp')
            return
        self.grip[a['side']] = GRIP_OPEN
        if kind in ('insert', 'putback'):
            self.holding = None
            self.grasp_off_r = None
        self.change('plan', f'{kind} finished')

    def fail_action(self, why: str) -> None:
        """The current action did not work: count it against the book and re-plan from the state."""
        a = self.act_plan or {}
        k = self.key(a.get('book', ''), a.get('kind', ''))
        # a sag while the book is still held replans the insert; it must not burn the attempt cap and open the hand
        if not ('sagged' in why and a.get('kind') == 'insert' and self.holding is not None):
            self.attempts[k] = self.attempts.get(k, 0) + 1
        self.servo = None
        if a.get('kind') != 'insert' or self.holding is None:
            self.grip = {'left': GRIP_OPEN, 'right': GRIP_OPEN}
            self.holding = None if a.get('kind') != 'insert' else self.holding
        self.act_plan = None
        self.change('plan', why)
