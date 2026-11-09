"""Phases settle, retreat, steady, done, failed and plan: which book and which action (push, pinch, insert) next.
Mixin of logic.Planner of the task 55 fair planner; the methods were moved verbatim from logic.py."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from ..world import BOOKS, State
from ..params import GRIP_CLOSED, GRIP_OPEN, MAX_ATTEMPTS, SETTLE_STEPS, STEADY_STEPS, VARIANT


class PlanMixin:
    # ---- phases ----------------------------------------------------------------------------------------------------------------------
    def phase_settle(self, action, q, s: State) -> None:
        if self.grid is None:
            self.build_grid(s)
        if self.age == 1:
            self.relooks = 0
        self.to_rest(action, q)
        if self.age >= SETTLE_STEPS:
            self.change('plan', 'settled')

    def phase_retreat(self, action, q, s: State) -> None:
        self.to_rest(action, q)
        if self.age > 40:
            self.change('plan', 'retreated')

    def phase_steady(self, action, q, s: State) -> None:
        """Base still, gripper closed. The pinch command is replayed; rewriting it from the measured joints lets the arm sag."""
        self.to_rest(action, q, hold=self.hold_side)
        self.grip[self.hold_side] = GRIP_CLOSED
        action[K.GRIPPER_ACTION[self.hold_side]] = GRIP_CLOSED
        if self.age == 1:
            self.hold_cmd = None
        if self.age == 1 or self.grasp_off_r is None:
            self.capture_grasp(q, s)
        self.freeze_hold_arm(action, q)
        if not self.check(('held', self.holding, self.hold_side), s):
            self.holding = None
            self.hold_cmd = None
            self.fail_action('the book left the fingers while the grasp settled')
            return
        if self.age >= STEADY_STEPS:
            self.capture_grasp(q, s)
            self.change('plan', 'grasp settled')

    def phase_done(self, action, q, s: State) -> None:
        self.to_rest(action, q)
        if not self.reason.startswith('stopped'):
            self.reason = 'all books inside'

    def phase_failed(self, action, q, s: State) -> None:
        self.to_rest(action, q)

    def phase_plan(self, action, q, s: State) -> None:
        self.to_rest(action, q, hold=self.carry_side())
        if self.age == 1:
            self.soft = False
            self.build_grid(s)          # the loose furniture may have moved
        if self.holding is not None and float(s.books[self.holding].box.lo[2]) < float(s.table_box.hi[2]) + 0.03 and self.age > 5:
            self.holding = None          # the book is back on the desk or fell: not in the hand any more
            self.grip = {'left': GRIP_OPEN, 'right': GRIP_OPEN}
        if self.holding is not None and s.inside(self.holding):
            self.holding = None          # placed: the hand lets go
            self.grip = {'left': GRIP_OPEN, 'right': GRIP_OPEN}
        if s.n_inside == 3:
            self.change('done', 'all three books inside')
            return
        todo = [b for b in BOOKS if not s.inside(b) and b not in self.dead_books and b in s.books]
        if not todo:
            self.change('failed', 'no book left to try')
            return
        if self.counts['plan_fail'] > 4:
            relooks = getattr(self, 'relooks', 0)
            if self.holding is None and relooks < 4:
                self.attempts.clear()
                self.put_back = set() if isinstance(self.put_back, set) else self.put_back
                # stale book estimates after an insert (looked: updated []): view the desk again before giving up
                self.relooks = relooks + 1
                self.counts['plan_fail'] = 0
                self.soft = False
                self.change('goview', f'no plan for {todo}: view the books again ({self.relooks})')
                return
            if self.holding is None and not getattr(self, 'desperate', False):
                # last resort before the terminal failure: allow very short strokes (0.14 / 0.09 / 0.05 m) that clear the lamp or a neighbour
                self.desperate = True
                self.attempts.clear()
                self.relooks = 2
                self.counts['plan_fail'] = 0
                self.soft = False
                self.change('goview', f'no plan for {todo}: last resort, short strokes allowed')
                return
            self.change('failed', 'planning failed too often')
            return
        plan = self.next_action(s, q, todo)
        if plan is None and not self.soft and self.holding is None:
            self.soft = True            # no plan around the loose furniture: allow the base to push it aside (for this action only)
            self.build_grid(s)
            plan = self.next_action(s, q, todo)
        if plan is None and self.holding is None and self.age < 80:
            # arm still walking to rest from the last action: IK is seeded with these joints, wait before counting a failure
            if self.soft:
                self.soft = False
                self.build_grid(s)
            return
        if self.notes:
            print(f'PLAN_NOTES step={self.step} ' + ' | '.join(self.notes), flush=True)
        if plan is None and self.holding is not None and 'putback' in VARIANT and self.holding not in self.put_back:
            plan = self.plan_putback(s, q)
        if plan is None:
            self.counts['plan_fail'] += 1
            self.reason = f'no plan for {todo}'
            if self.holding is not None:
                self.attempts[self.key(self.holding, 'insert')] = self.attempts.get(self.key(self.holding, 'insert'), 0) + 1
                if self.attempts[self.key(self.holding, 'insert')] >= MAX_ATTEMPTS:
                    self.dead_books.add(self.holding)
                    self.holding = None
                    self.grip = {'left': GRIP_OPEN, 'right': GRIP_OPEN}
            return
        self.begin(plan, s)

    def begin(self, plan: Dict[str, Any], s: State) -> None:
        self.act_plan, self.wi, self.servo, self.slide_goal, self.dwell = plan, 0, None, None, 0
        self.counts[{'push': 'pushes', 'pinch': 'pinches', 'insert': 'inserts', 'putback': 'putbacks'}[plan['kind']]] += 1
        k = self.key(plan['book'], plan['kind'])
        if plan['kind'] != 'insert':
            self.attempts[k] = self.attempts.get(k, 0) + 1     # every try counts, so a push that does not help is not repeated for ever
        self.change('go', f"{plan['kind']} {plan['book']}: {plan['why']}; base {np.round(plan['base'][:2], 2).tolist()} yaw {plan['base'][2]:.2f} arm {plan['side']}")

    # ---- planning ---------------------------------------------------------------------------------------------------------------------
    def next_action(self, s: State, q: np.ndarray, todo: List[str]) -> Optional[Dict[str, Any]]:
        self.notes.clear()
        if self.holding is not None and self.holding in todo:
            return self.plan_insert(s, q, self.holding)
        order = sorted(todo, key=lambda b: float(np.linalg.norm(s.base[:2] - s.books[b].box.centre[:2])))
        if s.n_inside >= 1 and len(order) > 1:
            # after the first insert a book whose full push is blocked (rear face pressed on a row neighbour, lamp on the sweep,
            # no free end hand box) still wins the nearest-first order and repeats inert micro-strokes to the horizon; a book
            # that can still be pushed to its overhang is tried first (ties and all-blocked cases keep the distance order)
            order = sorted(order, key=lambda b: (-self.push_feasible(s, b),
                                                 float(np.linalg.norm(s.base[:2] - s.books[b].box.centre[:2]))))
        for b in order:
            for fn in (self.plan_pinch, self.plan_push):
                plan = fn(s, q, b)
                if plan is not None:
                    return plan
        return None
