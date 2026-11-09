"""Phase hand: the hand waypoints of an action (go, slide, dwell, close and open the gripper, stop conditions).
Mixin of logic.Planner of the task 55 fair planner; the methods were moved verbatim from logic.py."""
from __future__ import annotations

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from ..world import State
from ..params import GRIP_OPEN, REACH_ORI_TOL, REST_STEP, SLIDE, WAYPOINT_TIMEOUT


class HandMixin:
    def phase_hand(self, action, q, s: State) -> None:
        a = self.act_plan
        side = a['side']
        w = a['waypoints'][self.wi]
        gate = 'jaws' in self._feat() and a.get('kind') == 'pinch' and w['name'] == 'close'
        between = (not gate) or self._between(s, a['book'], str(a.get('edge', '')), q, side)
        self.grip[side] = float(w['grip']) if between else GRIP_OPEN
        action[K.GRIPPER_ACTION[side]] = self.grip[side]
        other = 'right' if side == 'left' else 'left'
        target_other = K.rest(other)
        cur_other = q[list(K.ARM_DOFS[other])]
        action[K.ARM_ACTION[other]] = cur_other + np.clip(target_other[4:] - cur_other, -REST_STEP, REST_STEP)
        target = np.asarray(w['p'], float)
        speed = float(w['speed']) if w['mode'] == 'slide' else 0.
        if self.holding is not None and a['kind'] == 'insert' and float(w['grip']) < 0.:
            if not self.check(('held', self.holding, self.hold_side), s):
                self.fail_action('the book left the fingers while the hand moved')
                return
        first = self.slide_goal is None
        if first:
            self.hand0 = np.asarray(self.finger_mid_world(side, q, s), float)
            self.slide_goal = self.hand0.copy() if w['mode'] == 'slide' else target.copy()
        if w['mode'] == 'slide':
            d = target - self.slide_goal
            n = float(np.linalg.norm(d))
            self.slide_goal = target.copy() if n <= speed else self.slide_goal + d / n * speed
            goal_world = self.slide_goal
        else:
            goal_world = target
        goal, rot = G.to_robot(s.base, s.base_z, goal_world), G.rotation_to_robot(s.base, w['R'])
        self.err = self.servo.step(action, q, goal, rot, plan=first and w['mode'] == 'go')
        if self.holding is not None and a['kind'] == 'insert' and float(w['grip']) < 0. and self.err[1] > 0.25:
            self.fail_action('the wrist started to twist against the shelf')
            return
        if first and self.servo.plan_failed:
            self.counts['ik_none'] += 1
        pos_err = float(np.linalg.norm(np.asarray(self.finger_mid_world(side, q, s)) - target))
        self.reason = f"{a['kind']} {a['book']} waypoint {self.wi} {w['name']}: tip err {pos_err:.3f} m, ori {self.err[1]:.2f} rad"
        stop = w.get('stop')
        if stop and self.check(stop, s):
            if stop[0] == 'sheet':
                tip = np.asarray(self.finger_mid_world(side, q, s), float)
                for nxt in a['waypoints'][self.wi + 1:]:
                    if nxt.get('name') == 'clear':
                        nxt['p'] = tip + G.UP * 0.10          # lift straight up; sliding on would shove the book off
                        break
                self.note(f'push {a["book"]} {a.get("edge")}: stopped once the overhang was deep enough')
            self.next_waypoint(a, f"{w['name']}: stop condition met")
            return
        if w.get('guard'):
            e = self.strays(s, a['book'], w['guard'])
            if e is not None:
                if w.get('backoff') is not None:
                    tip = np.asarray(self.finger_mid_world(side, q, s), float)
                    back = tip + w['backoff']
                    a['waypoints'][self.wi + 1]['p'] = back + G.UP * 0.10
                    a['waypoints'].insert(self.wi + 1, dict(name='back', p=back, R=w['R'], grip=w['grip'], mode='slide', speed=SLIDE['retract'],
                                                            tol=0.02))
                self.next_waypoint(a, f"{w['name']}: the book reached the {e} edge, push stopped")
                return
        done = pos_err < w['tol'] and (self.err[1] < REACH_ORI_TOL or w['mode'] == 'slide')
        # A slide whose goal has not yet reached the waypoint is still moving. Speed 0 is a hold: finish once the tip is inside tolerance
        # (r11 left the gripper closed for the whole timeout 6 mm short of the goal and never lifted).
        if w['mode'] == 'slide' and float(np.linalg.norm(self.slide_goal - target)) > 1e-6 and not (speed <= 0. and pos_err < w['tol']):
            done = False
        if done and stop and stop[0] == 'inside' and not self.check(stop, s):
            self.slide_extra += 1
            if self.slide_extra <= 6:
                w['p'] = np.asarray(w['p'], float) + np.array([0., 0.025, 0.])
                self.slide_goal, self.dwell = None, 0
                self.reason = f'insert deeper ({self.slide_extra})'
                return
            self.fail_action('the book centre stayed outside the cubby')
            return
        if done and gate and not between:
            if self.age > 220:
                self.fail_action('the book is not between the fingers')
                return
            self.reason += '; jaws open until the book is between the fingers'
            return
        if done:
            self.dwell += 1
            if self.dwell > w.get('dwell', 0):
                chk = w.get('check')
                if chk and not self.check(chk, s):
                    self.counts['lift_fail'] += 1
                    self.fail_action(f"{w['name']}: check {chk[0]} failed")
                    return
                self.next_waypoint(a, f"{w['name']} reached")
        elif w['mode'] == 'slide' and w['name'] == 'lower' and pos_err < 0.045 and float(np.linalg.norm(self.slide_goal - target)) <= 1e-6:
            # the book rests on the shelf a little above the planned height: let go after 60 controls instead of pressing until the wrist twists
            self.stall = getattr(self, 'stall', 0) + 1 if getattr(self, 'stall_wp', None) == (id(a), self.wi) else 1
            self.stall_wp = (id(a), self.wi)
            if self.stall > 60:
                self.stall = 0
                self.counts['lower_go'] = self.counts.get('lower_go', 0) + 1
                self.next_waypoint(a, f"lower: rests {pos_err:.3f} m above the goal, release")
        elif self.age > WAYPOINT_TIMEOUT:
            self.fail_action(f"waypoint {w['name']} not reached ({pos_err:.3f} m)")
