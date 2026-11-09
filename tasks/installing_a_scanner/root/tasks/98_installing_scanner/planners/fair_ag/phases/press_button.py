"""Phases reach_button, press_button, retreat_button: put the fingertip on the perceived button and hold it there.
reach_button   wait a few controls for a clean look at the scanner from the final base pose, freeze the plan, servo the fingertip to the pre-press
               point (12 cm out along the face normal, pitched 0.6 rad down). Success: tip within REACH_POS_TOL and REACH_ORI_TOL.
press_button   move the tip 1 mm per control along the normal to the press point (8 mm behind the marker centre, into the face), then hold it
               PRESS_DWELL controls. Success: the dwell is over. The button state is never observed (the marker colour is not read for decisions):
               the press is made once, since a second entry into the marker volume would switch the button off again.
retreat_button back to 16 cm out. Success: tip within 4 cm of that point."""
from __future__ import annotations

import copy

import numpy as np

from .. import geometry as G
from .. import params as P

LOOK_STEPS = 9


def frozen_plan(c, w):
    return G.poke_plan(c.press_world, c.tilt)


def step_reach_button(c, action, q, w) -> None:
    c.to_rest(action, q)
    if c.age < LOOK_STEPS:
        c.reason = 'looking at the scanner from the base pose'
        return
    if c.age == LOOK_STEPS:
        if not w.ready:
            c.change('plan_button', 'scanner lost at the base pose')
            return
        c.press_world = copy.deepcopy(w)
        c.servo = c.make_servo()
    if c.servo is None:
        c.servo = c.make_servo()
    plan = frozen_plan(c, w)
    goal, rot = c.tool_goal(plan['pre'], plan['rotation'], c.press_world)      # the base stands still: its pose at the look is the reference
    c.err = c.servo.step(action, q, goal, rot, plan=c.age == LOOK_STEPS + 1)
    c.counts['ik_none'] += int(c.age == LOOK_STEPS + 1 and c.servo.plan_failed)
    c.reason = f'reaching the pre-press point, err {c.err[0]:.3f} m {c.err[1]:.2f} rad'
    if c.age > LOOK_STEPS + 2 and c.err[0] < P.REACH_POS_TOL and c.err[1] < P.REACH_ORI_TOL:
        c.press_s, c.press_hold = P.PRE_OFFSET_M, 0
        c.change('press_button', 'tip at the pre-press point')
    elif c.age > P.REACH_TIMEOUT:
        c.next_candidate('go_button', 'pre-press point not reached')


def step_press_button(c, action, q, w) -> None:
    plan = frozen_plan(c, w)
    final = P.PRESS_DEPTH_M
    c.press_s = max(final, c.press_s - P.PRESS_SPEED_M)
    if c.press_s <= final:
        c.press_hold += 1
    point = plan['marker'] + plan['out'] * c.press_s
    goal, rot = c.tool_goal(point, plan['rotation'], c.press_world)
    c.err = c.servo.step(action, q, goal, rot, plan=False)
    c.reason = f'pressing, depth {c.press_s:.3f}, hold {c.press_hold}, err {c.err[0]:.3f}'
    if c.press_hold > P.PRESS_DWELL:
        c.pressed = True
        c.fold_from = None
        c.change('retreat_button', f'press held {P.PRESS_DWELL} controls at depth {final}')
    elif c.age > P.PRESS_TIMEOUT:
        c.pressed = True
        c.fold_from = None
        c.change('retreat_button', 'press timeout')


def step_retreat_button(c, action, q, w) -> None:
    plan = frozen_plan(c, w)
    goal, rot = c.tool_goal(plan['marker'] + plan['out'] * 0.16, plan['rotation'], c.press_world)
    fold = getattr(c, 'fold_from', None)
    if fold is None:
        c.err = c.servo.step(action, q, goal, rot, plan=False)
        c.reason = f'retreating from the marker, err {c.err[0]:.3f}'
        if c.err[0] < 0.04 or c.age > 150:
            c.servo = None
            c.fold_from = c.age
            c.reason = 'tip clear of the scanner, folding the arms'
        return
    # fold both arms and the trunk back to the rest pose before the grasp planning starts from a clean posture
    worst = c.to_rest(action, q)
    c.reason = f'folding the arms after the press, worst joint {worst:.3f}'
    if worst < 0.06 or c.age - fold > 120:
        c.fold_from = None
        c.change('plan_grasp', 'tip clear of the scanner, arms folded')


PHASES = {'reach_button': step_reach_button, 'press_button': step_press_button, 'retreat_button': step_retreat_button}
