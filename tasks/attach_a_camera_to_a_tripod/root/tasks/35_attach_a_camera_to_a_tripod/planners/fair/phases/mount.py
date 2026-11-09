"""Mount phases of the fair planner of task 35: move the held camera along a straight line until its M frame (model prior, through the grasp
transform) is PRE_M above the estimated F frame with F's orientation, lower it along F's vertical past F (native AttachedTo joins them on
contact within 5 cm and 15 degrees; the episode ends on success, which the planner never observes), back up and try the next attempt
(another tripod yaw hypothesis, then small horizontal dithers), finally let the camera go on the head. Subgoals pre_mount and attached."""
from __future__ import annotations

import math

from core.skills_lib import r1pro_kinematics as K

from .. import ik
from .. import params as P
from .carry import attempts, mount_T
from .grasp import drop


def _reachable_here(c, w, q) -> bool:
    import numpy as np
    from .. import geometry as G
    T = mount_T(c, w, 0.)
    masks, _ = ik.scan(c.side, [G.raised(T, P.PRE_M), T], np.array([w.base]), 0., K.FINGER_MID_POINT)
    return bool(masks.all())


def to_pre_mount(c, action, q, w) -> None:
    if not c.holds(q, c.side):
        drop(c, w, 'camera lost before the mount')
        return
    c.other_rest(action, q)
    T = mount_T(c, w, P.PRE_M)
    if c.age == 1:
        c.line0 = c.mid_world(q, c.side)
        if P.MOUNT_SOLVE:       # joint target of the pose over the mount (solve_near from the current joints), walked with every joint together
            goal, rot = c.tool_goal(T[:3, 3], T[:3, :3], w)
            x = c.servo.solve(q, goal, rot)
            c.servo.target, c.servo.sync = x, x is not None
    if P.MOUNT_SOLVE and c.servo.target is not None:
        goal, rot = c.tool_goal(T[:3, 3], T[:3, :3], w)
        c.err = c.servo.step(action, q, goal, rot)
        lam, done = 1., False
        c.line0 = c.mid_world(q, c.side)
    else:
        lam, done = c.line_to(action, q, w, T, P.LINE_SPEED)
    c.reason = f'to the pre-mount {lam:.2f}, err {c.err[0]:.3f} m {c.err[1]:.2f} rad, attempt {c.attempt}'
    if done and c.err[0] < 0.012 and c.err[1] < 0.08:
        c.change('insert', f'camera above the mount, attempt {c.attempt} {attempts(w)[min(c.attempt, len(attempts(w)) - 1)]}')
    elif c.age > 200 and c.err[0] < 0.02 and c.err[1] < 0.12:
        c.change('insert', f'camera near the mount after {c.age} steps (err {c.err[0]:.3f} m), attempt {c.attempt}')
    elif c.age > P.PRE_MOUNT_TIMEOUT:
        next_attempt(c, w, q, f'pre-mount not reached (err {c.err[0]:.3f} m {c.err[1]:.2f} rad)')


def insert(c, action, q, w) -> None:
    if not c.holds(q, c.side):
        drop(c, w, 'camera lost during the insert')
        return
    c.other_rest(action, q)
    depth = P.PRE_M + P.INSERT_BELOW_M
    lam = min(depth, c.age * P.INSERT_SPEED)
    T = mount_T(c, w, P.PRE_M - lam)
    goal, rot = c.tool_goal(T[:3, 3], T[:3, :3], w)
    c.err = c.servo.step(action, q, goal, rot, plan=False)
    c.reason = f'lowering onto the mount {lam:.3f} of {depth:.3f} m, err {c.err[0]:.3f} m {c.err[1]:.2f} rad, attempt {c.attempt}'
    if lam >= depth and c.age > depth / P.INSERT_SPEED + P.INSERT_DWELL:
        c.counts['mount_tries'] += 1
        c.mount_log.append(dict(step=c.step, attempt=c.attempt, err=round(c.err[0], 4), deg=round(math.degrees(c.err[1]), 2)))
        c.change('retract', f'insert done without the episode ending (attempt {c.attempt})')


def next_attempt(c, w, q, why: str) -> None:
    c.attempt += 1
    if c.attempt >= len(attempts(w)):
        c.change('release', why + '; every attempt tried, letting go on the head')
    elif not _reachable_here(c, w, q):
        c.perc.freeze['tripod'] = False
        c.change('plan_mount', why + f'; attempt {c.attempt} not reachable from this stand')
    else:
        c.change('to_pre_mount', why + f'; attempt {c.attempt}')


def retract(c, action, q, w) -> None:
    c.other_rest(action, q)
    if c.age == 1:
        c.line0 = c.mid_world(q, c.side)
    T = mount_T(c, w, P.PRE_M)
    lam, done = c.line_to(action, q, w, T, P.LINE_SPEED)
    c.reason = f'back to the pre-mount {lam:.2f}, err {c.err[0]:.3f}'
    if done and (c.err[0] < 0.015 or c.age > 200):
        next_attempt(c, w, q, 'retracted')


def release(c, action, q, w) -> None:
    c.grip[c.side] = K.GRIPPER_OPEN
    c.reason = 'opening over the tripod head'
    if c.age > 60:
        c.perc.released()
        c.change('failed', 'released without observing the end of the episode')


def done(c, action, q, w) -> None:
    c.reason = 'done'


def failed(c, action, q, w) -> None:
    if c.hold_q is not None and c.holds(q, c.side):
        c.hold_arm(action)


PHASES = dict(to_pre_mount=to_pre_mount, insert=insert, retract=retract, release=release, done=done, failed=failed)
