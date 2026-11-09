"""Phases plan_carry, carry, pre_place, lower, release, retreat_grasp: carry the held scanner next to the laptop and put it down.
plan_carry     resting poses of the scanner next to the perceived laptop (pick_geometry.place_poses, the hand pose from the grasp transform), stands
               on rings around each, one GPU IK scan of the place and the pre-place (10 cm above) hand poses, A* routes for the cheapest. Before
               the press the button should face the place stand. None: lower it where it is.
carry          drive to the stand with the trunk and the holding arm frozen (the scanner keeps its pose in the hand). Lost hold: re-grasp.
pre_place      the hand along a straight line (and a rotation slerp) from its pose to the pre-place pose, Jacobian steps (no IK branch jumps).
lower          straight down 2.5 mm per control until the scanner bottom (hand pose and grasp transform) is RELEASE_GAP_M above the desk.
release        open the hand; the placed scanner pose (hand pose and grasp transform) replaces the estimate, perception refines it from there.
retreat_grasp  hand 10 cm back along the tool axis and 5 cm up. Then: next to the laptop by the perceived boxes -> done (or the press when the
               button is not pressed yet); else a new pick (MAX_PLACE_ROUNDS)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import ik
from .. import params as P
from .. import pick_geometry as PG
from ..world import Box
from . import pick_core as PC
from .grasp_scanner import fail_grasp
from .grasp_scanner import LIFT_M
from .push_to_laptop import est_nextto

CARRY_VMAX, CARRY_WMAX, CARRY_TIMEOUT = 0.25, 0.3, 1500   # base speed (fraction), turn rate and time limit while carrying
PLACE_ABOVE_M, LOWER_SPEED = 0.10, 0.0025                 # pre-place height above the resting pose; m per control
PLACE_STOP_ERR_M = 0.015          # lowering: the servo lags this far behind its goal near the desk, the scanner rests on it
RELEASE_WAIT = 12
LINE_SPEED, RELEASE_GAP_M, LOWER_TIMEOUT = 0.004, 0.012, 400   # m per control to the pre-place; scanner bottom gap of the release


def place_candidates(c, w, q):
    pk = c.pick
    grow = np.full(3, 0.02)
    boxes = [Box(b.lo - grow, b.hi + grow) for b in (w.table_box, w.laptop_box)]
    marker = PG.marker_local(w.scanner)
    out = []
    for t in PG.place_poses(w):
        mid = t['T'] @ pk['rel']
        pre = mid.copy()
        pre[2, 3] += PLACE_ABOVE_M
        stands, ecost = PC.stands_around(c, w, t['centre'])
        c.counts['ik_checks'] += 2 * len(stands)
        masks, qs = ik.scan(c.side, [pre, mid], stands, 0., K.FINGER_MID_POINT)
        for i in np.flatnonzero(masks.all(0)):
            if c.arm_clear(c.side, qs[0, i], q, stands[i], boxes) and c.arm_clear(c.side, qs[1, i], q, stands[i], boxes):
                extra = 0. if c.pressed else PG.press_after_place_cost(t['T'], marker, stands[i])
                out.append(dict(stand=stands[i].copy(), ecost=float(ecost[i]), cost=float(ecost[i]) + t['cost'] + extra, place=t, T_mid=mid,
                                T_pre=pre, side=c.side, x_from=pk['hold_q'].copy(), x_to=qs[0, i].copy()))
    return out


def lost(c, q, w, where) -> bool:
    if PC.holding(c, q):
        return False
    c.pick['drops'] += 1
    fail_grasp(c, w, f'scanner dropped {where}')
    return True


def step_plan_carry(c, action, q, w) -> None:
    PC.hold_arm(c, action)
    pk = c.pick
    pk['carry'] = None
    for soft in (False, True):
        c.build_grid(w, soft=soft)
        pk['carry'] = PC.pick_routed(c, w, place_candidates(c, w, q))
        if pk['carry'] is not None:
            break
    if pk['carry'] is None:
        c.change('lower', 'no stand reaches a place next to the laptop; putting it down here')
        return
    cr = pk['carry']
    c.servo = None
    c.change('carry', f"to stand {np.round(cr['stand'], 2).tolist()}, scanner to {np.round(cr['place']['centre'][:2], 2).tolist()} "
                      f"yaw {cr['place']['yaw']:.2f} ({cr['place']['quarter']} quarter turns), route cost {cr['total']:.2f}")


def step_carry(c, action, q, w) -> None:
    if lost(c, q, w, 'while carrying'):
        return
    g = c.pick['carry']['stand']
    dist, yaw_err = c.navigate(action, q, w, g[:2], g[2], rest=False, vmax_far=CARRY_VMAX, wmax=CARRY_WMAX)
    PC.hold_arm(c, action)
    c.err = (dist, yaw_err)
    pk = c.pick
    if c.arrived(dist, yaw_err):
        c.change('pre_place', f'at the place stand, {dist:.3f} m, yaw err {yaw_err:.2f}')
    elif c.age > CARRY_TIMEOUT:
        tries = pk.get('carry_replans', 0)
        if dist > 0.25 and tries < 2:
            pk['carry_replans'] = tries + 1
            c.servo = None
            c.change('plan_carry', f'carry timeout {dist:.2f} m from the stand; planning the carry anew ({tries + 1})')
        else:
            c.change('pre_place', f'carry timeout at {dist:.2f} m')


def step_pre_place(c, action, q, w) -> None:
    if lost(c, q, w, 'before the place'):
        return
    pk = c.pick
    if c.servo is None:
        c.servo = c.make_servo(mid=True)
    T = pk['carry']['T_pre']
    if c.age == 1:
        pk['line0'] = PC.mid_pose(q, w, c.side)
    T0 = pk['line0']
    turn = R.from_matrix(T[:3, :3] @ T0[:3, :3].T)
    span = max(float(np.linalg.norm(T[:3, 3] - T0[:3, 3])), turn.magnitude() * 0.2, 1e-6)
    lam = min(1., c.age * LINE_SPEED / span)
    rot = R.from_rotvec(turn.as_rotvec() * lam).as_matrix() @ T0[:3, :3]
    goal, rotr = c.tool_goal(T0[:3, 3] + (T[:3, 3] - T0[:3, 3]) * lam, rot, w)
    c.err = c.servo.step(action, q, goal, rotr, plan=False)
    left = float(np.linalg.norm(T[:3, 3] - PC.mid_pose(q, w, c.side)[:3, 3]))
    c.reason = f'scanner to {PLACE_ABOVE_M:.2f} m above its place along a line, {lam:.2f} done, {left:.3f} m left'
    if lam >= 1. and left < 0.02:
        c.change('lower', 'scanner above its place')
    elif c.age > P.REACH_TIMEOUT:
        c.change('lower', f'pre-place not reached ({left:.3f} m left); lowering from here')


def step_lower(c, action, q, w) -> None:
    pk = c.pick
    if c.servo is None:
        c.servo = c.make_servo(mid=True)
    T = PC.mid_pose(q, w, c.side)
    gap = float((T @ np.linalg.inv(pk['rel']))[2, 3] - w.table_box.hi[2])
    if c.age == 1:
        pk['line0'], pk['depth'] = T, gap + 0.01
    T0 = pk['line0']
    lam = min(pk['depth'], c.age * LOWER_SPEED)
    goal, rotr = c.tool_goal(T0[:3, 3] - G.UP * lam, T0[:3, :3], w)
    c.err = c.servo.step(action, q, goal, rotr, plan=False)
    c.reason = f'lowering {lam:.3f} of {pk["depth"]:.3f} m, scanner about {gap:.3f} m above the desk, err {c.err[0]:.3f}'
    if gap < RELEASE_GAP_M or (lam >= pk['depth'] and c.err[0] > PLACE_STOP_ERR_M):
        c.change('release', f'scanner on the desk (gap {gap:.3f} m, err {c.err[0]:.3f} m)')
    elif c.age > LOWER_TIMEOUT:
        c.change('release', f'lowering stalled, scanner about {gap:.3f} m above the desk')


def step_release(c, action, q, w) -> None:
    pk = c.pick
    c.grip[c.side] = K.GRIPPER_OPEN
    if c.age == 1:
        T = PC.mid_pose(q, w, c.side)
        placed = T @ np.linalg.inv(pk['rel'])
        c.perc.scanner = PG.scanner_from_T(placed, pk['world'].scanner, c.perc.step)
        pk['retreat'], pk['rot'] = T[:3, 3] + T[:3, :3] @ np.array([0., 0., 0.10]) + G.UP * 0.05, T[:3, :3].copy()
    c.reason = f'opening, finger gap {PC.finger_gap(q, c.side):.4f}'
    if c.age >= RELEASE_WAIT:
        c.perc.held = None
        pk['placed'] = True
        c.change('retreat_grasp', 'released')


def step_retreat_grasp(c, action, q, w) -> None:
    pk = c.pick
    c.grip[c.side] = K.GRIPPER_OPEN
    goal, rot = c.tool_goal(pk['retreat'], pk['rot'], w)
    c.err = c.servo.step(action, q, goal, rot, plan=False)
    c.reason = f'retreating from the scanner, err {c.err[0]:.3f}'
    if c.err[0] < 0.03 or c.age > 150:
        c.servo = None
        c.close_grips()
        if pk['carry'] is not None and est_nextto(c, w):
            c.done_age = 0
            c.change('done' if c.pressed else 'plan_button', 'perceived next to the laptop')
        else:
            pk['place_rounds'] += 1
            c.change('plan_grasp', 'placed without the perceived nextto; re-grasping')


PHASES = {'plan_carry': step_plan_carry, 'carry': step_carry, 'pre_place': step_pre_place, 'lower': step_lower, 'release': step_release,
          'retreat_grasp': step_retreat_grasp}
