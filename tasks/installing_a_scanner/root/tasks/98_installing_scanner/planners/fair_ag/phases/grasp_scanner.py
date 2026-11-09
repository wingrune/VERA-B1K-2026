"""Phases plan_grasp, go_grasp, pre_grasp, reach_grasp, close_grasp, back_off, lift: pinch a top edge of the perceived scanner (the native
assisted grasp holds it) and lift it.
plan_grasp   edge pinches of every face but the button face (pick_geometry), stands on rings around the scanner, one GPU IK scan of the pre-grasp
             and grasp points over all stands, A* routes for the cheapest. None: push the scanner instead (plan_push).
go_grasp     drive to the stand (kept relative to the scanner, so a refined estimate moves it). Success: at the base pose.
pre_grasp    look at the scanner from the stand, freeze that World for the arm phases, open the hand, servo the finger midpoint to 10 cm back
             along the tool axis (solve_near seeded with the current joints; the joint path must stay clear of desk, laptop and scanner).
reach_grasp  move in along the tool axis 3 mm per control.
close_grasp  close; success: the fingers stopped on the scanner (finger gap from proprio) for HOLD_SETTLE controls. Else back_off.
lift         raise the hand LIFT_M straight up, the hand rotation kept. Lost hold (the hand closed fully): next pinch."""
from __future__ import annotations

import copy
import math

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import ik
from .. import params as P
from .. import pick_geometry as PG
from . import pick_core as PC
from .push_to_laptop import est_nextto

LOOK_STEPS = 9
THETA_RANK_COST, OFFSET_COST = 0.25, 1.0   # stand cost per rank of the jaw tilt, per metre of offset along the edge
GRASP_SPREAD_M = 0.03                      # each finger this far from the midpoint while approaching
APPROACH_SPEED, LIFT_SPEED, LIFT_M = 0.003, 0.003, 0.12   # m per control; lift height
GRIP_WAIT, HOLD_SETTLE = 60, 6             # controls after the close without a hold: failed; controls of a steady hold before the lift
MAX_GRASP_TRIES, MAX_PLACE_ROUNDS = 10, 3


def grasp_candidates(c, w, q):
    f = G.ScannerFrame(w)
    opts = PG.grasp_options(f)
    stands, ecost = PC.stands_around(c, w, f.centre)
    boxes = c.boxes_for_clearance(w)
    out = []
    for side in ('left', 'right'):
        goals = []
        for o in opts:
            goals += [PG.pose_T(o['rotation'], o['pre']), PG.pose_T(o['rotation'], o['mid'])]
        if not goals:
            break
        c.counts['ik_checks'] += len(goals) * len(stands)
        masks, qs = ik.scan(side, goals, stands, 0., K.FINGER_MID_POINT)
        for j, o in enumerate(opts):
            rank = o['rank']
            for i in np.flatnonzero(masks[2 * j] & masks[2 * j + 1]):
                key = (o['face'], round(o['offset'], 2), round(o['theta'], 2), o['inset'], side, tuple(np.round(stands[i][:2], 1)))
                if key in c.pick['bad'] or not (c.arm_clear(side, qs[2 * j, i], q, stands[i], boxes) and
                                                c.arm_clear(side, qs[2 * j + 1, i], q, stands[i], boxes)):
                    continue
                here = i == 0 and float(np.linalg.norm(stands[0][:2] - w.base[:2])) < 0.03
                out.append(dict(stand=stands[i].copy(), side=side, opt=o, key=key, ecost=float(ecost[i]),
                                cost=float(ecost[i]) + THETA_RANK_COST * rank + OFFSET_COST * abs(o['offset']),
                                x_from=q[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])].copy() if here else K.rest(side), x_to=qs[2 * j, i].copy()))
    return out


def step_plan_grasp(c, action, q, w) -> None:
    c.to_rest(action, q)
    c.close_grips()
    pk = c.pick
    if not w.ready:
        c.change('scan', 'the world is not ready for planning')
        return
    if pk['placed'] and est_nextto(c, w):
        c.change('done' if c.pressed else 'plan_button', 'perceived next to the laptop')
        return
    if pk['tries'] >= MAX_GRASP_TRIES or pk['place_rounds'] >= MAX_PLACE_ROUNDS:
        c.change('plan_push', 'grasp tries or place rounds exhausted; pushing instead')
        return
    for soft in (False, True):
        c.build_grid(w, soft=soft)
        pk['cands'] = grasp_candidates(c, w, q)
        if pk['cands']:
            break
    next_grasp(c, w, 'edge pinch')


def path_ok(c, w, q, cand) -> bool:
    if 'path_ok' not in cand:
        cand['path_ok'] = PC.arm_path_clear(w, cand['side'], q, cand['x_from'], cand['x_to'], cand['stand'])
    return cand['path_ok']


def next_grasp(c, w, why) -> None:
    pk = c.pick
    q = K.q_from_proprio(c.perc.proprio)
    pk['grasp'] = PC.pick_routed(c, w, pk['cands'], lambda cand: path_ok(c, w, q, cand)) if pk['cands'] else None
    if pk['grasp'] is None:
        c.change('plan_push', why + '; no reachable edge pinch, pushing instead')
        return
    g = pk['grasp']
    pk['tries'] += 1
    rot = np.array([[math.cos(-w.scanner.yaw), -math.sin(-w.scanner.yaw)], [math.sin(-w.scanner.yaw), math.cos(-w.scanner.yaw)]])
    g['rel_xy'], g['rel_yaw'] = rot @ (g['stand'][:2] - w.scanner.centre), nav_grid.wrap(g['stand'][2] - w.scanner.yaw)
    c.side, c.servo = g['side'], None
    o = g['opt']
    text = (f"{why}: face {o['face']} offset {o['offset']:+.2f} jaw {math.degrees(o['theta']):.0f} deg, arm {c.side}, "
            f"stand {np.round(g['stand'], 2).tolist()}{' (chairs movable)' if c.soft_clutter else ''}")
    here = PC.route_len(c, w, g['stand'][:2]) == 0. and abs(nav_grid.wrap(g['stand'][2] - w.base[2])) < P.NAV_YAW_TOL
    c.change('pre_grasp' if here else 'go_grasp', text + (' (from here)' if here else ''))


def fail_grasp(c, w, why) -> None:
    pk = c.pick
    pk['bad'].add(pk['grasp']['key'])
    pk['fails'] += 1
    c.grip[c.side] = K.GRIPPER_CLOSED
    c.servo, c.perc.held = None, None
    if pk['cands']:
        next_grasp(c, w, why + '; next pinch')
    else:
        c.change('plan_grasp', why + '; re-planning')


def live_grasp(c):
    o = c.pick['grasp']['opt']
    return PG.edge_grasp(G.ScannerFrame(c.pick['world']), o['face'], o['offset'], o['theta'], o['inset'], o['bite'])


def step_go_grasp(c, action, q, w) -> None:
    g = c.pick['grasp']
    if w.scanner is not None:
        cs, sn = math.cos(w.scanner.yaw), math.sin(w.scanner.yaw)
        g['stand'][:2] = w.scanner.centre + np.array([[cs, -sn], [sn, cs]]) @ g['rel_xy']
        g['stand'][2] = nav_grid.wrap(w.scanner.yaw + g['rel_yaw'])
    dist, yaw_err = c.navigate(action, q, w, g['stand'][:2], g['stand'][2])
    c.err = (dist, yaw_err)
    if c.arrived(dist, yaw_err):
        c.change('pre_grasp', 'at the grasp stand')
    elif c.counts['nav_stuck'] >= P.NAV_MAX_STUCK * (1 + c.counts['candidates_failed']) or c.age > P.NAV_TIMEOUT or c.no_route > 60:
        c.no_route = 0
        c.counts['candidates_failed'] += 1
        fail_grasp(c, w, 'grasp stand not reached')


def step_pre_grasp(c, action, q, w) -> None:
    if c.age < LOOK_STEPS:
        c.to_rest(action, q)
        c.reason = 'looking at the scanner from the grasp stand'
        return
    if c.age == LOOK_STEPS:
        if not w.ready:
            c.change('plan_grasp', 'scanner lost at the grasp stand')
            return
        c.pick['world'] = copy.deepcopy(w)
        c.servo = c.make_servo(mid=True)
    c.open_grip(GRASP_SPREAD_M)
    o = live_grasp(c)
    goal, rot = c.tool_goal(o['pre'], o['rotation'], c.pick['world'])
    first = c.age == LOOK_STEPS
    c.err = c.servo.step(action, q, goal, rot, plan=first)
    if first:
        dofs = list(c.servo.dofs)
        if c.servo.target is None or not PC.arm_path_clear(c.pick['world'], c.side, q, q[dofs], c.servo.target, c.pick['world'].base):
            c.counts['ik_none'] += int(c.servo.target is None)
            fail_grasp(c, w, 'no clear joint path to the pre-grasp point')
            return
    c.reason = f'reaching the pre-grasp point, err {c.err[0]:.3f} m {c.err[1]:.2f} rad'
    if c.err[0] < 0.015 and c.err[1] < 0.2:
        c.change('reach_grasp', 'fingers at the pre-grasp point')
    elif c.age > P.REACH_TIMEOUT:
        fail_grasp(c, w, 'pre-grasp point not reached')


def step_reach_grasp(c, action, q, w) -> None:
    o = live_grasp(c)
    lam = min(PG.GRASP_BACK_M, c.age * APPROACH_SPEED)
    goal, rot = c.tool_goal(o['pre'] + o['axis'] * lam, o['rotation'], c.pick['world'])
    c.err = c.servo.step(action, q, goal, rot, plan=False)
    c.reason = f"moving in along the tool axis, {lam:.3f} of {PG.GRASP_BACK_M:.2f} m, err {c.err[0]:.3f}"
    if lam >= PG.GRASP_BACK_M and (c.err[0] < 0.012 or c.age > PG.GRASP_BACK_M / APPROACH_SPEED + 60):
        c.pick['gaps'], c.pick['held_age'] = [], 0
        c.change('close_grasp', f'fingers at the edge, err {c.err[0]:.3f} m')


def step_close_grasp(c, action, q, w) -> None:
    pk = c.pick
    o = live_grasp(c)
    c.grip[c.side] = K.GRIPPER_CLOSED
    goal, rot = c.tool_goal(o['mid'], o['rotation'], pk['world'])
    c.err = c.servo.step(action, q, goal, rot, plan=False)
    pk['held_age'] = pk['held_age'] + 1 if PC.update_hold(c, q) else 0
    c.reason = f"closing on the edge, held {pk['held_age']}, finger gap {PC.finger_gap(q, c.side):.4f}"
    if pk['held_age'] >= HOLD_SETTLE:
        T = PC.mid_pose(q, w, c.side)
        pk['lift0'], pk['rot'] = T[:3, 3].copy(), T[:3, :3].copy()
        pk['rel'] = np.linalg.inv(PG.scanner_T(pk['world'].scanner)) @ T      # the hand in the scanner frame, fixed by the assisted grasp
        sc = pk['world'].scanner
        c.perc.held = dict(side=c.side, rel=pk['rel'], half=tuple(sc.half), height=sc.z_hi - sc.z_lo)
        c.change('lift', f"the fingers stopped on the scanner (gap {PC.finger_gap(q, c.side):.4f}) after {c.age} controls")
    elif c.age > GRIP_WAIT:
        c.grip[c.side] = K.GRIPPER_OPEN
        c.change('back_off', f'the hand closed without a hold (gap {PC.finger_gap(q, c.side):.4f})')


def step_back_off(c, action, q, w) -> None:
    c.grip[c.side] = K.GRIPPER_OPEN
    o = live_grasp(c)
    goal, rot = c.tool_goal(o['pre'] + G.UP * 0.03, o['rotation'], c.pick['world'])
    c.err = c.servo.step(action, q, goal, rot, plan=False)
    c.reason = f'backing off the edge, err {c.err[0]:.3f}'
    if (c.err[0] < 0.03 and c.age > 10) or c.age > 150:
        fail_grasp(c, w, 'pinch failed')


def step_lift(c, action, q, w) -> None:
    pk = c.pick
    if not PC.holding(c, q):
        pk['drops'] += 1
        fail_grasp(c, w, 'hold lost while lifting')
        return
    lam = min(LIFT_M, c.age * LIFT_SPEED)
    goal, rot = c.tool_goal(pk['lift0'] + G.UP * lam, pk['rot'], w)
    c.err = c.servo.step(action, q, goal, rot, plan=False)
    c.reason = f"lifting {lam:.3f} of {LIFT_M:.2f} m, finger gap {PC.finger_gap(q, c.side):.4f}, err {c.err[0]:.3f}"
    if lam >= LIFT_M and (c.err[0] < 0.015 or c.age > LIFT_M / LIFT_SPEED + 80):
        pk['lift1'] = PC.mid_pose(q, w, c.side)[:3, 3].copy()
        pk['base1'] = w.base.copy()
        pk['hold_q'] = q[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[c.side])].copy()
        c.change('plan_carry', f'scanner lifted, finger gap {PC.finger_gap(q, c.side):.4f}')


PHASES = {'plan_grasp': step_plan_grasp, 'go_grasp': step_go_grasp, 'pre_grasp': step_pre_grasp, 'reach_grasp': step_reach_grasp,
          'close_grasp': step_close_grasp, 'back_off': step_back_off, 'lift': step_lift}
