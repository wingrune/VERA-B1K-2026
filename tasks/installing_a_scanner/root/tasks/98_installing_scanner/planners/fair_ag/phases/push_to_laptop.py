"""Phases plan_push, go_push, reach_push, push, retreat_push, done: slide the scanner along the desk to the laptop with the two open fingertips of
one arm, the base following (the fallback when no edge pinch for the assisted grasp is reachable, see grasp_scanner).
plan_push     choose a push (face, direction, travel from the perceived AABB gap) and a base pose with a free lane; ends the task when the perceived
              gap is at most `gap_goal` plus slack. Success of the plan: a feasible push (IK of the approach and touch points).
go_push       drive to the base pose of the push.
reach_push    tip to 7 cm behind the face, jaw along the face. Success: tip error below REACH_POS_TOL.
push          the tip advances 1.2 mm per control while the scanner follows (never more than PUSH_LEAD_M ahead of the face); the pusher line follows
              the scanner's sideways drift slowly. Ends on: gap reached, desk edge, scanner turned more than PUSH_YAW_MAX, no progress, timeout.
retreat_push  tip back 16 cm and up. Then plan again.
done          hold still; if the episode is still running after a while (the evaluation ends it at success), push 1.5 cm closer, then press the button
              once more."""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from core.skills_lib import nav_grid

from .. import geometry as G
from .. import params as P

NEXTTO_MARGIN_M = 0.02       # the perceived gap must be this much inside the native NextTo threshold
DONE_WAIT = 90
RECOVER_AFTER, MAX_RECOVERIES = 40, 4
REPUSH_STEP_M = 0.015


def gap_now(w) -> float:
    return G.box_gap_xy(w.scanner_box, w.laptop_box)


def est_nextto(c, w) -> bool:
    """Perceived NextTo with a margin.
    The native predicate on the perceived boxes, with a safety margin that shrinks as the episode refuses to end (gap_goal falls)."""
    return G.nextto_estimate(w, margin=NEXTTO_MARGIN_M * min(1., c.gap_goal / P.GAP_GOAL_M))


def step_plan_push(c, action, q, w) -> None:
    c.to_rest(action, q)
    c.close_grips()
    if not w.ready:
        c.change('scan', 'the world is not ready for planning')
        return
    if est_nextto(c, w):
        c.done_age = 0
        c.change('done', f'perceived gap {gap_now(w):.3f} m, next to the laptop')
        return
    if c.counts['push_rounds'] >= P.PUSH_MAX_ROUNDS:
        c.change('failed', 'push rounds exhausted')
        return
    c.push_stats = dict(options=0, ranked=0, lane_ok=0, ik_tried=0, ik_ok=0, bad=0)
    for reaching_only in (True, False):
        for soft in (False, True):
            if c.grid is None or soft != c.soft_clutter or c.counts['push_rounds'] != c.grid_round:
                c.build_grid(w, soft=soft)
                c.grid_round = c.counts['push_rounds']
            if find_push(c, w, q, reaching_only):
                return
    c.change('failed', f'no feasible push {c.push_stats}')


def find_push(c, w, q, reaching_only: bool = False) -> bool:
    boxes = c.boxes_for_clearance(w)
    front = G.ScannerFrame(w).front
    for opt in G.push_options(w, c.gap_goal):
        if opt['travel'] < P.MIN_PUSH_M or (reaching_only and not opt['reaches']):
            continue
        if c.pressed and float(front @ opt['normal']) > 0.9:   # a touch of the marker while pushing switches the scanner off again
            continue
        e, contact = opt['e'], opt['contact']
        c.push_stats['options'] += 1
        ranked = []
        for cand in c.rank_candidates(w, contact[:2] - e[:2] * 0.3, radii=P.PUSH_RINGS):
            lane = c.lane_length(cand['xy'], e, opt['travel'])
            if lane >= opt['travel'] - P.ARM_EXTRA_M:
                ranked.append(dict(cand, lane=lane, rank=cand['cost'] - 2.0 * lane))
        c.push_stats['ranked'] += len(ranked)
        ranked.sort(key=lambda cand: cand['rank'])
        for cand in ranked[:P.CAND_LIMIT]:
            base = np.array([cand['xy'][0], cand['xy'][1], cand['yaw']])
            to_contact = G.horiz(contact - np.r_[cand['xy'], 0.])
            for variant, a_h in (('toward', to_contact), ('along', e)):
                for tilt in P.PUSH_TILTS:
                    axis = G.tool_axis(a_h, tilt)
                    rot = G.jaw_rotation(axis, np.cross(G.UP, opt['normal']))
                    pre, touch = contact - e * 0.07, contact - e * 0.008
                    for side in c.sides_for_point(contact, cand):
                        if (variant, side, tuple(np.round(cand['xy'], 1))) in c.bad_push or round_key(opt, cand['xy']) in c.bad_rounds:
                            c.push_stats['bad'] += 1
                            continue
                        c.push_stats['ik_tried'] += 1
                        if c.ik_feasible(side, base, q, [(pre, rot), (touch, rot)], boxes, mid=True):
                            c.push = dict(opt, a_h=a_h, variant=variant, tilt=tilt, side=side, base0=base, lane=cand['lane'])
                            c.goal = dict(cand, side=side, tilt=tilt)
                            c.side, c.tilt, c.servo, c.push_p = side, tilt, None, 0.
                            c.counts['push_rounds'] += 1
                            c.change('go_push', f"push {opt['face']} face along {np.round(e[:2], 2).tolist()}, travel {opt['travel']:.2f} m, "
                                                f"lane {cand['lane']:.2f} m, base {cand['xy'].round(2).tolist()}, arm {side}, {variant}")
                            return True
    return False


def round_key(opt: Dict[str, Any], xy: np.ndarray) -> tuple:
    return (opt['face'], tuple(np.round(opt['e'][:2], 1)), tuple(np.round(np.asarray(xy)[:2], 1)))


def step_go_push(c, action, q, w) -> None:
    dist, yaw_err = c.navigate(action, q, w, c.goal['xy'], c.goal['yaw'])
    c.err = (dist, yaw_err)
    if c.arrived(dist, yaw_err):
        c.change('reach_push', 'at the push base pose')
    elif c.counts['nav_stuck'] >= P.NAV_MAX_STUCK * (2 + c.counts['push_rounds']) or c.age > P.NAV_TIMEOUT or c.no_route > 60:
        c.no_route = 0
        c.bad_rounds.add(round_key(c.push, c.push['base0']))
        c.change('plan_push', 'push base pose not reached')


def push_rotation(c) -> np.ndarray:
    return G.jaw_rotation(G.tool_axis(c.push['a_h'], c.tilt), c.push['t0'])


def push_point(c, w, along: float) -> np.ndarray:
    """Pusher point.
    Pusher point: `along` on the push direction e, the line across e fixed in the desk frame (as e and the jaw)."""
    point = c.push['e'] * along + c.push['t0'] * c.push['across0']
    point[2] = w.scanner.z_lo + P.PUSH_HEIGHT_M
    return point


def touch_along(c, w) -> float:
    """Where the pusher line meets the scanner face.
    Along-e coordinate at which the pusher line meets the live face plane of the scanner."""
    f = G.ScannerFrame(w)
    n, centre, _ = f.faces()[c.push['face']]
    e, t0 = c.push['e'], c.push['t0']
    ne = float(n[:2] @ e[:2])
    if ne > -0.2:
        return float(f.centre @ e) - f.halfspan(e)
    return float((n[:2] @ centre[:2] - c.push['across0'] * (n[:2] @ t0[:2])) / ne)


def step_reach_push(c, action, q, w) -> None:
    if c.servo is None:
        c.servo = c.make_servo(mid=True)
        c.open_grip(c.push['spread'])
        c.push_yaw0 = w.scanner.yaw
        f = G.ScannerFrame(w)
        c.push['t0'] = G.unit(np.cross(G.UP, c.push['e']))
        c.push['across0'] = float(c.push['contact'][:2] @ c.push['t0'][:2])
        c.push['lat0'] = c.push['across0'] - float(f.centre[:2] @ c.push['t0'][:2])
        c.push_scanner0, c.push_base0 = f.centre.copy(), w.base[:2].copy()
        c.face_filt = touch_along(c, w)
    c.face_filt += 0.3 * (touch_along(c, w) - c.face_filt)          # the tip follows the live (smoothed) face of the scanner, not a line fixed at the start
    goal, rotr = c.tool_goal(push_point(c, w, c.face_filt - 0.07), push_rotation(c), w)
    c.err = c.servo.step(action, q, goal, rotr, plan=c.age == 1)
    c.counts['ik_none'] += int(c.age == 1 and c.servo.plan_failed)
    c.reason = f'reaching the push start, err {c.err[0]:.3f} m {c.err[1]:.2f} rad'
    if c.err[0] < P.REACH_POS_TOL and c.err[1] < P.REACH_ORI_TOL:
        c.push_p, c.push_stall, c.push_moved = -0.07, 0, -1.
        c.tip_along = c.face_filt - 0.07
        c.change('push', 'tip behind the scanner face')
    elif c.age > P.REACH_TIMEOUT:
        c.servo = None
        c.bad_push.add((c.push['variant'], c.side, tuple(np.round(c.push['base0'][:2], 1))))
        c.change('plan_push', 'push start not reached')


def step_push(c, action, q, w) -> None:
    if est_nextto(c, w):
        c.change('retreat_push', f'perceived gap {gap_now(w):.3f} m, next to the laptop')
        return
    e = c.push['e']
    f = G.ScannerFrame(w)
    moved = float((f.centre - c.push_scanner0) @ e)
    target = float(f.centre[:2] @ c.push['t0'][:2]) + c.push['lat0']
    c.push['across0'] += float(np.clip(target - c.push['across0'], -P.PUSH_ACROSS_RATE_M, P.PUSH_ACROSS_RATE_M))
    c.face_filt += 0.3 * (touch_along(c, w) - c.face_filt)
    along = c.face_filt - 0.008 + c.push_p
    if c.tip_along is not None:      # the tip goal advances at most 2.5 times the push speed per control (a smooth push, no jerk from the estimate)
        along = float(np.clip(along, c.tip_along - 2.5 * P.PUSH_SPEED_M, c.tip_along + 2.5 * P.PUSH_SPEED_M))
    c.tip_along = along
    goal, rotr = c.tool_goal(push_point(c, w, along), push_rotation(c), w)
    c.err = c.servo.step(action, q, goal, rotr, plan=False)
    cap = P.PUSH_LEAD_M + P.PUSH_CAP_STEP_M * min(c.push_stall // P.PUSH_CAP_EVERY, P.PUSH_CAP_STEPS)     # no motion: press harder, a centimetre at a time
    if c.err[0] < 0.03 and c.push_p < cap:       # push_p: how far the tip goal lies beyond the live face (the arm presses with its stiffness)
        c.push_p += P.PUSH_SPEED_M
    desired = c.push_base0 + e[:2] * min(max(moved, 0.), c.push['lane'])
    v = np.clip((desired - w.base[:2]) * 1.5, -0.25, 0.25)
    cs, sn = math.cos(w.base[2]), math.sin(w.base[2])
    action[0] = (cs * v[0] + sn * v[1]) / nav_grid.BASE_SPEED
    action[1] = (-sn * v[0] + cs * v[1]) / nav_grid.BASE_SPEED
    action[2] = np.clip(nav_grid.wrap(c.push['base0'][2] - w.base[2]) * P.NAV_YAW_GAIN, -0.2, 0.2)
    c.reason = f"pushing, tip {c.push_p:.3f}, scanner moved {moved:.3f} of {c.push['travel']:.2f} m, gap {gap_now(w):.3f}, err {c.err[0]:.3f}"
    if moved > c.push_moved + P.PUSH_PROGRESS_M:
        c.push_moved, c.push_stall = moved, 0
    else:
        c.push_stall += 1
    if G.edge_limit(w.scanner_box, w.table_box, e, 0.0) < P.PUSH_EDGE_STOP_M:
        c.change('retreat_push', 'scanner at the desk edge')
    elif moved >= c.push['travel'] + 0.02:
        c.change('retreat_push', 'planned travel done')
    elif abs(nav_grid.wrap(w.scanner.yaw - c.push_yaw0)) > P.PUSH_YAW_MAX:
        c.bad_rounds.add(round_key(c.push, c.push['base0']))
        c.change('retreat_push', 'scanner rotated during the push')
    elif c.push_stall > P.PUSH_STALL_STEPS:
        c.bad_rounds.add(round_key(c.push, c.push['base0']))
        c.change('retreat_push', 'scanner not moving')
    elif c.age > P.PUSH_TIMEOUT:
        c.change('retreat_push', 'push timeout')


def step_retreat_push(c, action, q, w) -> None:
    if c.servo is None:
        c.change('plan_push', 'no servo')
        return
    goal, rotr = c.tool_goal(push_point(c, w, touch_along(c, w) - 0.16) + G.UP * 0.04, push_rotation(c), w)
    c.err = c.servo.step(action, q, goal, rotr, plan=False)
    c.reason = f'retreating from the push, err {c.err[0]:.3f}'
    if c.err[0] < 0.04 or c.age > 150:
        c.servo, c.push_moved = None, -1.
        c.close_grips()
        c.change('plan_push', 'tip clear of the scanner')


def step_done(c, action, q, w) -> None:
    c.to_rest(action, q)
    c.done_age += 1
    c.reason = f'next to the laptop (perceived gap {gap_now(w):.3f} m); waiting for the evaluation to end the episode'
    if c.done_age > DONE_WAIT:
        c.done_age = 0
        if c.gap_goal > 0.015:
            c.gap_goal = max(0.01, c.gap_goal - REPUSH_STEP_M)
            c.change('plan_push', f'the episode goes on: push closer (gap goal {c.gap_goal:.3f})')
        elif P.REPRESS_ALLOWED and not c.repressed:     # a second press may switch the button off again, so it is off by default
            c.repressed, c.press_s = True, P.PRE_OFFSET_M
            c.change('plan_button', 'the episode goes on after the push: press the button once more')


def step_failed(c, action, q, w) -> None:
    """Hold still, then recover.
    Hold still, then recover: forget the failed pushes and base poses, look around again and plan anew (RECOVERIES times)."""
    c.to_rest(action, q)
    if c.age > RECOVER_AFTER and c.recoveries < MAX_RECOVERIES:
        c.recoveries += 1
        c.bad_push, c.bad_rounds, c.cands = set(), set(), []
        c.counts['push_rounds'] = 0
        c.servo, c.scan_total, c.scan_last_yaw = None, 0., c.perc.odo.pose[2]
        c.change('scan', f'recovery {c.recoveries}: planning anew after a look around')


PHASES = {'plan_push': step_plan_push, 'go_push': step_go_push, 'reach_push': step_reach_push, 'push': step_push,
          'retreat_push': step_retreat_push, 'done': step_done, 'failed': step_failed}
