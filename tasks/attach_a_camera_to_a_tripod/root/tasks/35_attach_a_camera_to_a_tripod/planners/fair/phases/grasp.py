"""Grasp phases of the fair planner of task 35: choose a top pinch across a narrow part of the fitted camera model (lens barrel first) and a
stand that reaches it (GPU reachable_stands mask over rings around the camera), walk there, refit the camera from the stand, reach above
the pinch (solve_near servo), move in along the tool axis, close (the native assisted grasp holds the camera), confirm the hold by the
finger gap (proprio), lift. Subgoals grasped and lifted."""
from __future__ import annotations

import math

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import ik
from .. import params as P
from .pinch import approach_clear, arm_path_clear, candidates, check


def plan_grasp(c, action, q, w) -> None:
    c.to_rest(action, q)
    if w.camera is None:
        c.turned, c.scan_from = 0., None
        c.change('scan', 'camera estimate lost; turning again')
        return
    if c.counts['grasp_tries'] >= P.MAX_GRASP_TRIES:
        c.change('failed', 'grasp tries exhausted')
        return
    c.counts['plans'] += 1
    c.build_grid(w, [w.camera.xy] + ([w.tripod.xy] if w.tripod is not None else []))
    c.grasp_cands = candidates(c, w, q)
    next_grasp(c, w, f'top pinch ({len(c.grasp_cands)} candidates)')


def next_grasp(c, w, why: str) -> None:
    c.grasp = c.pick(w, c.grasp_cands, check(w)) if c.grasp_cands else None
    if c.grasp is None:
        c.change('failed', why + f'; no reachable camera pinch {c.pick_stats}')
        return
    c.counts['grasp_tries'] += 1
    c.side, c.servo = c.grasp['side'], None
    o = c.grasp['opt']
    c.change('go_grasp', f"{why}: {o['site']} width {o['width']:.3f} tilt {math.degrees(o['tilt']):.0f} sign {o['sign']:+.0f}, arm {c.side}, "
                         f"stand {np.round(c.grasp['stand'], 2).tolist()}")


def fail_grasp(c, w, why: str) -> None:
    c.bad_grasp.add(c.grasp['key'])
    c.counts['grasp_fail'] += 1
    c.grip[c.side] = K.GRIPPER_CLOSED
    c.servo = None
    c.perc.freeze['camera'] = False
    if c.grasp_cands:
        next_grasp(c, w, why + '; next pinch')
    else:
        c.change('plan_grasp', why + '; re-planning')


def go_grasp(c, action, q, w) -> None:
    g = c.grasp['stand']
    dist, yaw_err = c.navigate(action, q, w, g[:2], g[2])
    c.err = (dist, yaw_err)
    if c.arrived(dist, yaw_err):
        c.change('look_grasp', 'at the grasp stand; refitting the camera')
    elif c.counts['nav_stuck'] >= P.NAV_MAX_STUCK * (1 + c.counts['candidates_failed']) or c.age > P.NAV_TIMEOUT:
        c.counts['candidates_failed'] += 1
        fail_grasp(c, w, 'grasp stand not reached')


def look_grasp(c, action, q, w) -> None:
    """Stand still with the arms at rest, forget the camera estimate and refit it from this stand (consistent with the odometry pose the
    arm will use), then re-choose the pinch from this stand on the new estimate."""
    worst = c.to_rest(action, q)
    if c.look_t0 is None:
        if worst < P.REST_TOL or c.age > P.REST_TIMEOUT:
            c.perc.refresh('camera')
            c.look_t0 = c.age
        return
    if c.age - c.look_t0 < P.LOOK_STEPS or (w.camera is None and c.age - c.look_t0 < 3 * P.LOOK_STEPS):
        return
    c.look_t0 = None
    if w.camera is None:
        fail_grasp(c, w, 'camera not seen from the stand')
        return
    here = np.array([w.base]), np.zeros(1)
    cands = [x for x in candidates(c, w, q, *here) if approach_clear(w, x['opt'])]
    if not cands:
        c.change('plan_grasp', 'no pinch reachable from this stand on the refitted camera; re-planning')
        return
    cands.sort(key=lambda x: x['cost'])
    c.grasp = cands[0]
    c.side, c.servo = c.grasp['side'], None
    c.perc.freeze['camera'] = True
    c.change('pre_grasp', f"pinch {c.grasp['opt']['site']} arm {c.side} on the refitted camera")


def pre_grasp(c, action, q, w) -> None:
    if c.servo is None:
        c.servo = ik.make_servo(c.side, 'mid')
    c.open_grip(c.side, P.GRASP_SPREAD_M)
    c.other_rest(action, q)
    o = c.grasp['opt']
    goal, rot = c.tool_goal(o['pre'], o['rotation'], w)
    if c.age == 1:
        x = c.servo.solve(q, goal, rot)
        if x is None:
            c.counts['ik_none'] += 1
            fail_grasp(c, w, 'no solve_near solution for the pre-grasp point')
            return
        if P.ARM_PATH_CHECK and not arm_path_clear(c, w, q, x):
            c.counts['approach_rejects'] += 1
            fail_grasp(c, w, 'the arm path to the pre-grasp point crosses the tripod')
            return
        c.servo.target = x
        c.servo.sync = True
    c.err = c.servo.step(action, q, goal, rot)
    c.reason = f'reaching the pre-grasp point, err {c.err[0]:.3f} m {c.err[1]:.2f} rad'
    if c.age == 1 or c.err[0] < c.best_err - P.STALL_GAIN_M:
        c.best_err, c.best_age = c.err[0], c.age
    if c.err[0] < 0.015 and c.err[1] < 0.15:
        c.change('descend', 'fingers above the pinch')
    elif P.STALL_STEPS and c.age - c.best_age > P.STALL_STEPS and c.servo.target is None:
        fail_grasp(c, w, f'pre-grasp reach stalled (err {c.err[0]:.3f} m for {P.STALL_STEPS} controls)')
    elif c.age > P.REACH_TIMEOUT:
        fail_grasp(c, w, f'pre-grasp point not reached (err {c.err[0]:.3f})')


def descend(c, action, q, w) -> None:
    o = c.grasp['opt']
    c.open_grip(c.side, P.GRASP_SPREAD_M)
    c.other_rest(action, q)
    lam = min(G.GRASP_BACK_M, c.age * P.APPROACH_SPEED)
    goal, rot = c.tool_goal(o['pre'] + o['axis'] * lam, o['rotation'], w)
    if P.DESCEND_SOLVE and c.age % P.DESCEND_SOLVE_EVERY == 1:
        # joint target of a point DESCEND_LEAD ahead on the line (solve_near from the current joints): the resolved-rate servo alone
        # lags by centimetres near the joint limits
        ahead = min(G.GRASP_BACK_M, lam + P.DESCEND_LEAD_M)
        g2, r2 = c.tool_goal(o['pre'] + o['axis'] * ahead, o['rotation'], w)
        x = c.servo.solve(q, g2, r2)
        if x is not None:
            c.servo.target, c.servo.sync = x, True
    c.err = c.servo.step(action, q, goal, rot, plan=False)
    c.reason = f'moving in along the tool axis, {lam:.3f} of {G.GRASP_BACK_M:.2f} m, err {c.err[0]:.3f}'
    if lam >= G.GRASP_BACK_M and (c.err[0] < 0.010 or c.age > G.GRASP_BACK_M / P.APPROACH_SPEED + 60):
        c.gap_hist = []
        c.change('close', f'fingers at the pinch, err {c.err[0]:.3f} m')


def close(c, action, q, w) -> None:
    o = c.grasp['opt']
    c.grip[c.side] = K.GRIPPER_CLOSED
    c.other_rest(action, q)
    goal, rot = c.tool_goal(o['mid'], o['rotation'], w)
    c.err = c.servo.step(action, q, goal, rot, plan=False)
    gap = c.finger_gap(q, c.side)
    c.gap_hist = (getattr(c, 'gap_hist', []) + [gap])[-8:]
    still = len(c.gap_hist) >= 8 and max(c.gap_hist) - min(c.gap_hist) < 0.001
    c.reason = f'closing, finger gap {gap:.4f} m'
    if still and gap > P.HELD_GAP_M:
        c.held_age += 1
    if c.held_age >= P.HOLD_SETTLE:
        c.held_age = 0
        c.perc.grasped(q, c.side, w.camera.T)
        c.perc.freeze['camera'] = False
        c.lift0, c.rot_hold = c.mid_world(q, c.side)
        c.change('lift', f'fingers stopped {gap:.3f} m apart: holding the camera after {c.age} controls')
    elif (still and gap <= P.HELD_GAP_M) or c.age > P.GRIP_WAIT + 30:
        c.change('back_off', f'fingers closed on nothing (gap {gap:.4f} m)')


def back_off(c, action, q, w) -> None:
    c.grip[c.side] = K.GRIPPER_OPEN
    o = c.grasp['opt']
    goal, rot = c.tool_goal(o['pre'], o['rotation'], w)
    c.err = c.servo.step(action, q, goal, rot, plan=False)
    c.reason = f'backing off, err {c.err[0]:.3f}'
    if (c.err[0] < 0.03 and c.age > 10) or c.age > 150:
        fail_grasp(c, w, 'pinch failed')


def drop(c, w, why: str) -> None:
    c.counts['drops'] += 1
    c.perc.released()
    c.grip[c.side] = K.GRIPPER_CLOSED
    c.servo = None
    c.turned, c.scan_from = 0., None
    c.change('scan', why + '; looking for the camera again')


def lift(c, action, q, w) -> None:
    if not c.holds(q, c.side):
        drop(c, w, 'fingers closed: camera lost while lifting')
        return
    c.other_rest(action, q)
    lam = min(P.LIFT_M, c.age * P.LIFT_SPEED)
    goal, rot = c.tool_goal(c.lift0 + G.UP * lam, c.rot_hold, w)
    c.err = c.servo.step(action, q, goal, rot, plan=False)
    c.reason = f'lifting {lam:.3f} of {P.LIFT_M:.2f} m, err {c.err[0]:.3f}'
    if lam >= P.LIFT_M and (c.err[0] < 0.015 or c.age > P.LIFT_M / P.LIFT_SPEED + 80):
        c.servo = ik.make_servo(c.side, 'mid')
        near = w.tripod is not None and float(np.linalg.norm(w.tripod.xy - w.base[:2])) < P.BACKAWAY_TRIGGER_M
        if P.BACKAWAY_M > 0 and near:
            c.hold_q = q[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[c.side])].copy()
            c.change('back_away', 'camera lifted next to the tripod; backing away before the tuck')
        else:
            c.change('tuck', 'camera lifted')


def back_away(c, action, q, w) -> None:
    """Drive the base straight away from the tripod (arm held as lifted) so that the tuck swing cannot reach it."""
    if not c.holds(q, c.side):
        drop(c, w, 'camera lost while backing away')
        return
    c.hold_arm(action)
    c.other_rest(action, q)
    if c.age == 1:
        d = w.base[:2] - w.tripod.xy
        c.away_goal = w.base[:2] + d / max(float(np.linalg.norm(d)), 1e-6) * P.BACKAWAY_M
        c.away_yaw = float(w.base[2])
    action[:3] = nav_grid.drive_command(w.base, c.away_goal, c.away_yaw, P.NAV_GAIN, P.NAV_VNEAR, P.NAV_YAW_GAIN, P.NAV_WMAX)
    dist = float(np.linalg.norm(c.away_goal - w.base[:2]))
    c.reason = f'backing away from the tripod, {dist:.2f} m left'
    if dist < 0.04 or c.age > P.BACKAWAY_TIMEOUT:
        c.change('tuck', f'backed away ({dist:.2f} m left)')


PHASES = dict(plan_grasp=plan_grasp, go_grasp=go_grasp, look_grasp=look_grasp, pre_grasp=pre_grasp, descend=descend, close=close, back_away=back_away,
              back_off=back_off, lift=lift)
