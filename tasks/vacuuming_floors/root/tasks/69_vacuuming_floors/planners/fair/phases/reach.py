"""Reach phases of the fair planner of task 69: refit the vacuum at the stand and plan the pinch again from there (look_grasp), walk the arm to the pre-grasp
point (pre_grasp), in along the handle (approach), close the fingers (close, the native assisted grasp holds the vacuum, read from the finger gap) and back
off after a pinch on nothing (back_off). Subgoal grasped."""
from __future__ import annotations

import math

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import ik
from .. import params as P
from .base import Walker
from .grasp import fail_grasp
from .pinch import candidates, frames_of

def look_grasp(c, action, q, w) -> None:
    """Stand still with the arms at rest and refit the vacuum from this stand.

    Stand still with the arms at rest and refit the vacuum from this stand (consistent with the odometry pose the arm will use), then
    choose the pinch again from this stand on the new estimate."""
    worst = c.to_rest(action, q)
    if c.look_t0 is None:
        if worst < P.REST_TOL or c.age > P.REST_TIMEOUT:
            c.look_t0 = c.age
        return
    waited = c.age - c.look_t0
    if waited < P.LOOK_STEPS or (w.vacuum is None and waited < 3 * P.LOOK_STEPS):
        return
    c.look_t0 = None
    if w.vacuum is None:
        c.turned, c.scan_from = 0., None
        c.change('scan', 'vacuum not seen from the stand; turning')
        return
    c.perc.yaw_lock = True            # the yaw fitted before the first stand look is held from here on
    if c.plan_vac is not None:                 # arrived at the stand the last look chose: the estimate must not have moved since
        dv = float(np.linalg.norm(w.vacuum.T[:2, 3] - c.plan_vac[:2, 3]))
        dy = abs(G.wrap(G.yaw_of(w.vacuum.T) - G.yaw_of(c.plan_vac)))
        c.look_log.append(dict(step=c.step, again=True, shift=round(dv, 4), yaw_shift=round(dy, 3)))
        if dv < P.LOOK_SHIFT_M and dy < P.LOOK_SHIFT_RAD:
            c.plan_vac = None
            c.perc.freeze['vacuum'] = True
            begin_pre(c, q)
            return
        c.plan_vac = None
    if c.inflated is None:
        c.build_grid(w, [w.vacuum.xy], keep_vacuum=True)
    near = np.array([[w.base[0] + dx, w.base[1] + dy, G.wrap(w.base[2] + dyaw)] for dx, dy, dyaw in P.LOOK_OFFSETS])
    near = near[[c.free_pose(st[:2]) for st in near]]
    cands = candidates(c, w, q, stands=near)
    c.look_log.append(dict(step=c.step, stands=len(near), cands=len(cands), press=getattr(c, 'press_check_log', None),
                           approach=getattr(c, 'approach_check_log', None), vac=np.round(w.vacuum.T[:2, 3], 3).tolist(),
                           fix=c.perc.marker.get('fix_stand')))
    if not cands:
        c.change('plan_grasp', 'no pinch reachable from this stand or near it on the refitted vacuum; re-planning')
        return
    for x in cands:
        x['ecost'] = float(np.linalg.norm(x['stand'][:2] - w.base[:2])) + 0.5 * abs(G.wrap(x['stand'][2] - w.base[2]))
        x['cost'] = x['ecost'] + 0.5 * x['frame']['pitch'] + 0.2 * x['press_dtrunk'] + P.TILT_COST * (abs(x['frame']['roll']) + abs(x['frame']['yaw']))
    cands.sort(key=lambda x: x['cost'])
    c.grasp = cands[0]
    c.grasp['total'] = c.grasp['cost']
    c.side = c.grasp['side']
    c.perc.freeze['vacuum'] = True
    move = float(np.linalg.norm(c.grasp['stand'][:2] - w.base[:2])) + abs(G.wrap(c.grasp['stand'][2] - w.base[2]))
    c.looks += 1
    if move > 0.012 and c.looks < 4:
        c.leg, c.plan_vac = 'final', w.vacuum.T.copy()
        c.perc.freeze['vacuum'] = False
        c.change('go_grasp', f"small move to {np.round(c.grasp['stand'], 2).tolist()} ({move:.2f}) so that the refitted vacuum is pinchable")
        return
    begin_pre(c, q)

def begin_pre(c, q) -> None:
    c.take_arm(q, c.side)
    c.start_q = q.copy()
    c.change('pre_grasp', f"pinch pitch {math.degrees(c.grasp['frame']['pitch']):.0f} arm {c.side} on the refitted vacuum, press face {c.grasp['press_face']['name']}")

def live_frame(c, w):
    f0 = c.grasp['frame']
    for f in frames_of(w.vacuum.T):
        if f['sign'] == f0['sign'] and abs(f['pitch'] - f0['pitch']) < 1e-9 and abs(f['roll'] - f0['roll']) < 1e-9 and abs(f['yaw'] - f0['yaw']) < 1e-9:
            return f
    return f0

def pre_grasp(c, action, q, w) -> None:
    if c.walker is None:
        f = live_frame(c, w)
        c.grasp['live'] = f
        x = c.solve_hand(w, G.pose_T(f['rot'], f['pre']), start=c.grasp['q_pre'])    # the scan's branch, whose trunk the press check used
        if x is None:
            fail_grasp(c, w, 'no IK for the pre-grasp at the reached stand')
            return
        c.walker = Walker(c.arm_cmd, x, P.PRE_STEP)
        c.wait_fix, c.fix0 = 0, c.perc.counts.get('marker_fix_stand', 0)
    c.arm_cmd, done = c.walker.next()
    c.reason = f'to the pre-grasp, joint gap {c.walker.gap:.3f}'
    c.grip[c.side] = K.GRIPPER_OPEN
    c.grip[c.free_side] = K.GRIPPER_CLOSED
    if done and c.age < P.PRE_TIMEOUT and not c.fixed_here(w):
        if c.wait_fix == 0:
            c.wait_fix = c.age
        if c.age - c.wait_fix < P.MARKER_WAIT:
            return                      # a few more controls: the marker blob in the wrist camera corrects the vacuum pose
    if done:
        f = live_frame(c, w)
        c.grasp['live'] = f
        c.walker = None
        c.approach_walk = c.grasp.get('mode') != 'line' and joint_approach(c, w, f)
        if not c.approach_walk:
            a = ik.hand_fk(c.side, c.arm_cmd[:4], c.arm_cmd[4:], w.base)[:3, 3]
            c.start_line(a, f['point'], f['rot'], n=len(c.grasp['chain']) if c.grasp.get('chain') else None)
            c.line_seeds = c.grasp.get('chain')
        c.change('approach', 'at the pre-grasp' + (' (joint-space approach)' if c.approach_walk else ''))
    elif c.age > P.PRE_TIMEOUT:
        fail_grasp(c, w, 'pre-grasp walk timed out')

def joint_approach(c, w, f) -> bool:
    """One solve_near at the grasp point seeded with the pre-grasp command.

    One solve_near at the grasp point seeded with the pre-grasp command; accepted when the joint-space path keeps the finger midpoint
    within JOINT_PATH_TOL of the straight approach line (forward kinematics, 8 samples): then a single joint walk."""
    x = c.solve_hand(w, G.pose_T(f['rot'], f['point']), start=c.grasp.get('xp'))     # from the planned joints: the trunk stays where the press was checked
    c.approach_diag = dict(step=c.step, solved=x is not None)
    if x is None or float(np.max(np.abs(x - c.arm_cmd))) > P.JOINT_PATH_MAX:
        c.approach_diag['max_dq'] = None if x is None else round(float(np.max(np.abs(x - c.arm_cmd))), 3)
        return False
    a, b = ik.hand_fk(c.side, c.arm_cmd[:4], c.arm_cmd[4:], w.base)[:3, 3], f['point']
    ab = b - a
    for lam in np.linspace(0.0, 1.0, 9)[1:-1]:
        v = c.arm_cmd + (x - c.arm_cmd) * lam
        p = ik.hand_fk(c.side, v[:4], v[4:], w.base)[:3, 3]
        t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0., 1.))
        dev = float(np.linalg.norm(p - (a + t * ab)))
        c.approach_diag['dev'] = max(dev, c.approach_diag.get('dev', 0.))
        if dev > P.JOINT_PATH_TOL:
            return False
    c.walker = Walker(c.arm_cmd, x, P.APPROACH_STEP, P.WALK_SETTLE)
    return True

def approach(c, action, q, w) -> None:
    c.grip[c.side] = K.GRIPPER_OPEN
    if c.approach_walk:
        c.arm_cmd, done = c.walker.next()
        c.reason = f'joint-space approach, gap {c.walker.gap:.3f}'
        r = True if done else None
    else:
        r = c.follow_line(w)
        c.reason = f'approach {c.line_i}/{len(c.line)}'
    if r is False:
        fail_grasp(c, w, f'no IK on the approach line ({c.line_i}/{len(c.line)})')
    elif r:
        c.line = c.walker = None
        c.held_age = 0
        c.vac_at_close = w.vacuum.T.copy()
        c.change('close', 'fingers at the handle pinch')
    elif c.age > P.APPROACH_TIMEOUT:
        fail_grasp(c, w, 'approach timed out')

def close(c, action, q, w) -> None:
    c.grip[c.side] = K.GRIPPER_CLOSED
    gap = c.finger_gap(q, c.side)
    c.gap_hist = (getattr(c, 'gap_hist', []) + [gap])[-6:]
    still = len(c.gap_hist) >= 6 and max(c.gap_hist) - min(c.gap_hist) < 0.0008
    c.held_age = c.held_age + 1 if (still and gap > P.HELD_GAP_M) else 0
    c.reason = f'closing, finger gap {gap:.4f} m, held {c.held_age}, marker {w.toggled}'
    if c.held_age >= P.HOLD_SETTLE:
        c.grasp_step = c.grasp_step or c.step
        T = w.vacuum.T if w.vacuum is not None else c.vac_at_close
        c.perc.grasped(q, c.side, T)
        c.perc.freeze['vacuum'] = False
        c.lift_round = 0
        c.line = None
        c.change('lift', f'fingers stopped {gap:.3f} m apart: holding the vacuum after {c.age} controls')
    elif (still and gap <= P.HELD_GAP_M and c.age > 20) or c.age > P.CLOSE_MAX:
        c.change('back_off', f'fingers closed on nothing (gap {gap:.4f} m)')

def back_off(c, action, q, w) -> None:
    c.grip[c.side] = K.GRIPPER_OPEN
    if c.age == 1:
        f = c.grasp.get('live') or c.grasp['frame']
        mid = c.hand(q)
        c.start_line(mid[:3, 3], mid[:3, 3] - f['approach'] * 0.10, mid[:3, :3])
    r = c.follow_line(w)
    if r is not None or c.age > 200:
        fail_grasp(c, w, 'pinch failed, backed off')


PHASES = dict(look_grasp=look_grasp, pre_grasp=pre_grasp, approach=approach, close=close, back_off=back_off)
