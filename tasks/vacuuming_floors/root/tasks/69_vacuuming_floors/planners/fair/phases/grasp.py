"""Grasp phases of the fair planner of task 69: choose the pinch of the handle loop from its -x side and a stand that reaches it (GPU
reachable_stands over rings around the handle, the free hand's press checked in the same plan), drive there (go_grasp, then reach.py: refit, pre-grasp,
approach, close), retract after a failed pinch, lift the vacuum about a centimetre so that its projection cube spans the floor top (lift)."""
from __future__ import annotations

import math

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import params as P
from ..world import VacuumEst
from .base import Walker
from .pinch import candidates, frame_key


def plan_grasp(c, action, q, w) -> None:
    c.to_rest(action, q)
    if w.vacuum is None:
        c.turned, c.scan_from = 0., None
        c.change('scan', 'vacuum estimate lost; turning again')
        return
    if c.counts['grasp_tries'] >= P.MAX_GRASP_TRIES:
        c.change('failed', 'grasp tries exhausted')
        return
    c.counts['plans'] += 1
    c.wp_looked = False
    c.perc.yaw_lock = True            # the far views fit the yaw to a few degrees, the near partial views worse: keep it from here on (a free fit at the waypoint turned it 50 degrees on 307)
    c.build_grid(w, [w.vacuum.xy], keep_vacuum=True)
    c.grasp_cands = candidates(c, w, q)
    next_grasp(c, w, f'handle pinch ({len(c.grasp_cands)} candidates)')


def next_grasp(c, w, why: str) -> None:
    c.grasp = c.pick(w, c.grasp_cands) if c.grasp_cands else None
    if c.grasp is None:
        if c.cand_relax < P.MAX_RELAX:      # no candidate: plan again with the bans and the arm path check lifted
            c.cand_relax += 1
            c.change('plan_grasp', why + f'; no candidate, filters relaxed to level {c.cand_relax}')
            return
        c.change('failed', why + f'; no reachable handle pinch with a route {c.pick_stats}')
        return
    c.counts['grasp_tries'] += 1
    c.looks, c.leg, c.plan_vac = 0, None, None
    c.wp_looked = getattr(c, 'wp_looked', False)
    c.side = c.grasp['side']
    g, f = c.grasp, c.grasp['frame']
    c.change('go_grasp', f"{why}: arm {g['side']}, pitch {math.degrees(f['pitch']):.0f}, sign {f['sign']:+.0f}, stand {np.round(g['stand'], 2).tolist()}, "
                         f"cost {g['total']:.2f}, press face {g['press_face']['name']}")


def regrasp(c, q, w, why: str, press: bool = True) -> None:
    """Put the held vacuum down and grasp it again.

    Put the held vacuum down where it is (the grip opens), ban this arm at this stand, plan the pinch again."""
    held = c.perc.held_T(q)
    n = 'regrasps' if press else 'tilt_regrasps'
    if c.counts[n] >= (P.MAX_REGRASPS if press else P.MAX_TILT_REGRASPS) or c.grasp is None or held is None:
        if press:
            c.change('failed', why)
        return
    c.counts[n] += 1
    c.counts['grasp_tries'], c.cand_relax = 0, 0
    if press:
        c.banned.append(dict(side=c.side, stand=c.grasp['stand'][:2].copy(), key=frame_key(c.grasp['frame'])))
    c.perc.released()
    c.perc.was_on = not press and bool(w.toggled)
    c.perc.vacuum = VacuumEst(T=held, cost=0., inlier=1., n_cloud=0, seen_step=c.step)
    c.perc.hint['vacuum'] = None
    c.grasp_cands, c.press_log, c.press_faces, c.searched = [], [], [], False
    c.perc.yaw_lock = True
    fail_grasp(c, w, why + '; the vacuum is put down')


def fail_grasp(c, w, why: str) -> None:
    """The pinch failed: the arm walks back to the travel posture.

    The pinch failed: the arm walks back to the travel posture, the vacuum is looked at again (it may have moved), a new plan follows."""
    c.counts['grasp_fail'] += 1
    if c.grasp is not None:        # the same stand, arm and pinch frame failing again and again (no IK, a pinch on nothing): ban it, the others get their turn
        key = (c.side, tuple(np.round(c.grasp['stand'][:2] / 0.2).astype(int)), frame_key(c.grasp['frame']))
        c.fail_count[key] = c.fail_count.get(key, 0) + 1
        if c.fail_count[key] >= P.BAN_AFTER:
            c.banned.append(dict(side=c.side, stand=c.grasp['stand'][:2].copy(), key=key[2]))
    c.grip = {'left': K.GRIPPER_OPEN, 'right': K.GRIPPER_OPEN}
    c.walker = c.line = None
    c.perc.freeze['vacuum'] = False       # the estimate stays (a near view cannot find the vacuum anew), the refits go on
    if c.arm_cmd is not None:
        c.walker = Walker(c.arm_cmd, c.rest_pose(c.side), P.PRE_STEP, 0)
    c.retract_why = why
    c.change('retract', why + '; retracting the arm')


def retract(c, action, q, w) -> None:
    if c.walker is not None:
        c.arm_cmd, done = c.walker.next()
        c.reason = f'retracting the arm: {c.retract_why}'
        if not done and c.age < 400:
            return
    c.walker = None
    c.release_arms()
    c.look_t0 = None
    c.change('look_grasp', c.retract_why + '; looking at the vacuum again')


def look_wp(c, action, q, w) -> None:
    """Stand at the waypoint, refit the vacuum, plan again.

    Standing at the waypoint with the arms at rest: refit the vacuum (the marker blob corrects it when seen), plan the pinch again on that
    estimate and drive on to the stand it chooses; the look at that stand then only checks the estimate has not moved."""
    worst = c.to_rest(action, q)
    if c.look_t0 is None:
        if worst < P.REST_TOL or c.age > P.REST_TIMEOUT:
            c.look_t0 = c.age
        return
    waited = c.age - c.look_t0
    if waited < P.LOOK_STEPS and w.vacuum is not None:
        return
    if w.vacuum is None:
        c.change('plan_grasp', 'the vacuum is not seen from the waypoint; re-planning')
        return
    c.build_grid(w, [w.vacuum.xy], keep_vacuum=True)
    cands = candidates(c, w, q)
    if not cands:
        c.leg = 'final'
        c.change('go_grasp', 'no pinch from the waypoint; on to the stand chosen before')
        return
    c.grasp_cands = cands
    plan_vac, tries = w.vacuum.T.copy(), c.counts['grasp_tries']
    next_grasp(c, w, f'handle pinch from the waypoint ({len(cands)} candidates)')
    c.wp_looked, c.plan_vac = True, plan_vac
    c.counts['grasp_tries'] = tries


def go_grasp(c, action, q, w) -> None:
    """Drive to the grasp stand: along the grid route to a...

    Drive to the grasp stand: along the grid route to a waypoint 0.35 m farther out on the line from the vacuum through the stand, then
    straight in (the route planner may cut a corner of the vacuum's footprint; the straight leg is radial, away from it)."""
    c.to_rest(action, q)
    g = c.grasp['stand']
    if c.leg is None:
        c.leg = 'final'
        if w.vacuum is not None and float(np.linalg.norm(w.base[:2] - g[:2])) > P.WAYPOINT_M + 0.1:
            wp = c.waypoint_for(w, g[:2])        # the radial waypoint first, then turned about the stand (a wall behind it)
            if wp is not None:
                c.wp, c.leg = wp, 'wp'
    if c.leg == 'wp':
        dist, yaw_err = c.navigate(action, w, c.wp, g[2], far=True)
        c.reason = f'driving to the waypoint of the grasp stand, {dist:.2f} m'
        if dist < 0.10 or c.counts['nav_stuck'] >= P.NAV_MAX_STUCK or c.age > P.NAV_TIMEOUT // 2:
            c.leg, c.path, c.stuck_ref, c.counts['nav_stuck'] = 'final', None, None, 0
            if P.WAYPOINT_LOOK and not c.wp_looked:
                c.wp_looked, c.look_t0 = True, None
                c.change('look_wp', 'at the waypoint: looking at the vacuum from close by before the final stand is chosen')
        return
    dist, yaw_err = float(np.linalg.norm(g[:2] - w.base[:2])), G.wrap(g[2] - w.base[2])
    if dist > 0.8 and w.vacuum is not None:      # no waypoint was free: a straight line from here would cut through the standing vacuum
        dist, yaw_err = c.navigate(action, w, g[:2], g[2], far=True)
    else:
        action[:3] = nav_grid.drive_command(w.base, g[:2], g[2], P.NAV_GAIN, P.NAV_VNEAR if dist < 0.5 else 0.25, P.NAV_YAW_GAIN, P.NAV_WMAX)
    if c.stuck_ref is None or np.linalg.norm(c.stuck_ref[0] - w.base[:2]) > 0.03 or abs(G.wrap(c.stuck_ref[2] - w.base[2])) > 0.1:
        c.stuck_ref = (w.base[:2].copy(), c.age, float(w.base[2]))
    c.err = (dist, yaw_err)
    c.reason = f'driving in to the grasp stand, {dist:.2f} m, yaw err {yaw_err:.2f}'
    if c.arrived(dist, yaw_err):
        c.look_t0 = None
        c.change('look_grasp', f'at the grasp stand ({dist:.3f} m, yaw err {yaw_err:+.3f}); refitting the vacuum')
    elif c.age > P.NAV_TIMEOUT:
        if c.grasp_cands:
            next_grasp(c, w, f'grasp stand not reached ({dist:.2f} m); next candidate')
        else:
            c.change('plan_grasp', f'grasp stand not reached ({dist:.2f} m); re-planning')


















def floor_w(w) -> float:
    """Height (cube coordinates, the cube's thickness as unit)...

    Height (cube coordinates, the cube's thickness as unit) of the floor point under the cube centre: 0 means the floor passes through the
    cube's mid plane, |w| < 0.5 means a flake on the floor is inside the cube."""
    centre, rot, scale = w.cube()
    p = np.array([centre[0], centre[1], w.floor_z])
    return float((rot.T @ (p - centre) / scale)[2])


def lift(c, action, q, w) -> None:
    if not c.holds(q, c.side):
        c.counts['drops'] += 1
        c.perc.released()
        c.grasp_cands = []
        fail_grasp(c, w, 'hold lost while lifting')
        return
    if c.line is None:
        if c.lift_round == 0:
            c.cand_relax = 0
        wv = floor_w(w)
        c.lift_dz = float(np.clip((wv - P.CUBE_W_TARGET) * float(w.cube()[2][2]), -0.03, 0.05))
        mid = c.hand(q)
        c.start_line(mid[:3, 3], mid[:3, 3] + G.UP * c.lift_dz, mid[:3, :3], step=P.LIFT_STEP_M)
        c.lift_round += 1
    r = c.follow_line(w)
    c.reason = f'levelling the cube: round {c.lift_round}, dz {c.lift_dz:+.4f} m, floor w {floor_w(w):+.2f}, marker {w.toggled}'
    if r is False and abs(floor_w(w) - P.CUBE_W_TARGET) > P.LIFT_GIVE_UP_W:
        c.line = None
        regrasp(c, q, w, 'no IK to lift the held vacuum')
    elif r is not None:
        c.line = None
        wv = floor_w(w)
        if abs(wv - P.CUBE_W_TARGET) > P.CUBE_W_TOL and c.lift_round < P.LIFT_ROUNDS:
            return
        c.hold_arm = c.arm_cmd.copy()
        c.raise_pose(w)
        if w.toggled:
            c.change('plan_sweep', f'cube levelled (floor w {wv:+.2f}), the marker is green')
        elif not c.press_log or c.counts['press_tries'] < 2:
            c.change('press_plan', f'cube levelled (floor w {wv:+.2f}), marker {w.toggled}: pressing it with the free hand')
        else:
            regrasp(c, q, w, 'the vacuum stays off after the presses')


PHASES = dict(plan_grasp=plan_grasp, go_grasp=go_grasp, look_wp=look_wp, retract=retract, lift=lift)
