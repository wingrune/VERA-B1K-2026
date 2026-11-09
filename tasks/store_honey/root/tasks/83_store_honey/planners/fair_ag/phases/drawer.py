"""Phases of the top drawer: look at the whole cabinet top from a stop in front of it (go_view, drawer_view), choose a base stand in front of a pull ledge (plan_drawer), drive there (go_drawer), bring the open fingers in front of
the ledge and onto it (drawer_pre, drawer_reach), pinch it (drawer_grip), pull it out by driving the base back with the arm held (drawer_pull),
let go (drawer_release) and back off (drawer_retreat).

The stand is kept relative to the cabinet estimate (ledge k, distance along f, offset towards the centre), so a better view of the cabinet while
driving moves the goal with it; the estimate is frozen from drawer_pre on. Success of the pull: the base, which carries the pinching hand, moved
DRAWER_OPEN_TARGET_M along f with the fingers still closed on the ledge (the drawer itself is not observed). Stand reachability is one
batched GPU IK scan per hand (reachable_stands); the arm servo is solve_near seeded with the current joints."""
from __future__ import annotations

import math

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import params as P

SIDE_OF_LEDGE = {1: 'right', -1: 'left'}      # +lateral of the cabinet is on the right of a base that faces its front


def stand_pose(cab, k: int, d: float, lat: float):
    ledge = cab.ledge(k)
    return ledge[:2] + cab.f * d - cab.lateral * k * lat, cab.yaw_facing


def drawer_rotation(cab) -> np.ndarray:
    """Fingers into the cabinet (along -f), jaw vertical: the fingers close from above and below on the horizontal ledge."""
    return G.jaw_rotation(np.r_[-cab.f, 0.], G.UP)


def view_pose(cab, d: float):
    return cab.centre + cab.f * d, cab.yaw_facing


def plan_view(c, w, cab) -> bool:
    """A free stop with a route in front of the cabinet centre, to look at its whole top before choosing the drawer stand."""
    pg = c.pose_grid(w, cab.centre)
    for d in P.VIEW_DISTANCES_M:
        xy, yaw = view_pose(cab, d)
        if pg.free(xy, yaw) and pg.plan(w.base[:2], w.base[2], xy, yaw) is not None:
            c.goal = dict(view_d=d)
            c.change('go_view', f'view stop {np.round(xy, 2).tolist()} at {d:.1f} m in front of the cabinet')
            return True
    return False


def step_go_view(c, action, q, w) -> None:
    xy, yaw = view_pose(w.cabinet, c.goal['view_d'])
    if 'goal_xy' in c.mark and np.linalg.norm(xy - c.mark['goal_xy']) > 0.05:
        c.nav.reset()
    c.mark['goal_xy'] = xy
    dist, yaw_err = c.navigate(action, q, w, xy, yaw)
    if c.nav.arrived(dist, yaw_err):
        c.change('drawer_view', 'at the view stop')
    elif c.nav.no_route > P.NO_ROUTE_MAX:
        view_done(c, 'no route to the view stop')


def step_drawer_view(c, action, q, w) -> None:
    c.to_rest(action, q)
    if c.age == 1:
        c.mark['seen0'] = w.cabinet.seen_step
    if c.age >= P.VIEW_STEPS:
        view_done(c, f"looked at the cabinet {P.VIEW_STEPS} controls, estimate updated {w.cabinet.seen_step > c.mark.get('seen0', 0)}")


def view_done(c, why: str) -> None:
    c.cab_viewed = True
    c.change('plan_drawer', why)


def step_plan_drawer(c, action, q, w) -> None:
    c.to_rest(action, q)
    cab = w.cabinet
    if cab is None:
        c.change('scan', 'no cabinet to plan on')
        return
    if not c.cab_viewed:
        if plan_view(c, w, cab):
            return
        c.cab_viewed = True
    cands = []
    for k in (1, -1):
        for d in P.DRAWER_STANDS:
            for lat in P.DRAWER_LATERALS:
                xy, yaw = stand_pose(cab, k, d, lat)
                cost = float(np.linalg.norm(xy - w.base[:2])) + 0.5 * abs(lat - P.DRAWER_LATERALS[0]) + 0.5 * abs(d - P.DRAWER_STANDS[0]) + (0.3 if k < 0 else 0.)
                cands.append(dict(k=k, d=d, lat=lat, side=SIDE_OF_LEDGE[k], cost=cost))
    bad = getattr(c, 'bad_stands', [])
    if bad:
        fresh = [v for v in cands if not any(v['k'] == b[0] and abs(v['d'] - b[1]) < 0.01 and abs(v['lat'] - b[2]) < 0.01 for b in bad)]
        if fresh:
            cands = fresh
    cands.sort(key=lambda v: v['cost'])
    pg = c.pose_grid(w, cab.centre)
    rot = drawer_rotation(cab)
    rows = []
    for shrink in (0.0, 0.03, 0.06):            # no free stand: retry with a narrower footprint
        g0 = pg if shrink == 0 else c.pose_grid(w, cab.centre, shrink)
        for cand in cands:
            xy, yaw = stand_pose(cab, cand['k'], cand['d'], cand['lat'])
            if g0.free(xy, yaw):
                rows.append(dict(cand, base=np.array([xy[0], xy[1], yaw])))
        if rows:
            break
    relaxed = False
    if not rows:                                # still none: the grid may be polluted; judge by IK and route only
        relaxed = True
        for cand in cands:
            xy, yaw = stand_pose(cab, cand['k'], cand['d'], cand['lat'])
            rows.append(dict(cand, base=np.array([xy[0], xy[1], yaw])))
    ok, diag = [], dict(free=len(rows), ik=0, relaxed=relaxed)
    for side in ('right', 'left'):              # one batched GPU scan per hand: pre point and ledge of every free stand of that hand
        sub = [r for r in rows if r['side'] == side]
        if not sub:
            continue
        ledges = np.array([cab.ledge(r['k']) for r in sub])
        mask, _ = c.scan(side, np.array([r['base'] for r in sub]), [(ledges + np.r_[cab.f * P.PRE_GAP_M, 0.], rot), (ledges, rot)])
        for r, m in zip(sub, mask):
            if m:
                ok.append(r)
    diag['ik'] = len(ok)
    ok.sort(key=lambda v: v['cost'])
    routed = []
    for shrink in P.NAV_SHRINKS:                # stands with a route first (the narrower footprint only when the full one has none)
        g = pg if shrink == 0 else c.pose_grid(w, cab.centre, shrink)
        routed = [r for r in ok if g.plan(w.base[:2], w.base[2], r['base'][:2], r['base'][2]) is not None][:3]
        if routed:
            break
    diag['routed'] = len(routed)
    ok = routed or ok[:3]
    if not ok:
        c.fail(f'no free, reachable stand in front of the drawer ledges {diag}')
        return
    c.cands = ok
    c.goal = c.cands.pop(0)
    c.side = c.goal['side']
    xy, _ = stand_pose(cab, c.goal['k'], c.goal['d'], c.goal['lat'])
    c.change('go_drawer', f"stand {np.round(xy, 2).tolist()} hand {c.side} ledge {np.round(cab.ledge(c.goal['k']), 3).tolist()} {diag}")


def next_candidate(c, why: str) -> None:
    c.counts['candidates_failed'] += 1
    g = c.goal
    if isinstance(g, dict) and 'k' in g:
        bad = list(getattr(c, 'bad_stands', []))
        bad.append((g['k'], g['d'], g['lat']))
        c.bad_stands = bad
    c.perc.freeze_cabinet = False
    c.grip[c.side] = K.GRIPPER_OPEN
    if c.counts['drawer_tries'] >= P.MAX_DRAWER_TRIES:
        c.fail(why + '; out of drawer tries')
    elif c.cands:
        c.goal = c.cands.pop(0)
        c.side = c.goal['side']
        c.change('go_drawer', why + '; next stand')
    else:
        c.change('plan_drawer', why + '; re-planning')


def step_go_drawer(c, action, q, w) -> None:
    cab = w.cabinet
    xy, yaw = stand_pose(cab, c.goal['k'], c.goal['d'], c.goal['lat'])
    if 'goal_xy' in c.mark and np.linalg.norm(xy - c.mark['goal_xy']) > 0.05:
        c.nav.reset()                                   # the cabinet estimate moved the stand: plan anew
    c.mark['goal_xy'] = xy
    dist, yaw_err = c.navigate(action, q, w, xy, yaw)
    if c.nav.arrived(dist, yaw_err):
        c.grip[c.side] = K.GRIPPER_OPEN
        c.perc.freeze_cabinet = True
        c.change('drawer_pre', f'at the drawer stand, cabinet seen {w.step - cab.seen_step} controls ago')
    elif c.nav.no_route > P.NO_ROUTE_MAX:
        c.counts['nav_stuck'] += 1
        next_candidate(c, 'no route to the stand')


def nudge_in(c, action, w, cab, tip, target, perr: float) -> None:
    """Hand stuck beyond its target along f (base too far from the cabinet): creep the base towards the cabinet, at most 0.12 m."""
    if c.age < 150 or perr < 0.03:
        return
    base0 = c.mark.setdefault('nudge0', w.base[:2].copy())
    used = float(np.dot(w.base[:2] - base0, -cab.f))
    out = float((tip[:2] - target[:2]) @ cab.f)
    if out > 0.03 and used < 0.12:
        drive_along(action, w, -cab.f, 0.06)


def step_drawer_pre(c, action, q, w) -> None:
    cab = w.cabinet
    pre = c.goal_ledge(cab) + np.r_[cab.f * P.PRE_GAP_M, 0.]
    c.grip[c.side] = K.GRIPPER_OPEN
    perr, oerr = c.servo_to(action, q, w, pre, drawer_rotation(cab))
    tip = c.tip_point(q, w)
    c.reason = (f'to the pre point of the ledge, err {perr:.3f} m {oerr:.2f} rad, hand {float((tip[:2] - pre[:2]) @ cab.f):+.3f} m out of it, '
                f'dz {tip[2] - pre[2]:+.3f}')
    nudge_in(c, action, w, cab, tip, pre, perr)
    if perr < 0.015 and oerr < 0.15:
        c.change('drawer_reach', 'pre point reached')


def step_drawer_reach(c, action, q, w) -> None:
    cab = w.cabinet
    c.grip[c.side] = K.GRIPPER_OPEN
    perr, oerr = c.servo_to(action, q, w, c.goal_ledge(cab), drawer_rotation(cab), max_step=0.006)
    c.reason = f'fingers onto the ledge, err {perr:.3f} m {oerr:.2f} rad'
    if perr < 0.008 and oerr < 0.15:
        c.change('drawer_grip', 'ledge between the fingers')


def step_drawer_grip(c, action, q, w) -> None:
    cab = w.cabinet
    c.servo_to(action, q, w, c.goal_ledge(cab), drawer_rotation(cab), max_step=0.004)
    c.grip[c.side] = K.GRIPPER_CLOSED
    fing = c.finger_mean(q, c.side)
    c.reason = f'closing on the ledge, fingers {fing:.4f}'
    if c.age >= P.GRIP_WAIT:
        if P.PINCH_FINGER_BAND[0] <= fing <= P.PINCH_FINGER_BAND[1]:
            c.hold = c.joints(q)
            c.change('drawer_pull', f'ledge pinched, fingers {fing:.4f}')
            c.mark['base0'] = w.base[:2].copy()
        else:
            c.counts['drawer_tries'] += 1
            next_candidate(c, f'pinch missed, fingers {fing:.4f}')


def drive_along(action, w, direction: np.ndarray, speed: float) -> None:
    """Base command that moves along a unit xy direction of the odometry frame without turning."""
    c, s = math.cos(w.base[2]), math.sin(w.base[2])
    action[0] = (c * direction[0] + s * direction[1]) * speed
    action[1] = (-s * direction[0] + c * direction[1]) * speed
    action[2] = 0.


def step_drawer_pull(c, action, q, w) -> None:
    cab = w.cabinet
    c.hold_joints(action, c.hold)
    c.grip[c.side] = K.GRIPPER_CLOSED
    fing = c.finger_mean(q, c.side)
    moved = float(np.dot(w.base[:2] - c.mark['base0'], cab.f))
    prev = c.mark.get('prev', moved)
    c.mark['stall'] = c.mark.get('stall', 0) + 1 if abs(moved - prev) < 0.0008 else 0
    c.mark['prev'] = moved
    c.reason = f'pulling, base moved {moved:.3f} m along the front normal, fingers {fing:.4f}'
    if moved >= P.DRAWER_OPEN_TARGET_M:
        c.drawer_open_m = moved
        c.change('drawer_release', f'drawer out {moved:.3f} m (base travel)')
        return
    if fing < 0.0015 or fing > 0.02:
        c.counts['drawer_tries'] += 1
        next_candidate(c, f'the ledge slipped out of the fingers at {moved:.3f} m, {fing:.4f}')
        return
    if c.mark['stall'] > P.PULL_STALL_STEPS and c.age > 90:
        if moved > 0.25:
            c.drawer_open_m = moved
            c.change('drawer_release', f'the base stopped at {moved:.3f} m (drawer at its end)')
        else:
            c.counts['drawer_tries'] += 1
            next_candidate(c, f'the base does not move back while pulling ({moved:.3f} m)')
        return
    drive_along(action, w, cab.f, P.PULL_SPEED)


def step_drawer_release(c, action, q, w) -> None:
    c.hold_joints(action, c.hold)
    c.grip[c.side] = K.GRIPPER_OPEN
    if c.age >= 40:
        c.change('drawer_retreat', 'fingers open')


def step_drawer_retreat(c, action, q, w) -> None:
    """Hand straight back off the ledge, then the base backs away from the open drawer, then the arms go home."""
    cab = w.cabinet
    c.grip[c.side] = K.GRIPPER_OPEN
    stage = c.mark.setdefault('stage', 0)
    if stage == 0:
        tip = c.tip_point(q, w)
        tip0 = c.mark.setdefault('tip0', tip.copy())
        target = tip0 + np.r_[cab.f, 0.] * min(P.RETREAT_HAND_M + 0.05, 0.004 * max(c.age, 1))
        perr, _ = c.servo_to(action, q, w, target, drawer_rotation(cab), plan=False)
        along = float(np.dot((tip - tip0)[:2], cab.f))
        c.reason = f'backing the hand off the ledge, {along:.3f} m, err {perr:.3f}'
        if along > P.RETREAT_HAND_M or c.age >= 80:
            c.mark.update(stage=1, base0=w.base[:2].copy(), prev=w.base[:2].copy(), stall=0)
            c.hold = c.joints(q)
        return
    if stage == 1:
        c.hold_joints(action, c.hold)
        moved = float(np.dot(w.base[:2] - c.mark['base0'], cab.f))
        c.mark['stall'] = 0 if np.linalg.norm(w.base[:2] - c.mark['prev']) > 0.0015 else c.mark['stall'] + 1
        c.mark['prev'] = w.base[:2].copy()
        drive_along(action, w, cab.f, P.PULL_SPEED)
        c.reason = f'backing the base away from the open drawer, {moved:.3f} m'
        if moved >= P.RETREAT_BASE_M or c.mark['stall'] >= 35:
            c.mark.update(stage=2, home=0)
        return
    worst = c.to_rest(action, q)
    c.mark['home'] += 1
    c.reason = f'arms home, err {worst:.2f}'
    if worst < 0.18 or c.mark['home'] > 160:
        c.change('drawer_done', f'drawer pulled {c.drawer_open_m:.3f} m, arms home err {worst:.2f}')


def step_drawer_done(c, action, q, w) -> None:
    """The drawer is out: the cabinet estimate stays frozen (the open drawer would grow the top rectangle); on to the jar."""
    c.to_rest(action, q)
    c.perc.freeze_cabinet = True
    c.change('plan_jar' if w.jar is not None else 'find_jar', f'drawer open {c.drawer_open_m:.3f} m, jar known {w.jar is not None}')


PHASES = {'go_view': step_go_view, 'drawer_view': step_drawer_view, 'plan_drawer': step_plan_drawer, 'go_drawer': step_go_drawer, 'drawer_pre': step_drawer_pre, 'drawer_reach': step_drawer_reach,
          'drawer_grip': step_drawer_grip, 'drawer_pull': step_drawer_pull, 'drawer_release': step_drawer_release,
          'drawer_retreat': step_drawer_retreat, 'drawer_done': step_drawer_done}
