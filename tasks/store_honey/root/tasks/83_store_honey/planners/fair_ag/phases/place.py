"""Phases of the place: choose a base stand in front of the open top drawer (plan_place), carry the jar there (go_place), bring it above the
drawer cavity and lower it in (place_above, place_lower), open the fingers (place_release), back the hand out (place_retreat), stop (done).

The cavity is not observed: it follows from the cabinet estimate (frozen since the pull) and the drawer travel of the pull (base travel), with the
asset priors HULL_* of the top drawer. The place line keeps the rolled side pinch of the carry and the trunk at its carry posture when a
trunk-locked solution exists; the hand advances along a fixed straight line in the odometry frame only while it stays within 3 cm of it (a
sagging loaded arm does not drag the target down). The jar counts as held while the fingers stay apart (JAR_HELD_MIN)."""
from __future__ import annotations

import math

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from .. import ik
from .. import params as P
from .base import world_T
from .jar import horiz, support_rotation

CAVITY_ALONG = (0.0, -0.05, 0.05, -0.10)          # along the place axis from the cavity centre (negative: toward the robot)
CAVITY_LATERAL = (0.0, 0.1, -0.1, 0.2, -0.2)      # along the drawer front
CAVITY_RISE = (P.PLACE_ABOVE_M, 0.14)             # height of the above point over the place point
STALL_WINDOW = 150                                # controls without STALL_GAIN_M of progress toward the above point
STALL_GAIN_M = 0.005
STALLS_BEFORE_ALT = 2                             # stalled visits that keep the default target before an alternative is chosen


def place_targets(c, cab) -> dict:
    hull = cab.centre + cab.f * (P.HULL_OUT_OF_CENTRE_M + c.drawer_open_m)
    floor_z = cab.top_z - P.HULL_FLOOR_BELOW_TOP_M
    place = np.array([hull[0], hull[1], floor_z + P.JAR_HEIGHT_M / 2 + P.PLACE_CLEAR_M])
    axis = horiz(np.r_[-cab.f, 0.])
    rot = support_rotation(axis)
    return dict(hull=np.array([hull[0], hull[1], cab.top_z - P.HULL_BELOW_TOP_M]), place=place + axis * P.JAR_DEEP_M,
                above=place + axis * P.JAR_DEEP_M + np.array([0., 0., P.PLACE_ABOVE_M]), rotation=rot, axis=axis)


def trunk_locked_ok(c, side: str, base: np.ndarray, q: np.ndarray, targets) -> bool:
    """Every target has a solve_near solution from `base` that keeps the trunk where it is now (seed: current trunk, rest arm)."""
    trunk = np.asarray(q, float)[list(K.TRUNK_DOFS)]
    seed = np.r_[trunk, K.rest(side)[4:]]
    for point, rotation in targets:
        res = ik.near(side, world_T(point, rotation, 1), np.asarray(base, float).reshape(1, 3), 0.0, seed)
        if res is None or not res[0][0] or np.max(np.abs(res[1][0, :4] - trunk)) > P.LOCK_TOL:
            return False
    return True


def holding(c, q) -> bool:
    return c.finger_mean(q, c.side) >= P.JAR_HELD_MIN


def active_targets(c, w) -> dict:
    """Place targets; an alternative cavity point chosen after a stall replaces above/place while the base stays where it was chosen."""
    t = place_targets(c, w.cabinet)
    alt = getattr(c, 'place_alt', None)
    if alt is None:
        return t
    base = np.asarray(w.base, float)
    if float(np.linalg.norm(base[:2] - alt['base'][:2])) > 0.08 or abs(math.remainder(base[2] - alt['base'][2], 2 * math.pi)) > 0.1:
        return t
    return dict(t, above=alt['above'].copy(), place=alt['place'].copy(), alt=True)


def replace_target(c, q, w, t, tip) -> bool:
    """The hand stalled on the way to t['above']: remember that point as rejected, scan the cavity points from the measured base pose in one
    batch and switch to the reachable one nearest to the stalled hand (or straight down from the hand when it is already over the cavity)."""
    cab = w.cabinet
    rejected = list(getattr(c, 'place_rejected', []))
    rejected.append(np.asarray(t['above'], float).copy())
    c.place_rejected = rejected
    axis = np.asarray(t['axis'], float)
    lat = np.r_[np.asarray(cab.lateral, float)[:2], 0.]
    lat = lat / max(float(np.linalg.norm(lat)), 1e-9)
    base_place = np.asarray(place_targets(c, cab)['place'], float)
    above, place = [], []
    for a in CAVITY_ALONG:
        for l in CAVITY_LATERAL:
            p = base_place + axis * a + lat * l
            for rise in CAVITY_RISE:
                above.append(p + np.array([0., 0., rise]))
                place.append(p.copy())
    rel = np.asarray(tip, float) - base_place
    along, side = float(np.dot(rel, axis)), float(np.dot(rel, lat))
    column = None
    if abs(along) <= 0.12 and abs(side) <= 0.3:
        p = base_place + axis * along + lat * side
        a_pt = p.copy()
        a_pt[2] = float(np.clip(tip[2], p[2] + 0.10, p[2] + P.PLACE_ABOVE_M))
        column = (a_pt, p)
        above.append(a_pt)
        place.append(p.copy())
    keep = [i for i in range(len(above)) if all(float(np.linalg.norm(above[i] - r)) > 0.04 for r in rejected)]
    if not keep:
        return False
    A = np.array([above[i] for i in keep])
    Pl = np.array([place[i] for i in keep])
    n = len(keep)
    stands = np.repeat(np.asarray(w.base, float).reshape(1, 3), n, 0)
    rots = np.repeat(np.asarray(t['rotation'], float)[None], n, 0)
    mask, _ = c.scan(c.side, stands, [(A, rots), (Pl, rots)])
    ok = [i for i in range(n) if mask[i]]
    if ok:
        best = min(ok, key=lambda i: float(np.linalg.norm(A[i] - tip)))
        pick, how = (A[best], Pl[best]), f'{len(ok)} of {n} reachable'
    elif column is not None and all(float(np.linalg.norm(column[0] - r)) > 0.04 for r in rejected):
        pick, how = column, f'none of {n} reachable; straight down from the hand over the cavity'
    else:
        return False
    c.place_alt = dict(base=np.asarray(w.base, float).copy(), above=np.asarray(pick[0], float).copy(), place=np.asarray(pick[1], float).copy())
    c.servo = None
    c.change('place_above', f'stalled at {np.round(tip, 3).tolist()}; alternative cavity point above {np.round(pick[0], 3).tolist()} '
                            f'place {np.round(pick[1], 3).tolist()} ({how}, rejected {len(rejected)})')
    return True


def step_plan_place(c, action, q, w) -> None:
    c.hold_joints(action, c.hold)
    cab = w.cabinet
    if cab is None:
        c.fail('no cabinet estimate for the place')
        return
    t = place_targets(c, cab)
    c.place_alt = None
    c.place_stalls = 0
    c.place_rejected = []
    heading = math.atan2(t['axis'][1], t['axis'][0])
    pg = c.pose_grid(w, t['hull'][:2])
    out = []
    for d in P.PLACE_STANDS:
        for lat in P.PLACE_LATERALS:
            xy = t['hull'][:2] + cab.f * d + cab.lateral * lat
            if pg.free(xy, heading):
                out.append(dict(xy=xy, yaw=heading, cost=float(np.linalg.norm(xy - w.base[:2])) + 0.3 * d + 0.5 * abs(lat)))
    out.sort(key=lambda v: v['cost'])
    rows = out[:P.SCAN_STANDS]
    c.diag = dict(free=len(out), ik=0, routed=0, locked=0)
    if not rows:
        c.fail(f'no free stand in front of the open drawer {c.diag}')
        return
    stands = np.array([[v['xy'][0], v['xy'][1], v['yaw']] for v in rows])
    pair = [(t['above'], t['rotation']), (t['place'], t['rotation'])]
    mask, _ = c.scan(c.side, stands, pair)
    ok = []
    for i, v in enumerate(rows):
        if len(ok) >= P.CAND_LIMIT:
            break
        if not mask[i]:
            continue
        c.diag['ik'] += 1
        if pg.plan(w.base[:2], w.base[2], v['xy'], v['yaw']) is None:
            continue
        c.diag['routed'] += 1
        v['lock'] = trunk_locked_ok(c, c.side, stands[i], q, pair)
        c.diag['locked'] += int(v['lock'])
        ok.append(v)
    if not ok:
        c.fail(f'no reachable, routed stand at the open drawer {c.diag}')
        return
    ok.sort(key=lambda v: (not v['lock'], v['cost']))
    c.cands = ok[1:]
    c.goal = ok[0]
    c.change('go_place', f"stand {np.round(c.goal['xy'], 2).tolist()} lock {c.goal['lock']} hull {np.round(t['hull'], 3).tolist()} {c.diag}")


def next_place_candidate(c, why: str) -> None:
    c.counts['candidates_failed'] += 1
    if c.counts['place_tries'] >= P.MAX_PLACE_TRIES:
        c.fail(why + '; out of place tries')
    elif c.cands:
        c.goal = c.cands.pop(0)
        c.change('go_place', why + '; next stand')
    else:
        c.change('plan_place', why + '; re-planning')


def step_go_place(c, action, q, w) -> None:
    c.grip[c.side] = K.GRIPPER_CLOSED
    if not holding(c, q):
        c.fail(f'jar dropped while carrying, fingers {c.finger_mean(q, c.side):.4f}')
        return
    dist, yaw_err = c.navigate(action, q, w, c.goal['xy'], c.goal['yaw'], carry=True)
    if not c.mark.get('turn_done'):
        if abs(yaw_err) > 0.6:
            action[2] = float(np.clip(yaw_err, -0.30, 0.30))     # one start turn, then the path follower owns the base
            c.reason = f'turning toward the drawer while driving, yaw err {yaw_err:.2f}'
        else:
            c.mark['turn_done'] = True
    if c.nav.arrived(dist, yaw_err):
        c.change('place_above', 'at the drawer stand')
    elif c.nav.no_route > P.NO_ROUTE_MAX:
        c.counts['nav_stuck'] += 1
        next_place_candidate(c, 'no route to the drawer stand')


def track_toward(c, action, q, w, goal: np.ndarray, rotation: np.ndarray, step: float):
    """Advance the reference along the straight line to `goal` while the fingers stay within 3 cm of it; one resolved-rate control there."""
    c.grip[c.side] = K.GRIPPER_CLOSED
    tip = c.tip_point(q, w)
    ref = c.mark.setdefault('ref', tip.copy())
    delta = np.asarray(goal, float) - ref
    n = float(np.linalg.norm(delta))
    if float(np.linalg.norm(tip - ref)) < 0.03:
        ref = np.asarray(goal, float).copy() if n <= step else ref + delta * (step / n)
        c.mark['ref'] = ref
    perr, oerr = c.servo_to(action, q, w, ref, rotation, max_step=0.004, plan=False, lock_trunk=bool(c.goal.get('lock', False)))
    return tip, perr, oerr


def step_place_above(c, action, q, w) -> None:
    t = active_targets(c, w)
    if not holding(c, q):
        c.fail(f'jar lost above the drawer, fingers {c.finger_mean(q, c.side):.4f}')
        return
    tip, perr, oerr = track_toward(c, action, q, w, t['above'], t['rotation'], 0.003)
    gerr = float(np.linalg.norm(tip - t['above']))
    c.reason = (f'jar to above the drawer, err {gerr:.3f} m {oerr:.2f} rad, goal {np.round(t["above"], 3).tolist()}'
                f'{" (alternative)" if t.get("alt") else ""}, stalls {getattr(c, "place_stalls", 0)}')
    if gerr < 0.02 and oerr < (0.3 if t.get('alt') else 0.2):
        c.change('place_lower', 'jar above the drawer' + (' (alternative cavity point)' if t.get('alt') else ''))
        return
    if gerr < c.mark.get('best', np.inf) - STALL_GAIN_M:
        c.mark['best'] = gerr
        c.mark['best_age'] = c.age
    elif c.age - c.mark.get('best_age', c.age) > STALL_WINDOW and not c.mark.get('stalled'):
        c.mark['stalled'] = True
        c.place_stalls = getattr(c, 'place_stalls', 0) + 1
        if c.place_stalls > STALLS_BEFORE_ALT:
            replace_target(c, q, w, t, tip)


def step_place_lower(c, action, q, w) -> None:
    t = active_targets(c, w)
    tip, perr, oerr = track_toward(c, action, q, w, t['place'], t['rotation'], 0.002)
    prev = c.mark.get('prev_z', tip[2])
    c.mark['stall'] = c.mark.get('stall', 0) + 1 if abs(prev - tip[2]) < 0.0005 else 0
    c.mark['prev_z'] = tip[2]
    c.reason = f'lowering the jar into the drawer, tip z {tip[2]:.3f}, goal z {t["place"][2]:.3f}, stall {c.mark["stall"]}'
    low = tip[2] <= t['place'][2] + 0.015
    stuck = c.mark['stall'] > 40 and tip[2] < t['above'][2] - 0.08
    if low or stuck:
        c.hold = c.joints(q)
        c.change('place_release', f'jar down in the drawer, tip z {tip[2]:.3f} ({"at the goal" if low else "stopped"})')


def step_place_release(c, action, q, w) -> None:
    c.hold_joints(action, c.hold)
    c.grip[c.side] = K.GRIPPER_OPEN
    c.reason = f'opening the fingers, {c.finger_mean(q, c.side):.4f}'
    if c.age >= 60:
        c.change('place_retreat', 'fingers open')


def step_place_retreat(c, action, q, w) -> None:
    t = place_targets(c, w.cabinet)
    c.grip[c.side] = K.GRIPPER_OPEN
    tip0 = c.mark.setdefault('tip0', c.tip_point(q, w))
    pt = tip0 - t['axis'] * min(0.15, 0.002 * c.age) + np.array([0., 0., min(0.10, 0.002 * c.age)])
    c.servo_to(action, q, w, pt, t['rotation'], max_step=0.008, plan=False)
    c.reason = 'backing the hand out of the drawer'
    if c.age > 90:
        c.change('done', 'jar left in the drawer')


def step_done(c, action, q, w) -> None:
    c.reason = 'done'


PHASES = {'plan_place': step_plan_place, 'go_place': step_go_place, 'place_above': step_place_above, 'place_lower': step_place_lower,
          'place_release': step_place_release, 'place_retreat': step_place_retreat, 'done': step_done}
