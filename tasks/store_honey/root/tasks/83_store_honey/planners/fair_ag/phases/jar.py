"""Phases of the jar: look for it (find_jar), choose a base stand on a ring around it from which a hand reaches it over the counter (plan_jar),
drive there (go_jar), re-measure it from the stand (jar_look), side-pinch it with a horizontal jaw (jar_pre, jar_reach, jar_grasp; the native
assisted grasp holds it once the fingers close on it), lift it (jar_lift), draw it back over the counter, roll the wrist so the lower finger is
under it and bring it to the carry posture (jar_retract).

The grasp is judged from the finger gap in the proprio only: closed on the 8 cm jar the fingers stop in JAR_FINGER_BAND; below JAR_HELD_MIN
they hold nothing. The jar estimate is frozen from jar_pre on (the hand then hides it and the self filter cuts the points next to the fingers)."""
from __future__ import annotations

import math

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import params as P
from .explore import UNSTICK_MAX, backoff_step, start_backoff


def jar_targets(jar, axis: np.ndarray) -> dict:
    """Pre point, grasp point and rotation of the side pinch along the horizontal approach axis (from the stand to the jar)."""
    c = np.array([jar.xy[0], jar.xy[1], jar.grasp_z])
    rot = G.jaw_rotation(axis, np.cross(G.UP, axis))
    return dict(centre=c, pre=c - axis * P.JAR_PRE_GAP_M, grasp=c + axis * P.JAR_DEEP_M, rotation=rot)


def support_rotation(axis: np.ndarray, roll: float = P.CARRY_ROLL) -> np.ndarray:
    """The closed side pinch rolled about the finger axis so the lower finger moves under the jar (roll 0 is the horizontal jaw)."""
    axis = G.unit(axis)
    horizontal = np.cross(G.UP, axis)
    jaw = horizontal * math.cos(roll) + np.cross(axis, horizontal) * math.sin(roll)
    return G.jaw_rotation(axis, jaw)


def horiz(v) -> np.ndarray:
    v = np.asarray(v, float).copy()
    v[2] = 0.
    return G.unit(v)


def crossing_cost(side: str, base: np.ndarray, point: np.ndarray) -> float:
    """Penalty for a hand reaching across the body: the target lies on the other side of the base than the hand."""
    lateral = float(G.to_robot(base, point)[1])
    wrong = (side == 'left' and lateral < -0.05) or (side == 'right' and lateral > 0.05)
    return 3.0 + abs(lateral) if wrong else 0.0


def step_find_jar(c, action, q, w) -> None:
    c.to_rest(action, q)
    if c.age > 300:
        c.perc.jar_dense = True
        c.perc.jar_min = 2
    if w.jar is not None and w.jar.sightings >= min(P.JAR_MIN_SIGHTINGS, getattr(c.perc, 'jar_min', P.JAR_MIN_SIGHTINGS)):
        c.perc.jar_dense = False
        c.change('plan_jar', f'jar at {np.round(w.jar.xy, 3).tolist()} on a surface at {w.jar.support:.3f} m ({w.jar.sightings} sightings)')
        return
    turned = c.mark.setdefault('turned', 0.0) + abs(nav_grid.wrap(w.base[2] - c.mark.setdefault('yaw', w.base[2])))
    c.mark.update(turned=turned, yaw=w.base[2])
    if turned >= 2 * P.FIND_JAR_TURN_RAD:
        c.fail(f'no jar after {turned:.1f} rad of turning')
        return
    action[2] = 0.5 * P.SCAN_RATE / nav_grid.BASE_TURN
    c.reason = f'turning to find the jar, {turned:.1f} rad, sightings {len(c.perc.jar_sightings)}'


def step_plan_jar(c, action, q, w) -> None:
    if backoff_step(c, action, q, w):
        return
    c.to_rest(action, q)
    jar = w.jar
    if jar is None:
        c.change('find_jar', 'no jar to plan on')
        return
    pg = c.pose_grid(w, jar.xy)
    out = []
    for ring in P.JAR_RINGS + ((0.55, 1.15, 1.25, 1.35) if getattr(c, 'wide_jar', False) else ()):
        for bearing in np.arange(0., 2 * math.pi, P.JAR_BEARING_STEP_RAD):
            xy = jar.xy + ring * np.array([math.cos(bearing), math.sin(bearing)])
            yaw = nav_grid.wrap(bearing + math.pi)
            if pg.free(xy, yaw):
                out.append(dict(xy=xy, yaw=yaw, cost=float(np.linalg.norm(xy - w.base[:2])) + 0.8 * ring))
    out.sort(key=lambda v: v['cost'])
    rej = c.mark.get('rejected_stands') or []
    if rej:
        keep = [v for v in out if all(float(np.linalg.norm(v['xy'] - np.asarray(r, float)[:2])) > 0.30 for r in rej)]
        out = keep or out          # a stand that wedged the base or had no route is not offered again while others exist
    rows = out[:P.SCAN_STANDS]
    c.diag = dict(free=len(out), routed=0, ik=0, clear=0)
    if not rows:
        c.fail(f'no free stand around the jar {c.diag}')
        return
    stands = np.array([[v['xy'][0], v['xy'][1], v['yaw']] for v in rows])
    tg = [jar_targets(jar, horiz(np.r_[jar.xy - v['xy'], 0.])) for v in rows]
    pre = np.array([t['pre'] for t in tg])
    grasp = np.array([t['grasp'] for t in tg])
    rots = np.array([t['rotation'] for t in tg])
    scans = {sd: c.scan(sd, stands, [(pre, rots), (grasp, rots)]) for sd in ('right', 'left')}
    ok = []
    for i, v in enumerate(rows):
        if len(ok) >= P.CAND_LIMIT:
            break
        base = stands[i]
        sides = sorted(('right', 'left'), key=lambda sd: crossing_cost(sd, base, tg[i]['centre']))
        sides = [sd for sd in sides if scans[sd][0][i]]
        if not sides:
            continue
        c.diag['ik'] += 1
        sides = [sd for sd in sides if all(c.arm_clear(sd, sol[i], q, base, jar.xy, P.JAR_DIAMETER_M) for sol in scans[sd][1])]
        if not sides:
            continue
        c.diag['clear'] += 1
        if pg.plan(w.base[:2], w.base[2], v['xy'], v['yaw']) is None:
            continue
        c.diag['routed'] += 1
        ok.append(dict(v, side=sides[0]))
    if not ok:
        if not getattr(c, 'wide_jar', False):
            c.wide_jar = True
            c.change('plan_jar', f'no reachable, clear and routed stand at the jar {c.diag}; wider rings')
            return
        if c.counts.get('plan_retries', 0) < 2 and c.counts.get('unsticks', 0) < UNSTICK_MAX and not c.mark.get('backoff'):
            c.counts['plan_retries'] = c.counts.get('plan_retries', 0) + 1
            for v in rows[:3]:
                c.mark.setdefault('rejected_stands', []).append(np.asarray(v['xy'], float))
            c.mark['rejected_stands'] = c.mark['rejected_stands'][-8:]
            start_backoff(c, w, away=np.asarray(jar.xy[:2], float), next_phase='plan_jar',
                          why=f'no reachable, clear and routed stand at the jar {c.diag}; backed off and re-planning')
            return
        c.fail(f'no reachable, clear and routed stand at the jar {c.diag}')
        return
    c.mark.pop('go_best', None)
    c.mark['go_stall'] = 0
    c.cands = ok[1:]
    c.goal = ok[0]
    c.side = c.jar_side = c.goal['side']
    c.change('go_jar', f"stand {np.round(c.goal['xy'], 2).tolist()} yaw {c.goal['yaw']:.2f} hand {c.side} jar {np.round(jar.xy, 3).tolist()} {c.diag}")


def next_jar_candidate(c, why: str) -> None:
    c.counts['candidates_failed'] += 1
    c.perc.freeze_jar = False
    c.grip[c.side] = K.GRIPPER_OPEN
    if c.counts['jar_tries'] >= P.MAX_JAR_TRIES:
        c.fail(why + '; out of jar tries')
    elif c.cands:
        c.goal = c.cands.pop(0)
        c.side = c.jar_side = c.goal['side']
        c.change('go_jar', why + '; next stand')
    else:
        c.change('plan_jar', why + '; re-planning')


def step_go_jar(c, action, q, w) -> None:
    if backoff_step(c, action, q, w):
        return
    if c.age == 1:
        c.mark.pop('go_best', None)
        c.mark['go_stall'] = 0
    dist, yaw_err = c.navigate(action, q, w, c.goal['xy'], c.goal['yaw'])
    if c.nav.arrived(dist, yaw_err):
        c.change('jar_look', 'at the jar stand')
        return
    best = c.mark.get('go_best')
    if best is None or dist < best - 0.02:
        c.mark['go_best'], c.mark['go_stall'] = float(dist), 0
    else:
        c.mark['go_stall'] = int(c.mark.get('go_stall', 0)) + 1
    stalled = c.mark['go_stall'] > 300
    if not stalled and c.nav.no_route <= P.NO_ROUTE_MAX:
        return
    c.counts['nav_stuck'] += 1
    rej = c.mark.setdefault('rejected_stands', [])
    rej.append(np.asarray(c.goal['xy'], float))
    c.mark['rejected_stands'] = rej[-8:]
    why = ('stuck driving to the jar stand, no progress for 300 controls' if stalled
           else f'no route to the jar stand (no_route {c.nav.no_route})')
    if c.counts.get('unsticks', 0) >= UNSTICK_MAX:
        next_jar_candidate(c, why)
        return
    start_backoff(c, w, away=np.asarray(c.goal['xy'], float), next_phase='plan_jar',
                  why=f'{why}; backed off, stand rejected and re-planned')


def step_jar_look(c, action, q, w) -> None:
    """Stand still with the arms at rest while the head measures the jar from close by; then fix the grasp targets from this base pose."""
    c.to_rest(action, q)
    if c.age == 1:
        c.mark['old'] = w.jar
        c.perc.forget_jar(c.step)
        return
    if c.age < P.JAR_LOOK_STEPS:
        c.reason = f'measuring the jar from the stand, {len(c.perc.jar_sightings)} sightings'
        return
    jar = w.jar if w.jar is not None else c.mark.get('old')
    if jar is None:
        c.change('find_jar', 'the jar is not seen from the stand')
        return
    axis = horiz(np.r_[jar.xy - w.base[:2], 0.])
    t = jar_targets(jar, axis)
    mask, _ = c.scan(c.side, w.base[None], [(t['pre'], t['rotation']), (t['grasp'], t['rotation'])])
    if not mask[0]:
        other = c.other(c.side)
        mask_o, _ = c.scan(other, w.base[None], [(t['pre'], t['rotation']), (t['grasp'], t['rotation'])])
        if not mask_o[0]:
            c.counts['jar_tries'] += 1
            next_jar_candidate(c, f'the re-measured jar {np.round(jar.xy, 3).tolist()} is out of reach from here')
            return
        c.side = c.jar_side = other
    c.jar_goal = dict(t, axis=axis, jar=jar)
    c.perc.freeze_jar = True
    c.change('jar_pre', f"jar re-measured at {np.round(jar.xy, 3).tolist()} support {jar.support:.3f} top {jar.top:.3f} "
                        f"({jar.sightings} sightings, fresh {w.jar is not None}), hand {c.side}")


def step_jar_pre(c, action, q, w) -> None:
    t = c.jar_goal
    c.grip[c.side] = K.GRIPPER_OPEN
    c.grip[c.other(c.side)] = K.GRIPPER_CLOSED
    perr, oerr = c.servo_to(action, q, w, t['pre'], t['rotation'])
    c.reason = f'to the pre point behind the jar, err {perr:.3f} m {oerr:.2f} rad'
    if perr < 0.015 and oerr < 0.15:
        c.change('jar_reach', 'pre point reached')


def step_jar_reach(c, action, q, w) -> None:
    t = c.jar_goal
    c.grip[c.side] = K.GRIPPER_OPEN
    perr, oerr = c.servo_to(action, q, w, t['grasp'], t['rotation'], max_step=0.008)
    c.reason = f'fingers around the jar, err {perr:.3f} m {oerr:.2f} rad'
    if perr < 0.010 and oerr < 0.15:
        c.change('jar_grasp', 'jar between the fingers')


def step_jar_grasp(c, action, q, w) -> None:
    t = c.jar_goal
    c.servo_to(action, q, w, t['grasp'], t['rotation'], max_step=0.004)
    c.grip[c.side] = K.GRIPPER_CLOSED
    fing = c.finger_mean(q, c.side)
    c.reason = f'closing on the jar, fingers {fing:.4f}'
    if c.age >= P.GRIP_WAIT:
        if P.JAR_FINGER_BAND[0] <= fing <= P.JAR_FINGER_BAND[1]:
            c.change('jar_lift', f'jar gripped, fingers {fing:.4f}')
        else:
            c.counts['jar_tries'] += 1
            next_jar_candidate(c, f'grasp missed, fingers {fing:.4f}')


def step_jar_lift(c, action, q, w) -> None:
    t = c.jar_goal
    c.grip[c.side] = K.GRIPPER_CLOSED
    tool = np.array(t['grasp'], float)
    tool[2] += min(P.JAR_LIFT_M, 0.0008 * c.age)
    perr, _ = c.servo_to(action, q, w, tool, t['rotation'], max_step=0.004)
    fing = c.finger_mean(q, c.side)
    c.reason = f'lifting, tool {tool[2] - t["grasp"][2]:.3f} m up, err {perr:.3f}, fingers {fing:.4f}'
    if fing < P.JAR_HELD_MIN:
        c.counts['jar_tries'] += 1
        next_jar_candidate(c, f'the jar slipped out while lifting, fingers {fing:.4f}')
        return
    if c.age > 220:
        c.change('jar_retract', f'jar lifted, fingers {fing:.4f}')
        c.mark['lift_tool'] = tool.copy()


def carry_point(c, w, z: float) -> np.ndarray:
    side = 1.0 if c.side == 'left' else -1.0
    p = G.to_odom(w.base, np.array([P.CARRY_POINT[0], side * P.CARRY_POINT[1], 0.]))
    p[2] = z
    return p


def step_jar_retract(c, action, q, w) -> None:
    """Straight back over the counter at lift height with the closed side pinch, then the wrist roll, then the carry point at the same height."""
    t = c.jar_goal
    c.grip[c.side] = K.GRIPPER_CLOSED
    axis = t['axis']
    lift = np.asarray(c.mark['lift_tool'], float)
    stage = c.mark.setdefault('stage', 0)
    c.mark['stage_age'] = c.mark.get('stage_age', 0) + 1
    sa = c.mark['stage_age']
    back = lift.copy()
    back[:2] = back[:2] - axis[:2] * 0.28
    fing = c.finger_mean(q, c.side)
    if fing < P.JAR_HELD_MIN:
        c.counts['jar_tries'] += 1
        next_jar_candidate(c, f'jar lost while drawing it back, fingers {fing:.4f}')
        return
    if stage == 0:
        pt = lift.copy()
        pt[:2] = pt[:2] - axis[:2] * min(0.28, 0.001 * max(sa, 1))
        perr, _ = c.servo_to(action, q, w, pt, t['rotation'], max_step=0.004)
        pulled = float(np.dot(lift[:2] - c.tip_point(q, w)[:2], axis[:2]))
        c.reason = f'drawing the jar back over the counter, {pulled:.3f} m, err {perr:.3f}, fingers {fing:.4f}'
        if pulled > 0.18 or sa >= 450:
            c.mark.update(stage=1, stage_age=0)
            c.servo = None
    elif stage == 1:
        perr, oerr = c.servo_to(action, q, w, back, support_rotation(axis), max_step=0.003)
        c.reason = f'rolling the wrist so the lower finger is under the jar, err {perr:.3f} m {oerr:.2f} rad, fingers {fing:.4f}'
        if (perr < 0.04 and oerr < 0.35 and sa > 40) or sa >= 700:
            c.mark.update(stage=2, stage_age=0)
            c.servo = None
    else:
        perr, oerr = c.servo_to(action, q, w, carry_point(c, w, float(lift[2])), support_rotation(axis), max_step=0.003)
        c.reason = f'to the carry posture, err {perr:.3f} m {oerr:.2f} rad, fingers {fing:.4f}'
        if (perr < 0.035 and oerr < 0.4 and sa > 25) or sa > 400:
            c.hold = c.joints(q)
            c.change('plan_place', f'jar carried, fingers {fing:.4f}')


PHASES = {'find_jar': step_find_jar, 'plan_jar': step_plan_jar, 'go_jar': step_go_jar, 'jar_look': step_jar_look, 'jar_pre': step_jar_pre,
          'jar_reach': step_jar_reach, 'jar_grasp': step_jar_grasp, 'jar_lift': step_jar_lift, 'jar_retract': step_jar_retract}
