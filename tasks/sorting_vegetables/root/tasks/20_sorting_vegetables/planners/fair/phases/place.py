"""Placing the held vegetable in the bowl of its group (the oracle's drop, on perception): plan_drop (a stand by the bowl whose arm reaches
over it from the carry pose, body-checked against the perceived occupancy), go_bowl, look_bowl (the bowl seen again from the stand),
bowl_pre (joint walk to over the bowl; under a wall shelf to a point in front of it, then bowl_in straight in), bowl_lower (straight down
until the vegetable's bottom is under the rim), veg_release (open and wait), veg_retract (hand up over the rim), then plan_veg. A vegetable no stand reaches the bowl with is put down low (veg_putdown) and
picked again later. Held means the fingers stay apart (the native assisted grasp holds the vegetable between them)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import params as P
from ..world import BowlEst, World


def holding(c, q) -> bool:
    return c.finger_gap(q, c.task['side']) > P.HELD_GAP_M


def bowl(c, w: World) -> BowlEst:
    return w.bowls[c.task['bowl']]


def bowl_xy(b: BowlEst) -> np.ndarray:
    """The newest sighting of the bowl (the running mean lags the odometry drift)."""
    return (b.last if b.last is not None else b.centre)[:2].copy()


def target_xy(c, w: World) -> np.ndarray:
    """The bowl centre the drop aims at: as seen from the drop stand (look_bowl), else the bowl track's newest sighting."""
    t = c.task
    return np.asarray(t['bowl_xy'], float).copy() if t.get('bowl_xy') is not None else bowl_xy(bowl(c, w))


def held_rot(c, base: np.ndarray) -> np.ndarray:
    """Odometry rotation of the hand with the vegetable at the base pose `base`: after the raise it keeps its rotation in the robot frame."""
    return G.rot_yaw(base[2]) @ c.task['carry_rot']


def ceiling(c, b: BowlEst, xy: np.ndarray) -> float:
    """Highest finger midpoint over the bowl that keeps the wrist of a hand pointing down under something hanging over it (a wall shelf):
    the lowest occupied voxel over the bowl's footprint whose column is empty from the rim up to 15 cm under it (a wall is occupied all the
    way up, a shelf only in its band). inf without one."""
    vox = c.perc.vox
    occ = vox.occupied()
    r = P.BOWL_RIM_R_M + P.CEIL_MARGIN_M
    i0, j0 = vox.cell2(xy - r)
    i1, j1 = vox.cell2(xy + r)
    k_rim = int((b.rim + 0.05) / P.VOX_M)
    k_min = int((b.rim + P.CEIL_MIN_ABOVE_M) / P.VOX_M)
    low = np.inf
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            ks = np.flatnonzero(occ[i, j, k_min:]) + k_min
            if not len(ks):
                continue
            k = int(ks[0])
            if not occ[i, j, k_rim:max(k_rim, k - 3)].any():
                low = min(low, k * P.VOX_M)
    return low - P.BODY_MARGIN_M - P.WRIST_ABOVE_MID_M - P.CEILING_SLACK_M


def release_xy(c, w: World) -> np.ndarray:
    """Where the vegetable's centre goes: the bowl centre for the first one, then SPREAD_M off it, turning by a third each time."""
    t = c.task
    n = c.in_bowl.get(t['bowl'], 0)
    if n == 0:
        return target_xy(c, w)
    a = 2. * math.pi * n / 3. + 0.5 * (n // 3)
    return target_xy(c, w) + P.SPREAD_M * np.array([math.cos(a), math.sin(a)])


def drop_points(c, w: World, stand: np.ndarray, lower: float = 0.) -> Tuple[np.ndarray, np.ndarray]:
    """(finger midpoint over the bowl where the walk ends, the end of the straight line down) for the base pose `stand`: the vegetable, not
    the fingers, over the release point (its offset from the fingers turns with the base: the hand keeps its pose in the robot frame).
    `lower` takes the walk's end that much lower (not under the release height)."""
    t, b = c.task, bowl(c, w)
    off = (held_rot(c, stand) @ t['held_off'])[:2]
    xy = release_xy(c, w) - off
    rel = b.rim + P.DROP_CLEAR_M + t['held_below']
    above = max(rel, min(rel + P.BOWL_ABOVE_M, ceiling(c, b, target_xy(c, w))) - lower)
    return np.r_[xy, above], np.r_[xy, b.rim - P.IN_BOWL_M + t['held_below']]


def drop_pre(c, w: World, stand: np.ndarray, above: np.ndarray) -> np.ndarray:
    """Where the walk over a bowl under a shelf ends: the point over the bowl moved SHELF_IN_M toward the base pose `stand`, at its height
    (walking straight to over bowl_3 of 302 the hand rose into the shelf, oracle); then a straight line in. The point itself without a
    shelf."""
    if not np.isfinite(ceiling(c, bowl(c, w), target_xy(c, w))):
        return above
    d = stand[:2] - above[:2]
    return np.r_[above[:2] + P.SHELF_IN_M * d / max(float(np.linalg.norm(d)), 1e-6), above[2]]


def bowl_box(xy: np.ndarray, b: BowlEst):
    r = P.BOWL_RIM_R_M + 0.04
    return (np.r_[xy - r, 0.], np.r_[xy + r, b.rim + 0.04])


def reach(c, w: World, q: np.ndarray, stand: np.ndarray, start: np.ndarray) -> Tuple[Optional[np.ndarray], str, float]:
    """Trunk + arm joints of the reach over the bowl from the base pose `stand`, the walk's end as high as REACH_LOWER allows (lower
    ends keep the elbow under wall cabinets: every reach of 303 had the elbow or the shoulder at 1.35-1.40 m in them, t20f19):
    (joints, how, lower) or (None, why, 0)."""
    hits = []
    for lower in P.REACH_LOWER:
        x, how = reach_at(c, w, q, stand, start, lower)
        if x is not None:
            return x, how + (f', {lower:.2f} m lower' if lower else ''), lower
        hits.append(how)
    return None, hits[0], 0.


def reach_at(c, w: World, q: np.ndarray, stand: np.ndarray, start: np.ndarray, lower: float) -> Tuple[Optional[np.ndarray], str]:
    """Trunk + arm joints of the reach over the bowl from the base pose `stand` in the carry rotation, clear of the perceived occupancy at
    the end and on the way from `start`, with the least joint change (at most MAX_JUMP_RAD): (joints, how) or (None, why). Seeds: the
    joints now, the same with the trunk upright."""
    t = c.task
    dofs = list(K.TRUNK_DOFS) + list(K.ARM_DOFS[t['side']])
    above, _ = drop_points(c, w, stand, lower)
    pre = drop_pre(c, w, stand, above)
    goal, rot = G.to_robot(stand, pre), t['carry_rot']
    q0 = np.array(q, float)
    q0[dofs] = start
    upright = q0.copy()
    upright[list(K.TRUNK_DOFS)] = K.rest(t['side'])[:4]
    ok, hits = [], []
    ignore = [bowl_box(target_xy(c, w), bowl(c, w))]
    for name, s in (('near', q0), ('near upright', upright)):     # solve_near only (the scan's joints only gate stands: AGENTS.md)
        x = c.servo.solve(s, goal, rot)
        hit = 'no IK' if x is None else c.body_hits(x, t['side'], stand, ignore, start, P.REACH_FRACTIONS)
        if hit is None and not np.allclose(pre, above):          # then straight in under the shelf to over the bowl
            seed_in = np.array(q, float)
            seed_in[dofs] = x
            x_in = c.servo.solve(seed_in, G.to_robot(stand, above), rot)
            hit = 'in: no IK' if x_in is None else (lambda h: h and f'in: {h}')(c.body_hits(x_in, t['side'], stand, ignore, x, P.REACH_FRACTIONS))
        jump = 0. if x is None else float(np.max(np.abs(x - start)))
        if hit is None and jump > P.MAX_JUMP_RAD:
            hit = f'joints {jump:.1f} rad'
        if hit is None:
            ok.append((jump, name, x))
        else:
            hits.append(f'{name}: {hit}')
    if ok:
        jump, name, x = min(ok, key=lambda o: o[0])
        return x, f'{name}, joints {jump:.2f} rad'
    return None, '; '.join(hits)


def stands_by(c, w: World, centre: np.ndarray, tally: Optional[dict] = None) -> np.ndarray:
    """Base poses on rings round `centre` facing it (with small yaw offsets), free in the inflated grid, clear of the obstacles, with a
    route; nearest to the centre and the base first. `tally` counts the ring points rejected per reason."""
    rows = []
    tally = {} if tally is None else tally
    for r in P.DROP_RINGS:
        for bearing in np.arange(P.N_BEARINGS) * 2 * math.pi / P.N_BEARINGS:
            xy = centre[:2] + r * np.array([math.cos(bearing), math.sin(bearing)])
            why = 'blocked' if not c.free_pose(xy) else 'clearance' if c.grid_raw.clearance(xy) < P.DROP_STAND_CLEAR_M else None
            d = c.route(xy) if why is None else np.inf
            why = why or (None if np.isfinite(d) else 'route')
            if why:
                tally[why] = tally.get(why, 0) + 1
                continue
            for dy in P.YAW_OFFSETS:
                rows.append((d + 2. * r + 0.15 * abs(dy), xy[0], xy[1], G.wrap(bearing + math.pi + dy)))
    rows.sort()
    return np.array([r[1:] for r in rows[:P.MAX_STANDS]], float).reshape(-1, 3)


def step_plan_drop(c, action, q, w: World) -> None:
    t = c.task
    action[K.TRUNK_ACTION], action[K.ARM_ACTION[t['side']]] = c.hold_q[:4], c.hold_q[4:]
    c.tuck_other(action, q, t['side'])
    if not holding(c, q):
        lost(c, 'vegetable lost before the drop plan')
        return
    b = bowl(c, w)
    t['bowl_xy'] = None
    c.build_grid(w, [bowl_xy(b)])
    centre = np.r_[bowl_xy(b), 0.]
    rejected: dict = {}
    stands = stands_by(c, w, centre, rejected)
    tally = dict(stands=len(stands), reach=0, ring=rejected)
    skip = t.get('skip_stands', 0)
    # solve_near per stand, nearest first: the hand's world rotation turns with each stand's yaw, so one batched scan goal does not fit all
    for i in range(min(len(stands), P.DROP_SOLVES)):
        x, how, _ = reach(c, w, q, stands[i], c.hold_q)
        if x is None:
            tally['body'] = tally.get('body', 0) + 1
            c.debug['drop_last_reject'] = how[:200]
            continue
        tally['reach'] += 1
        if skip:
            skip -= 1
            continue
        t.update(drop_stand=stands[i].copy(), drop_q=x, reach_how=how)
        c.change('go_bowl', f"drop stand {np.round(stands[i], 2).tolist()} by bowl {t['bowl']} at {np.round(bowl_xy(b), 2).tolist()} ({how}), {tally}")
        return
    key = (t['bowl'], t['side'])
    c.bad_drop[key] = c.bad_drop.get(key, 0) + 1
    if not len(stands) and not c.in_bowl.get(t['bowl']):      # nowhere to stand by this bowl and nothing in it yet: the group takes
        c.no_stand.add(t['bowl'])                              # a free bowl, or the empty bowl of a smaller group (any bowl serves any group)
        g, old_bowl = t['group'], t['bowl']
        free = [k for k in distinct(w) if k not in c.no_stand and k not in c.bowl_of.values()]
        swap = [h for h, k in c.bowl_of.items() if k not in c.no_stand and not c.in_bowl.get(k) and P.GROUP_SIZE[h] < P.GROUP_SIZE[g]]
        if free:
            c.bowl_of[g] = min(free, key=lambda k: float(np.linalg.norm(bowl_xy(w.bowls[k]) - w.base[:2])))
        elif swap:
            h = min(swap, key=lambda h: P.GROUP_SIZE[h])
            c.bowl_of[g], c.bowl_of[h] = c.bowl_of[h], old_bowl
        if c.bowl_of[g] != old_bowl:
            t['bowl'] = c.bowl_of[g]
            c.change('plan_drop', f"no stand by bowl {old_bowl} {tally}: group {g} goes to bowl {t['bowl']} (groups to bowls {c.bowl_of})")
            return
    put_down(c, q, w, f"no stand reaches bowl {t['bowl']} {tally}, last: {c.debug.get('drop_last_reject', '')[:120]}")


def step_go_bowl(c, action, q, w: World) -> None:
    t = c.task
    action[K.TRUNK_ACTION], action[K.ARM_ACTION[t['side']]] = c.hold_q[:4], c.hold_q[4:]
    c.tuck_other(action, q, t['side'])
    if not holding(c, q):
        lost(c, 'vegetable lost on the way to the bowl')
        return
    dist, yaw_err = c.navigate(action, w, t['drop_stand'][:2], t['drop_stand'][2])
    if c.arrived(dist, yaw_err):
        c.change('look_bowl', 'at the drop stand')
    elif c.age > c.nav_budget(t['drop_stand'][:2]):
        retry_drop(c, q, w, f'drive to the bowl timed out, {dist:.2f} m left')


def retry_drop(c, q, w: World, why: str) -> None:
    t = c.task
    t['drop_tries'] = t.get('drop_tries', 0) + 1
    t['skip_stands'] = t.get('skip_stands', 0) + 1
    if t['drop_tries'] <= P.DROP_TRIES:
        c.hold_q = q[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[t['side']])].copy()
        c.change('plan_drop', f"{why}: another stand ({t['drop_tries']})")
    else:
        put_down(c, q, w, why)


def step_look_bowl(c, action, q, w: World) -> None:
    """Stand still: the head sees the bowl again from the stand (detections run every control here)."""
    t = c.task
    action[K.TRUNK_ACTION], action[K.ARM_ACTION[t['side']]] = c.hold_q[:4], c.hold_q[4:]
    c.tuck_other(action, q, t['side'])
    if c.age < P.LOOK_STEPS:
        return
    b = bowl(c, w)
    planned = target_xy(c, w)
    fresh = [x for x in w.bowls if x.seen_step >= w.step - P.LOOK_STEPS and np.linalg.norm(bowl_xy(x) - planned) < P.BOWL_RELOOK_M]
    if fresh:     # the bowl as seen from here, whatever track the sighting went to (a drifted pose opens a new track)
        s = min(fresh, key=lambda x: float(np.linalg.norm(bowl_xy(x) - planned)))
        t['bowl_xy'] = bowl_xy(s)
    t['bowl_seen'] = bool(fresh)
    c.change('bowl_pre', f"bowl {t['bowl']} {'seen' if fresh else 'NOT seen'} at {np.round(target_xy(c, w), 3).tolist()} "
             f"({np.linalg.norm(target_xy(c, w) - planned):.3f} m from the plan), rim {b.rim:.3f}")


def step_bowl_pre(c, action, q, w: World) -> None:
    t = c.task
    c.tuck_other(action, q, t['side'])
    if not holding(c, q):
        lost(c, 'vegetable lost reaching over the bowl')
        return
    rot = t['carry_rot']
    if c.age == 1:
        now = q[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[t['side']])].copy()
        x, how, lower = reach(c, w, q, w.base, now)
        t['drop'], t['drop_low'] = drop_points(c, w, w.base, lower)
        t['drop_pre'] = drop_pre(c, w, w.base, t['drop'])
        t['reach_how'] = how
        if x is None:
            retry_drop(c, q, w, f'no clear reach over the bowl from the pose reached ({how[:120]})')
            return
        c.servo.target, c.servo.sync, c.servo.cmd = x, True, None
    c.err = c.servo.step(action, q, G.to_robot(w.base, t['drop_pre']), rot)
    if c.err[0] < P.BOWL_ABOVE_TOL_M and c.servo.target is None or c.err[0] < 0.02:
        c.servo.sync, c.servo.target, c.servo.cmd = False, None, None
        c.line0 = c.mid_world(q, t['side'])
        if not np.allclose(t['drop_pre'], t['drop']):
            c.change('bowl_in', f"in front of the shelf over the bowl, {c.err[0]:.3f} m off ({t['reach_how'][:60]})")
            return
        c.change('bowl_lower', f"over the bowl, {c.err[0]:.3f} m off ({t['reach_how'][:60]})")
    elif c.age > P.REACH_TIMEOUT or c.stalled():
        retry_drop(c, q, w, f"drop point not reached, err {c.err[0]:.3f} m")


def step_bowl_in(c, action, q, w: World) -> None:
    """Straight in under the shelf at the walk's height to over the bowl, then down."""
    t = c.task
    c.tuck_other(action, q, t['side'])
    if not holding(c, q):
        lost(c, 'vegetable lost under the shelf')
        return
    if c.line_done(action, q, w, t['drop'], held_rot(c, w.base), P.LIFT_SPEED):
        c.line0 = c.mid_world(q, t['side'])
        c.change('bowl_lower', f"under the shelf, {c.err[0]:.3f} m from the point")


def step_bowl_lower(c, action, q, w: World) -> None:
    """Straight down into the bowl until the vegetable's bottom is IN_BOWL_M under the rim, or it touches (the error grows)."""
    t = c.task
    c.tuck_other(action, q, t['side'])
    goal = np.r_[c.line0[:2], t['drop_low'][2]]
    done = c.line_to(action, q, w, goal, held_rot(c, w.base), P.LIFT_SPEED)
    if done and c.err[0] < 0.015 or c.err[0] > P.LOWER_CONTACT_M or c.age > np.linalg.norm(goal - c.line0) / P.LIFT_SPEED + 60:
        c.grip[t['side']] = K.GRIPPER_OPEN            # the native assisted grasp lets go only at the fully open command
        c.change('veg_release', f"{np.linalg.norm(c.mid_world(q, t['side']) - c.line0):.3f} m down, err {c.err[0]:.3f}")


def step_veg_release(c, action, q, w: World) -> None:
    t = c.task
    c.servo.hold(action, q)
    c.tuck_other(action, q, t['side'])
    if c.age >= P.RELEASE_WAIT:
        c.line0 = c.mid_world(q, t['side'])
        c.in_bowl[t['bowl']] = c.in_bowl.get(t['bowl'], 0) + 1
        c.counts['released'] = c.counts.get('released', 0) + 1
        c.change('veg_retract', 'hand up out of the bowl')


def step_veg_retract(c, action, q, w: World) -> None:
    t = c.task
    c.tuck_other(action, q, t['side'])
    b = bowl(c, w)
    top = min(max(c.line0[2] + P.VEG_RETRACT_M, b.rim + P.RETRACT_OVER_RIM_M), max(c.line0[2], ceiling(c, b, target_xy(c, w))))
    top = max(top, b.rim + P.STOW_OVER_RIM_M)     # out of the bowl before the hand goes back: under a low ceiling the hand stayed in bowl_1
    # of 302 and the move back dragged the bowl 0.44 m (native t20f14)
    if c.line_done(action, q, w, np.r_[c.line0[:2], top], held_rot(c, w.base), P.LIFT_SPEED * 2):
        c.perc.held_cut = None
        c.line0 = c.mid_world(q, t['side'])
        c.change('veg_stow', f"{t['kind']} released over bowl {t['bowl']}")


def step_veg_stow(c, action, q, w: World) -> None:
    """The empty hand on a straight line to STOW_AT (close beside the torso, over the counter's height), then both arms tucked; the base
    waits for it (a hand tucking from over a bowl, or a base turning with the arm out, pushed bowl_2 of 302 0.65 m and its broccoli fell
    into the sink: native t20f11; the oracle found the same)."""
    t = c.task
    if t.get('stowed') is None:      # a straight line to close beside the torso, over the counter's height: the joint path of the tuck from
        side_y = P.STOW_AT[1] if t['side'] == 'left' else -P.STOW_AT[1]    # over the counter swung the hand down into bowl_2 (t20f15)
        here = G.to_robot(w.base, c.line0)
        goal = G.to_odom(w.base, np.array([min(here[0], P.STOW_AT[0]), side_y, max(here[2], P.STOW_AT[2])]))
        c.tuck_other(action, q, t['side'])
        if c.line_done(action, q, w, goal, held_rot(c, w.base), P.LIFT_SPEED * 3) or c.age > P.STOW_MAX:
            t['stowed'] = c.age
        return
    if c.tucked(c.to_rest(action, q)) or c.age - t['stowed'] > P.STOW_MAX:
        c.change('plan_veg', f"{t['kind']} released over bowl {t['bowl']}, arm tucked")


def put_down(c, q, w: World, why: str) -> None:
    """No stand reaches the bowl: lower the vegetable before letting go, so it lies where it can be picked again (let go at the carry height,
    an onion landed on the shelf over the counter, oracle native 301)."""
    t = c.task
    mid = c.mid_world(q, t['side'])
    t['put_at'] = np.r_[mid[:2], P.PUT_FLOOR_M]
    c.line0 = mid
    c.change('veg_putdown', why + ': putting it down')


def step_veg_putdown(c, action, q, w: World) -> None:
    t = c.task
    c.tuck_other(action, q, t['side'])
    if c.line_done(action, q, w, t['put_at'], held_rot(c, w.base), P.LIFT_SPEED * 2) or c.err[0] > 0.05:
        c.grip[t['side']] = K.GRIPPER_OPEN
        c.counts['put_down'] = c.counts.get('put_down', 0) + 1
        c.perc.held_cut = None
        c.line0 = c.mid_world(q, t['side'])
        c.change('putdown_wait', 'let go low')


def step_putdown_wait(c, action, q, w: World) -> None:
    t = c.task
    c.servo.hold(action, q)
    c.tuck_other(action, q, t['side'])
    if c.age >= P.RELEASE_WAIT:
        c.change('plan_veg', 'vegetable put down')


def distinct(w: World):
    from .pick import distinct_bowls
    return distinct_bowls(w)


def lost(c, why: str) -> None:
    from .pick import veg_fail
    c.counts['lost'] = c.counts.get('lost', 0) + 1
    veg_fail(c, why)


PHASES = {'plan_drop': step_plan_drop, 'go_bowl': step_go_bowl, 'look_bowl': step_look_bowl, 'bowl_pre': step_bowl_pre, 'bowl_in': step_bowl_in,
          'bowl_lower': step_bowl_lower, 'veg_release': step_veg_release, 'veg_retract': step_veg_retract, 'veg_stow': step_veg_stow,
          'veg_putdown': step_veg_putdown,
          'putdown_wait': step_putdown_wait}
