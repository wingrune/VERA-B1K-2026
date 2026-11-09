"""Picking one vegetable (the oracle's one-by-one loop, on perception): plan_veg (vegetable, arm and stand), go_veg, look_veg (refresh the
estimate from the stand), veg_pre (open hand above it), veg_look (wrist view from above), veg_descend, veg_close (the native assisted
grasp holds it: the fingers stay apart), veg_lift, veg_raise, veg_carry (tucked in front of the torso for the drive)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import ik
from .. import params as P
from ..world import VegEst, World
from .base import other_side
from . import grasp


def distinct_bowls(w: World) -> List[int]:
    """Up to three bowl tracks, the most seen first, each BOWL_APART_M from those taken (a drifting odometry splits one bowl into tracks)."""
    out: List[int] = []
    for k in sorted(range(len(w.bowls)), key=lambda k: -w.bowls[k].n):
        if w.bowls[k].n >= P.BOWL_MIN_SIGHTINGS and all(np.linalg.norm(w.bowls[k].xy - w.bowls[j].xy) > P.BOWL_APART_M for j in out):
            out.append(k)
    return out[:3]


def assign_bowls(c, w: World) -> None:
    """Group -> bowl track for the groups seen and not yet assigned: the free bowls, the permutation of least total distance from the
    mean of each group's vegetables (any bowl serves any group). Groups already assigned keep their bowl."""
    import itertools
    used = set(c.bowl_of.values())
    bowls = [k for k in distinct_bowls(w) if k not in used and k not in c.no_stand]
    groups = sorted({v.group for v in w.vegs if v.group is not None} - set(c.bowl_of))
    if not bowls or not groups:
        return
    where = {g: np.mean([v.xy for v in w.vegs if v.group == g], axis=0) for g in groups}
    best = None
    for perm in itertools.permutations(bowls, min(len(bowls), len(groups))):
        cost = sum(float(np.linalg.norm(w.bowls[b].xy - where[g])) for g, b in zip(groups, perm))
        if best is None or cost < best[0]:
            best = (cost, perm)
    c.bowl_of.update({g: b for g, b in zip(groups, best[1])})


def bad(c, v: VegEst) -> bool:
    """A spot where a pick failed in the last BAD_SPOT_STEPS controls (the pile may have changed since)."""
    return any(np.linalg.norm(v.xy - b[:2]) < P.BAD_SPOT_M and c.step - b[2] < b[3] for b in c.bad_spots)


def in_bowl(w: World, v: VegEst) -> bool:
    return any(np.linalg.norm(v.xy - b.xy) < P.BOWL_RIM_R_M + 0.05 for b in w.bowls)


def candidates(c, w: World, v: VegEst, q: np.ndarray) -> List[Dict[str, Any]]:
    """(arm, stand, pinch) whose pre-grasp and grasp are reachable from the stand (GPU scan) and whose arm walks clear of the perceived
    occupancy; the current base pose first, then rings round the vegetable facing it; cheapest first."""
    opts = grasp.pinch_options(v, w)[:P.VEG_OPTIONS]
    if not opts:
        c.debug['veg_tally'] = dict(pinches=0)
        return []
    rows = [w.base.copy()]
    for r in P.VEG_STAND_RINGS:
        for bearing in np.arange(P.N_BEARINGS) * 2 * math.pi / P.N_BEARINGS:
            xy = v.xy + r * np.array([math.cos(bearing), math.sin(bearing)])
            if c.free_pose(xy) and c.grid_raw.clearance(xy) >= P.VEG_STAND_CLEAR_M and not stand_barred(c, xy):
                rows.append(np.array([xy[0], xy[1], G.wrap(bearing + math.pi)]))
    stands = np.array(rows[:P.MAX_STANDS]).reshape(-1, 3)
    ignore = [(np.r_[v.xy - 0.35, 0.], np.r_[v.xy + 0.35, v.top + 0.30])]   # the basket and the vegetables round the pinch
    out, tally = [], dict(pinches=len(opts), stands=len(stands), reach=0)
    for side in ('left', 'right'):
        goals = []
        for o in opts:
            goals += [G.pose_T(o['rotation'], o['pre']), G.pose_T(o['rotation'], o['mid'])]
        masks, qs = ik.scan(side, goals, stands, 0.0, K.FINGER_MID_POINT)
        for j, o in enumerate(opts):
            for i in np.flatnonzero(masks[2 * j] & masks[2 * j + 1]):
                tally['reach'] += 1
                here = bool(i == 0)
                d = 0. if here else c.route(stands[i][:2])
                off = v.xy - stands[i][:2]
                why = ('face' if abs(G.wrap(math.atan2(off[1], off[0]) - stands[i][2])) > P.FACE_MAX else
                       'backwards' if min(qs[2 * j, i, 0], qs[2 * j + 1, i, 0]) < 0. else
                       'route' if not np.isfinite(d) else
                       'sight' if not c.sees(stands[i], np.r_[v.xy, v.top]) else
                       'body' if c.body_hits(qs[2 * j, i], side, stands[i], ignore) or c.body_hits(qs[2 * j + 1, i], side, stands[i], ignore) else None)
                if why:
                    tally[why] = tally.get(why, 0) + 1
                    continue
                out.append(dict(side=side, stand=stands[i].copy(), here=here, pinch=o, x_pre=qs[2 * j, i].copy(),
                                cost=d + o['cost'] + P.SIDE_COST[side] + 0.3 * abs(G.wrap(stands[i][2] - w.base[2]))
                                + P.DROP_FAIL_COST * c.bad_drop.get((c.bowl_of.get(v.group), side), 0)))
    out.sort(key=lambda o: o['cost'])
    verified = []                 # the servo walks to solve_near from the tucked rest the robot holds at the stand: check that solution,
    for o in out[:P.PICK_VERIFY]:  # not the scan's (its joints only gate stands; a near solution put the shoulder over the island, t20f12)
        x = near_from_rest(c, o['side'], o['stand'], o['pinch'])
        why = 'near: no IK' if x is None else c.body_hits(x, o['side'], o['stand'], ignore, rest_x(o['side']))
        if why is None:
            verified.append(dict(o, x_near=x))
        else:
            tally['near body'] = tally.get('near body', 0) + 1
    c.debug['veg_tally'] = tally
    return verified


def rest_x(side: str) -> np.ndarray:
    return np.r_[K.rest(side)[:4], P.TUCK_ARM]


def near_from_rest(c, side: str, stand: np.ndarray, pinch: Dict[str, Any]) -> Optional[np.ndarray]:
    """solve_near to the pre-grasp from the tucked rest pose at the base pose `stand`."""
    q = np.zeros(22)
    q[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])] = rest_x(side)
    q[list(K.ARM_DOFS[other_side(side)])] = P.TUCK_ARM
    servo = c.servo if c.servo is not None and c.servo.side == side else ik.make_servo(side, 'mid')
    return servo.solve(q, G.to_robot(stand, pinch['pre']), G.rotation_to_robot(stand, pinch['rotation']))


def step_plan_veg(c, action, q, w: World) -> None:
    if not c.tucked(c.to_rest(action, q)) or c.age < P.LOOK_STEPS:   # no drive before the arms are tucked
        return
    assign_bowls(c, w)
    c.build_grid(w, [v.xy for v in w.vegs] + [b.centre for b in w.baskets])
    left = [v for v in w.vegs if v.group is not None and v.group in c.bowl_of and not bad(c, v) and not in_bowl(w, v)]
    if not left:
        why: Dict[str, int] = {}
        for v in w.vegs:
            k = ('no basket' if v.basket is None else 'no group' if v.group is None else 'no bowl' if v.group not in c.bowl_of else
                 'failed here' if bad(c, v) else 'in a bowl')
            why[k] = why.get(k, 0) + 1
        c.history.append((c.step, 'plan_veg', f'{len(w.vegs)} vegetable tracks, none to move: {why}'))
        unseen = [k for k, b in enumerate(w.baskets) if k not in c.looked and b.n >= P.BASKET_MIN_SIGHTINGS]
        if unseen:
            c.change('plan_view', f'no vegetable to move yet: look into basket {unseen[0]}')
        elif c.counts.get('scans', 0) <= P.MAX_SCANS:
            c.looked = set()                    # look again from somewhere else: what is left in the baskets, and vegetables on the floor
            c.change('survey', 'no vegetable left to move: scan again from elsewhere')   # (301 gave up after one try, t20f18)
        else:
            c.change('done', 'no vegetable left to move')
        return
    left.sort(key=lambda v: float(np.linalg.norm(v.xy - w.base[:2])))
    for v in left[:P.VEG_PLAN_MAX]:
        cands = candidates(c, w, v, q)
        if cands:
            best = cands[0]
            c.task = dict(kind=v.kind, group=v.group, veg_xy=v.xy, veg=v, bowl=c.bowl_of[v.group], tries=0, **best)
            c.servo = ik.make_servo(best['side'], 'mid')
            c.counts['plans'] += 1
            c.change('look_veg' if best['here'] else 'go_veg', f"{v.kind} (group {v.group}) at {np.round(v.xy, 2).tolist()} with the {best['side']} "
                     f"arm, stand {np.round(best['stand'], 2).tolist()}, cost {best['cost']:.2f}, {c.debug.get('veg_tally')}")
            return
        c.bad_spots.append(np.r_[v.xy, c.step, P.NO_STAND_STEPS])   # no stand from this view: tried again soon (all five vegetables of
        # a basket were barred for 1500 controls this way, t20f16)
        c.history.append((c.step, 'plan_veg', f"no stand for {v.kind} at {np.round(v.xy, 2).tolist()}: {c.debug.get('veg_tally')}"))
    c.counts['plan_fail'] = c.counts.get('plan_fail', 0) + 1
    if c.counts['plan_fail'] > P.MAX_PLAN_FAILS:
        c.change('failed', 'no reachable vegetable')


STALL_CTRL = 220          # controls without STALL_GAIN_M of progress (straight or route) end the drive
STALL_GAIN_M = 0.12
NOROUTE_CTRL = 45         # consecutive controls of 'no route' (the base gets no command then) end the drive
STALL_EXEMPT_M = 0.6      # the final approach (creep, yaw alignment) is never cut
STAND_BAR_M, STAND_BAR_STEPS, STAND_BARS_KEPT = 0.4, 3000, 12
REDRIVES = 2              # other stands tried for the same vegetable before veg_fail


def stand_barred(c, xy) -> bool:
    """A stand within STAND_BAR_M of one whose drive made no progress in the last STAND_BAR_STEPS controls."""
    xy = np.asarray(xy, float)[:2]
    return any(np.linalg.norm(xy - b[:2]) < STAND_BAR_M and c.step - b[2] < STAND_BAR_STEPS for b in getattr(c, 'stand_bars', []))


def redrive(c, w: World, q, why: str) -> None:
    """The drive to the stand gave up: remember the stand, replan the same vegetable from where the robot is now (another stand, another
    arm or the pose reached), and only after REDRIVES such tries bar the vegetable (veg_fail)."""
    t = c.task
    if not hasattr(c, 'stand_bars'):
        c.stand_bars = []
    c.stand_bars.append(np.r_[t['stand'][:2], c.step])
    del c.stand_bars[:-STAND_BARS_KEPT]
    c.counts['drive_aborts'] = c.counts.get('drive_aborts', 0) + 1
    n = t.get('redrives', 0)
    if n < REDRIVES:
        v = nearest(w, t['kind'], t['veg_xy'], P.VEG_REFIND_M)
        v = t['veg'] if v is None else v
        c.build_grid(w, [v.xy] + [b.centre for b in w.baskets])
        cands = [o for o in candidates(c, w, v, q) if o['here'] or not stand_barred(c, o['stand'][:2])]
        if cands:
            best = cands[0]
            nt = dict(t)
            nt.pop('prog', None)
            nt.update(best)
            nt.update(veg=v, veg_xy=v.xy, redrives=n + 1)
            c.task = nt
            c.servo = ik.make_servo(best['side'], 'mid')
            c.path, c.stuck_ref = None, None
            c.change('look_veg' if best['here'] else 'go_veg', f"{why}: re-drive {n + 1} for the {v.kind} with the {best['side']} arm, stand "
                     f"{np.round(best['stand'], 2).tolist()}, cost {best['cost']:.2f}, {c.debug.get('veg_tally')}")
            return
        c.history.append((c.step, 'go_veg', f"{why}: no other stand for the {v.kind}: {c.debug.get('veg_tally')}"))
    veg_fail(c, why)


def step_go_veg(c, action, q, w: World) -> None:
    t = c.task
    c.to_rest(action, q)
    goal = t['stand']
    dist, yaw_err = c.navigate(action, w, goal[:2], goal[2])
    if c.arrived(dist, yaw_err):
        c.change('look_veg', 'at the stand')
        return
    r = c.route(goal[:2])
    p = t.get('prog')
    if p is None:
        p = t['prog'] = dict(d=dist, r=r if np.isfinite(r) else float('inf'), at=c.step, noroute=0)
    better = dist < p['d'] - STALL_GAIN_M or (np.isfinite(r) and r < p['r'] - STALL_GAIN_M)
    if better:
        p['d'] = min(p['d'], dist)
        if np.isfinite(r):
            p['r'] = min(p['r'], r)
        p['at'] = c.step
    p['noroute'] = p['noroute'] + 1 if getattr(c, 'reason', '') == 'no route to the base pose' else 0
    if dist > STALL_EXEMPT_M and p['noroute'] > NOROUTE_CTRL:
        redrive(c, w, q, f'no route to the stand for {p["noroute"]} controls, {dist:.2f} m left')
    elif dist > STALL_EXEMPT_M and c.step - p['at'] > STALL_CTRL:
        redrive(c, w, q, f'no progress for {c.step - p["at"]} controls, {dist:.2f} m left')
    elif c.age > c.nav_budget(goal[:2]):
        redrive(c, w, q, f'navigation timeout, {dist:.2f} m left')


def veg_fail(c, why: str) -> None:
    t = c.task
    t['tries'] = t.get('tries', 0) + 1
    c.counts['grasp_fail'] += 1
    c.bad_spots.append(np.r_[t['veg_xy'], c.step, P.BAD_SPOT_STEPS])
    if len(c.bad_spots) > P.BAD_SPOTS_KEPT:
        c.bad_spots.pop(0)
    c.grip[t['side']] = K.GRIPPER_OPEN     # a wrong catch is let go where it is (oracle: a closed hand dragged a leek over a bowl)
    c.perc.held_cut = None
    c.change('plan_veg', why)


def nearest(w: World, kind: str, xy: np.ndarray, within: float) -> Optional[VegEst]:
    best = [v for v in w.vegs if v.kind == kind and np.linalg.norm(v.xy - xy) < within]
    return min(best, key=lambda v: float(np.linalg.norm(v.xy - xy))) if best else None


def step_look_veg(c, action, q, w: World) -> None:
    """Stand still with the head over the vegetable: the estimate is refreshed from here (the odometry drifts on the way)."""
    t = c.task
    c.to_rest(action, q)
    if c.age < P.LOOK_STEPS:
        return
    fresh = [v for v in w.vegs if v.kind == t['kind'] and w.step - v.seen_step <= P.LOOK_STEPS + 5
             and np.linalg.norm(v.xy - t['veg_xy']) < P.VEG_REFIND_M]
    # the planned pinch's vegetable is the fresh one with a pinch nearest the plan (a blob's centre moves when neighbours of its colour
    # merge into it or split from it: 'not seen from the stand' four times on 301, t20f18)
    every = [(float(np.linalg.norm(o['mid'][:2] - t['pinch']['mid'][:2])), k, o) for k, v in enumerate(fresh) for o in grasp.pinch_options(v, w)]
    near = [n for n in every if n[0] < P.VEG_REFIND_M]
    if not near:
        veg_fail(c, f"{t['kind']} not seen from the stand near {np.round(t['veg_xy'], 2).tolist()} ({len(fresh)} fresh tracks, "
                    f"{len(every)} pinches, nearest {min([n[0] for n in every], default=float('nan')):.2f} m)")
        return
    v = fresh[min(near, key=lambda n: n[0])[1]]
    opts = sorted([n[2] for n in near if fresh[n[1]] is v], key=lambda o: float(np.linalg.norm(o['mid'][:2] - t['pinch']['mid'][:2])))
    opts = opts[:P.VEG_OPTIONS]                   # the pinches again from this view, from the base pose reached
    goals = []
    for o in opts:
        goals += [G.pose_T(o['rotation'], o['pre']), G.pose_T(o['rotation'], o['mid'])]
    masks, qs = ik.scan(t['side'], goals, w.base[None], 0.0, K.FINGER_MID_POINT) if opts else (np.zeros((0, 1), bool), None)
    ok = [j for j in range(len(opts)) if masks[2 * j, 0] and masks[2 * j + 1, 0]]
    if not ok:
        veg_fail(c, f"no pinch of the {v.kind} in reach from the base pose reached ({len(opts)} options)")
        return
    j = ok[0]
    pinch = opts[j]
    t.update(veg=v, veg_xy=v.xy, pinch=pinch, x_pre=qs[2 * j, 0].copy(), x_near=near_from_rest(c, t['side'], w.base, pinch))
    c.change('veg_pre', f"{v.kind} seen at {np.round(v.centre, 3).tolist()}, width {v.width:.3f}, pinch {np.round(pinch['mid'], 3).tolist()}")


def step_veg_pre(c, action, q, w: World) -> None:
    t = c.task
    c.tuck_other(action, q, t['side'])
    c.grip[t['side']] = K.GRIPPER_OPEN     # any command below +1 is 'applying' for the assisted grasp (oracle: a half-open hand grabbed a bowl)
    pinch = t['pinch']
    goal, rot = G.to_robot(w.base, pinch['pre']), G.rotation_to_robot(w.base, pinch['rotation'])
    if c.age == 1:      # solve_near seeded with the joints now, then with the trunk upright (the scan's joints only gate stands: AGENTS.md)
        ignore = [(np.r_[t['veg_xy'] - 0.35, 0.], np.r_[t['veg_xy'] + 0.35, pinch['pre'][2] + 0.1])]
        now = q[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[t['side']])]
        upright = np.array(q, float)
        upright[list(K.TRUNK_DOFS)] = K.rest(t['side'])[:4]
        x, hits = None, []
        for name, q0 in (('near', q), ('near upright', upright), ('planned', None)):   # 'planned': the solve_near solution from the tucked
            y = c.servo.solve(q0, goal, rot) if q0 is not None else t.get('x_near')   # rest checked when the stand was chosen
            hit = 'no IK' if y is None else c.body_hits(y, t['side'], w.base, ignore, now)
            if hit is None:
                x = y
                break
            hits.append(f'{name}: {hit}')
        if x is None:
            veg_fail(c, f"no clear reach to the pre-grasp ({'; '.join(hits)[:160]})")
            return
        c.servo.target, c.servo.sync, c.servo.cmd = x, True, None
    c.err = c.servo.step(action, q, goal, rot)
    if c.err[0] < 0.015 and c.err[1] < 0.15:
        c.servo.sync, c.servo.target, c.servo.cmd = False, None, None
        c.perc.wrist = t['side']
        c.change('veg_look', 'above the vegetable: wrist view')
    elif c.age > P.REACH_TIMEOUT or c.stalled():
        veg_fail(c, f"pre-grasp not reached, err {c.err[0]:.3f} m")


def step_veg_look(c, action, q, w: World) -> None:
    """The wrist camera looks down at the vegetable from the pre-grasp: the pinch is taken again from its view (closer, no occlusion by
    the basket rim)."""
    t = c.task
    c.tuck_other(action, q, t['side'])
    c.servo.hold(action, q)
    if c.age < P.WRIST_LOOK_STEPS:
        return
    c.perc.wrist = None
    v = nearest(w, t['kind'], t['veg_xy'], P.VEG_FOLLOW_M)
    how = 'not seen by the wrist'
    if v is not None and w.step - v.seen_step <= P.WRIST_LOOK_STEPS:
        opts = grasp.pinch_options(v, w)
        same = [o for o in opts if (v.kind == 'onion' or abs(float(o['axis'] @ t['pinch']['axis'])) > 0.9)   # the planned slab seen closer
                and np.linalg.norm(o['mid'][:2] - t['pinch']['mid'][:2]) < P.VEG_FOLLOW_M * 0.5]     # (an onion has no long axis)
        how = f'wrist: {len(opts)} options, {len(same)} at the plan'
        if same:
            pinch = min(same, key=lambda o: float(np.linalg.norm(o['mid'][:2] - t['pinch']['mid'][:2])))
            how += f", moved {np.linalg.norm(pinch['mid'] - t['pinch']['mid']):.3f} m"
            t.update(veg=v, veg_xy=v.xy, pinch=dict(pinch, pre=t['pinch']['pre']))
    c.grip[t['side']] = K.GRIPPER_OPEN    # fully open down to the pinch: opened to the top's width + 2 cm a fingertip landed on onions
    c.line0 = c.mid_world(q, t['side'])   # and leeks (native t20f4); the oracle found the same on broccoli and onions
    c.change('veg_descend', f"pinch {np.round(t['pinch']['mid'], 3).tolist()} width {t['pinch']['width']:.3f} ({how})")


def step_veg_descend(c, action, q, w: World) -> None:
    t = c.task
    c.tuck_other(action, q, t['side'])
    if c.walk_done(action, q, w, t['pinch']['mid'], t['pinch']['rotation'], P.APPROACH_SPEED, 0.01, P.DESCEND_WALK_MAX):
        if c.err[0] > P.DESCEND_SHORT_M:       # stopped on something over the vegetable: closing there catches nothing (t20f5: 7, 11 cm)
            veg_fail(c, f"descent stopped {c.err[0]:.3f} m short of the pinch")
            return
        c.grip[t['side']] = K.GRIPPER_CLOSED
        c.change('veg_close', f"at the vegetable, err {c.err[0]:.3f} m")


def step_veg_close(c, action, q, w: World) -> None:
    t = c.task
    c.servo.hold(action, q)
    c.tuck_other(action, q, t['side'])
    if c.age < P.GRIP_WAIT:
        return
    gap = c.finger_gap(q, t['side'])
    if gap > P.HELD_GAP_M:
        T = c.perc.mid_T(q, t['side'])
        c.line0 = T[:3, 3].copy()
        v, pinch = t['veg'], t['pinch']
        # where the vegetable sits in the hand (the assisted grasp holds it rigidly): its centre in the finger-midpoint frame (level with the
        # fingers) and its bottom under them, both from the sighting it was pinched by
        t['held_off'] = T[:3, :3].T @ (np.r_[v.xy, T[2, 3]] - T[:3, 3])
        t['held_below'] = float(max(T[2, 3] - (pinch['top'] - pinch['thick']), 0.01))
        c.perc.held_cut = (t['side'], P.HELD_CUT_M)   # a carried leek (0.31 m) cut 0.10 m round the hand painted walls ahead of the robot on
        # every drive, and the stands by the bowls read as blocked (native t20f6, t20f8): the whole reach of a held vegetable is cut
        c.change('veg_lift', f"fingers {gap:.3f} m apart: held; centre {np.round(t['held_off'], 3).tolist()} in the hand, bottom {t['held_below']:.3f} m under it")
    else:
        veg_fail(c, f'no grasp: fingers {gap:.3f} m apart')


def holding(c, q, t) -> bool:
    return c.finger_gap(q, t['side']) > P.HELD_GAP_M


def step_veg_lift(c, action, q, w: World) -> None:
    t = c.task
    c.tuck_other(action, q, t['side'])
    if not holding(c, q, t):
        veg_fail(c, 'vegetable lost while lifting')
        return
    if c.walk_done(action, q, w, c.line0 + np.array([0., 0., P.VEG_LIFT_M]), t['pinch']['rotation'], P.LIFT_SPEED * 2, 0.03):
        c.line0 = c.mid_world(q, t['side'])
        c.perc.vegs = [x for x in c.perc.vegs if not (x.kind == t['kind'] and np.linalg.norm(x.xy - t['veg_xy']) < P.VEG_FOLLOW_M)]   # in the hand now
        c.change('veg_raise', 'out of the basket')


def step_veg_raise(c, action, q, w: World) -> None:
    """Straight up to the carry height, clear of the counter edge."""
    t = c.task
    c.tuck_other(action, q, t['side'])
    if not holding(c, q, t):
        veg_fail(c, 'vegetable lost while raising')
        return
    top = np.r_[c.line0[:2], P.CARRY_AT[2]]
    if c.walk_done(action, q, w, top, t['pinch']['rotation'], P.LIFT_SPEED * 2, 0.04, 250):
        c.change('veg_carry', 'at the carry height')


def carry_pose(side: str):
    """(trunk + arm joints, finger-midpoint pose 4x4 in the robot frame) of the fixed carry pose of `side` (the other arm tucked)."""
    x = np.asarray(P.CARRY_Q[side], float)
    qq = np.zeros(22)
    qq[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])] = x
    qq[list(K.ARM_DOFS[other_side(side)])] = P.TUCK_ARM
    servo = K.ArmServo(side)
    servo.use_fingers_mid()
    return x, servo.pose(qq)


def step_veg_carry(c, action, q, w: World) -> None:
    """A joint walk to the fixed carry pose (torso yaw 0: the head looks ahead on the drive and at the bowl from the drop stand)."""
    t = c.task
    c.tuck_other(action, q, t['side'])
    if not holding(c, q, t):
        veg_fail(c, 'vegetable lost while tucking it')
        return
    x, T = carry_pose(t['side'])
    if c.age == 1:
        c.servo.target, c.servo.sync, c.servo.cmd = x.copy(), True, None
    c.err = c.servo.step(action, q, T[:3, 3], T[:3, :3])
    if c.servo.target is None or c.age > P.CARRY_WALK_MAX:
        c.servo.sync, c.servo.target, c.servo.cmd = False, None, None
        t['carry_rot'] = T[:3, :3].copy()
        c.hold_q = x.copy()
        c.change('plan_drop', f'vegetable in the carry pose, {c.err[0]:.3f} m off')


PHASES = {'plan_veg': step_plan_veg, 'go_veg': step_go_veg, 'look_veg': step_look_veg, 'veg_pre': step_veg_pre, 'veg_look': step_veg_look,
          'veg_descend': step_veg_descend, 'veg_close': step_veg_close, 'veg_lift': step_veg_lift, 'veg_raise': step_veg_raise,
          'veg_carry': step_veg_carry}
