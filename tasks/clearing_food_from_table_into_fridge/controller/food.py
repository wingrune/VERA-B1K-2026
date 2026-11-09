"""(z02) The half chicken and the half apple pie from their plates on the breakfast table into a tupperware on the bar.

The goal wants both foods in tupperware.n.01_1. Which of the two identical tupperwares that is cannot be seen; on the
public starts 301-310 it is the western one (the smaller x) on 8 of 10, so the foods go into the western one.

Each food is taken from above by a pinch across its narrow side (the pie is 0.17 x 0.09 x 0.04 m, a half circle: 8.7
cm across its middle; the chicken 0.20 x 0.12 x 0.14 m: 8.5-9 cm across 7 cm from its narrower end), found as a group
of not-white points 3.3-20 cm over the table's top (the plates are white and lower), its long way from the group's
principal axis. Then it is carried to the tupperware's stance and let fall into it from 5 cm over the rim.
Privileged trial (dev job food_trial.py, 301, true poses): both pinched and carried; the pie fell into the tupperware
(inside); the chicken, let go over the pie already in it, came to rest on it and was not inside. So the chicken goes
first.
"""
import time
import numpy as np
from manip import hold, move_to, pose2, BASE_Z
from car_skills import tuck, park
import decor
import kitchen as K
from table_view import TableView, groups
from observer import HEAD

FOOD_FROM = K.TABLE['top'] + .033   # (the plates' rims end 2.4 cm over the table's top)
FOOD_TO = K.TABLE['top'] + .20
CHICKEN_TOP = .085                  # a group whose top stands higher than this over the table's top is the chicken (0.16), else the pie (0.05)
LOOK_FROM = (((5.75, 1.45), np.pi), ((4.30, 3.00), -1.2), ((5.75, .55), 2.7))   # the table seen from its east side, 1-2 m off; then from the north and the south-east (the pie, 4 cm high, near the image's lower edge from the first, f 301)
DROP_OVER = .05                     # the food's lowest point this far over the tupperware's rim when let go


def white_mask(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return (r > 150) & (g > 150) & (b > 140) & (np.abs(r - b) < 45)


def food_things(ob, base_pose, image):
    """The foods this look shows on the table: [dict(kind ('chicken', 'pie'), xy, major (unit, map), top, n, ends
    {+1/-1: width 7 cm from the middle along major})]."""
    local = ob.points('head', 1); h = local[:, 2] + BASE_Z
    local = local[(h > FOOD_FROM) & (h < FOOD_TO)]
    if len(local) < 20:
        return []
    local = ob.clear_of_robot(local)
    x, y, yaw = base_pose; c, s = np.cos(yaw), np.sin(yaw)
    xy = np.c_[x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]]; z = local[:, 2] + BASE_Z
    (x0, x1), (y0, y1) = K.TABLE['x'], K.TABLE['y']
    ok = (xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)
    rgb = TableView.colours(ob, local, image)
    ok &= ~white_mask(rgb)
    xy, z = xy[ok], z[ok]
    out = []
    for idx in groups(xy, 15):
        g = xy[idx]; extent = float(max(np.ptp(g, axis=0)))
        if not .06 < extent < .30:
            continue
        centre = (g.max(0) + g.min(0)) / 2; d = g - g.mean(0)
        major = np.linalg.svd(d, full_matrices=False)[2][0]; minor = np.array([-major[1], major[0]])
        a = (g - centre) @ major; top = float(np.percentile(z[idx], 97))
        ends = {}
        for sgn in (1., -1.):
            m = np.abs(a - sgn * .07) < .015
            ends[sgn] = float(np.ptp((g[m] - centre) @ minor)) if m.sum() >= 4 else 9.
        kind = 'chicken' if top - K.TABLE['top'] > CHICKEN_TOP else 'pie'
        out.append(dict(kind=kind, xy=centre, major=major, top=top, n=int(len(idx)), ends=ends, cut=bool(ob.at_border(local[ok][idx]))))
    return out


def grip_of(food, base):
    """Fingers' ends (base frame) and rotation (fingers down, closing across the food's narrow side) for @food."""
    major = np.asarray(food['major'], float); minor = np.array([-major[1], major[0]])
    if food['kind'] == 'chicken':
        sgn = min(food['ends'], key=food['ends'].get)
        xy = np.asarray(food['xy'], float) + sgn * .07 * major; z = food['top'] - .075   # (its ends stand 6-8 cm lower than its top)
    else:
        xy = np.asarray(food['xy'], float); z = food['top'] - .018
    x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = xy - [x, y]
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    closing = float(np.arctan2(minor[1], minor[0])) - yaw
    return np.r_[local, max(z, K.TABLE['top'] + .03) - BASE_Z], decor.top_rotation(closing)


def look_foods(rt_ctx, est, frames=4):
    """Sightings of the foods from where the base stands (a few frames, merged by kind: the nearest sighting wins)."""
    ctx = rt_ctx; seen = {}
    for _ in range(frames):
        yield from hold(ctx, 3)
        base = est.base(); image = est.ob.data[HEAD + 'rgb'][..., :3]
        for f in food_things(est.ob, base, image):
            if f['cut']:
                continue
            f['range'] = float(np.linalg.norm(f['xy'] - np.asarray(base[:2])))
            have = seen.get(f['kind'])
            if have is None or f['n'] > have['n']:
                seen[f['kind']] = f
    return seen


def stances(ctx, food, arms=('right', 'left')):
    """Base poses at the table's ends from which a hand pinches @food: [(arm, pose, end)]."""
    out = []
    for end, spec in K.ENDS.items():
        for order, arm in enumerate(arms):
            aside = -.16 if arm == 'right' else .16
            ideal = food['xy'][0] - (aside if end == 'north' else -aside)
            for bx in sorted(np.arange(K.END_X[0], K.END_X[1] + .001, .05), key=lambda v: abs(v - ideal)):
                pose = (float(bx), spec['y'], spec['yaw'])
                if abs(bx - ideal) > .30 or not ctx.scene.footprint_free(*pose, FOOTPRINT):
                    continue
                far = abs(food['xy'][1] - spec['y'])
                out.append((far + .4 * abs(bx - ideal) + .3 * order, arm, pose, end))
    return [o[1:] for o in sorted(out, key=lambda o: o[0])]


from car_pipeline import FOOTPRINT  # noqa: E402


def take_food(ctx, est, food, PP):
    """Stand at a table end, look again, pinch @food and lift it. Returns arm or None."""
    started = time.time()
    for arm, pose, end in stances(ctx, food):
        if time.time() - started > 40.:
            break
        ctx.arm = arm; point, rot = grip_of(food, pose)
        if decor.plan_pinch(ctx, point, rot, base=pose2(*pose), hand_margin=-.03) is None:
            continue
        ctx.event('food_stance', food=food['kind'], arm=arm, pose=np.round(pose, 3).tolist(), end=end)
        status = yield from PP.approach_end(ctx, est, pose, end)
        if status != 'arrived':
            yield from PP.leave_end(ctx, est, end); continue
        seen = yield from look_foods(ctx, est)
        near = seen.get(food['kind'])
        if near is not None and float(np.linalg.norm(near['xy'] - food['xy'])) < .15:
            food = near
        point, rot = grip_of(food, est.base()); ctx.arm = arm
        plan = decor.plan_pinch(ctx, point, rot, hand_margin=-.03)
        if plan is None:
            ctx.event('food', ok=False, reason='no plan at the stance', food=food['kind']); yield from PP.leave_end(ctx, est, end); continue
        ok = yield from decor.pinch(ctx, plan, label='food_' + food['kind'], gap=.002, lift=.15)
        if not ok:
            ctx.set_grip(1., arm); yield from tuck(ctx, (arm,)); yield from PP.leave_end(ctx, est, end); return None, food
        side = 1. if arm == 'left' else -1.
        yield from park(ctx, [(.42, side * .10, 1.10), (.45, side * .05, 1.15)], [ctx.tip()[:3, :3]], label='carry_food', rate=.012)
        yield from PP.leave_end(ctx, est, end)
        if float(np.sum(ctx.fingers(arm))) <= .002:
            ctx.set_grip(1., arm); yield from tuck(ctx, (arm,)); return None, food
        return arm, food
    ctx.event('food', ok=False, reason='no stance', food=food['kind'])
    return None, food


def drop_into(ctx, est, arm, food, tup, PP):
    """Carry the food in @arm to a stance at tupperware @tup (an estimator item) and let it fall in. Returns True if
    let go over it."""
    item = dict(id=tup['id'], kind='tupperware', xy=np.asarray(tup['xy'], float))
    hang = (food['top'] - K.TABLE['top']) if food['kind'] == 'chicken' else .045
    started = time.time()
    for cost, a2, pose, end in PP.table_options(ctx, est, item, (arm,)):
        if time.time() - started > 40.:
            break
        ctx.arm = arm; local = PP.in_base(pose, item['xy'])
        drop = np.r_[local, K.BAR['top'] + K.TUPPERWARE['height'] + DROP_OVER + hang - BASE_Z]
        rot = ctx.tip()[:3, :3]
        if ctx.ik.solve(ctx.q, drop, rot, count=16, validator=ctx.validator(base=pose2(*pose), hand_margin=-.05)) is None:
            continue
        ctx.event('drop_stance', food=food['kind'], pose=np.round(pose, 3).tolist())
        status = yield from PP.approach_end(ctx, est, pose, end)
        if status != 'arrived':
            yield from PP.leave_end(ctx, est, end); continue
        yield from hold(ctx, 6)
        local = PP.in_base(est.base(), item['xy']); drop = np.r_[local, drop[2]]
        # (from 12 cm higher straight down: moved there at once, the food swept the tupperware 12 cm aside, f 301)
        above = drop + [0., 0., .12]
        sol = ctx.ik.solve(ctx.q, above, rot, count=24, validator=ctx.validator(hand_margin=-.05))
        if sol is None:
            ctx.event('drop', ok=False, reason='no pose at the stance'); yield from PP.leave_end(ctx, est, end); continue
        status = yield from move_to(ctx, sol, rate=.02, label='over_tupperware')
        from skills import interpolate_cartesian
        way = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], drop, rot, step=.02)
        if way is not None:
            from manip import follow
            yield from follow(ctx, way, rate=.01, max_steps=200)
        yield from hold(ctx, 8)
        ctx.set_grip(1., arm)
        yield from hold(ctx, 40)
        yield from tuck(ctx, (arm,))
        yield from PP.leave_end(ctx, est, end)
        ctx.event('drop', ok=True, food=food['kind'], over=status)
        return True
    ctx.event('drop', ok=False, reason='no stance', food=food['kind'])
    return False


def foods_stage(ctx, est, PP):
    """Both foods into the western tupperware (the smaller x). Returns how many were let go over it."""
    tups = sorted(est.known('tupperware'), key=lambda i: float(i['xy'][0]))
    if not tups:
        ctx.event('foods', ok=False, reason='no tupperware known'); return 0
    target = tups[0]; seen = {}
    for spot, yaw in LOOK_FROM:
        status = yield from PP.go(ctx, est, spot, yaw, tolerance=.10)
        yield from PP.turn_to(ctx, est, yaw)
        more = yield from look_foods(ctx, est, frames=6)
        for k, v in more.items():
            if k not in seen or v['n'] > seen[k]['n']:
                seen[k] = v
        if len(seen) >= 2:
            break
    ctx.event('foods_seen', seen={k: dict(xy=np.round(v['xy'], 3).tolist(), top=round(v['top'], 3), n=v['n']) for k, v in seen.items()}, target=np.round(target['xy'], 3).tolist())
    done = 0
    for kind in ('chicken', 'pie'):
        food = seen.get(kind)
        if food is None:
            continue
        arm, food = yield from take_food(ctx, est, food, PP)
        if arm is None:
            continue
        ok = yield from drop_into(ctx, est, arm, food, target, PP)
        done += int(bool(ok))
        if not ok and float(np.sum(ctx.fingers(arm))) > .002:
            ctx.set_grip(1., arm); yield from hold(ctx, 20); yield from tuck(ctx, (arm,))
    return done
