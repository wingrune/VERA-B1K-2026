"""Scrubbing the bathroom floor: the task as a sequence of skills.

    look round on the spot for the brush and the dirt (then from lookouts if the brush was not seen);
    take the brush: the base stands so that the hand, squatting beside it, comes down on the brush's middle from above,
    the fingers closing across its width; lifted 1.2 cm and the arm folded in front of the chest;
    for each dirt mark, the nearest first: the base stands so that the hand comes down beside it onto the mark, the
    brush 1.2 cm over the floor (it takes a dirt particle within 2 cm of its bounding box), moved 5 cm to and fro
    along the base's heading, and the arm folded again for the way on;
    look round again: marks still there or newly seen are scrubbed in turn, till none is left or time runs out.

Poses are (x, y, yaw) in the map frame M.
"""
import numpy as np
from manip import hold, drive, move_to, BASE_Z
from car_skills import tuck
from car_pipeline import straight, settle, FOOTPRINT
import decor
import sweep
from heading_ways import HeadingWays

wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
# (the squatting body over the base, 0.20 m to either side of its middle line above 0.55 m: sweep's side hold has 0.25)
sweep.HOLDS.setdefault('squat', dict(sweep.HOLDS['side'], torso=(-.45, .46, .20)))
# (from these two the floor where things lie is 0.85-2.4 m away, in the head camera's view, for 97 % of it; the strip
# by the bath, (23.4, 11.3) before, the route planner does not reach, and a brush 0.5 m from the start and from the
# other lookout was never seen, 309)
LOOKOUTS = ((22.3, 11.0), (21.0, 10.2), (21.9, 10.8))
SIDES = ('right', 'left')
HANDS = ((0., .55), (.15, .50), (-.15, .50), (0., .45), (.25, .40), (0., .65), (.15, .62), (-.15, .62))   # where the squatting hand comes down beside the base (base frame, its side +y)
REACH = ((.50, .05), (.45, .15), (.55, .0), (.40, .05), (.35, .10))     # where the hand comes down ahead of the base, the trunk bent over (narrow places: by the toilet)
# (the bent trunk's box ahead of the base for each reach: as far again as the hand and 0.45 m beyond it, 0.30 m to
# either side; the broom's forward hold, 1.06 m for a hand 0.5 m ahead, kept every stance off the floor by the
# toilet, where the arm planner's own check of the body against the walls is what tells)
for _ahead in sorted({r[0] for r in REACH}):
    sweep.HOLDS.setdefault(f'reach_{_ahead:.2f}', dict(sweep.HOLDS['forward'], torso=(.24, _ahead + .45, .30)))
HOVER = .012                        # the brush's bottom this high over the floor while scrubbing
BRUSH_TOP = .03                     # the brush's back is this high; the fingers' ends take it 1 cm below
STROKE = .05                        # to and fro along the base's heading, each way


def in_base(base, xy):
    x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = np.asarray(xy, float)[:2] - [x, y]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


def turn_to(ctx, est, yaw):
    here = est.base()
    status = yield from drive(ctx, here[:2], yaw, direct=True, face_travel=False, tolerance=.08, yaw_tolerance=.04, max_steps=500)
    return status


def look(ctx, est, steps=4):
    yield from hold(ctx, steps)
    return est.look()


def look_round(ctx, est):
    start = est.base()[2]
    for quarter in range(4):
        if quarter:
            yield from turn_to(ctx, est, wrap(start + quarter * np.pi / 2))
        yield from look(ctx, est)


def go(ctx, est, goal, yaw, **options):
    goal = np.asarray(goal, float)[:2]; here = np.asarray(est.base()[:2], float)
    if ctx.scene.path(here, goal) is None:
        ctx.event('route', ok=False, goal=np.round(goal, 2).tolist()); return 'no_route'
    status = yield from drive(ctx, goal, yaw, lookahead=.25, **options)
    return status


def find_brush(ctx, est):
    yield from look_round(ctx, est)
    # (the head camera sees the floor only from 0.75 m out: a brush that the look round on the spot did not show lies
    # near the start in most starts (in 158 of the 160 such of the 280 training starts not held out, offline). Seen
    # from where the most of the floor near the start not yet seen shows: in the bathroom only two such places are in
    # the route planner's reach, (21.3, 10.3) and (22.5, 11.1), between them they show 158 of the 160. The lookouts'
    # look rounds took 800-1200 steps more and missed a brush 0.25 m off one, 301, 302, 312, 313, 314)
    start = np.asarray(est.base()[:2]).copy(); tried = []
    while est.brush is None and len(tried) < 5 and hasattr(est, 'floor_xy'):
        # (the floor near the start first, then all the floor not yet seen: a brush in the nook by the toilet was
        # looked for at the lookouts, 2000 steps, 313)
        unseen = est.floor_xy[(np.linalg.norm(est.floor_xy - start, axis=1) < .9) & ~est.floor_seen]
        if len(unseen) < 10 or len(tried) >= 2:
            unseen = est.floor_xy[~est.floor_seen]
        if len(unseen) < 10:
            break
        here = np.asarray(est.base()[:2]); chosen = None
        for o in vantages(ctx, est, unseen)[:16]:
            if any(float(np.hypot(o[1][0] - t[0], o[1][1] - t[1])) < .3 for t in tried):
                continue
            if float(np.linalg.norm(np.asarray(o[1][:2]) - here)) < .15 or ctx.scene.path(here, np.asarray(o[1][:2])) is not None:
                chosen = o; break
        if chosen is None or chosen[2] < 10:
            break
        _, pose, n = chosen; tried.append(pose)
        ctx.event('brush_search', pose=np.round(pose, 2).tolist(), shows=n, unseen=len(unseen))
        if not (yield from look_from(ctx, est, pose)):
            continue
        yield from look(ctx, est)
        ways = heading_ways(ctx); ci, cj = ways.cell(est.base()[:2])
        for side in (.5, -.5):
            if est.brush is not None:
                break
            # (only where the chassis has room at the new heading, and at the one between)
            inside = 0 <= ci < ways.shape[0] and 0 <= cj < ways.shape[1]
            if not inside or not (ways.free(wrap(pose[2] + side))[ci, cj] and ways.free(wrap(pose[2] + side / 2))[ci, cj]):
                continue
            yield from turn_to(ctx, est, wrap(pose[2] + side)); yield from look(ctx, est)
    left = list(LOOKOUTS)
    while est.brush is None and left:
        here = np.asarray(est.base()[:2]); left.sort(key=lambda p: float(np.linalg.norm(np.asarray(p) - here))); spot = left.pop(0)
        yield from go(ctx, est, spot, est.base()[2], tolerance=.12)
        yield from look_round(ctx, est)
    return est.brush is not None


def posture(ctx):
    """The trunk squatting over the base, not bent out ahead of or behind it (as for the broom: sweep.POSTURE)."""
    names = ctx.collision.names; radii = np.asarray(ctx.collision.radii)
    arms = np.array([n.startswith(('left_arm', 'right_arm')) for n in names])
    trunk = np.array([not n.startswith(('left_', 'right_', 'base')) and 'wheel' not in n for n in names])

    def ok(full, poses):
        c = ctx.collision.centers(poses)
        return ((c[trunk, 0] + radii[trunk]).max() < sweep.POSTURE['trunk'] and (c[arms, 0] + radii[arms]).max() < sweep.POSTURE['arm']
                and (c[trunk, 0] - radii[trunk]).min() > sweep.POSTURE['rear'])
    return ok


FLOOR = (np.array([19.73, 9.71]) + .15, np.array([24.99, 11.97]) - .15)   # where the base's middle can be: the bathroom's floor


# The chassis' balls in the robot's collision model (base frame: x, y, z, radius), the same in every posture.
CHASSIS = np.array([[.10, -.205, .165, .17], [.10, 0., .165, .17], [.10, .205, .165, .17], [-.08, -.155, .175, .185], [-.08, .155, .175, .185],
                    [-.23, 0., .185, .20], [.169, .28, .075, .075], [.169, -.28, .075, .075], [-.327, 0., .075, .075]])


def squat_free(scene, pose, height=.55, step=.04):
    """The squatting trunk's box (sweep.HOLDS['squat']) clear of what stands up from the floor higher than @height.
    What hangs over the floor it passes under: the basins' counter and the shower's head (their columns' tops are all
    the map has of them); counted, they left no stance for a brush by the shower, 307."""
    x, y, yaw = pose; c, s = np.cos(yaw), np.sin(yaw); front, reach, half = sweep.HOLDS['squat']['torso']
    u, v = np.meshgrid(np.r_[np.arange(front, reach, step), reach], np.r_[np.arange(-half, half, step), half], indexing='ij')
    i, j = scene.cell(np.c_[x + c * u.ravel() - s * v.ravel(), y + s * u.ravel() + c * v.ravel()])
    if not scene.inside(i, j).all():
        return False
    return not bool(np.any((scene.chassis_distance[i, j] <= 0.) & (scene.top[i, j] > height)))


def under_nothing(scene, pose):
    """Does the chassis at @pose keep clear of the map's heights as the arm planner sees it? (The map's heights are a
    column's top: the chassis fits under the shower's head and the basins' counter, but for the arm planner it then
    stands inside them and no arm move is valid: four stances by the shower, 317.)"""
    x, y, yaw = pose; c, s = np.cos(yaw), np.sin(yaw)
    centres = np.c_[x + c * CHASSIS[:, 0] - s * CHASSIS[:, 1], y + s * CHASSIS[:, 0] + c * CHASSIS[:, 1], CHASSIS[:, 2] + BASE_Z]
    return bool(np.all(scene.sphere_clearance(centres, CHASSIS[:, 3], ignore_below=np.full(len(CHASSIS), .06)) >= 0.))


def stances(ctx, est, point, side, avoid=(), slacks=(0.,), with_cost=False):
    """Base poses that put the squatting @side hand on the floor at @point (map xy), the nearest drive first:
    [(pose, staging)]."""
    sign = 1. if side == 'left' else -1.; here = np.asarray(est.base()[:2]); out = []
    # (standing in a narrow place, a stance at the heading it has, straight ahead or aside, takes no way out and back
    # in: out of the strip by the bath and in again for the next mark 0.4 m off cost 950 steps a mark, 308)
    here_yaw = float(est.base()[2]); narrow = not ctx.scene.base_free(here, margin=.02)
    yaws = list(np.arange(-np.pi, np.pi, np.pi / 12)) + ([here_yaw] if narrow else [])
    for (ahead, aside), yaw, slack in ((h, y_, k) for k in slacks for h in HANDS + REACH for y_ in yaws):
        # (slack: the brush, 15 cm long across the base's heading in the hand, may come down that far to either side of
        # the point and still cover it: in the corner by the toilet only so was there room, 303, 310)
        bent = (ahead, aside) in REACH
        c, s = np.cos(yaw), np.sin(yaw); fwd = np.array([c, s]); left = np.array([-s, c])
        base = np.asarray(point, float) - ahead * fwd - (sign * aside + slack) * left; pose = (float(base[0]), float(base[1]), float(yaw))
        if any(float(np.hypot(pose[0] - f[0], pose[1] - f[1])) < .05 and abs(wrap(pose[2] - f[2])) < .2 for f in avoid):
            continue
        if not np.all(base > FLOOR[0]) or not np.all(base < FLOOR[1]):
            continue                    # (the map is open outside the bathroom's walls: stances there are not stances)
        if not ctx.scene.footprint_free(*pose, FOOTPRINT, .02) or not (sweep.torso_free(ctx.scene, pose, hold=f'reach_{ahead:.2f}') if bent else squat_free(ctx.scene, pose)):
            continue
        if not under_nothing(ctx.scene, pose):
            continue
        # (from 0.3 m behind if there is room there, else straight to it: where nothing is behind, in the corners,
        # two in three of the places had no stance with the room asked for, 306)
        staging = base - .30 * fwd
        if not ctx.scene.base_free(staging, margin=-.05):
            staging = base
        hand = base + ahead * fwd + sign * aside * left
        # (facing south the head camera looks through the window's and the sliding door's glass into the garden: the
        # scans fit nothing there, and sliding along the strip by the bath so the pose went 0.16-0.28 m off, 317)
        # (all of it only where the base slides in at the stance's heading: where the route planner reaches the
        # staging the base turns there and comes 0.3 m straight in; with the mirrors' 3 the brush in 320 was taken
        # from three stances with no way to it instead of the one that took it)
        slides = not ctx.scene.base_free(staging, margin=.02)
        glass = 3. if np.cos(yaw + np.pi / 2) > np.cos(np.radians(55.)) else 0.
        # (nor north where the mirrors hang, over the basins and over the toilet: the depth camera sees the room again
        # behind them, and those points are left out of the fit, so little is left to fit, 317)
        if np.cos(yaw - np.pi / 2) > np.cos(np.radians(55.)) and (21.6 < base[0] < 24.0 or base[0] < 21.0):
            glass = 3.
        # (into the strip between the bath and the counter the base slides in at the stance's heading: only along the
        # strip, east or west, does the camera keep the room's ends in view; sliding in facing north-north-east the
        # pose went 0.5 m off, 319, and facing north-west it was lost, 317)
        if base[0] > 22.3 and base[1] > 10.70 and abs(np.sin(yaw)) > np.sin(np.radians(35.)):
            glass += 8.
        direct = narrow and abs(wrap(yaw - here_yaw)) < .02 and slide_free(ctx, here, base, yaw)
        # (a short slide at the heading the base already has keeps the pose; the long slide in is what lost it: the
        # penalty drove the base out of the strip and in again for the next mark 0.24 m off, 1000 steps, r8 308, 319)
        glass = glass if slides and not direct else 0.
        out.append((float(np.linalg.norm(staging - here)) + .3 * abs(wrap(yaw - est.base()[2])) + .5 * (HANDS + REACH).index((ahead, aside)) + (2. if bent else 0.) + 3. * abs(slack) + glass - (4. if direct else 0.),
                    pose, (float(staging[0]), float(staging[1])), bent, hand))
    out.sort(key=lambda o: o[0])
    if with_cost:
        return [(cost, pose, staging, bent, hand) for cost, pose, staging, bent, hand in out]
    return [(pose, staging, bent, hand) for _, pose, staging, bent, hand in out]


WAYS = {}


def slide_free(ctx, a, b, yaw):
    """Is the straight slide from @a to @b (map xy) at the heading @yaw clear for the chassis all along? (The cells
    near the start count with the loose margin: where the chassis stands it fits.)"""
    ways = heading_ways(ctx); free = ways.free(yaw); loose = heading_ways(ctx, loose=True).free(yaw)
    a = np.asarray(a, float)[:2]; b = np.asarray(b, float)[:2]; n = max(2, int(np.linalg.norm(b - a) / .02) + 1)
    for t in np.linspace(0., 1., n):
        i, j = ways.cell(a + t * (b - a))
        if not (0 <= i < ways.shape[0] and 0 <= j < ways.shape[1]):
            return False
        if not (free[i, j] or (t * float(np.linalg.norm(b - a)) < .08 and loose[i, j])):
            return False
    return True


def heading_ways(ctx, loose=False):
    key = 'loose' if loose else 'bathroom'
    if key not in WAYS:
        WAYS[key] = HeadingWays(ctx.scene, FOOTPRINT, (19.6, 9.6), (25.1, 12.1), margin=-.015 if loose else .01)
    return WAYS[key]


MIDDLE = np.array([21.9, 10.8])     # the main room's middle, where the route planner gets from everywhere it reaches


def to_open(ctx, est, reach=.6):
    """Where the base stands too near something for the route planner's disc (after a squat by the bath): slide
    straight, at the heading it has, to the nearest place where the disc stands. Returns True if there or already so."""
    here = np.asarray(est.base()[:2]); yaw = est.base()[2]
    # (where the disc stands but the planner gets out from nowhere: in the nook by the toilet the disc fits, the
    # nook's mouth it does not, 310)
    if ctx.scene.base_free(here, margin=.02) and ctx.scene.path(here, MIDDLE) is not None:
        return True
    ways = heading_ways(ctx); free = ways.free(yaw)
    # (out of the nook by the toilet, or the strip by the bath, the way it came: at the heading it has, to the
    # nearest place where it can turn; straight within 0.6 m there was none, and every stance after was out of
    # reach, 310)
    # (the chassis' own cell counts as free with a little less room than the planner keeps: where it stands it fits)
    # (the way keeps the planner's room but for the start: with less room all along, the slide out of the nook cut the
    # wall's corner and stalled on it, five times over, 301)
    legs = ways.escape(here, yaw, accept=lambda xy: ctx.scene.path(np.asarray(xy), MIDDLE) is not None, start_free=heading_ways(ctx, loose=True).free(yaw))
    if legs:
        ctx.event('to_open', legs=[np.round(w, 2).tolist() for w in legs])
        for w in legs:
            status = yield from straight(ctx, w, yaw, speed=.2, tolerance=.03)
            if status != 'arrived':
                break
        if ctx.scene.base_free(np.asarray(est.base()[:2]), margin=.0):
            return True
    for r in np.arange(.05, reach + 1e-6, .05):
        for a in np.arange(0., 2 * np.pi, np.pi / 8):
            goal = here + r * np.array([np.cos(a), np.sin(a)])
            if not ctx.scene.base_free(goal, margin=.04):
                continue
            # (the chassis clear all along the slide at this heading)
            if all(free[ways.cell(here + t * (goal - here))] for t in np.linspace(0., 1., max(2, int(r / .02)))):
                ctx.event('to_open', goal=np.round(goal, 2).tolist())
                status = yield from straight(ctx, goal, yaw, speed=.12, tolerance=.03)
                return status == 'arrived'
    ctx.event('to_open', ok=False)
    return False


def come_to(ctx, est, pose, staging):
    """To @staging, turned to @pose's heading, then straight in to @pose. Returns True if there.
    Where the route planner does not reach @staging (the nook by the toilet, the strip between the bath and the
    counter, the shower), the base turns to @pose's heading where it has room to and slides in at that heading."""
    here = est.base()
    if not ctx.scene.base_free(np.asarray(here[:2]), margin=.02) and abs(wrap(pose[2] - here[2])) < .02 and slide_free(ctx, here[:2], pose[:2], pose[2]):
        # (in a narrow place already at the stance's heading: straight there)
        ctx.event('slide', goal=np.round(pose, 2).tolist())
        status = yield from straight(ctx, pose[:2], pose[2], speed=.15, tolerance=.02)
        yield from settle(ctx, est, 4)
        return float(np.hypot(est.base()[0] - pose[0], est.base()[1] - pose[1])) < .06
    yield from to_open(ctx, est)
    here = np.asarray(est.base()[:2])
    if ctx.scene.path(here, np.asarray(staging)) is None:
        options = heading_ways(ctx).plan(here, pose[:2], pose[2])
        for start, legs in options:
            if float(np.linalg.norm(start - here)) > .15 and ctx.scene.path(here, start) is None:
                continue
            ctx.event('heading_way', start=np.round(start, 2).tolist(), legs=[np.round(w, 2).tolist() for w in legs], yaw=round(float(pose[2]), 3))
            if float(np.linalg.norm(start - here)) > .15:
                status = yield from go(ctx, est, start, pose[2], tolerance=.05)
            yield from turn_to(ctx, est, pose[2])
            for p_ in list(legs) + [np.asarray(pose[:2], float)]:
                status = yield from straight(ctx, p_, pose[2], speed=.22, tolerance=.02)
                # (sliding sideways the pose drifts: 0.14 m and 6 degrees along the strip by the bath, 317; the
                # estimator puts it right standing still)
                yield from settle(ctx, est, 4)
                if status != 'arrived':
                    break
            yield from turn_to(ctx, est, pose[2])
            status = yield from straight(ctx, pose[:2], pose[2], speed=.10, tolerance=.02)
            return float(np.hypot(est.base()[0] - pose[0], est.base()[1] - pose[1])) < .06
        ctx.event('route', ok=False, goal=np.round(pose, 2).tolist(), reason='no way at the heading'); return False
    status = yield from go(ctx, est, staging, pose[2], tolerance=.05 if float(np.hypot(staging[0] - pose[0], staging[1] - pose[1])) > .01 else .03)
    if status != 'arrived' and float(np.hypot(est.base()[0] - pose[0], est.base()[1] - pose[1])) > .25:
        return False
    yield from turn_to(ctx, est, pose[2]); yield from settle(ctx, est, 4)
    status = yield from straight(ctx, pose[:2], pose[2], speed=.15, tolerance=.02)
    return status == 'arrived'


def closing_across(direction, base_yaw):
    """The fingers' closing direction (base frame angle) across a thing lying along @direction (map unit xy)."""
    along = float(np.arctan2(direction[1], direction[0])) - base_yaw
    return (along + np.pi / 2 + np.pi / 2) % np.pi - np.pi / 2


def take_brush(ctx, est):
    """Take the brush by its back from above with the hand that has a stance. Returns the hand, or None."""
    failed = []; tries = 0
    for attempt in range(12):
        b = est.brush
        if b is None or tries >= 4:
            return None
        # (either hand, the cheaper stance first)
        options = sorted(((st[0], side, st[1:]) for side in SIDES for st in stances(ctx, est, b['xy'], side, failed, with_cost=True)[:3]), key=lambda o: o[0])
        options = [(side, st) for _, side, st in options]
        for k, (side_, st) in enumerate(options[:4]):
            if reachable(ctx, side_, st[0], b['xy'], BRUSH_TOP - .01 - BASE_Z, decor.top_rotation(closing_across(b['direction'], st[0][2])), st[2]):
                options = options[k:] + options[:k]; break
        if not options:
            ctx.event('stance', ok=False, thing='brush'); return None
        side, (pose, staging, bent, _) = options[0]
        ctx.event('stance', thing='brush', arm=side, pose=np.round(pose, 3).tolist())
        here0 = np.asarray(est.base()[:2]).copy()
        ok = yield from come_to(ctx, est, pose, staging)
        if not ok:
            # (a pinch from 0.3 m off the stance finds no way and costs as much as a fresh stance, 317)
            ctx.event('brush', ok=False, reason='stance not reached', at=np.round(est.base(), 3).tolist()); failed.append(pose)
            tries += float(np.hypot(est.base()[0] - here0[0], est.base()[1] - here0[1])) > .2      # (a stance the base never set out for costs nothing)
            continue
        tries += 1
        yield from look(ctx, est); b = est.brush
        ctx.arm = side; base = est.base(); local = in_base(base, b['xy'])
        rotation = decor.top_rotation(closing_across(b['direction'], base[2]))
        plan = decor.plan_pinch(ctx, np.r_[local, BRUSH_TOP - .01 - BASE_Z], rotation, way_in=.10, hand_margin=-.03, also=None if bent else posture(ctx), count=64)
        if plan is None:
            ctx.event('brush', ok=False, reason='no way to it', why=why_not(ctx, np.r_[local, BRUSH_TOP - .01 - BASE_Z], rotation, bent), local=np.round(local, 3).tolist()); failed.append(pose); continue
        # (a last look from over it with the wrist camera: from 2 m the head camera's brush was a few degrees and
        # centimetres off, the fingers came down on it and turned it, four times over, 317)
        status = yield from move_to(ctx, plan['pre'], rate=.04, label='brush')
        yield from hold(ctx, 4)
        seen = est.wrist_brush(side, ctx.tip()[:3, 3]) if status == 'done' else None
        if seen is not None:
            centre, axis, length = seen
            closing = (float(np.arctan2(axis[1], axis[0])) + np.pi) % np.pi - np.pi / 2
            better = decor.plan_pinch(ctx, np.r_[centre, BRUSH_TOP - .01 - BASE_Z], decor.top_rotation(closing), way_in=.10, hand_margin=-.03, also=None if bent else posture(ctx), count=64)
            ctx.event('brush_look', shift=np.round(centre - local, 3).tolist(), length=round(length, 3), replanned=better is not None)
            if better is not None:
                plan = better
        else:
            ctx.event('brush_look', seen=False, debug=getattr(est, 'wrist_debug', None))
        ok = yield from decor.pinch(ctx, plan, label='brush', gap=.002, lift=HOVER + .005)
        if ok:
            est.holding_brush = True; est.correct_bent = True
            # (a ball for the brush in the arm planner while it is carried, and cut out of the scans: folded without
            # it, the arm took the brush under the basins' counter and stalled there three times, 320; it is
            # dropped for the way down onto the floor, where it goes on purpose, 301)
            hold_brush(ctx, side)
            ctx.event('taken', thing='brush', arm=side, fingers=ctx.fingers().round(4).tolist())
            ctx.holding = {side: 'brush'}
            return side
        failed.append(pose)
        yield from tuck(ctx, (side,))
        yield from look(ctx, est)
    return None


def reachable(ctx, side, pose, hand, z, rotation, bent):
    """Would the arm planner find the @side hand's way down to @hand (map xy, @z over the base frame's origin) from
    the base at @pose? Asked before the base drives there: stances that pass the map's checks but have no way for
    the arm cost 150-300 steps each (the trunk met the bath's rim, 317)."""
    from manip import pose2
    held = ctx.payload.pop(side, None); arm = ctx.arm; ctx.arm = side
    try:
        plan = decor.plan_pinch(ctx, np.r_[in_base(pose, hand), z], rotation, base=pose2(*pose), way_in=.0001, hand_margin=-.03,
                                also=None if bent else posture(ctx), count=64)
    finally:
        ctx.arm = arm
        if held is not None:
            ctx.payload[side] = held
    return plan is not None


def hold_brush(ctx, side):
    """The brush in @side's hand as one ball for the arm planner (15 x 7 cm, its back in the fingers)."""
    tip = ctx.tip(side)[:3, 3]
    ctx.attach(side, [tip - [0., 0., .01]], [.085])


def fold(ctx, side):
    """The hand with the brush folded in front of the chest, the trunk upright. When the arm stalls on the way (the
    brush, no part of the arm planner's model, met the body: 320) the hand goes up first and it is tried again."""
    status = yield from tuck(ctx, (side,))
    for again in range(2):
        if status == 'done':
            break
        ctx.arm = side
        yield from decor.slide(ctx, [0., 0., .15], rate=.02)
        status = yield from tuck(ctx, (side,))
    return status


def why_not(ctx, point, rotation, bent):
    """Why the arm planner found no way to @point: no IK at all, or the nearest of the robot to the map."""
    sol = ctx.ik.solve(ctx.q, point, rotation, count=32)
    if sol is None:
        return 'no ik'
    q = ctx.q.copy(); q[ctx.ik.columns] = sol; poses = ctx.body.fk(q)[0]
    out = dict(valid=bool(ctx.validator(hand_margin=-.03)(q, poses)), posture=None if bent else bool(posture(ctx)(q, poses)), own=bool(ctx.collision.valid(poses, None)))
    world = ctx.base_matrix(); centres = ctx.collision.centers(poses) @ world[:3, :3].T + world[:3, 3]
    gaps = ctx.scene.sphere_clearance(centres, ctx.collision.radii, ignore_below=np.where(ctx.base_part, .06, 0.))
    out['nearest'] = [(ctx.collision.names[k], round(float(gaps[k]), 3), np.round(centres[k], 2).tolist()) for k in np.argsort(gaps)[:3]]
    return out


def hand_xy(ctx, est):
    base = est.base(); c, s = np.cos(base[2]), np.sin(base[2]); t = ctx.tip()[:3, 3]
    return np.array([base[0] + c * t[0] - s * t[1], base[1] + s * t[0] + c * t[1]])


def scrub_at(ctx, est, side, mark):
    """Bring the held brush down onto @mark (a known dirt dict) and move it to and fro. Returns True if it got there."""
    failed = []; tries = 0
    for attempt in range(8):
        if tries >= 2:
            break
        options = stances(ctx, est, mark['xy'], side, failed, slacks=(0., .05, -.05))[:6]
        for k, (pose_, _, bent_, hand_) in enumerate(options[:4]):
            if reachable(ctx, side, pose_, hand_, HOVER + (BRUSH_TOP - .01) - BASE_Z, decor.top_rotation(0.), bent_):
                options = options[k:] + options[:k]; break
        if not options:
            ctx.event('stance', ok=False, thing='dirt', xy=np.round(mark['xy'], 2).tolist()); mark['unreachable'] = True; return False
        pose, staging, bent, hand = options[0]
        here0 = np.asarray(est.base()[:2]).copy()
        ok = yield from come_to(ctx, est, pose, staging)
        if not ok:
            failed.append(pose); tries += float(np.hypot(est.base()[0] - here0[0], est.base()[1] - here0[1])) > .2; continue
        tries += 1
        ctx.arm = side; base = est.base(); local = in_base(base, hand)
        ctx.detach(side)
        rotation = decor.top_rotation(0.)       # (fingers straight down: the brush lies flat in them as it was taken)
        # (the brush's bottom HOVER over the floor: the hand holds its back 2 cm over its bottom)
        z = HOVER + (BRUSH_TOP - .01) - BASE_Z
        plan = decor.plan_pinch(ctx, np.r_[local, z], rotation, way_in=.0001, hand_margin=-.03, also=None if bent else posture(ctx), count=64)
        if plan is None:
            sol = ctx.ik.solve(ctx.q, np.r_[local, z], rotation, count=32)
            why = 'no ik'
            if sol is not None:
                q = ctx.q.copy(); q[ctx.ik.columns] = sol; poses = ctx.body.fk(q)[0]
                why = dict(valid=bool(ctx.validator(hand_margin=-.03)(q, poses)), posture=bool(posture(ctx)(q, poses)), own=bool(ctx.collision.valid(poses, None)))
                world = ctx.base_matrix(); centres = ctx.collision.centers(poses) @ world[:3, :3].T + world[:3, 3]
                gaps = ctx.scene.sphere_clearance(centres, ctx.collision.radii, ignore_below=np.where(ctx.base_part, .06, 0.))
                why['nearest'] = [(ctx.collision.names[k], round(float(gaps[k]), 3), np.round(centres[k], 2).tolist()) for k in np.argsort(gaps)[:3]]
            ctx.event('dirt', ok=False, reason='no way down', xy=np.round(mark['xy'], 2).tolist(), local=np.round(local, 3).tolist(), why=why,
                      q=np.round(ctx.q[ctx.ik.columns], 2).tolist(), base=np.round(est.base(), 3).tolist())
            failed.append(pose); hold_brush(ctx, side); continue
        # (a mark took 200 steps even 8 cm from the one before: the way down, the strokes and the way up at their former
        # rates (0.03, 0.01, 0.02) were half of it, v1)
        status = yield from move_to(ctx, plan['pre'], rate=.05, label='to_dirt')
        est.hand_xy = hand_xy(ctx, est)
        for shift in ((STROKE, 0., 0.), (-2 * STROKE, 0., 0.), (STROKE, 0., 0.)):
            yield from decor.slide(ctx, shift, rate=.02)
        yield from hold(ctx, 3)
        ctx.event('scrubbed', xy=np.round(mark['xy'], 3).tolist(), hand=np.round(hand_xy(ctx, est), 3).tolist(), status=status)
        if status == 'done' and float(np.linalg.norm(hand_xy(ctx, est) - mark['xy'])) < .04:
            # (scrubbed where it was: taken again only if seen there twice more; from beside it the floor under
            # the hand is out of view, and a mark already gone was scrubbed a second time for 700 steps, 306)
            mark['scrubbed_at'] = est.ob.steps; mark['seen_since'] = 0
            # (and the other marks the brush passed over: 15 cm across the heading, the strokes 5 cm each way along it,
            # the remover's 2 cm round it; a mark 8 cm off the one before was scrubbed again for 240 steps, v1 318)
            yaw_ = est.base()[2]; c_, s_ = np.cos(yaw_), np.sin(yaw_); h_ = hand_xy(ctx, est)
            for other in est.dirt:
                if other is mark or 'scrubbed_at' in other:
                    continue
                r_ = np.asarray(other['xy'], float) - h_; along = c_ * r_[0] + s_ * r_[1]; across = -s_ * r_[0] + c_ * r_[1]
                if abs(along) < STROKE + .035 + .01 and abs(across) < .075 + .01:
                    other['scrubbed_at'] = est.ob.steps; other['seen_since'] = 0
                    ctx.event('scrubbed', xy=np.round(other['xy'], 3).tolist(), by=np.round(mark['xy'], 3).tolist())
        yield from decor.slide(ctx, [0., 0., .10], rate=.04)
        hold_brush(ctx, side)
        est.hand_xy = None
        yield from fold(ctx, side)
        return True
    return False


SURVEY = dict(left=.03, steps=450, grid=.20, least=30)   # look from new places till less than this fraction of the floor is unseen, or for at most so many steps


def vantages(ctx, est, unseen=None, least=10):
    """Base poses to look at the floor not yet seen from (@unseen: these floor points; the dirt bands' points not yet
    seen if not given), the best first: [(score, pose, points it would show)]."""
    from scrub_observed import project, COVER
    cam = getattr(est, 'cam_tall', None)
    if cam is None:
        return []
    unseen = est.cover_xy[~est.covered] if unseen is None else unseen
    if not len(unseen):
        return []
    here = np.asarray(est.base()[:2]); out = []; ways = heading_ways(ctx)
    for x in np.arange(FLOOR[0][0], FLOOR[1][0] + 1e-6, SURVEY['grid']):
        for y in np.arange(FLOOR[0][1], FLOOR[1][1] + 1e-6, SURVEY['grid']):
            if not ctx.scene.base_free(np.array([x, y]), margin=-.02):
                continue
            for yaw in np.arange(-np.pi, np.pi, np.pi / 6):
                # (the chassis has room at that heading there: turning on the spot to look from where it had not, the
                # base stuck on the wall by the toilet while dead reckoning turned the pose by 90 degrees, 309)
                ci, cj = ways.cell((x, y))
                if not (0 <= ci < ways.shape[0] and 0 <= cj < ways.shape[1]) or not ways.free(yaw)[ci, cj]:
                    continue
                u, v, z, ok, rng = project(unseen, (x, y, yaw), cam, est.shape)
                k = np.flatnonzero(ok & (rng < COVER['reach']) & (rng > .8))     # (nearer, the floor is below the picture: a brush 0.72 m off a look was not seen, 304)
                if len(k) < least:
                    continue
                # (the line of sight from the camera over the map's heights: the bath hides the floor behind it)
                c, s = np.cos(yaw), np.sin(yaw); eye = np.array([x + c * cam[0] - s * cam[1], y + s * cam[0] + c * cam[1]])
                seen = np.ones(len(k), bool)
                for t in (.3, .5, .7, .85, .95):
                    q = eye + t * (unseen[k] - eye); i, j = ctx.scene.cell(q)
                    seen &= ctx.scene.top[np.clip(i, 0, ctx.scene.shape[0] - 1), np.clip(j, 0, ctx.scene.shape[1] - 1)] < cam[2] * (1 - t) - .02
                n = int(seen.sum())
                if n >= least:
                    out.append((n / (1. + float(np.linalg.norm(np.array([x, y]) - here)) / .4), (float(x), float(y), float(yaw)), n))
    out.sort(key=lambda o: -o[0])
    return out


def look_from(ctx, est, pose):
    """To @pose (a place to look from) and turned to its heading: where the chassis cannot turn round on the spot, it
    turns where it can and slides there at that heading. Returns False if there is no such way."""
    here = np.asarray(est.base()[:2]); ways = heading_ways(ctx)
    i, j = ways.cell(pose[:2])
    turnable = 0 <= i < ways.shape[0] and 0 <= j < ways.shape[1] and bool(ways.turnable()[i, j])
    if turnable or float(np.linalg.norm(np.asarray(pose[:2]) - here)) < .15 and abs(wrap(pose[2] - est.base()[2])) < .3:
        if float(np.linalg.norm(np.asarray(pose[:2]) - here)) > .15:
            yield from go(ctx, est, pose[:2], pose[2], tolerance=.10)
        yield from turn_to(ctx, est, pose[2])
        return True
    for start, legs in ways.plan(here, pose[:2], pose[2]):
        if float(np.linalg.norm(start - here)) > .15 and ctx.scene.path(here, start) is None:
            continue
        if float(np.linalg.norm(start - here)) > .15:
            yield from go(ctx, est, start, pose[2], tolerance=.05)
        yield from turn_to(ctx, est, pose[2])
        for p_ in list(legs):
            status = yield from straight(ctx, p_, pose[2], speed=.22, tolerance=.03)
            if status != 'arrived':
                break
        return True
    return False


# (floor no survey showed, where marks lay unseen at the end of whole runs: under the base's start (the head camera sees
# the floor from 0.75 m on), the nook's far end by the toilet, the east end of the strip by the bath; r7/r8 304, 308,
# 309, 310, 316, 317, 319)
HOT = ((19.75, 10.75, 20.75, 11.30), (24.00, 11.35, 24.60, 11.90))
HOT_START = .45                     # ... and this near the base's start


def hot_unseen(est):
    """The dirt bands' points not yet seen in the hot spots."""
    xy = est.cover_xy; un = ~est.covered
    hot = np.zeros(len(xy), bool)
    for x0, y0, x1, y1 in HOT:
        hot |= (xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)
    start = getattr(est, 'start_xy', None)
    if start is not None:
        hot |= np.linalg.norm(xy - start, axis=1) < HOT_START
    return xy[hot & un]


# (looks from inside the narrow places, at the floor no look from the room showed: the nook's far end by the toilet,
# from the nook's mouth facing north (from the room the mark there, 0.8-1.2 m off past the nook's wall, was never seen:
# v1/v2 304, 308, 309); the strip's east end by the bath, from the strip facing east (v2 310)
NOOK = dict(box=(19.75, 10.75, 20.75, 11.30), poses=((20.30, 10.05, np.pi / 2), (20.30, 10.15, np.pi / 2), (20.45, 10.10, np.pi / 2), (20.70, 10.30, np.pi / 2)))
CORNER = dict(box=(24.00, 11.35, 24.60, 11.90), poses=((23.55, 11.30, 0.), (23.65, 11.25, 0.), (23.45, 11.35, 0.)))


def unseen_in(est, box):
    xy = est.cover_xy; x0, y0, x1, y1 = box
    return int(np.sum(~est.covered & (xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)))


def place_look(ctx, est, place, name):
    """A look from the first of @place's poses the base gets to, if its box has floor not yet seen. True if looked."""
    if unseen_in(est, place['box']) < 3:
        return False
    for pose in place['poses']:
        ok = yield from look_from(ctx, est, pose)
        if ok and float(np.hypot(est.base()[0] - pose[0], est.base()[1] - pose[1])) < .15:
            before = unseen_in(est, place['box'])
            yield from look(ctx, est)
            ctx.event('place_look', place=name, pose=np.round(pose, 2).tolist(), unseen=[before, unseen_in(est, place['box'])], dirt=len(est.dirt))
            return True
    ctx.event('place_look', place=name, ok=False)
    return False


def hot_looks(ctx, est, most=3, budget=500):
    """Looks at the hot spots' floor not yet seen, from the places that show most of it (a few points is enough)."""
    started = est.ob.steps; tried = est.__dict__.setdefault('looked_from', [])
    for _ in range(most):
        hot = hot_unseen(est)
        if len(hot) < 3 or est.ob.steps - started > budget:
            break
        options = [o for o in vantages(ctx, est, unseen=hot, least=3) if not any(float(np.hypot(o[1][0] - t[0], o[1][1] - t[1])) < .3 and abs(wrap(o[1][2] - t[2])) < .6 for t in tried)]
        here = np.asarray(est.base()[:2]); chosen = None
        for o in options[:12]:
            if float(np.linalg.norm(np.asarray(o[1][:2]) - here)) < .15 or ctx.scene.path(here, np.asarray(o[1][:2])) is not None:
                chosen = o; break
        if chosen is None:
            ctx.event('hot_look', ok=False, hot=len(hot)); break
        _, pose, n = chosen; tried.append(pose)
        ctx.event('hot_look', pose=np.round(pose, 2).tolist(), shows=n, hot=len(hot))
        if not (yield from look_from(ctx, est, pose)):
            continue
        yield from look(ctx, est)


def survey(ctx, est):
    """Look at the floor from where the most of what is still unseen shows, till little is left unseen."""
    # (the places looked from are kept over all surveys: a later survey went back to the same two headings, which
    # showed nothing new for the 60 points they were to show, four times, 1000 steps, 312)
    tried = est.__dict__.setdefault('looked_from', []); unreachable = set()
    yield from to_open(ctx, est)
    # (the steps from where the slide out of a narrow place ends: that slide took 400 of the 450, 310)
    started = est.ob.steps
    while est.ob.steps - started < SURVEY['steps'] and (~est.covered).mean() > SURVEY['left']:
        # (a place looked from is tried for that heading only: facing east from where it had looked south-west it
        # would have shown the strip by the bath, where the mark left lay, 305)
        options = [o for o in vantages(ctx, est) if not any(float(np.hypot(o[1][0] - t[0], o[1][1] - t[1])) < .3 and abs(wrap(o[1][2] - t[2])) < .6 for t in tried)]
        here = np.asarray(est.base()[:2]); chosen = None; checked = 0
        for o in options:
            # (only where the route planner reaches: the nook by the toilet and the shower's corner it does not, 320)
            key = (round(o[1][0], 2), round(o[1][1], 2))
            if key in unreachable:
                continue
            if float(np.linalg.norm(np.asarray(o[1][:2]) - here)) < .15 or ctx.scene.path(here, np.asarray(o[1][:2])) is not None:
                chosen = o; break
            unreachable.add(key); checked += 1
            if checked >= 12:
                break
        if chosen is not None and chosen[2] < SURVEY['least']:
            chosen = None               # (the last looks showed 12-50 points of floor each, 150 steps apiece, 301)
        if chosen is None:
            ctx.event('survey', ok=False, options=len(options), cam=getattr(est, 'cam_tall', None) is not None, unseen=int((~est.covered).sum()))
            break
        _, pose, n = chosen; tried.append(pose)
        ctx.event('survey', pose=np.round(pose, 2).tolist(), shows=n, unseen=round(float((~est.covered).mean()), 3))
        before = int((~est.covered).sum())
        if not (yield from look_from(ctx, est, pose)):
            continue
        yield from look(ctx, est)
        if before - int((~est.covered).sum()) < .3 * n:
            # (what the map's heights said would show did not: something not in the map stands in the way from there;
            # the places within 0.3 m of it are not tried again either)
            tried += [(pose[0], pose[1], a) for a in np.arange(-np.pi, np.pi, np.pi / 6)]
    ctx.event('stage', done='survey', unseen=round(float((~est.covered).mean()), 3), dirt=[np.round(d['xy'], 2).tolist() for d in est.dirt])


def full_task(ctx, est):
    ctx.holding = {}; est.start_xy = np.asarray(est.base()[:2], float).copy()
    yield from tuck(ctx)
    found = yield from find_brush(ctx, est)
    ctx.event('stage', done='find', brush=found, dirt=[np.round(d['xy'], 2).tolist() for d in est.dirt])
    if not found:
        return 'no brush'
    side = yield from take_brush(ctx, est)
    ctx.event('stage', done='take', ok=side is not None, arm=side)
    if side is None:
        return 'brush not taken'
    yield from fold(ctx, side)
    yield from survey(ctx, est)
    yield from place_look(ctx, est, NOOK, 'nook')
    done = 0; tried = {}
    # (each mark once a round: one that is still there after the look round that ends the round is tried again in
    # the next; tried twice in a row it was there twice for nothing, 301, 303)
    # (when no mark is known any more and there is time, the floor not yet seen is looked at again: three starts
    # ended with 500-1500 steps to spare and a mark never seen in the nook, 306, 310, 318)
    for round_ in range(6):
        pending = lambda d: not d.get('unreachable') and ('scrubbed_at' not in d or d.get('seen_since', 0) >= 2)
        if round_ > 0 and not any(pending(d) for d in est.dirt) and est.ob.steps < 4000 and unseen_in(est, CORNER['box']) >= 3:
            looked = yield from place_look(ctx, est, CORNER, 'corner')
            if looked and any(pending(d) for d in est.dirt):
                continue
        if round_ > 0 and not any(pending(d) for d in est.dirt) and est.ob.steps < 4000 and (~est.covered).mean() > SURVEY['left']:
            unseen = int((~est.covered).sum())
            yield from survey(ctx, est)
            if int((~est.covered).sum()) >= unseen:
                break
        visited = set()
        while True:
            here = np.asarray(est.base()[:2])
            # (in the first round only marks seen in two looks or more: shadows by the squatting robot were seen once
            # each and scrubbed for nothing, 800 steps in 319; one seen once is taken in a later round)
            left = [d for d in est.dirt if d['id'] not in visited and tried.get(d['id'], 0) < 2 and not d.get('unreachable') and (round_ > 0 or d['seen'] >= 2) and ('scrubbed_at' not in d or d.get('seen_since', 0) >= 2)]
            if not left:
                break
            # (marks never scrubbed first, the nearest of them; one scrubbed and seen there again only when none is
            # left: going back for a mark already scrubbed, seen again in the hand's shadow, cost 770 steps and the
            # last real one was not reached, 301)
            # (and those in the strip by the bath together, after the others or first if the base is in it: in and out
            # for each cost 760-1250 steps a mark, v1 301, 304, 308, 309)
            in_strip = lambda xy: xy[0] > 22.4 and xy[1] > 10.75
            mark = min(left, key=lambda d: ('scrubbed_at' in d, in_strip(d['xy']) != in_strip(here), float(np.linalg.norm(d['xy'] - here))))
            visited.add(mark['id']); tried[mark['id']] = tried.get(mark['id'], 0) + 1
            ok = yield from scrub_at(ctx, est, side, mark)
            done += bool(ok)
        if not visited and round_ > 0 and any(pending(d) for d in est.dirt):
            break                       # (marks left that no round took: none will)
        if visited or round_ == 0:
            yield from look_round(ctx, est)
    ctx.event('stage', done='scrub', visits=done, left=[np.round(d['xy'], 2).tolist() for d in est.dirt])
    return f'{done} marks scrubbed'
