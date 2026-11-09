"""Sweeping the garage: the task as a sequence of skills.

    find the broom on the garage floor (a look round on the spot, then from lookouts);
    stand beside the end of its handle, the base's heading across it with the head to the left; the left hand takes the
    handle from above and lifts the broom a centimetre and a half, lying flat (where the base has no room there, the
    car or a wall being in the way, it stands the other way round and the right hand takes it, the head to the right);
    drive the legs of the sweep plan for that hand (sweep.py) with it, in the order that takes least time (each leg
    either way along its heading): between legs the base goes, keeping its heading, to a spot where it can turn with
    the broom held out to its side, turns, and goes on to the next leg's start.

Poses are (x, y, yaw) in the map frame M.
"""
import time
import numpy as np
from manip import hold, drive, move_to, BASE_Z
from car_skills import tuck
from car_pipeline import straight, settle, FOOTPRINT
import decor
import sweep

wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
LOOKOUTS = ((-5.0, 4.0), (-7.7, 4.0), (-7.7, 7.0), (-3.0, 4.0))
SPEED = .45                         # m/s along a leg (the broom is held off the floor: nothing drags)
TURN_S = 6.                         # about this long a quarter turn with the broom takes, with the settling (s)


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


def broom_keepout(est):
    b = est.broom
    if b is None or getattr(est, 'holding_broom', False):
        return []
    return [(b['head'] + t * b['direction'], .25) for t in np.arange(0., 1.25, .25)]


def go(ctx, est, goal, yaw, **options):
    """drive() to @goal by a planned route round the broom lying on the floor. Returns drive's status or 'no_route'."""
    goal = np.asarray(goal, float)[:2]; here = np.asarray(est.base()[:2], float)
    # (round the broom with as much room as there is: dropped all at once where the full room was not there, the
    # way went over the broom and turned it by 50 degrees, 312)
    for extra in (.35, .25, .15, .05, None):
        if extra is None:
            keep = []
            if ctx.scene.path(here, goal) is None:
                ctx.event('route', ok=False, goal=np.round(goal, 2).tolist()); return 'no_route'
            break
        keep = [(xy, max(0., min(r + extra, float(np.linalg.norm(here - xy)) - .05, float(np.linalg.norm(goal - xy)) - .05))) for xy, r in broom_keepout(est)]
        if ctx.scene.path(here, goal, keepout=keep) is not None:
            break
    status = yield from drive(ctx, goal, yaw, keepout=keep, lookahead=.25, **options)
    return status


def find_broom(ctx, est):
    """Know where the broom is: a look round on the spot, then from the lookouts, nearest first."""
    start = est.base()[2]
    for quarter in range(4):
        if est.broom is not None:
            break
        if quarter:
            yield from turn_to(ctx, est, wrap(start + quarter * np.pi / 2))
        yield from look(ctx, est)
    left = list(LOOKOUTS)
    while est.broom is None and left:
        here = np.asarray(est.base()[:2]); left.sort(key=lambda p: float(np.linalg.norm(np.asarray(p) - here))); spot = left.pop(0)
        status = yield from go(ctx, est, spot, est.base()[2], tolerance=.12)
        start = est.base()[2]
        for quarter in range(4):
            if est.broom is not None:
                break
            yield from turn_to(ctx, est, wrap(start + quarter * np.pi / 2)); yield from look(ctx, est)
    return est.broom is not None


def stance_for(ctx, broom, side='left', hold='side', avoid=()):
    """The base pose by the handle's end for the @hold (sweep.HOLDS): heading across the handle, the head to the
    @side, the grip at that hand; none near the poses in @avoid (where the arm found no way). Returns (pose, staging,
    grip) or None."""
    d = broom['direction']; sign = 1. if side == 'left' else -1.
    yaw = float(np.arctan2(-d[1], -d[0])) if hold == 'along' else float(np.arctan2(d[1], d[0])) + sign * np.pi / 2
    c, s = np.cos(yaw), np.sin(yaw); fwd = np.array([c, s]); left = np.array([-s, c])
    hand_ahead, hand_aside = sweep.HOLDS[hold]['hand']
    if hold == 'side':
        # (the squatting hand reaches the floor 0.2 m behind to 0.1 m ahead of the base's middle: sweep_legs then
        # shifts the legs by where the hand took the handle)
        # (where the squatting arm reaches the floor keeping the posture: whole-body IK, both hands at 0 ahead, the
        # left also 0.1 ahead and 0.2 behind; elsewhere the arm found no way, 304)
        options = [(0., .55), (0., .60)] + ([(.1, .55), (.1, .60), (-.2, .55)] if side == 'left' else [(-.2, .60)])
    else:
        options = [(hand_ahead, hand_aside), (hand_ahead + .05, hand_aside), (hand_ahead - .05, hand_aside), (hand_ahead, hand_aside + .06), (hand_ahead + .05, hand_aside - .06)]
    for ahead, aside in options:
        for along in (0., .05, -.05):
            grip = broom['head'] + (sweep.HOLDS[hold]['along'] + along) * d
            base = grip - ahead * fwd - sign * aside * left; pose = (float(base[0]), float(base[1]), yaw)
            staging = base - .45 * fwd
            if any(float(np.hypot(pose[0] - f[0], pose[1] - f[1])) < .04 and abs(wrap(pose[2] - f[2])) < .1 for f in avoid):
                continue
            if ctx.scene.footprint_free(*pose, FOOTPRINT, .02) and ctx.scene.base_free(staging, margin=-.05) and sweep.torso_free(ctx.scene, pose, hold=hold):
                return pose, (float(staging[0]), float(staging[1])), grip
    return None


def posture(ctx):
    """The condition on the whole body's configurations take_broom keeps: the trunk squatting over the base, not bent
    out ahead of it (sweep.POSTURE)."""
    names = ctx.collision.names; radii = np.asarray(ctx.collision.radii)
    arms = np.array([n.startswith(('left_arm', 'right_arm')) for n in names])
    trunk = np.array([not n.startswith(('left_', 'right_', 'base')) and 'wheel' not in n for n in names])

    def ok(full, poses):
        c = ctx.collision.centers(poses)
        return ((c[trunk, 0] + radii[trunk]).max() < sweep.POSTURE['trunk'] and (c[arms, 0] + radii[arms]).max() < sweep.POSTURE['arm']
                and (c[trunk, 0] - radii[trunk]).min() > sweep.POSTURE['rear'])
    return ok


SIDES = ('right', 'left')            # the hand tried first (the right hand's plan sweeps more starts clean)
LIFT = dict(side=.015, forward=.04)  # how high the broom is held over the floor in each hold


def take_broom(ctx, est, how='side'):
    """Take the broom's handle in the hold @how (sweep.HOLDS) with the hand the base has room for and hold the broom flat
    over the floor (LIFT). (@how: not 'hold', the name of the skill that waits.) Returns the hand that holds it, or None."""
    failed = []
    for attempt in range(3):
        b = est.broom
        if b is None:
            return None
        holds = [how] if how == 'side' else ['forward', 'forward_8', 'forward_6', 'forward_4', 'along']
        options = [(side, h) for h in holds for side in SIDES if stance_for(ctx, b, side, h, failed) is not None]
        if not options:
            ctx.event('stance', ok=False, thing='broom', hold=how); return None
        # (a stance on the far side of the broom's line takes a way round the handle's end, and where there was no room
        # for the keepout the base went round it close and knocked the broom by 0.66 rad, 312: this side first, then
        # the order of the holds)
        here = np.asarray(est.base()[:2]); dd = np.asarray(b['direction'], float); nn = np.array([-dd[1], dd[0]])
        own = float(np.sign((here - np.asarray(b['head'], float)) @ nn))
        def far_side(o):
            return float(np.sign((np.asarray(stance_for(ctx, b, o[0], o[1], failed)[0][:2]) - np.asarray(b['head'], float)) @ nn)) != own
        found = min(options, key=lambda o: (far_side(o), options.index(o)))
        side, how = found; est.hold = how
        pose, staging, grip = stance_for(ctx, b, side, how, failed)
        ctx.event('stance', thing='broom', arm=side, hold=how, pose=np.round(pose, 3).tolist(), grip=np.round(grip, 3).tolist())
        status = yield from go(ctx, est, staging, pose[2], tolerance=.05)
        if status != 'arrived':
            continue
        yield from turn_to(ctx, est, pose[2]); yield from settle(ctx, est, 6)
        yield from look(ctx, est)
        b = est.broom; found = stance_for(ctx, b, side, how, failed) or (pose, staging, grip); pose, staging, grip = found
        status = yield from straight(ctx, pose[:2], pose[2], speed=.15, tolerance=.015)
        yield from hold(ctx, 6)
        ctx.arm = side; base = est.base(); local = in_base(base, grip)
        # (the handle runs from the head's top, 0.16 m up 0.15 m from the head's middle, down to the floor at its end
        # 1.21 m from it: taken nearer the head it stands higher; at its end's height there the fingers closed on
        # nothing: 302, 317)
        along = sweep.HOLDS[how]['along']
        z = (b.get('grip_z') or sweep.handle_z(along) + .017) - .005 if how == 'side' else sweep.handle_z(along) - .005
        handle = float(np.arctan2(b['direction'][1], b['direction'][0])) - base[2]
        closing = (handle + np.pi / 2 + np.pi / 2) % np.pi - np.pi / 2           # (across the handle; either hand the fingers close the same way)
        rotation = decor.top_rotation(closing)
        # (the squat that keeps the posture is one of few whole-body solutions: more seeds; with 24 the arm found
        # none three times over, 317)
        # (the broom is a thing lying about for the arm planner: on its way to the start of the pinch the arm swept the
        # handle round by 11 degrees and the fingers came down on the floor, 302; only the last straight stretch of
        # the hand meets it)
        ctx.obstacles['broom'] = broom_balls(b) + (True,)

        def replan(point, how=how, rotation=rotation):
            return decor.plan_pinch(ctx, np.asarray(point, float), rotation, way_in=.10, hand_margin=-.03, also=posture(ctx) if how == 'side' else None, count=64 if how == 'side' else 24)
        plan = replan(np.r_[local, max(z, .02) - BASE_Z])
        if plan is None:
            ctx.obstacles.pop('broom', None)
            ctx.event('broom', ok=False, reason='no way to the handle'); failed.append(pose); yield from straight(ctx, staging, pose[2], speed=.2); yield from look(ctx, est); continue
        # (the wrist camera, the hand 10 cm over the handle, shows where the handle is and how high: the base stands a
        # few centimetres off, the broom gets knocked on the way in, and the head camera's height at the grip
        # misses by a centimetre; the fingers closed on nothing in 4 of 20 held-out starts, s01)
        across_dir = np.array([np.cos(handle), np.sin(handle)]); seen_top = {}

        def wrist(tip, arm=side, direction=across_dir):
            seen = est.wrist_handle(arm, tip, direction)
            if seen is None:
                return None
            seen_top['z'] = seen[1]
            return seen[0]

        def replan_seen(point):
            z_ = point[2] if 'z' not in seen_top else max(.02, seen_top['z'] - .012) - BASE_Z
            return replan(np.r_[point[:2], z_])
        ctx.obstacles['broom'] = ctx.obstacles['broom'][:2] + (False,)
        # (the hand over where the handle should be; if the wrist camera shows no handle there the broom is not where
        # it was believed (knocked on the way in, or the estimate was off): no pinch in the air, which knocked it
        # further, but a fresh look from 0.6 m back; 4 of the 20 held-out starts of s02 were lost so)
        ctx.set_grip(1.)
        status = yield from move_to(ctx, plan['pre'], rate=.04, label='broom')
        yield from hold(ctx, 6)
        seen = wrist(ctx.tip()[:3, 3]) if status == 'done' else None
        if seen is None:
            ctx.obstacles.pop('broom', None)
            ctx.event('broom', ok=False, reason='no handle under the hand', debug=getattr(est, 'wrist_debug', None))
            failed.append(pose); yield from tuck(ctx, (side,))
            yield from look_from_back(ctx, est)
            continue
        better = replan_seen(np.r_[seen, plan['point'][2]])
        ok = yield from decor.pinch(ctx, better or plan, label='broom', gap=.002, lift=LIFT['side' if how == 'side' else 'forward'], farthest=.15)
        ctx.obstacles.pop('broom', None)
        if ok:
            est.holding_broom = True
            # The broom goes into what the hand holds: its points are then cut out of the scans the localiser matches
            # against the map (without it the pose drifted by 0.6-0.9 m while sweeping).
            tip = ctx.tip()[:3, 3]; base = est.base()
            head = -np.asarray(b['direction'], float); c, s_ = np.cos(base[2]), np.sin(base[2])
            u = np.array([c * head[0] + s_ * head[1], -s_ * head[0] + c * head[1], 0.])
            along = sweep.HOLDS[how]['along']; ts = np.arange(along - 1.21, along + .16, .1)
            pts = [tip + t * u + [0., 0., .06] for t in ts]
            radii = [.20 if t > along - .1 else .08 for t in ts]
            ctx.attach(side, np.array(pts), np.array(radii)); est.correct_bent = True
            est.hand_at = np.array([tip[0], tip[1] * (1. if side == 'left' else -1.)])      # (base frame, the hand's side +y)
            ctx.event('taken', thing='broom', arm=side, hold=how, fingers=ctx.fingers().round(4).tolist())
            return side
        yield from tuck(ctx, (side,))
        yield from straight(ctx, staging, pose[2], speed=.2)
        yield from look(ctx, est)
    return None


# ---- with the broom held ---------------------------------------------------------------------------------------
def pose_free(scene, pose, side='left', hold='side'):
    return scene.footprint_free(*pose, FOOTPRINT, .02) and sweep.broom_free(scene, pose, side=side, hold=hold) and sweep.torso_free(scene, pose, hold=hold)


def segment_free(scene, a, b, yaw, step=.05, side='left', hold='side', skip=0.):
    """Is the way from @a to @b free, leaving out its first @skip metres?"""
    a, b = np.asarray(a, float), np.asarray(b, float); length = float(np.linalg.norm(b - a)); n = max(2, int(np.ceil(length / step)) + 1)
    return all(pose_free(scene, (*(a + t * (b - a)), yaw), side, hold) for t in np.linspace(0., 1., n) if t * length >= skip)


def way(scene, a, b, yaw, side='left', hold='side', skip=0.):
    """Straight, or with one corner along the scene's axes, from @a to @b at heading @yaw with the broom held: a list
    of points to drive to, or None. skip: the first metres of the way are not checked (the base stands where it is)."""
    if segment_free(scene, a, b, yaw, side=side, hold=hold, skip=skip):
        return [tuple(b)]
    for corner in ((b[0], a[1]), (a[0], b[1])):
        if segment_free(scene, a, corner, yaw, side=side, hold=hold, skip=skip) and segment_free(scene, corner, b, yaw, side=side, hold=hold):
            return [corner, tuple(b)]
    return None


def broom_from_hand(ctx, est, side):
    """The broom as it lies (or would lie) on the floor from where the hand holds it now: dict as BroomView.broom."""
    base = est.base(); c, s = np.cos(base[2]), np.sin(base[2]); tip = ctx.tip()[:3, 3]; sign = 1. if side == 'left' else -1.
    grip = np.array([base[0] + c * tip[0] - s * tip[1], base[1] + s * tip[0] + c * tip[1]])
    out_ = np.array([-s, c]) * sign if getattr(est, 'hold', 'forward') != 'along' else np.array([c, s])     # (from the hand towards the head)
    head = grip + sweep.HOLDS[getattr(est, 'hold', 'forward')]['along'] * out_; return dict(head=head, direction=-out_, length=1.36, grip=grip, grip_z=None, n=0, cut=False, range=9.)


def variants(broom, shift=.05, turn=np.radians(4.)):
    """@broom as it may lie: as given, shifted by @shift each way, turned by @turn each way about its head."""
    out = [broom]
    for dx, dy in ((shift, 0.), (-shift, 0.), (0., shift), (0., -shift)):
        out.append({**broom, 'head': np.asarray(broom['head']) + [dx, dy]})
    for a in (turn, -turn):
        c, s = np.cos(a), np.sin(a); d = np.asarray(broom['direction'])
        out.append({**broom, 'direction': np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])})
    return out


def look_from_back(ctx, est, back=.6):
    """Back off along the heading and look at the broom again from where the whole of it is in view."""
    base = est.base(); c, s_ = np.cos(base[2]), np.sin(base[2])
    yield from straight(ctx, np.asarray(base[:2]) - back * np.array([c, s_]), base[2], speed=.2)
    yield from settle(ctx, est, 6)
    old = est.broom
    if est.broom is not None:
        est.broom = dict(est.broom, range=9.)        # (whatever is seen now replaces it)
    # (facing where the broom was last seen, then to either side of that: knocked by the base on the way to the
    # stance it lay turned by 0.66 rad, beside the base, and the look straight ahead kept the old pose for three
    # stances, 312)
    headings = [est.base()[2]]
    if old is not None:
        middle = np.asarray(old['head'], float) + .5 * np.asarray(old['direction'], float)
        to = float(np.arctan2(*(middle - np.asarray(est.base()[:2]))[::-1]))
        headings = [to, wrap(to + .7), wrap(to - .7)]
    for heading in headings:
        if abs(wrap(heading - est.base()[2])) > .1:
            yield from turn_to(ctx, est, heading)
        seen = None
        for _ in range(2):
            seen = yield from look(ctx, est)
            if seen is not None and not seen['cut']:
                break
        if seen is not None and not seen['cut']:
            break
    ctx.event('looked_again', seen=None if est.broom is None else dict(head=np.round(est.broom['head'], 3).tolist(), direction=np.round(est.broom['direction'], 3).tolist()))


def broom_balls(broom):
    """The broom lying on the floor as balls for the arm planner (map): (points N x 3, radii N). The handle runs from
    the head's top (0.16 m up, 0.15 m from its middle) down to the floor at its end."""
    d = np.asarray(broom['direction'], float); n = np.array([-d[1], d[0]]); head = np.asarray(broom['head'], float)
    pts = [np.r_[head + a * d, sweep.handle_z(a)] for a in np.arange(.15, 1.22, .03)]
    radii = [.015] * len(pts)
    pts += [np.r_[head + a * d + b * n, .08] for a in (-.08, .08) for b in np.arange(-.18, .19, .09)]
    radii += [.08] * (len(pts) - len(radii))
    return np.array(pts), np.array(radii)


def broom_cells(broom, shift=np.zeros(2), step=.04):
    """Points (map xy) over the broom lying at @broom moved by @shift: its handle and its head."""
    d = np.asarray(broom['direction'], float); n = np.array([-d[1], d[0]]); head = np.asarray(broom['head'], float) + shift
    pts = [head + t * d for t in np.arange(.15, 1.22, step)]
    pts += [head + a * d + b * n for a in np.arange(-.15, .16, step) for b in np.arange(-.225, .226, step)]
    return np.array(pts)


def pull_stances(ctx, est, distance=.6):
    """Base poses from which a hand pinches the handle near the head (from above) and, the base sliding back the
    handle's way by @distance, pulls the broom out: [(pose, staging, grip, along)], the nearest first."""
    b = est.broom; d = np.asarray(b['direction'], float); n = np.array([-d[1], d[0]]); here = np.asarray(est.base()[:2]); out = []
    lying = broom_cells(b); moved = broom_cells(b, -distance * d)
    i, j = ctx.scene.cell(moved)
    if not ctx.scene.inside(i, j).all() or np.any(ctx.scene.top[i, j] > .05):
        return []                       # (pulled so far the broom would meet something)
    for along in (.25, .35):
        grip = np.asarray(b['head'], float) + along * d
        for dist in (.45, .55):
            for side in (1., -1.):
                for skew in np.radians([0., -30., 30.]):
                    u = side * n; c, s_ = np.cos(skew), np.sin(skew); u = np.array([c * u[0] - s_ * u[1], s_ * u[0] + c * u[1]])
                    base = grip - dist * u; yaw = float(np.arctan2(u[1], u[0])); end = base - distance * d
                    if not (ctx.scene.footprint_free(base[0], base[1], yaw, FOOTPRINT, .02) and ctx.scene.footprint_free(end[0], end[1], yaw, FOOTPRINT, .02)):
                        continue
                    # (the chassis keeps 5 cm off the broom where it stands and where it slides to with it)
                    cc, ss = np.cos(yaw), np.sin(yaw)
                    def inside(points, at):
                        q = points - at; u_ = q @ np.array([cc, ss]); v_ = q @ np.array([-ss, cc])
                        return np.any((u_ > FOOTPRINT[0] - .05) & (u_ < FOOTPRINT[1] + .05) & (np.abs(v_) < FOOTPRINT[2] + .05))
                    if inside(lying, base) or inside(moved, end):
                        continue
                    staging = base - .35 * u
                    if not ctx.scene.base_free(staging, margin=-.05):
                        continue
                    out.append((float(np.linalg.norm(staging - here)) + .3 * float(abs(skew)), (float(base[0]), float(base[1]), yaw), (float(staging[0]), float(staging[1])), grip, along))
    out.sort(key=lambda o: o[0])
    return [o[1:] for o in out]


def pull_out(ctx, est, distance=.6):
    """Where no hold has room by the handle's end (under the car's side, 307, 312): take the handle near the head from
    above, slide the base back the handle's way by @distance pulling the broom out, let it go and look at it again.
    Returns True if the broom was pulled."""
    b = est.broom; d = np.asarray(b['direction'], float)
    for pose, staging, grip, along in pull_stances(ctx, est, distance)[:3]:
        ctx.event('pull', pose=np.round(pose, 3).tolist(), grip=np.round(grip, 3).tolist(), along=along)
        status = yield from go(ctx, est, staging, pose[2], tolerance=.05)
        if status != 'arrived':
            continue
        yield from turn_to(ctx, est, pose[2]); yield from settle(ctx, est, 6)
        status = yield from straight(ctx, pose[:2], pose[2], speed=.12, tolerance=.015)
        yield from hold(ctx, 6)
        base = est.base(); local = in_base(base, grip); z = sweep.handle_z(along) - .005
        handle = float(np.arctan2(d[1], d[0])) - base[2]; rotation = decor.top_rotation((handle + np.pi) % np.pi - np.pi / 2)
        for arm in SIDES:
            ctx.arm = arm
            plan = decor.plan_pinch(ctx, np.r_[local, z - BASE_Z], rotation, way_in=.10, hand_margin=-.03, count=24)
            if plan is not None:
                break
        if plan is None:
            ctx.event('pull', ok=False, reason='no way to the handle'); yield from straight(ctx, staging, pose[2], speed=.2); continue
        seen_top = {}

        def wrist(tip, arm=ctx.arm, direction=np.array([np.cos(handle), np.sin(handle)])):
            seen = est.wrist_handle(arm, tip, direction)
            if seen is None:
                return None
            seen_top['z'] = seen[1]
            return seen[0]

        def replan(point, rotation=rotation):
            z_ = point[2] if 'z' not in seen_top else max(.02, seen_top['z'] - .012) - BASE_Z
            return decor.plan_pinch(ctx, np.r_[point[:2], z_], rotation, way_in=.10, hand_margin=-.03, count=24)
        ok = yield from decor.pinch(ctx, plan, label='pull', gap=.002, lift=.02, look=wrist, replan=replan, debug=lambda: getattr(est, 'wrist_debug', None), farthest=.15)
        if not ok:
            yield from tuck(ctx, (ctx.arm,)); yield from straight(ctx, staging, pose[2], speed=.2); continue
        here = np.asarray(est.base()[:2])
        status = yield from straight(ctx, here - distance * d, pose[2], speed=.10, tolerance=.03)
        ctx.set_grip(1.); yield from hold(ctx, 15)
        yield from decor.slide(ctx, [0., 0., .10], rate=.02)
        yield from tuck(ctx, (ctx.arm,))
        moved = float(np.linalg.norm(np.asarray(est.base()[:2]) - here))
        est.broom = dict(b, head=np.asarray(b['head'], float) - moved * d, grip=np.asarray(b['grip'], float) - moved * d, range=9.)
        ctx.event('pull', ok=True, status=status, moved=round(moved, 3))
        base = est.base(); c, s_ = np.cos(base[2]), np.sin(base[2])
        yield from straight(ctx, np.asarray(base[:2]) - .5 * np.array([c, s_]), base[2], speed=.2)
        yield from settle(ctx, est, 6)
        for _ in range(3):
            seen = yield from look(ctx, est)
            if seen is not None and not seen['cut']:
                break
        return True
    return False


def relocate(ctx, est, side):
    """With the broom in the forward hold (taken where the base had no room beside the handle), carry it, keeping the
    heading, to the nearest spot from where it lies, once put down, with room beside its handle for the side hold;
    put it down there and let go. Returns True if put down so."""
    base = est.base(); yaw = base[2]; c, s = np.cos(yaw), np.sin(yaw)
    # (back first the way the base came: the broom held out ahead and the trunk bent over it often have no room where
    # it took the handle, a wall or the car ahead; from there the way is checked but for its first 0.35 m)
    yield from straight(ctx, np.asarray(base[:2]) - .45 * np.array([c, s]), yaw, speed=.15)
    here = np.asarray(est.base()[:2], float)
    now = broom_from_hand(ctx, est, side); shift = lambda xy: {**now, 'head': now['head'] + (xy - here), 'grip': now['grip'] + (xy - here)}
    spots = sorted(((float(np.linalg.norm(np.array([x, y]) - here)), np.array([x, y])) for x in np.arange(-8.5, -1.9, .15) for y in np.arange(2.8, 5.6, .15)), key=lambda c_: c_[0])
    started = time.time(); chosen = None
    for _, xy in spots:
        if time.time() - started > 20.:
            break
        lying = shift(xy)
        # (with room for the side hold even if it lies a few centimetres or degrees off: put down where there was
        # just room, it had none once looked at again, 304)
        if not all(any(stance_for(ctx, v, other, 'side') is not None for other in SIDES) for v in variants(lying)):
            continue
        if not pose_free(ctx.scene, (xy[0], xy[1], yaw), side, est.hold):
            continue
        to = way(ctx.scene, here, xy, yaw, side, est.hold, skip=.35)
        if to is not None:
            chosen = (xy, to); break
    if chosen is None:
        ctx.event('relocate', ok=False, reason='no spot'); return False
    xy, to = chosen
    ctx.event('relocate', spot=np.round(xy, 2).tolist(), way=len(to))
    status = yield from drive_way(ctx, est, to, yaw, speed=.25)
    yield from settle(ctx, est, 6)
    yield from decor.slide(ctx, [0., 0., -(LIFT['forward'] - .008)], rate=.01)
    ctx.set_grip(1.); yield from hold(ctx, 20)
    lying = broom_from_hand(ctx, est, side)
    ctx.detach(side); est.holding_broom = False; est.correct_bent = False
    yield from decor.slide(ctx, [0., 0., .12], rate=.02)
    yield from tuck(ctx, (side,))
    est.broom = dict(lying, range=9.)
    ctx.event('put_down', thing='broom', status=status, head=np.round(lying['head'], 3).tolist(), direction=np.round(lying['direction'], 3).tolist())
    # (where the hand let it go is 7 cm off where it lies, and the base came in to the side hold so close to the
    # handle's end that it knocked the broom round, 317: looked at from 0.6 m back, where the whole broom is in view)
    base = est.base(); c, s_ = np.cos(base[2]), np.sin(base[2])
    yield from straight(ctx, np.asarray(base[:2]) - .6 * np.array([c, s_]), base[2], speed=.2)
    yield from settle(ctx, est, 6)
    for _ in range(3):
        seen = yield from look(ctx, est)
        if seen is not None and not seen['cut']:
            break
    ctx.event('looked_again', seen=None if seen is None else dict(head=np.round(seen['head'], 3).tolist(), direction=np.round(seen['direction'], 3).tolist(), cut=bool(seen['cut'])))
    return True


def turn_free(scene, xy, start, end, step=np.radians(8.), side='left', hold='side'):
    """Can the base turn at @xy from heading @start to @end (the given way round) with the broom held?"""
    span = end - start; n = max(2, int(np.ceil(abs(span) / step)) + 1)
    return all(pose_free(scene, (xy[0], xy[1], start + t * span), side, hold) for t in np.linspace(0., 1., n))


def turn_plan(scene, here, start, goal_xy, end, side='left', skip=0.):
    """Where to turn from heading @start to @end on the way from @here to @goal_xy: (spot, signed span, way to the
    spot, way on) or None. Spots are tried in the order of the way's length through them."""
    spans = [wrap(end - start)]; spans.append(spans[0] - 2 * np.pi if spans[0] > 0 else spans[0] + 2 * np.pi)
    spots = [(float(np.linalg.norm(np.array([x, y]) - here) + np.linalg.norm(np.array([x, y]) - goal_xy)), (float(x), float(y)))
             for x in np.arange(-8.4, -2.0, .2) for y in np.arange(2.9, 8.2, .2)]
    spots.sort(key=lambda c: c[0])
    for cost, xy in spots[:400]:
        if not pose_free(scene, (xy[0], xy[1], start), side) or not pose_free(scene, (xy[0], xy[1], end), side):
            continue
        for span in spans:
            if not turn_free(scene, xy, start, start + span, side=side):
                continue
            to = way(scene, here, xy, start, side, skip=skip)
            on = way(scene, xy, goal_xy, end, side) if to is not None else None
            if to is not None and on is not None:
                return xy, span, to, on
    return None


def drive_way(ctx, est, points, yaw, speed=SPEED):
    for p in points:
        status = yield from straight(ctx, p, yaw, speed=speed, tolerance=.04)
        if status != 'arrived':
            return status
    return 'arrived'


def turn_with_broom(ctx, est, span, step=1.5):
    start = est.base()[2]; n = max(1, int(np.ceil(abs(span) / step)))
    for k in range(1, n + 1):
        yield from turn_to(ctx, est, wrap(start + span * k / n))


def path_length(points):
    points = np.asarray(points, float); return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def move_time(scene, here, yaw, goal, heading, side='left', skip=0.):
    """About how long (s) the base takes from @here at @yaw to @goal at @heading with the broom held (the ways
    turn_plan and way find, at SPEED, and TURN_S a quarter turn), or None where they find none."""
    here = np.asarray(here, float)
    if abs(wrap(heading - yaw)) > .08:
        found = turn_plan(scene, here, yaw, np.asarray(goal, float), heading, side, skip)
        if found is None:
            return None
        spot, span, to, on = found
        return (path_length([here] + list(to)) + path_length([spot] + list(on))) / SPEED + TURN_S * abs(span) / (np.pi / 2)
    to = way(scene, here, goal, heading, side, skip=skip)
    return None if to is None else path_length([here] + list(to)) / SPEED


MOVES = {}


def moves(plan):
    """The times between the plan's legs (dev/sweep_moves.py made them): {(k, flip, j, flip): s or None}."""
    if plan not in MOVES:
        import json, os
        table = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'priors', 'sweep_moves.json')))[plan]
        MOVES[plan] = {tuple(int(v) for v in key.split(',')): t for key, t in table.items()}
    return MOVES[plan]


def leg_shift(heading, delta, side):
    """How far (map) the base drives off a leg at @heading for the hand holding the broom @delta (base frame, its side
    +y) off sweep.HAND: the broom then sweeps where the plan has it."""
    sign = 1. if side == 'left' else -1.; c, s = np.cos(heading), np.sin(heading)
    return -np.array([c * delta[0] - s * sign * delta[1], s * delta[0] + c * sign * delta[1]])


def order(scene, plan, here, yaw, side='left', only=None, delta=(0., 0.), ends=None, skip=0.):
    """The legs of sweep.PLANS[@plan] (those in @only) in the order, and each the way along its heading, that takes
    least time from the base at @here, heading @yaw; each leg shifted for the hand's @delta (leg_shift). A list of
    (index, heading, from, to); legs that no way reaches are left out."""
    legs = sweep.PLANS[plan]; n = len(legs); keep = [k for k in range(n) if only is None or k in only]
    if ends is None:
        ends = [tuple(np.asarray(p_, float) + leg_shift(h, delta, side) for p_ in (a, b)) for h, a, b in legs]
    table = moves(plan); best = {}
    for k in keep:
        for flip in (0, 1):
            t = move_time(scene, here, yaw, ends[k][flip], legs[k][0], side, skip)
            if t is not None:
                best[(1 << k, k, flip)] = (t + path_length(ends[k]) / SPEED, None)
    full = sum(1 << k for k in keep)
    for mask in range(1, 1 << n):
        if mask & ~full:
            continue
        for k in keep:
            for flip in (0, 1):
                if (mask, k, flip) not in best:
                    continue
                t = best[(mask, k, flip)][0]
                for j in keep:
                    if mask >> j & 1:
                        continue
                    for f in (0, 1):
                        move = table.get((k, flip, j, f))
                        if move is None:
                            continue
                        key = (mask | 1 << j, j, f); v = t + move + path_length(ends[j]) / SPEED
                        if key not in best or v < best[key][0]:
                            best[key] = (v, (mask, k, flip))
    if not best:
        return []
    # (the most legs that can be driven one after another, in the least time)
    key = min(best, key=lambda q: (-bin(q[0]).count('1'), best[q][0])); out = []
    while key is not None:
        _, k, flip = key; a, b = ends[k][flip], ends[k][1 - flip]
        out.append((k, legs[k][0], tuple(a), tuple(b))); key = best[key][1]
    return out[::-1]


def sweep_legs(ctx, est, plan, side='left'):
    """Drive the legs of sweep.PLANS[@plan], the order planned again from where the base is after each leg (a way
    found from the plan's ends was not always there from where the base stopped, 303). Returns how many were swept."""
    done = 0; left = set(range(len(sweep.PLANS[plan])))
    delta = getattr(est, 'hand_at', np.asarray(sweep.HAND)) - np.asarray(sweep.HAND)
    # (the legs shifted by where the hand took the handle, cut to where the base and the broom still have room: a leg
    # shifted 0.19 m on ran the trunk to the wall, and from there no way led on, 301)
    ends = []
    for k, (h, a, b) in enumerate(sweep.PLANS[plan]):
        a, b = (np.asarray(p_, float) + leg_shift(h, delta, side) for p_ in (a, b)); n = max(2, int(np.ceil(np.linalg.norm(b - a) / .05)) + 1)
        free = [t for t in np.linspace(0., 1., n) if pose_free(ctx.scene, (*(a + t * (b - a)), h), side)]
        if len(free) < 2:
            ends.append((a, a)); continue
        a, b = a + free[0] * (b - a), a + free[-1] * (b - a)
        # (and on past either end as far as there is room, up to 0.1 m: a particle 3 cm past a leg's end was left, 309)
        u = (b - a) / max(float(np.linalg.norm(b - a)), 1e-6)
        for sign_, end_ in ((-1., 0), (1., 1)):
            for more in (.10, .075, .05, .025):
                p_ = (a if end_ == 0 else b) + sign_ * more * u
                if segment_free(ctx.scene, a if end_ == 0 else b, p_, h, side=side):
                    a, b = (p_, b) if end_ == 0 else (a, p_); break
        ends.append((a, b))
    while left:
        here = np.asarray(est.base()[:2], float); yaw = est.base()[2]
        legs = order(ctx.scene, plan, here, yaw, side, only=left, ends=ends, skip=.25)
        ctx.event('order', plan=plan, legs=[k for k, *_ in legs], left=sorted(left), hand_off=np.round(delta, 3).tolist())
        if not legs:
            ctx.event('leg', ok=False, legs=sorted(left), reason='no way to any'); break
        k, heading, a, b = legs[0]; left.discard(k)
        if abs(wrap(heading - yaw)) > .08:
            found = turn_plan(ctx.scene, here, yaw, np.asarray(a, float), heading, side, skip=.25)
            if found is None:
                ctx.event('leg', ok=False, leg=k, reason='no turn'); continue
            spot, span, to, on = found
            status = yield from drive_way(ctx, est, to, yaw)
            yield from turn_with_broom(ctx, est, span)
            status = yield from drive_way(ctx, est, on, heading)
        else:
            to = way(ctx.scene, here, a, heading, side, skip=.25)
            if to is None:
                ctx.event('leg', ok=False, leg=k, reason='no way to the start'); continue
            status = yield from drive_way(ctx, est, to, heading)
        status = yield from straight(ctx, b, heading, speed=SPEED, tolerance=.04)
        ctx.event('leg', ok=status == 'arrived', leg=k, status=status, at=np.round(est.base(), 3).tolist())
        done += status == 'arrived'
    return done


def full_task(ctx, est):
    ctx.holding = {}
    yield from tuck(ctx)
    found = yield from find_broom(ctx, est)
    ctx.event('stage', done='find', ok=found, broom=None if est.broom is None else dict(head=np.round(est.broom['head'], 3).tolist(), direction=np.round(est.broom['direction'], 3).tolist()))
    if not found:
        return 'no broom'
    side = yield from take_broom(ctx, est, 'side')
    for pulls in range(2):
        if side is not None or est.broom is None or any(stance_for(ctx, est.broom, sd, h) is not None for h in ('forward', 'forward_8', 'forward_6', 'forward_4', 'along') for sd in SIDES):
            break
        # (no hold has room by the handle's end: pulled out by the handle near the head first, as far as the floor
        # behind it is free, 0.8 m or 0.5; pulled 0.57 m the handle's end still lay under the car's corner, 307)
        pulled = False
        for distance in (.8, .5):
            if pull_stances(ctx, est, distance):
                pulled = yield from pull_out(ctx, est, distance)
                break
        if not pulled:
            break
        side = yield from take_broom(ctx, est, 'side')
    for again in range(3):
        if side is not None or est.broom is None:
            break
        # (no room beside the handle's end for either hand: taken from in front, put down in the open, taken again)
        held_by = yield from take_broom(ctx, est, 'forward')
        if held_by is None:
            # (nor from in front: knocked on the way to the stance, the broom lay where no hold had room; pulled out by
            # the handle near the head, then tried again, 302)
            pulled = False
            for distance in (.8, .5):
                if pull_stances(ctx, est, distance):
                    pulled = yield from pull_out(ctx, est, distance)
                    break
            if not pulled:
                break
            side = yield from take_broom(ctx, est, 'side')
            continue
        if not (yield from relocate(ctx, est, held_by)):
            break
        side = yield from take_broom(ctx, est, 'side')
    ctx.event('stage', done='take', ok=side is not None, arm=side)
    if side is None:
        return 'broom not taken'
    done = yield from sweep_legs(ctx, est, 'default' if side == 'left' else 'right', side)
    ctx.event('stage', done='sweep', legs=done)
    return f'{done} legs swept'
