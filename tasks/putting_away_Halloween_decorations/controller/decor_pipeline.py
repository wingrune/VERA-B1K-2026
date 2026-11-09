"""Putting away the Halloween decorations: the task as a sequence of skills.

    find the two pumpkins, the three candles and the caldron on the living-room floor (look round, then from
    lookouts);
    for each pumpkin and candle, the nearest first (one that lies where a drawer slides out, before that drawer is
    opened): stand where a hand reaches it, the way in chosen from a look at it; take it by its stem or wick from
    straight above; hold it in front of the chest; stand in front of the cabinet; if the drawer its place is in is
    shut, the free hand pulls it out by its lip; let the thing go over its place in the open drawer;
    push the open drawers shut: the hand takes the lip and the base drives in;
    take the caldron by its rim and set it down next to the coffee table.

Poses are (x, y, yaw) in the map frame M. The cabinet stands against the room's east wall, its drawers sliding out
along -x; the robot faces it at yaw 0.
"""
import time
import numpy as np
from manip import hold, drive, pose2, follow, move_to, BASE_Z, TALL_TRUNK
from car_skills import tuck, park, down
from car_pipeline import straight, settle, stance_options, gap, FOOTPRINT, TURN_CLEAR
import decor

wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
other = lambda arm: 'right' if arm == 'left' else 'left'
RADIUS = dict(pumpkin=.09, candle=.065, caldron=.16)        # of the disc round each thing that the chassis keeps out of
ROOM = .10                          # room the chassis keeps to a thing it passes (m), where the way allows
REACH = .56; ASIDE = .16            # where a thing lies in the base frame when a hand takes it: ahead, and to the hand's side
CALDRON_REACH = .46                 # ... the caldron's middle (the hand takes its rim on the near side)
LOOKOUTS = ((4.6, 4.3), (4.6, 6.5), (5.2, 5.4), (4.2, 3.95), (4.2, 6.85))      # where the base looks round from when a thing has not shown
OPEN_FLOOR = (4.6, 5.3)             # the middle of the room's open floor: the base goes there when no stance has a way from where it stands
# (From a lookout between the coffee table's end and the wall no way led anywhere, and every thing had 'no stance'.)
CARRY_SPOTS = ((.36, .16, .82), (.40, .14, .78), (.34, .20, .86))               # where the hand with a thing is held on the way (base frame, height above the floor)
LIFT = dict(pumpkin=.40, candle=.40, caldron=.25)            # how far a thing is lifted straight up after the pinch
GRIP_HEIGHT = dict(pumpkin=.112, candle=.130)               # height of the fingers' ends on the stem, on the wick (held: candle 26 of 28 at 0.128-0.132 with the true wick, pumpkin 15 of 18 at 0.109-0.115)
RIM = dict(height=.215, inset=.125)                         # the caldron's rim: fingers' ends, and how far from its middle the wall is
# The places in the drawers, in order of use: (drawer, map y). The left drawer's main compartment runs from y 4.63 to
# 5.41, the right one's from 3.37 to 4.07: 26 and 38 cm between places for things 10 and 15 cm across.
SLOTS = (('left', 4.77), ('left', 5.03), ('left', 5.29), ('right', 3.54), ('right', 3.92))
DROP_X = 5.90                       # map x of a place: the middle of what a drawer shows when it is out (5.76 to 6.04)
DROP_ABOVE = .06                    # the thing's bottom this far above the drawer's floor when the hand lets it go
DRAWER_STANCE_X = 5.30              # the base's middle in front of the cabinet: its front 5 cm short of a drawer pulled out
DROP_STANCE_X = 5.24                # ... to let a thing go into a drawer that is out (from 5.30 the drawer was nudged 1-2 cm in at every drop)
PUSH = dict(ahead=.62, height=.25)  # the closed hand that pushes a drawer shut: how far ahead of the base, how high
SHUT_STANCE_X = 5.03                # ... to take the lip of a drawer that is out (the arm does not fold closer)
DRAWER_OUT = .287                   # how far out a hand pulls a drawer (its travel is 0.297)
# Where the caldron is set down next to the coffee table (its middle; the table is the same in every start): north of
# it, south of it, east of it, 10 cm from it (NextTo holds up to 18 cm); with the heading of a base that sets it down.
CALDRON_SPOTS = (((4.175, 5.29), np.pi), ((3.52, 6.375), np.pi), ((3.52, 4.21), np.pi), ((3.52, 4.21), np.pi / 2))
# (Not from north of the table facing south: the base's back is then under the wall shelf, which the arm planner's
# height field takes for solid down to the floor, and no way down was found.)


# ---- the things on the floor ------------------------------------------------------------------------------------
def obstacles(est, but=()):
    """The things on the floor as discs: [(xy, radius)], without the item ids in @but; the drawers that are out."""
    out = [(np.asarray(item['xy'], float), RADIUS[item['kind']]) for item in est.known() if item['id'] not in but]
    for side in getattr(est, 'drawers_out', ()):
        y0, y1 = decor.DRAWERS[side]['y']
        out += [(np.array([5.88, y]), .17) for y in np.arange(y0 + .12, y1 - .1, .2)]
    return out


def keepouts(est, here, goal, extra, but=()):
    """Discs the base's middle stays out of on a way from @here to @goal: each thing's disc with the chassis's half
    width and @extra round it; narrowed, not dropped, where the way starts or ends nearer than that."""
    return [(xy, max(0., min(radius + .345 + extra, float(np.linalg.norm(here - xy)) - .03, float(np.linalg.norm(goal - xy)) - .03))) for xy, radius in obstacles(est, but)]


def go(ctx, est, goal, yaw, but=(), **options):
    """drive() to @goal round the things on the floor; the way is planned again when a thing shows that was not
    known, or shows elsewhere. Returns drive's status, or 'no_route'."""
    goal = np.asarray(goal, float)[:2]
    for again in range(5):
        here = np.asarray(est.base()[:2], float); known = {item['id']: np.asarray(item['xy'], float).copy() for item in est.known()}; keep = None
        for extra in (ROOM, .04, 0.):
            keep = keepouts(est, here, goal, extra, but)
            if ctx.scene.path(here, goal, keepout=keep) is not None:
                break
            keep = None
        if keep is None:
            ctx.event('route', ok=False, goal=np.round(goal, 2).tolist()); return 'no_route'
        steps = drive(ctx, goal, yaw, keepout=keep, lookahead=.20 if keep else .35, **options)
        while True:
            try:
                action = next(steps)
            except StopIteration as stop:
                return stop.value
            yield action
            now = {item['id']: item['xy'] for item in est.known()}
            if again < 4 and (any(i not in known for i in now) or any(i in now and float(np.linalg.norm(now[i] - xy)) > .10 for i, xy in known.items())):
                break
        steps.close()
        ctx.event('route', again=True, goal=np.round(goal, 2).tolist())
    return 'no_route'


def clear(est, spot, but=()):
    """May the base stand and turn at map @spot without touching a thing on the floor?"""
    return all(float(np.linalg.norm(np.asarray(spot, float)[:2] - xy)) > TURN_CLEAR + radius for xy, radius in obstacles(est, but))


def turn_to(ctx, est, yaw):
    here = est.base()
    status = yield from drive(ctx, here[:2], yaw, direct=True, face_travel=False, tolerance=.08, yaw_tolerance=.05, max_steps=400)
    return status


def look(ctx, est, steps=4):
    yield from hold(ctx, steps)
    return est.look()


def missing(est):
    """How many things of each kind are neither known nor taken."""
    from decor_observed import COUNT
    return {kind: COUNT[kind] - len(est.known(kind)) - sum(1 for i in est.taken_ids if i[0] == kind) for kind in COUNT}


def all_known(est):
    return not any(v > 0 for v in missing(est).values())


def look_round(ctx, est):
    """Looks a quarter of a turn apart, all the way round, until all six things are known."""
    start = est.base()[2]
    for quarter in range(4):
        if quarter:
            yield from turn_to(ctx, est, wrap(start + quarter * np.pi / 2))
        yield from look(ctx, est)
        if all_known(est):
            return True
    return all_known(est)


def find(ctx, est):
    """Know where the things are: a look round where the base stands, then from the lookouts, the nearest first."""
    yield from look_round(ctx, est)
    ctx.event('find', where='start', known=len(est.known()), missing=missing(est))
    left = list(LOOKOUTS)
    while not all_known(est) and left:
        here = np.asarray(est.base()[:2], float); left.sort(key=lambda p: float(np.linalg.norm(np.asarray(p) - here))); spot = left.pop(0)
        if float(np.linalg.norm(np.asarray(spot) - here)) < .8 or not ctx.scene.base_free(np.asarray(spot), margin=.02) or not clear(est, spot):
            continue
        status = yield from go(ctx, est, spot, float(np.arctan2(spot[1] - here[1], spot[0] - here[0])), tolerance=.10)
        if status == 'no_route':
            continue
        yield from look_round(ctx, est)
        ctx.event('find', where=list(spot), status=status, known=len(est.known()), missing=missing(est))
    return missing(est)


# ---- taking a thing ---------------------------------------------------------------------------------------------
def carry(ctx, arm):
    """The hand with a thing to where it is held on the way: in front of the chest, the trunk upright."""
    side = 1. if arm == 'left' else -1.; ctx.arm = arm
    status = yield from park(ctx, [(x, side * y, z) for x, y, z in CARRY_SPOTS], [down(0.), down(np.pi / 2)], label='carry', rate=.015)       # (slowly: a candle hanging by its wick fell off on the way 2 times of 6 at twice the rate)
    return status


def in_base(base, xy):
    x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = np.asarray(xy, float)[:2] - [x, y]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


def grip(item, base, xy=None):
    """Where the fingers' ends go on @item seen from base pose @base (or at @xy in the base frame, if given), and the
    hand's rotation: (point in the base frame, rotation). The fingers close along the line of sight from the base's
    middle (what the camera's range tells least well is where the fingers have most room)."""
    xy = in_base(base, item['xy']) if xy is None else np.asarray(xy, float); sight = float(np.arctan2(xy[1], xy[0]))
    if item['kind'] == 'caldron':
        along = np.array([np.cos(sight), np.sin(sight)]); return np.r_[xy - RIM['inset'] * along, RIM['height'] - BASE_Z], decor.top_rotation(sight)
    return np.r_[xy, GRIP_HEIGHT[item['kind']] - BASE_Z], decor.top_rotation(sight)


def held_shape(kind, tip):
    """What the hand holds for the arm planner, in the base frame for the fingers' ends at @tip. The caldron hangs
    from its near rim: its middle is RIM['inset'] further from the base than the fingers."""
    if kind == 'caldron':
        away = np.r_[tip[:2] / max(float(np.linalg.norm(tip[:2])), 1e-6), 0.]
        return np.array([tip + RIM['inset'] * away + [0., 0., -.10]]), np.array([.15])
    return np.array([tip + [0., 0., -.06]]), np.array([.085 if kind == 'pumpkin' else .07])


def item_options(ctx, est, item, arms, here):
    """Stances from which a hand of @arms takes @item: [(cost, arm, (x, y, yaw), staging)] in order of choice."""
    xy = np.asarray(item['xy'], float); toward = float(np.arctan2(xy[1] - here[1], xy[0] - here[0]))
    headings = [wrap(toward + (k + 1) // 2 * (1 if k % 2 else -1) * 2 * np.pi / 16) for k in range(16)]
    reach = CALDRON_REACH if item['kind'] == 'caldron' else REACH
    floor = obstacles(est, but=(item['id'],)); out = []
    for order, arm in enumerate(arms):
        offset = (reach, ASIDE if arm == 'left' else -ASIDE)
        out += [(cost + 50. * order, arm, pose[:3], pose[3]) for cost, pose in stance_options(ctx, here, xy, offset, headings, floor, turn_clear=[(xy, RADIUS[item['kind']])])]
    return sorted(out, key=lambda o: o[0])


def take_item(ctx, est, item, arms):
    """Stand where a hand of @arms reaches @item, take it and hold it for the way. Returns (arm or None, why)."""
    failed = []; empty = 0; item_id = item['id']; kind = item['kind']
    for attempt in range(5):
        if item_id not in est.items:
            return None, 'lost'
        item = est.items[item_id]; here = np.asarray(est.base()[:2], float); chosen = None; started = time.time()
        for cost, arm, pose, staging in item_options(ctx, est, item, arms, here):
            if any(np.hypot(pose[0] - f[0], pose[1] - f[1]) < .10 and abs(wrap(pose[2] - f[2])) < .25 for f in failed) or time.time() - started > 30.:
                continue
            if ctx.scene.path(here, staging, keepout=keepouts(est, here, staging, .04, but=(item_id,))) is None:
                continue
            ctx.arm = arm; point, rotation = grip(item, pose)
            if decor.plan_pinch(ctx, point, rotation, base=pose2(*pose)) is not None:
                chosen = (arm, pose, staging); break
        if chosen is None:
            ctx.event('stance', ok=False, thing=kind, failed=len(failed)); return None, 'failed' if failed else 'no_stance'
        arm, (x, y, yaw), staging = chosen; ctx.arm = arm
        ctx.event('stance', thing=kind, arm=arm, pose=np.round([x, y, yaw], 3).tolist(), staging=np.round(staging, 3).tolist(), target=np.round(item['xy'], 3).tolist())
        status = yield from go(ctx, est, staging, yaw, but=(item_id,), tolerance=.04)
        if status != 'arrived':
            failed.append((x, y, yaw)); continue
        yield from settle(ctx, est, 8)
        yield from look(ctx, est)                   # (from here, 0.9-1.3 m: the look that the hand goes by)
        if item_id not in est.items:
            return None, 'lost'
        item = est.items[item_id]
        status = yield from straight(ctx, (x, y), yaw, speed=.2, tolerance=.02)
        if status != 'arrived':
            yield from straight(ctx, staging, yaw, speed=.2); failed.append((x, y, yaw)); continue
        yield from hold(ctx, 6)
        point, rotation = grip(item, est.base(), est.relative(item)); plan = decor.plan_pinch(ctx, point, rotation)
        if plan is None:
            ctx.event(kind, ok=False, reason='no way from the stance', arm=arm)
            yield from straight(ctx, staging, yaw, speed=.2); failed.append((x, y, yaw)); continue
        wrist = None if kind == 'caldron' else (lambda tip, arm=arm, kind=kind: est.wrist_look(arm, kind, tip))
        # (lifted straight up 0.40 m before the hand goes to the chest: from 0.20 the way there swung the thing over the
        # chassis' front edge, and a candle hanging by its wick was knocked off 2 times of 6)
        ok = yield from decor.pinch(ctx, plan, label=kind, gap=.002, lift=LIFT[kind], look=wrist, replan=lambda p, r=rotation: decor.plan_pinch(ctx, p, r), debug=lambda: dict(getattr(est, 'wrist_debug', {})), looks=3 if kind == 'pumpkin' else 2)
        if ok:
            ctx.attach(arm, *held_shape(kind, ctx.tip()[:3, 3])); ctx.holding[arm] = item_id; est.taken(item_id)
            status = yield from carry(ctx, arm)
            ctx.event('taken', thing=kind, arm=arm, carry=status, fingers=ctx.fingers().round(4).tolist())
            if not holds(ctx, arm):
                # Fell on the way to the chest: it lies near the stance, maybe on its side, and is looked for afresh (a
                # record kept where it had stood sent the hand there to close on nothing again and again).
                ctx.detach(arm); ctx.holding.pop(arm, None); ctx.set_grip(1., arm); est.taken_ids.discard(item_id)
                yield from tuck(ctx, (arm,))
                yield from straight(ctx, staging, yaw, speed=.2); yield from settle(ctx, est, 6); yield from look(ctx, est)
                return None, 'dropped'
            yield from straight(ctx, staging, yaw, speed=.2)
            return arm, 'taken'
        yield from tuck(ctx, (arm,))
        yield from straight(ctx, staging, yaw, speed=.2)
        yield from hold(ctx, 6)
        yield from look(ctx, est)                   # where is it now?
        failed.append((x, y, yaw)); empty += 1
        if empty >= 2 and item_id in est.items:
            est.items.pop(item_id, None); return None, 'lost'
    return None, 'failed'


def holds(ctx, arm):
    return float(np.sum(ctx.fingers(arm))) > .002


# ---- the cabinet ------------------------------------------------------------------------------------------------
def drawer_stances(ctx, est, side, arm, x, but=()):
    """Stances facing the cabinet from which @arm's hand is in front of drawer @side's lip (or over its places):
    [(x, y, 0.)] clear of the things on the floor, nearest the drawer's middle first."""
    y0, y1 = decor.DRAWERS[side]['y']; aside = ASIDE if arm == 'left' else -ASIDE; mid = .5 * (y0 + y1); out = []
    for y in sorted(np.arange(y0 + .12, y1 - .11, .05), key=lambda v: abs(v - mid)):
        pose = (x, float(y - aside), 0.)
        if ctx.scene.footprint_free(*pose, FOOTPRINT) and all(gap(pose, c, r) > .03 for c, r in obstacles(est, but)):
            out.append(pose)
    return out


def front_of_cabinet(ctx, est, pose, back=.45):
    """Drive to @pose (facing the cabinet) by a point @back behind it, the last stretch straight. Returns 'arrived' or
    what stopped it."""
    staging = (pose[0] - back, pose[1])
    status = yield from go(ctx, est, staging, 0., tolerance=.04)
    if status != 'arrived':
        return status
    yield from settle(ctx, est, 6)
    status = yield from straight(ctx, pose[:2], 0., speed=.2, tolerance=.015)
    return status


def pinch_lip(ctx, est, side, arm, out_by=0.):
    """The hand takes drawer @side's lip in front of it (the drawer @out_by out). Returns True, or False/None."""
    ctx.arm = arm; base = est.base(); aside = ASIDE if arm == 'left' else -ASIDE
    y0, y1 = decor.DRAWERS[side]['y']; lip_y = float(np.clip(base[1] + aside, y0 + .08, y1 - .08))
    target = np.array([decor.CABINET_FRONT + .006 - out_by, lip_y]); point = np.r_[in_base(base, target), decor.LIP['height'] - BASE_Z]
    plan = decor.plan_pinch(ctx, point, decor.lip_rotation(-base[2]), way_in=.08, hand_margin=decor.LIP['margin'])
    if plan is None:
        ctx.event('lip', ok=False, reason='no way to the lip', arm=arm, side=side); return None
    # The hand's last look along its fingers at the drawer's front puts the lip right in depth (the map pose is a few
    # centimetres off by then, and the lip is 2 cm deep: the hand closed in front of it, or on the front, 3 times of 6).
    rotation = decor.lip_rotation(-base[2])
    wrist = lambda tip: est.wrist_face(arm, tip, heading=-base[2])
    ok = yield from decor.pinch(ctx, plan, label='lip', gap=.004, most=.03, lift=0., look=wrist,
                                replan=lambda p: decor.plan_pinch(ctx, p, rotation, way_in=.08, hand_margin=decor.LIP['margin']))       # (on the 8 mm lip the fingers stay 1 cm apart)
    return ok


def in_front(item, side):
    """Does @item stand where drawer @side slides out (it would push it, and a hand going down to it then pushes the
    drawer back in: 1 run of 20)?"""
    y0, y1 = decor.DRAWERS[side]['y']; x, y = item['xy']
    return x + RADIUS[item['kind']] > 6.015 - DRAWER_OUT - .04 and y0 - .10 < y < y1 + .10


def open_drawer(ctx, est, side, arm):
    """Pull drawer @side out with @arm's hand by its lip. Returns True if it is out."""
    blocking = [i for i in est.known() if in_front(i, side)]
    if blocking:
        ctx.event('drawer_out', ok=False, side=side, arm=arm, reason='thing in front', things=[(i['kind'], np.round(i['xy'], 2).tolist()) for i in blocking])
        return False
    for pose in drawer_stances(ctx, est, side, arm, DRAWER_STANCE_X)[:3]:
        status = yield from front_of_cabinet(ctx, est, pose)
        if status != 'arrived':
            continue
        yield from hold(ctx, 6)
        ok = yield from pinch_lip(ctx, est, side, arm)
        if ok:
            status = yield from decor.slide(ctx, [-DRAWER_OUT * np.cos(-est.base()[2]), -DRAWER_OUT * np.sin(-est.base()[2]), 0.])
            yield from decor.let_go(ctx)
            ctx.event('drawer_out', side=side, arm=arm, status=status)
            est.drawers_out = list(getattr(est, 'drawers_out', [])) + [side]
            y0, y1 = decor.DRAWERS[side]['y']; est.ignore.append((5.60, y0 - .05, 6.20, y1 + .05))
        yield from tuck(ctx, (arm,))
        yield from straight(ctx, (pose[0] - .45, pose[1]), 0., speed=.2)
        if ok:
            return True
    ctx.event('drawer_out', ok=False, side=side, arm=arm)
    return False


def drawer_fronts(est):
    """The fronts of the drawers that are out, for the arm planner: (points N x 3 in M, radii, touch)."""
    points = []
    for side in getattr(est, 'drawers_out', ()):
        y0, y1 = decor.DRAWERS[side]['y']
        points += [(decor.CABINET_FRONT + .025 - DRAWER_OUT, y, z) for y in np.arange(y0 + .03, y1, .07) for z in (.12, .25, .37)]
    if not points:
        points = [(0., 0., -5.)]
    return np.array(points), np.full(len(points), .03), False


def drop_into(ctx, est, arm, kind, slot):
    """Let the thing in @arm's hand go over place @slot = (drawer, y) of an open drawer. Returns True if let go there."""
    side, y_slot = slot; aside = ASIDE if arm == 'left' else -ASIDE
    for pose in [p for p in [(DROP_STANCE_X, y_slot - aside, 0.)] if ctx.scene.footprint_free(*p, FOOTPRINT) and all(gap(p, c, r) > .03 for c, r in obstacles(est))] or drawer_stances(ctx, est, side, arm, DROP_STANCE_X)[:1]:
        status = yield from front_of_cabinet(ctx, est, pose)
        if status != 'arrived':
            continue
        yield from hold(ctx, 6)
        ctx.arm = arm; base = est.base(); xy = in_base(base, (DROP_X, y_slot))
        z = decor.DRAWER_FLOOR + DROP_ABOVE + GRIP_HEIGHT[kind] - BASE_Z
        plan = None; things = dict(ctx.obstacles); ctx.obstacles = dict(things, drawers=drawer_fronts(est))
        try:
            for closing in (0., np.pi / 2):
                plan = decor.plan_pinch(ctx, np.r_[xy, z], decor.top_rotation(closing), way_in=.30, hand_margin=decor.LIP['margin'])
                if plan is not None:
                    break
        finally:
            ctx.obstacles = things
        if plan is None:
            ctx.event('drop', ok=False, reason='no way over the place', arm=arm, slot=list(slot)); yield from straight(ctx, (pose[0] - .45, pose[1]), 0., speed=.2); continue
        status = yield from move_to(ctx, plan['pre'], rate=.03, label='drop')
        # Is the drawer out under the hand? (A drawer was once pushed back in on the way, and the next things were let go
        # over the floor in front of the cabinet.) Its floor is 8 cm up, the room's at 0.
        under = est.wrist_floor(arm, ctx.tip()[:3, 3]) if status == 'done' else None
        if under is not None and under < .04:
            ctx.event('drop', ok=False, reason='the drawer is not out', arm=arm, slot=list(slot), under=round(under, 3))
            est.drawers_out = [s_ for s_ in est.drawers_out if s_ != side]; est.ignore = [b for b in est.ignore if not (b[1] < y_slot < b[3])]
            yield from carry(ctx, arm); yield from straight(ctx, (pose[0] - .45, pose[1]), 0., speed=.2)
            return 'shut'
        if status == 'done':
            status = yield from follow(ctx, plan['inward'], rate=.015, max_steps=300)
        there = float(np.linalg.norm(ctx.tip()[:3, 3] - plan['point'])) < .04
        if status == 'done' or there:
            ctx.set_grip(1.); ctx.detach(arm); ctx.holding.pop(arm, None)
            yield from hold(ctx, 30)
        ctx.event('drop', ok=status == 'done' or there, arm=arm, slot=list(slot), status=status, fingers=ctx.fingers().round(4).tolist())
        yield from decor.slide(ctx, [0., 0., .12], rate=.02)
        yield from tuck(ctx, (arm,))
        yield from straight(ctx, (pose[0] - .45, pose[1]), 0., speed=.2)
        if status == 'done' or there:
            return True
    return False


def shut_drawer(ctx, est, side, arm='left'):
    """Push drawer @side shut: the closed hand level against its front, the base drives in until the hand is where the
    front is when the drawer is shut. (How far out the drawer is is not known to a few centimetres: the drops nudge it
    in. Taken by its lip as it was pulled out, the hand closed in front of the lip 4 times of 4.) Returns True if
    the base got in."""
    for arm_ in (arm, other(arm)):
        for pose in drawer_stances(ctx, est, side, arm_, SHUT_STANCE_X)[:2]:
            status = yield from front_of_cabinet(ctx, est, pose, back=.30)
            if status != 'arrived':
                continue
            yield from hold(ctx, 6)
            ctx.arm = arm_; base = est.base(); aside = ASIDE if arm_ == 'left' else -ASIDE
            y0, y1 = decor.DRAWERS[side]['y']; y = float(np.clip(base[1] + aside, y0 + .10, y1 - .10))
            start = np.array([PUSH['ahead'] + base[0], y]); point = np.r_[in_base(base, start), PUSH['height'] - BASE_Z]
            ctx.set_grip(-1.)
            yield from hold(ctx, 15)
            plan = decor.plan_pinch(ctx, point, decor.lip_rotation(-base[2]), way_in=.06, hand_margin=decor.LIP['margin'])
            ok = False
            if plan is not None:
                status = yield from move_to(ctx, plan['pre'], rate=.04, label='push')
                if status == 'done':
                    yield from follow(ctx, plan['inward'], rate=.015, max_steps=300)
                    reach = float(start[0] - base[0])
                    # (to 5 cm past where the shut drawer's front is: the arm gives a little; with 1.5 cm the drawer
                    # stayed 1.1-1.5 cm out, against 1.55 for "open", and with 3.5 cm still 1.4-1.5 cm 2 times of 20)
                    status = yield from straight(ctx, (decor.CABINET_FRONT + .05 - reach, base[1]), 0., speed=.1, tolerance=.008, max_steps=400)
                    yield from hold(ctx, 6)
                    ok = True
                    ctx.event('drawer_shut', side=side, arm=arm_, status=status, base=np.round(est.base(), 3).tolist())
                    est.drawers_out = [s_ for s_ in getattr(est, 'drawers_out', []) if s_ != side]
            ctx.set_grip(1.)
            yield from decor.slide(ctx, [-.06, 0., 0.], rate=.02)
            yield from tuck(ctx, (arm_,))
            yield from straight(ctx, (pose[0] - .30, pose[1]), 0., speed=.2)
            if ok:
                return True
    ctx.event('drawer_shut', ok=False, side=side)
    return False


# ---- the caldron ------------------------------------------------------------------------------------------------
def set_caldron_down(ctx, est, arm):
    """Set the caldron in @arm's hand down next to the coffee table. Returns True if it was let go there."""
    aside = ASIDE if arm == 'left' else -ASIDE
    here = np.asarray(est.base()[:2], float)
    for (centre, heading) in sorted(CALDRON_SPOTS, key=lambda s: float(np.linalg.norm(np.asarray(s[0]) - here))):
        c, s = np.cos(heading), np.sin(heading); centre = np.asarray(centre, float)
        pose = (centre[0] - CALDRON_REACH * c + aside * s, centre[1] - CALDRON_REACH * s - aside * c, heading)
        if not ctx.scene.footprint_free(*pose, FOOTPRINT) or any(gap(pose, xy, r) < .03 for xy, r in obstacles(est)):
            continue
        staging = (pose[0] - .45 * c, pose[1] - .45 * s)
        status = yield from go(ctx, est, staging, heading, tolerance=.04)
        if status != 'arrived':
            continue
        yield from settle(ctx, est, 6)
        status = yield from straight(ctx, pose[:2], heading, speed=.2, tolerance=.02)
        if status != 'arrived':
            yield from straight(ctx, staging, heading, speed=.2); continue
        ctx.arm = arm; base = est.base(); xy = in_base(base, centre); sight = float(np.arctan2(xy[1], xy[0]))
        point = np.r_[xy - RIM['inset'] * np.array([np.cos(sight), np.sin(sight)]), RIM['height'] + .015 - BASE_Z]
        # (the last stretch down is planned without the caldron in the model: next to the table the ball that stands
        # for it overlaps the table, and every way down was refused)
        held = ctx.payload.pop(arm, None)
        plan = decor.plan_pinch(ctx, point, decor.top_rotation(sight), way_in=.18)
        if held is not None:
            ctx.payload[arm] = held
        if plan is None:
            ctx.event('caldron_down', ok=False, reason='no way down', arm=arm); yield from straight(ctx, staging, heading, speed=.2); continue
        status = yield from move_to(ctx, plan['pre'], rate=.03, label='caldron_down')
        if status == 'done':
            status = yield from follow(ctx, plan['inward'], rate=.012, max_steps=300)
        ctx.set_grip(1.); ctx.detach(arm); ctx.holding.pop(arm, None)
        yield from hold(ctx, 30)
        ctx.event('caldron_down', ok=True, arm=arm, status=status, spot=np.round(centre, 3).tolist())
        yield from decor.slide(ctx, [0., 0., .15], rate=.02)
        yield from tuck(ctx, (arm,))
        yield from straight(ctx, staging, heading, speed=.2)
        return True
    return False


# ---- the whole task ---------------------------------------------------------------------------------------------
def in_the_way(item, drawers):
    """Does @item lie where one of @drawers slides out, or where the base stands to pull it (the chassis' back is at
    x 4.84 there)?"""
    x, y = item['xy']
    return any(x + RADIUS[item['kind']] > 4.84 and decor.DRAWERS[side]['y'][0] - .3 < y < decor.DRAWERS[side]['y'][1] + .3 for side in drawers)


def free_floor(ctx, est, xy, but=()):
    """May a thing be set down at map @xy: free floor the map knows, 12 cm from anything it has, in no drawer's way,
    0.25 m from the other things on the floor?"""
    xy = np.asarray(xy, float); i, j = ctx.scene.cell(xy[None]); scene = ctx.scene
    if not scene.inside(i, j)[0] or scene.top[i[0], j[0]] > .03 or scene.distance[i[0], j[0]] < .12:
        return False
    if in_the_way(dict(xy=xy, kind='caldron'), tuple(decor.DRAWERS)):
        return False
    return all(float(np.linalg.norm(xy - c)) > .25 for c, r in obstacles(est, but))


def put_down(ctx, est, arm):
    """Set the thing in @arm's hand down on the floor ahead of the base where it is in no drawer's way (no drawer could
    be pulled out while it was held, or the caldron had no place by the table). The estimator looks for it there again.
    Returns True if it was set down, False if the hand let it go where it was."""
    item_id = ctx.holding.get(arm); kind = item_id[0] if item_id else 'candle'; aside = ASIDE if arm == 'left' else -ASIDE
    reach = CALDRON_REACH if kind == 'caldron' else REACH
    for where in ('here', 'open floor'):
        if where == 'open floor':
            if float(np.linalg.norm(np.asarray(est.base()[:2]) - OPEN_FLOOR)) > .3:
                yield from go(ctx, est, OPEN_FLOOR, est.base()[2], tolerance=.12)
        start = est.base()[2]
        for turn in ((0.,) if where == 'here' else (0., np.pi / 2, -np.pi / 2, np.pi)):
            heading = wrap(start + turn); c, s = np.cos(heading), np.sin(heading); base = est.base()
            spot = np.array([base[0] + reach * c - aside * s, base[1] + reach * s + aside * c])
            if not free_floor(ctx, est, spot, but=(item_id,)):
                continue
            if turn:
                yield from turn_to(ctx, est, heading)
            ctx.arm = arm; base = est.base(); xy = in_base(base, spot); sight = float(np.arctan2(xy[1], xy[0]))
            if kind == 'caldron':
                point = np.r_[xy - RIM['inset'] * np.array([np.cos(sight), np.sin(sight)]), RIM['height'] + .015 - BASE_Z]
            else:
                point = np.r_[xy, GRIP_HEIGHT[kind] + .015 - BASE_Z]
            held = ctx.payload.pop(arm, None)           # (the ball that stands for the thing reaches into the floor)
            plan = decor.plan_pinch(ctx, point, decor.top_rotation(sight), way_in=.25)
            if held is not None:
                ctx.payload[arm] = held
            if plan is None:
                continue
            status = yield from move_to(ctx, plan['pre'], rate=.03, label='put_down')
            if status == 'done':
                status = yield from follow(ctx, plan['inward'], rate=.012, max_steps=300)
            ctx.set_grip(1.); ctx.detach(arm); ctx.holding.pop(arm, None)
            yield from hold(ctx, 30)
            ctx.event('put_down', thing=kind, arm=arm, status=status, spot=np.round(spot, 3).tolist())
            yield from decor.slide(ctx, [0., 0., .15], rate=.02)
            yield from tuck(ctx, (arm,))
            if item_id:
                est.put_back(item_id, spot)
            yield from look(ctx, est)
            return True
    ctx.event('put_down', ok=False, thing=kind, arm=arm)
    ctx.set_grip(1.); ctx.detach(arm); ctx.holding.pop(arm, None)
    yield from hold(ctx, 30); yield from tuck(ctx, (arm,))
    if item_id:
        base = est.base(); est.put_back(item_id, (base[0] + .4 * np.cos(base[2]), base[1] + .4 * np.sin(base[2])))
    yield from look(ctx, est)
    return False


def move_caldron(ctx, est):
    """Take the caldron and set it down next to the coffee table. Returns True if it stands there. (It is looked for
    again when the hand closed twice on nothing: the chassis had pushed it 0.24 m on the way, 1 run of 20.)"""
    for attempt in range(3):
        if not est.known('caldron'):
            yield from look_round(ctx, est)
            if not est.known('caldron'):
                return False
        caldron = est.known('caldron')[0]
        if by_the_table(caldron['xy']):
            ctx.event('stage', done='caldron', ok=True, why='already next to the table', at=np.round(caldron['xy'], 3).tolist()); return True
        arm, why = yield from take_item(ctx, est, caldron, free_arms(ctx))
        ctx.event('stage', done='take', thing='caldron', arm=arm, why=why)
        if arm is None:
            if why == 'no_stance' and float(np.linalg.norm(np.asarray(est.base()[:2]) - OPEN_FLOOR)) > .5:
                yield from go(ctx, est, OPEN_FLOOR, est.base()[2], tolerance=.15)
            continue
        ok = yield from set_caldron_down(ctx, est, arm)
        ctx.event('stage', done='caldron', ok=ok)
        if not ok:
            yield from put_down(ctx, est, arm)
        return ok
    return False


def free_arms(ctx):
    return tuple(a for a in ('left', 'right') if a not in ctx.holding)


TABLE = ((3.114, 4.457), (3.928, 6.128))      # the coffee table's outline (map x, y), the same in every start


def by_the_table(xy, within=.10):
    """Does a caldron whose middle is at @xy stand next to the coffee table already (its side @within of the table's
    outline; the simulator's NextTo holds up to 18 cm)?"""
    (x0, y0), (x1, y1) = TABLE; x, y = xy; dx = max(x0 - x, 0., x - x1); dy = max(y0 - y, 0., y - y1)
    return float(np.hypot(dx, dy)) - .147 < within


def full_task(ctx, est):
    ctx.holding = {}; est.drawers_out = []; slots = list(SLOTS); put = 0; searches = 0; misses = {}
    yield from tuck(ctx)
    lost = yield from find(ctx, est)
    ctx.event('stage', done='find', missing=lost, known=[(i['kind'], np.round(i['xy'], 2).tolist()) for i in est.known()])
    caldron_done = False; caldron_tries = 0; stuck = 0
    for turn in range(20):
        # The caldron goes to the table first if it stands where a drawer that is still to be pulled out slides out,
        # or where the base stands to pull it (2 runs of 20: no drawer could be pulled with a candle in the other hand).
        closed = sorted({side for side, _ in slots} - set(est.drawers_out)); caldron = est.known('caldron')
        if not caldron_done and caldron_tries < 3 and caldron and not ctx.holding and in_the_way(caldron[0], closed):
            caldron_tries += 1
            caldron_done = yield from move_caldron(ctx, est)
            continue
        items = [i for i in est.known() if i['kind'] != 'caldron']
        if not items:
            if put + sum(1 for i in ctx.holding.values()) < 5 and searches < 2:
                searches += 1
                yield from find(ctx, est); ctx.event('stage', done='find_again', missing=missing(est)); continue
            break
        closed = sorted({side for side, _ in slots} - set(est.drawers_out))
        here = np.asarray(est.base()[:2], float)
        items.sort(key=lambda i: (misses.get(i['id'], 0), not (in_the_way(i, closed) or any(in_front(i, side) for side in decor.DRAWERS)), float(np.linalg.norm(np.asarray(i['xy']) - here))))
        item = items[0]
        # (the arm keeps clear of the fronts of the drawers that are out: reaching for a pumpkin in front of one, it
        # pushed the drawer back in, 1 run of 20)
        things = dict(ctx.obstacles)
        if est.drawers_out:
            ctx.obstacles = dict(things, drawers=drawer_fronts(est))
        try:
            arm, why = yield from take_item(ctx, est, item, free_arms(ctx))
        finally:
            ctx.obstacles = things
        ctx.event('stage', done='take', thing=item['kind'], arm=arm, why=why)
        if arm is None:
            misses[item['id']] = misses.get(item['id'], 0) + 1
            if why == 'no_stance':
                if not caldron_done and caldron_tries < 3 and est.known('caldron'):
                    # (things round the caldron, or against it and a wall, have no stance till it is gone: a candle
                    # 0.4 m from it in a corner, 1 run of 20; it goes to the table anyway)
                    caldron_tries += 1
                    caldron_done = yield from move_caldron(ctx, est)
                elif float(np.linalg.norm(np.asarray(est.base()[:2]) - OPEN_FLOOR)) > .5:
                    yield from go(ctx, est, OPEN_FLOOR, est.base()[2], tolerance=.15)
                else:
                    # (from the open floor too: look from the lookout nearest the thing, which may show it elsewhere)
                    spot = min(LOOKOUTS, key=lambda p_: float(np.linalg.norm(np.asarray(p_) - item['xy'])))
                    if clear(est, spot):
                        status = yield from go(ctx, est, spot, float(np.arctan2(item['xy'][1] - spot[1], item['xy'][0] - spot[0])), tolerance=.12)
                        if status == 'arrived':
                            yield from turn_to(ctx, est, float(np.arctan2(item['xy'][1] - spot[1], item['xy'][0] - spot[0])))
                    yield from look(ctx, est)
            continue
        if not slots:
            break
        slot = slots[0]
        if slot[0] not in est.drawers_out:
            opened = yield from open_drawer(ctx, est, slot[0], other(arm))
            if not opened:
                # the other drawer, if it has places left
                rest = [s for s in slots if s[0] != slot[0]]
                if rest and rest[0][0] not in est.drawers_out:
                    opened = yield from open_drawer(ctx, est, rest[0][0], other(arm))
                    if opened:
                        slots = rest + [s for s in slots if s[0] == slot[0]]; slot = slots[0]
                if not opened:
                    # (no stance in front of either drawer: things on the floor stand there) The thing is set down out
                    # of the way, and the things in the way are taken first.
                    ctx.event('stage', done='drawer', ok=False); stuck += 1
                    misses[item['id']] = misses.get(item['id'], 0) + 1
                    yield from put_down(ctx, est, arm)
                    # (the caldron in front of a drawer had no stance while things stood round it; with one of them
                    # set aside it may have one now, 1 run of 20)
                    if not caldron_done:
                        caldron_done = yield from move_caldron(ctx, est)
                    if stuck >= 4:
                        break
                    continue
        if not holds(ctx, arm):
            ctx.event('lost', thing=item['kind'], arm=arm); ctx.detach(arm); ctx.holding.pop(arm, None); ctx.set_grip(1., arm)
            yield from tuck(ctx, (arm,)); est.taken_ids.discard(item['id'])
            yield from look(ctx, est); continue
        ok = yield from drop_into(ctx, est, arm, item['kind'], slot)
        if ok == 'shut':
            opened = yield from open_drawer(ctx, est, slot[0], other(arm))
            ok = (yield from drop_into(ctx, est, arm, item['kind'], slot)) if opened else False
            ok = ok is True
        ctx.event('stage', done='drawer', thing=item['kind'], slot=list(slot), ok=ok)
        if ok:
            slots.pop(0); put += 1
        elif arm in ctx.holding:
            misses[item['id']] = misses.get(item['id'], 0) + 1
            yield from put_down(ctx, est, arm)
    for arm in list(ctx.holding):
        yield from put_down(ctx, est, arm)
    for side in list(est.drawers_out):
        yield from shut_drawer(ctx, est, side)
    if not caldron_done:
        yield from move_caldron(ctx, est)
    return 'done' if put == 5 else f'{put} in'
