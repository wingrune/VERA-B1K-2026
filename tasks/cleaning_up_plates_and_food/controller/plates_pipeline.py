"""Cleaning up the plates and the food: the task as a sequence of skills.

    find the two plates (each with its pizza) and the two bowls on the breakfast table (looks at it from its ends);
    each bowl: stand at the table's nearer end, take the bowl by its rim, carry it to the sink and let it go over the
    basin;
    open the fridge's right door by its handle, in stages (the hand leads it round the hinge from one stance as far as
    the chassis lets the door pass, then takes it again from the next);
    each plate: take it by its rim at the table's end, level, its pizza on it; carry it to the fridge; slide it in onto
    a shelf of the west half and let go;
    push the door shut.

Poses are (x, y, yaw) in the map frame M.
"""
import time
import numpy as np
from manip import hold, drive, pose2, follow, move_to, BASE_Z, TALL_TRUNK
from car_skills import tuck, park, frame
from car_pipeline import straight, settle, FOOTPRINT
from skills import interpolate_cartesian
import decor
import kitchen as K

wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
other = lambda arm: 'right' if arm == 'left' else 'left'

LOOKOUTS = (((4.30, 3.00), -np.pi / 2), ((5.75, 1.45), np.pi), ((5.75, .55), 2.3), ((3.40, 2.90), -1.0))
EAST_GATE = (5.20, K.ENDS['south']['y'])   # the base enters the gap between the table's south end and the fridge from here
SOUTH_TURN = (5.75, .55)                    # room to turn east of that gap
LIFT = dict(bowl=.12, plate=.10)
CARRY = dict(bowl=((.40, .14, .95), (.44, .10, 1.00)), plate=((.34, .10, 1.00), (.38, .06, 1.05)))   # base frame, height above the floor; mirrored for the right hand
CLEAR_ABOVE_SHELF = .035                    # the plate's middle this far above the shelf's top on the way in
FRONT_EAST = .05                            # the plate's middle comes to the opening this far east of where it goes in
PUSH_TO = -.48                              # a plate let go short of this (its middle's y) is pushed in to it
IN_ENOUGH = -.33                            # a plate is let go in the fridge with its middle further in than this (y)
PUSH_PITCH = 15.                            # the closed hand's fingers point in, this far down (deg), pushing the plate's rim


# ---- moving about -----------------------------------------------------------------------------------------------
def door_keepouts(est):
    """Discs the base's middle keeps out of where the fridge door stands open: along its leaf."""
    angle = getattr(est, 'door', 0.)
    if angle < .05:
        return []
    d = np.array([np.cos(angle), np.sin(angle)])
    return [(K.HINGE + r * d, .10 + .40) for r in np.arange(.05, K.DOOR_LENGTH + .01, .12)]


def go(ctx, est, goal, yaw, **options):
    """drive() to @goal by a planned route, round the open door. Returns drive's status, or 'no_route'."""
    goal = np.asarray(goal, float)[:2]; here = np.asarray(est.base()[:2], float)
    keep = [(xy, max(0., min(r, float(np.linalg.norm(here - xy)) - .03, float(np.linalg.norm(goal - xy)) - .03))) for xy, r in door_keepouts(est)]
    if ctx.scene.path(here, goal, keepout=keep) is None:
        keep = []
        if ctx.scene.path(here, goal) is None:
            ctx.event('route', ok=False, goal=np.round(goal, 2).tolist()); return 'no_route'
    status = yield from drive(ctx, goal, yaw, keepout=keep, lookahead=.25, **options)
    return status


def turn_to(ctx, est, yaw):
    here = est.base()
    status = yield from drive(ctx, here[:2], yaw, direct=True, face_travel=False, tolerance=.08, yaw_tolerance=.05, max_steps=400)
    return status


def look(ctx, est, steps=4):
    yield from hold(ctx, steps)
    return est.look()


def all_known(est):
    return len(est.known('plate')) + sum(1 for i in est.taken_ids if i[0] == 'plate') >= 2 and len(est.known('bowl')) + sum(1 for i in est.taken_ids if i[0] == 'bowl') >= 2


def find(ctx, est):
    """Know where the things are: a look round on the spot, then from the lookouts round the table, nearest first."""
    start = est.base()[2]
    toward = float(np.arctan2(1.44 - est.base()[1], 4.29 - est.base()[0]))
    yield from turn_to(ctx, est, toward); yield from look(ctx, est)
    left = list(LOOKOUTS)
    while not all_known(est) and left:
        here = np.asarray(est.base()[:2], float); left.sort(key=lambda p: float(np.linalg.norm(np.asarray(p[0]) - here))); spot, yaw = left.pop(0)
        if float(np.linalg.norm(np.asarray(spot) - here)) < .5:
            yield from turn_to(ctx, est, yaw); yield from look(ctx, est); continue
        status = yield from go(ctx, est, spot, yaw, tolerance=.10)
        if status == 'no_route':
            continue
        yield from turn_to(ctx, est, yaw)
        for _ in range(3):
            yield from look(ctx, est)
        ctx.event('find', where=list(spot), known=[(i['kind'], np.round(i['xy'], 3).tolist()) for i in est.known()])
    return all_known(est)


# ---- at the table -----------------------------------------------------------------------------------------------
def in_base(base, xy):
    x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = np.asarray(xy, float)[:2] - [x, y]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


def rim_grip(kind, centre_local, pitch):
    """Fingers' ends and hand rotation (base frame) on the rim's near side of a thing whose middle is at
    @centre_local (base-frame xy), the fingers coming along the way to its middle, tilted down by @pitch (deg)."""
    heading = float(np.arctan2(centre_local[1], centre_local[0])); height, inset = K.GRIP[kind]
    radius = (K.PLATE if kind == 'plate' else K.BOWL)['radius']
    xy = np.asarray(centre_local, float) - (radius - inset) * np.array([np.cos(heading), np.sin(heading)])
    return np.r_[xy, K.TABLE['top'] + height - BASE_Z], decor.edge_rotation(heading, np.radians(pitch))


def under_another(est, item, pose, point):
    """Is the rim at @point (base frame, the base at @pose) under another plate or next to a bowl? (Two plates lie one
    half on the other on some starts: a hand that went for the lower one's rim where the upper one covers it pushed
    them both, and a pizza slid from one plate onto the other.)"""
    x, y, yaw = pose; c, s_ = np.cos(yaw), np.sin(yaw)
    rim = np.array([x + c * point[0] - s_ * point[1], y + s_ * point[0] + c * point[1]])
    for other in est.known():
        if other['id'] == item['id'] or (other['kind'] == item['kind'] and float(np.linalg.norm(np.asarray(other['xy']) - item['xy'])) < .10):
            continue                # (itself, or itself seen twice: one bowl seen 6 cm apart from 3.4 m kept both from a hand, 312)
        reach = (K.PLATE if other['kind'] == 'plate' else K.BOWL)['radius'] + .03
        if float(np.linalg.norm(rim - np.asarray(other['xy'], float))) < reach:
            return True
    return False


def table_options(ctx, est, item, arms):
    """Stances at the table's ends from which a hand of @arms takes @item: [(cost, arm, pose, end, pitch)]."""
    centre = np.asarray(item['xy'], float); out = []
    for end, spec in K.ENDS.items():
        yaw = spec['yaw']; far = abs(centre[1] - (K.TABLE['y'][0] if end == 'south' else K.TABLE['y'][1]))
        for order, arm in enumerate(arms):
            aside = .16 if arm == 'left' else -.16
            # the base's middle so that the thing's middle lies at 'aside' to the hand's side (facing north, left is -x)
            ideal = centre[0] + (aside if end == 'south' else -aside)
            for x in sorted(np.arange(K.END_X[0], K.END_X[1] + .001, .05), key=lambda v: abs(v - ideal)):
                pose = (float(x), spec['y'], yaw)
                if abs(x - ideal) > .30 or not ctx.scene.footprint_free(*pose, FOOTPRINT):
                    continue
                out.append((far + .4 * abs(x - ideal) + .3 * order, arm, pose, end))
    return sorted(out, key=lambda o: o[0])


def approach_end(ctx, est, pose, end):
    """Bring the base to @pose at the table's @end: by a point behind it (north end), or into the gap between table
    and fridge from its east side (south end), the last moves straight at the end's heading. Returns 'arrived' or
    what stopped it."""
    yaw = pose[2]
    if end == 'north':
        staging = (pose[0], pose[1] + .40)
        status = yield from go(ctx, est, staging, yaw, tolerance=.05)
        if status != 'arrived':
            return status
        yield from settle(ctx, est, 6); yield from look(ctx, est)
        return (yield from straight(ctx, pose[:2], yaw, speed=.15, tolerance=.015))
    here = np.asarray(est.base()[:2], float)
    if not (abs(here[1] - pose[1]) < .06 and here[0] < 5.3):
        status = yield from go(ctx, est, SOUTH_TURN, yaw, tolerance=.06)
        if status != 'arrived':
            return status
        yield from turn_to(ctx, est, yaw)
        status = yield from straight(ctx, EAST_GATE, yaw, speed=.15, tolerance=.02)
        if status != 'arrived':
            return status
    yield from settle(ctx, est, 6)
    return (yield from straight(ctx, pose[:2], yaw, speed=.15, tolerance=.015))


def leave_end(ctx, est, end):
    """Back out of the table's @end the way the base came in."""
    base = est.base()
    if end == 'north':
        return (yield from straight(ctx, (base[0], base[1] + .40), base[2], speed=.15))
    status = yield from straight(ctx, EAST_GATE, base[2], speed=.15, tolerance=.03)
    if status == 'arrived':
        status = yield from straight(ctx, SOUTH_TURN, base[2], speed=.15, tolerance=.05)
    return status


def carry(ctx, arm, kind):
    """The hand with a thing to where it is held on the way, the thing level as it was taken. A plate is first drawn
    in slowly, straight and level (drawn in from 0.93 m at the arm's planned pace, its pizza slid off, 1 of 4)."""
    side = 1. if arm == 'left' else -1.; ctx.arm = arm; rotation = ctx.tip()[:3, :3]
    if kind == 'plate':
        tip = ctx.tip()[:3, 3]; goal = np.array([CARRY[kind][0][0], side * CARRY[kind][0][1]])
        if float(np.linalg.norm(tip[:2] - goal)) > .05:
            status = yield from decor.slide(ctx, np.r_[goal - tip[:2], 0.], rate=.004, max_steps=1500)
            if status == 'no_way':
                ctx.event('carry', ok=False, reason='no straight way in', arm=arm)
    if kind != 'plate':
        status = yield from park(ctx, [(x, side * y, z) for x, y, z in CARRY[kind]], [rotation], label='carry', rate=.012)
        return status
    # A plate: up and in, level, slowly; where no such pose is found the hand stays as it is (the fall-back of park(),
    # the trunk set upright at once with the arm as it was, swung the plate and its pizza slid off, 1 of 10).
    from car_skills import UPRIGHT
    from manip import BASE_Z as Z0
    free = ctx.validator(relax_from=ctx.q)
    for x, y, z in CARRY[kind] + ((.45, .12, .95), (.40, .20, .90), (.50, .10, 1.00)):
        goal = ctx.ik.solve(ctx.q, np.array([x, side * y, z - Z0]), rotation, count=30, validator=free, locked=UPRIGHT)
        if goal is not None:
            status = yield from move_to(ctx, goal, rate=.005, label='carry')
            if status == 'done':
                return status
    ctx.event('carry', ok=False, reason='no level pose up and in: left as it is', arm=arm)
    return 'kept'


def holds(ctx, arm):
    return float(np.sum(ctx.fingers(arm))) > .002


def take(ctx, est, item, arms=('right', 'left'), pitches=None):
    """Stand at a table end where a hand reaches @item, take it by its rim, lift it and hold it for the way. Returns
    (arm or None, why, end)."""
    item_id = item['id']; kind = item['kind']; failed = []
    for attempt in range(4):
        if item_id not in est.items:
            return None, 'lost', None
        item = est.items[item_id]; chosen = None; started = time.time()
        for cost, arm, pose, end in table_options(ctx, est, item, arms):
            if any(abs(pose[0] - f[0]) < .04 and abs(pose[1] - f[1]) < .04 for f in failed) or time.time() - started > 30.:
                continue
            if end == 'south' and getattr(est, 'door', 0.) > .05:
                continue                # (the open door shuts the way into the gap between the table and the fridge)
            ctx.arm = arm; local = in_base(pose, item['xy'])
            for pitch in (pitches or K.GRIP_PITCH[kind]):
                point, rotation = rim_grip(kind, local, pitch)
                if kind == 'plate' and float(np.linalg.norm(point[:2])) > NORTH_REACH:
                    continue
                if under_another(est, item, pose, point):
                    continue
                if decor.plan_pinch(ctx, point, rotation, base=pose2(*pose), hand_margin=-.03) is not None:
                    chosen = (arm, pose, end, pitch); break
            if chosen:
                break
        if chosen is None:
            ctx.event('stance', ok=False, thing=kind, failed=len(failed)); return None, 'no_stance', None
        arm, pose, end, pitch = chosen; ctx.arm = arm
        ctx.event('stance', thing=kind, arm=arm, pose=np.round(pose, 3).tolist(), end=end, pitch=pitch, target=np.round(item['xy'], 3).tolist())
        status = yield from approach_end(ctx, est, pose, end)
        if status != 'arrived':
            failed.append(pose); yield from leave_end(ctx, est, end); continue
        yield from hold(ctx, 6)
        if item_id not in est.items:
            return None, 'lost', None
        item = est.items[item_id]
        point, rotation = rim_grip(kind, est.relative(item), pitch)
        plan = decor.plan_pinch(ctx, point, rotation, hand_margin=-.03)
        if plan is None:
            ctx.event(kind, ok=False, reason='no way from the stance', arm=arm); failed.append(pose); yield from leave_end(ctx, est, end); continue
        ok = yield from decor.pinch(ctx, plan, label=kind, gap=.002, lift=LIFT[kind])
        if ok:
            # where the thing's middle hangs from the fingers (tool frame): it moves with the hand from now on
            tip = ctx.tip(); centre = np.r_[est.relative(item), K.TABLE['top'] + .01 - BASE_Z]
            item['hold'] = tip[:3, :3].T @ (centre + [0., 0., LIFT[kind]] - tip[:3, 3]); item['pitch'] = pitch
            ctx.attach(arm, np.array([centre + [0., 0., LIFT[kind]]]), np.array([.07 if kind == 'bowl' else .12]))
            ctx.holding[arm] = item_id; est.taken(item_id); est.in_hand[arm] = item
            status = yield from carry(ctx, arm, kind)
            ctx.event('taken', thing=kind, arm=arm, carry=status, fingers=ctx.fingers().round(4).tolist())
            yield from leave_end(ctx, est, end)
            if not holds(ctx, arm):
                ctx.detach(arm); ctx.holding.pop(arm, None); ctx.set_grip(1., arm); est.taken_ids.discard(item_id)
                yield from tuck(ctx, (arm,)); yield from look(ctx, est)
                return None, 'dropped', end
            return arm, 'taken', end
        yield from tuck(ctx, (arm,))
        yield from leave_end(ctx, est, end)
        yield from look(ctx, est)
        failed.append(pose)
    return None, 'failed', None


NORTH_REACH = .86                   # a plate's near rim no further ahead than this (at 0.99 the hand did not get down to it 3 times; from 0.92-0.93 the plate was taken, but drawn in its pizza slid off 2 times of 3)


def north_reach(ctx, est, item, arms=('left',)):
    """Can a hand of @arms take @item from the table's north end (the south end is shut off by the open fridge door)?"""
    for cost, arm, pose, end in table_options(ctx, est, item, arms):
        if end != 'north':
            continue
        ctx.arm = arm; local = in_base(pose, item['xy'])
        for pitch in K.GRIP_PITCH[item['kind']]:
            point, rotation = rim_grip(item['kind'], local, pitch)
            if float(np.linalg.norm(point[:2])) > NORTH_REACH:
                continue
            if decor.plan_pinch(ctx, point, rotation, base=pose2(*pose), hand_margin=-.03) is not None:
                return True
    return False


def free_spot(est, but):
    """Where on the table's north half a plate can be set down: [(x, y)] clear of the other things, northmost first."""
    others = [(np.asarray(i['xy'], float), (K.PLATE if i['kind'] == 'plate' else K.BOWL)['radius']) for i in est.known() if i['id'] != but]
    spots = []
    for y in np.arange(1.85, 1.54, -.04):          # (from 1.97 the near rim was 0.32 m ahead of the base: the hand stalled going down)
        # (to 3.98 and 4.62: the rim a few centimetres over the table's edge; with the other plate at (4.31, 1.53)
        # there was no spot between 4.04 and 4.55, and the plate went on in the hand that then held it through the
        # door's opening, and the pizza fell, 312)
        for x in np.arange(3.98, 4.63, .04):
            room = min([float(np.linalg.norm(np.array([x, y]) - c)) - r - K.PLATE['radius'] for c, r in others] or [1.])
            if room > .02:
                spots.append((round(float(y), 2), room, (float(x), float(y))))
    spots.sort(key=lambda s_: (-s_[0], -s_[1]))
    return [s_[2] for s_ in spots]


def put_on_table(ctx, est, arm, spot):
    """Set the plate in @arm's hand down on the table with its middle at @spot, from the north end. Returns True if
    let go there; the plate is then known there again."""
    item = est.in_hand[arm]; aside = .16 if arm == 'left' else -.16
    pose = (float(spot[0] - aside), K.ENDS['north']['y'], K.ENDS['north']['yaw'])
    status = yield from approach_end(ctx, est, pose, 'north')
    if status != 'arrived':
        return False
    ctx.arm = arm; base = est.base(); rotation = ctx.tip()[:3, :3]
    centre = np.r_[in_base(base, spot), K.TABLE['top'] + K.PLATE['height'] / 2 + .012 - BASE_Z]
    above = centre + [0., 0., .08] - rotation @ item['hold']; down_to = centre - rotation @ item['hold']
    held = ctx.payload.pop(arm, None)
    plan = decor.plan_pinch(ctx, above, rotation, way_in=.0001, hand_margin=-.02); ok = False
    if plan is not None:
        status = yield from move_to(ctx, plan['pre'], rate=.012, label='to_table')
        way = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], down_to, rotation, step=.01) if status == 'done' else None
        if way is not None:
            yield from follow(ctx, way, rate=.006, max_steps=300); ok = True
    if held is not None:
        ctx.payload[arm] = held
    if not ok:
        ctx.event('to_table', ok=False, arm=arm, spot=np.round(spot, 3).tolist()); return False
    ctx.set_grip(1.); ctx.detach(arm); ctx.holding.pop(arm, None); est.in_hand.pop(arm, None)
    yield from hold(ctx, 30)
    yield from decor.slide(ctx, [0., 0., .10], rate=.02)
    yield from tuck(ctx, (arm,))
    est.put_back(item['id'], spot)
    ctx.event('to_table', ok=True, arm=arm, spot=np.round(spot, 3).tolist())
    yield from leave_end(ctx, est, 'north')
    yield from look(ctx, est)
    return True


# ---- the sink -----------------------------------------------------------------------------------------------------
def into_sink(ctx, est, arm):
    """Let the bowl in @arm's hand go over the sink's basin. Returns True if let go there."""
    stance = K.SINK['stance']
    status = yield from go(ctx, est, K.SINK['staging'], stance[2], tolerance=.05)
    if status != 'arrived':
        return False
    yield from settle(ctx, est, 6)
    status = yield from straight(ctx, stance[:2], stance[2], speed=.15, tolerance=.02)
    ctx.arm = arm; base = est.base(); over = np.asarray(K.SINK['over'], float)
    target = np.r_[in_base(base, over[:2]), over[2] - BASE_Z]
    # (the bowl hangs from its rim: its middle 7 cm further along the hand's way than the fingers)
    way = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], target, ctx.tip()[:3, :3], step=.03)
    ok = way is not None
    if ok:
        yield from follow(ctx, way, rate=.012, max_steps=600)
        yield from hold(ctx, 10)
    ctx.set_grip(1.); ctx.detach(arm); ctx.holding.pop(arm, None)
    yield from hold(ctx, 40)
    ctx.event('sink', ok=ok, arm=arm, status=status, tip=np.round(ctx.tip()[:3, 3], 3).tolist())
    yield from decor.slide(ctx, [0., 0., .10], rate=.02)
    yield from tuck(ctx, (arm,))
    yield from straight(ctx, K.SINK['staging'], stance[2], speed=.2)
    return ok


# ---- the fridge -----------------------------------------------------------------------------------------------------
def pose_of(base):
    x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0., x], [s, c, 0., y], [0., 0., 1., BASE_Z], [0., 0., 0., 1.]])


def plate_aside(ctx, est, arm, aside, ahead=.25, height=1.05):
    """Hold the plate in @arm's hand to the hand's side of the base (left: +y), level, its middle @aside from the base's
    middle line and @ahead of it, @height above the floor. Returns the follow status."""
    item = est.in_hand[arm]; side = 1. if arm == 'left' else -1.
    rotation = decor.edge_rotation(side * np.pi / 2, np.radians(item.get('pitch', 90.)))
    centre = np.array([ahead, side * aside, height - BASE_Z]); tip = centre - rotation @ item['hold']
    ctx.arm = arm
    plan = decor.plan_pinch(ctx, tip, rotation, way_in=.0001, hand_margin=-.02)
    if plan is None:
        ctx.event('plate_aside', ok=False, arm=arm, aside=aside); return 'no_plan'
    status = yield from move_to(ctx, plan['pre'], rate=.005, label='plate_aside')        # (slowly: turned at the arm's pace a pizza slid off, 1 of 6)
    return status


def turn_slowly(ctx, est, yaw, step=.15):
    """Turn the base to @yaw in small steps (a pizza on a plate in the hand stays on)."""
    for _ in range(30):
        error = wrap(yaw - est.base()[2])
        if abs(error) < .05:
            break
        yield from turn_to(ctx, est, wrap(est.base()[2] + float(np.clip(error, -step, step))))


def into_fridge(ctx, est, arm, shelf):
    """Slide the plate in @arm's (the left) hand in onto shelf @shelf (its top's height) of the fridge's west half and
    let it go there (the door is open). The base faces west in front of the fridge's east half, the plate to its left:
    it comes along y 0.55 with the plate's edge 5 cm clear of the east door's handle, the hand carries the plate west
    to the west half's opening, the base steps 0.15 m south, and the hand slides the plate in at the shelf's height.
    Returns True if let go there."""
    yaw = np.pi; stance = (5.45, .55); near = (5.45, .40)
    # the plate to the hand's side before the base turns to face west
    yield from plate_aside(ctx, est, arm, .35)
    ctx.event('fridge_phase', phase='aside')
    yield from turn_slowly(ctx, est, yaw)
    ctx.event('fridge_phase', phase='turned')
    status = yield from go(ctx, est, (5.95, 1.10), yaw, tolerance=.06, face_travel=False)
    ctx.event('fridge_phase', phase='by the fridge', status=status)
    if status != 'arrived':
        return False
    status = yield from straight(ctx, (5.95, stance[1]), yaw, speed=.15, tolerance=.03)
    status = yield from straight(ctx, stance, yaw, speed=.12, tolerance=.015)
    yield from settle(ctx, est, 6)
    ctx.event('fridge_phase', phase='at the stance')
    ctx.arm = arm; item = est.in_hand[arm]; hold_offset = item['hold']
    rotation = decor.edge_rotation(np.pi / 2, np.radians(item.get('pitch', 90.)))
    z = shelf + K.PLATE['height'] / 2 + CLEAR_ABOVE_SHELF

    def tip_for(base, centre_map, height):
        return np.r_[in_base(base, centre_map), height - BASE_Z] - rotation @ hold_offset

    import door
    door.keep_off(ctx, est)
    # The plate as the arm planner keeps it off things on the way to the opening: its rim, not the ball round its
    # middle that the carry has (that left the rim out, and the plate pushed the open door on to its stop, 2 of 3).
    held = ctx.payload.get(arm); tip = ctx.tip(); centre = tip[:3, 3] + tip[:3, :3] @ hold_offset
    rim = [centre + [.15 * np.cos(a), .15 * np.sin(a), 0.] for a in np.arange(0., 2 * np.pi, np.pi / 4)]
    ctx.attach(arm, np.array([centre] + rim), np.array([.10] + [.06] * len(rim))); ok = False
    # the plate west, in front of the opening (a few centimetres east of it), at the shelf's height
    base = est.base(); front = tip_for(base, (K.PLATE_AT[0] + FRONT_EAST, .25), z)
    plan = decor.plan_pinch(ctx, front, rotation, way_in=.0001, hand_margin=-.02)
    status = 'no_plan' if plan is None else (yield from move_to(ctx, plan['pre'], rate=.010, label='to_fridge'))
    if status != 'done':
        # (kept in the hand: let go here it fell on the floor)
        ctx.event('fridge', ok=False, reason='no way to the opening' if plan is None else status, arm=arm, shelf=shelf)
        if held is not None:
            ctx.payload[arm] = held
        return False
    status = yield from straight(ctx, near, yaw, speed=.08, tolerance=.012)
    yield from hold(ctx, 10)
    base = est.base()
    # Where the plate really is in the hand, from a look at it: the opening leaves it 2.5 cm a side, and the
    # plate held a few centimetres off what its grip put it at met the opening's side (2 of 4).
    planned = (pose_of(base) @ np.r_[ctx.tip()[:3, 3] + rotation @ hold_offset, 1.])[:2]
    seen = est.plate_look(planned, z) if hasattr(est, 'plate_look') else None
    ctx.event('plate_look', planned=np.round(planned, 3).tolist(), seen=None if seen is None else np.round(seen, 3).tolist(), debug=dict(getattr(est, 'plate_debug', {})))
    if seen is not None and float(np.linalg.norm(seen - planned)) < .08:
        shift_local = in_base((0., 0., base[2]), seen - planned)
        hold_offset = rotation.T @ (rotation @ hold_offset + np.r_[shift_local, 0.])
    # in line with the opening first, then straight in: as deep as the arm reaches (the second plate, on the first
    # one's pizza, had no way to the full depth), the rest of the way pushed (push_in)
    ahead = tip_for(base, (K.PLATE_AT[0], .10), z)
    way = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], ahead, rotation, step=.02)
    if way is not None:
        yield from follow(ctx, way, rate=.008, max_steps=300)
    for depth in (K.PLATE_AT[1], -.40, -.36, -.32):
        inside = tip_for(base, (K.PLATE_AT[0], depth), z)
        found = straight_in(ctx, ctx.tip()[:3, 3], inside, rotation)
        if found is not None:
            break
    way = None
    if found is not None:
        first, way = found
        if first is not None:
            # (the arm's other way of holding the plate here: the one that goes on in a straight line)
            status = yield from move_to(ctx, first, rate=.008, label='to_branch')
            if status != 'done':
                way = None
    ctx.payload.pop(arm, None)
    if way is not None:
        status = yield from follow(ctx, way, rate=.008, max_steps=900)
        short = float(np.linalg.norm(ctx.tip()[:3, 3] - inside))
        reached = (pose_of(est.base()) @ np.r_[ctx.tip()[:3, 3] + rotation @ hold_offset, 1.])[:2]
        ctx.event('fridge_way', ok=bool(reached[1] < IN_ENOUGH), status=status, depth=depth, short=round(short, 3), reached=np.round(reached, 3).tolist(), base=np.round(est.base(), 3).tolist())
        if reached[1] < IN_ENOUGH:
            down_to = ctx.tip()[:3, 3] - [0., 0., z - (shelf + K.PLATE['height'] / 2 + .008)]
            way = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], down_to, rotation, step=.01)
            if way is not None:
                yield from follow(ctx, way, rate=.006, max_steps=300)
            ok = True
        else:
            # (the plate met something on the way in: it comes back out held, not let go short of the shelf)
            back = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], front, rotation, step=.02)
            if back is not None:
                yield from follow(ctx, back, rate=.008, max_steps=600)
            if held is not None:
                ctx.payload[arm] = held
            return False
    else:
        ctx.event('fridge', ok=False, reason='no way in', arm=arm, shelf=shelf)
        if held is not None:
            ctx.payload[arm] = held
        return False
    centre = (pose_of(est.base()) @ np.r_[ctx.tip()[:3, 3] + rotation @ hold_offset, 1.])[:2]
    ctx.set_grip(1.); ctx.detach(arm); ctx.holding.pop(arm, None); est.in_hand.pop(arm, None)
    yield from hold(ctx, 40)
    ctx.event('fridge', ok=ok, arm=arm, shelf=shelf, tip=np.round(ctx.tip()[:3, 3], 3).tolist(), centre=np.round(centre, 3).tolist())
    tip = ctx.tip()[:3, 3]
    way = interpolate_cartesian(ctx.ik, ctx.q, tip, tip + [0., 0., .03], ctx.tip()[:3, :3], step=.01)
    if way is not None:
        yield from follow(ctx, way, rate=.01, max_steps=100)
    base = est.base(); back = ctx.tip()[:3, 3] + np.r_[in_base((0., 0., base[2]), (0., .30 if ok else .45)), 0.]
    # (kept off the open door: drawn straight back unchecked the arm pushed it from 102 degrees on to its stop, 301)
    way = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], back, ctx.tip()[:3, :3], step=.02, validator=ctx.validator(relax_from=ctx.q))
    if way is not None:
        yield from follow(ctx, way, rate=.012, max_steps=500)
    else:
        # (no straight way back that keeps off the door: a planned one to the same place; drawn back unchecked the
        # arm pushed the door on to its stop, where no hand reaches its edge, 305)
        plan = decor.plan_pinch(ctx, back, ctx.tip()[:3, :3], way_in=.0001, hand_margin=-.03)
        status = 'no_plan' if plan is None else (yield from move_to(ctx, plan['pre'], rate=.012, label='drawn_back'))
        ctx.event('fridge_phase', phase='drawn back by a plan', status=status)
    ctx.event('fridge_phase', phase='drawn back')
    if ok:
        yield from push_in(ctx, est, arm, centre, shelf)
    if held is not None and arm in ctx.holding:
        ctx.payload[arm] = held
    yield from tuck(ctx, (arm,))
    ctx.event('fridge_phase', phase='tucked')
    yield from straight(ctx, stance, yaw, speed=.15)
    return ok


def straight_in(ctx, start, goal, rotation, step=.02):
    """A straight way of the hand from @start (where it is now) to @goal at @rotation: (None, way) from the
    configuration it is in, or (configuration to go to first, way) on another branch of the arm's solutions that goes
    the whole way (from where the stacked plate's way began, the arm at its limits had none), or None."""
    way = interpolate_cartesian(ctx.ik, ctx.q, start, goal, rotation, step=step)
    if way is not None:
        return None, way
    for solution in ctx.ik.solve(ctx.q, goal, rotation, count=24, all_solutions=True) or []:
        at = ctx.q.copy(); at[ctx.ik.columns] = solution
        back = interpolate_cartesian(ctx.ik, at, goal, start, rotation, step=step)
        if back is not None:
            return back[-1], back[-2::-1] + [solution]
    return None


def push_in(ctx, est, arm, centre, shelf, to=PUSH_TO):
    """Push the plate let go on @shelf (its top's height) with its middle at @centre (map) further in, till its
    middle is at y @to: the closed hand comes to its near rim with the fingers pointing in and a little down, and
    pushes. (With the fingers straight down on the rim the wrist met the shelf above: plates were let go 5-7 cm short
    of where they were to go, their near edge in the door's way.) Returns the status."""
    if centre[1] <= to + .01:
        return 'deep enough'
    ctx.arm = arm; base = est.base(); ctx.set_grip(-1., arm)
    # (the fingers close across, level: closing up and down the lower finger met the fridge's face under the shelf)
    h = wrap(-np.pi / 2 - base[2]); p = np.radians(PUSH_PITCH)
    rotation = decor.frame(np.array([np.cos(h) * np.cos(p), np.sin(h) * np.cos(p), -np.sin(p)]), np.array([-np.sin(h), np.cos(h), 0.]))
    z = shelf + K.PLATE['height'] * .6 - BASE_Z; rim = centre[1] + K.PLATE['radius']
    start = np.r_[in_base(base, (centre[0], rim + .05)), z]; end = np.r_[in_base(base, (centre[0], rim - (centre[1] - to))), z]
    plan = decor.plan_pinch(ctx, start, rotation, way_in=.0001, hand_margin=-.02)
    status = 'no_plan' if plan is None else (yield from move_to(ctx, plan['pre'], rate=.012, label='push_in'))
    if status == 'done':
        way = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], end, rotation, step=.01)
        status = 'no_way' if way is None else (yield from follow(ctx, way, rate=.004, max_steps=700))
        short = float(np.linalg.norm(ctx.tip()[:3, 3] - end))
        ctx.event('push_in', ok=short < .03, status=status, short=round(short, 3), centre=np.round(centre, 3).tolist())
        yield from decor.slide(ctx, np.r_[in_base((0., 0., base[2]), (0., .20)), .02], rate=.012)
    else:
        ctx.event('push_in', ok=False, status=status, centre=np.round(centre, 3).tolist())
    ctx.set_grip(1., arm)
    return status


def open_door(ctx, est):
    """Open the fridge's right door (stages: see door.py). Returns True if it is open wide enough for a plate."""
    import door
    ok = yield from door.open_wide(ctx, est)
    return ok


def close_door(ctx, est, attempts=5):
    """Shut the fridge's right door (door.shut), once more from a fresh look at it while it is still open (the
    hand's pinch of the free edge closed beside it, 314). Returns True if it is shut."""
    import door
    ok = False
    for attempt in range(attempts):
        ok = yield from door.shut(ctx, est)        # (a push on the door's outer face had no way to its start from any stance: door.PUSH_SHUT unused)
        ctx.event('shut', ok=ok, attempt=attempt, believed=round(float(np.degrees(est.door)), 1))
        if ok:
            break
    return ok


# ---- the whole task -------------------------------------------------------------------------------------------------
def full_task(ctx, est, parts=('bowls', 'door', 'plates', 'shut')):
    ctx.holding = {}; est.in_hand = {}; est.door = getattr(est, "door", 0.)
    yield from tuck(ctx)
    found = yield from find(ctx, est)
    ctx.event('stage', done='find', all=found, known=[(i['kind'], np.round(i['xy'], 3).tolist()) for i in est.known()])
    sunk = 0; shelved = 0
    if 'bowls' in parts:
        for turn in range(5):
            bowls = est.known('bowl')
            if not bowls or sunk >= 2:
                break
            here = np.asarray(est.base()[:2]); bowls.sort(key=lambda i: float(np.linalg.norm(i['xy'] - here)))
            arm, why, end = yield from take(ctx, est, bowls[0])
            ctx.event('stage', done='take', thing='bowl', arm=arm, why=why)
            if arm is None:
                if why == 'no_stance':
                    yield from find(ctx, est)
                continue
            ok = yield from into_sink(ctx, est, arm)
            ctx.event('stage', done='sink', ok=ok); sunk += bool(ok)
    if 'plates' in parts and est.door < .05:
        # A plate that a hand reaches from the south end only goes to the north half first: with the door open the
        # way into the gap between the table and the fridge is shut.
        for turn in range(3):
            south = [i for i in est.known('plate') if not north_reach(ctx, est, i)]
            if not south or not free_spot(est, south[0]['id']):
                break
            arm, why, end = yield from take(ctx, est, south[0], arms=('left',))
            ctx.event('stage', done='take', thing='plate', arm=arm, why=why, to='north half')
            if arm is None:
                continue
            spots = free_spot(est, south[0]['id'])
            ok = bool(spots) and (yield from put_on_table(ctx, est, arm, spots[0]))
            ctx.event('stage', done='to_north', ok=ok, spot=np.round(spots[0], 3).tolist() if spots else None)
            if not ok:
                break
    if 'door' in parts:
        for attempt in range(3):
            opened = yield from open_door(ctx, est)
            ctx.event('stage', done='door', ok=opened, angle=round(float(est.door), 3), attempt=attempt)
            if opened:
                break
        if not opened:
            parts = tuple(p_ for p_ in parts if p_ != 'plates')     # (a plate let go at a shut door falls to the floor)
    if 'plates' in parts:
        # (the second shelf is out of the arm's reach from where the base stands; the second plate goes onto the first
        # one's pizza: both pizzas are then inside the fridge's lower part, each on its own plate)
        shelves = [K.SHELVES[0], K.SHELVES[0] + K.PLATE['height'] + K.STACK]
        misses = {}
        for turn in range(6):
            plates = est.known('plate')
            if not plates or not shelves:
                break
            here = np.asarray(est.base()[:2]); plates.sort(key=lambda i: (misses.get(i['id'], 0), float(np.linalg.norm(i['xy'] - here))))
            # (held with the fingers 75 degrees down the arm reaches in to the shelf's depth on the first one's pizza, 7 cm
            # higher, too; straight down it stopped at the opening there, and the wrist stood higher, nearer the
            # shelf above)
            pitches = (75., 90.)
            arm, why, end = yield from take(ctx, est, plates[0], arms=('left',), pitches=pitches)
            ctx.event('stage', done='take', thing='plate', arm=arm, why=why)
            if arm is None:
                misses[plates[0]['id']] = misses.get(plates[0]['id'], 0) + 1     # (the other plate next: one that lay under the other had no grip)
                continue
            ok = yield from into_fridge(ctx, est, arm, shelves[0])
            if not ok and arm in ctx.holding:
                ok = yield from into_fridge(ctx, est, arm, shelves[0])          # (once more, from the stance again)
            if not ok and arm in ctx.holding:
                spots = free_spot(est, est.in_hand[arm]['id'])
                if spots:
                    yield from put_on_table(ctx, est, arm, spots[0])
            ctx.event('stage', done='fridge', ok=ok, shelf=shelves[0])
            if ok:
                shelves.pop(0); shelved += 1
    if 'shut' in parts:
        shut = yield from close_door(ctx, est)
        ctx.event('stage', done='shut', ok=shut)
    return f'{sunk} bowls {shelved} plates'
