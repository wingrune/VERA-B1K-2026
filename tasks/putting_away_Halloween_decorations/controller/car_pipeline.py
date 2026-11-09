"""The car task as a sequence of skills, driven by an estimator.

The robot starts in the garage, where the car stands with its trunk shut. A toy box stands on the living-room
floor; a camera and a tennis racket lie on the coffee table there. At the end the box must be in the trunk, the
camera in the box and the racket in the trunk.

  1. open the trunk lid (both hands are free now, and the lid stays open by its own weight);
  2. through the door to the corridor (the chassis pushes it open; it stays open), to the living room;
  3. camera from the table into the box; box by a long wall, the camera in it; back to the car, box into the trunk;
  4. to the living room again: racket from the table, back to the car, racket into the trunk over the box.

Two trips, one thing in the hands at a time: with the racket upright in one hand the other arm found no way down
to the box (the trunk bends and the racket swings with it), and a hand that holds the box level cannot follow the
trunk down to the table. The way there and back is 2200 of the 28839 steps.

The estimator supplies
  base() = (x, y, yaw) in M; where(name), direction(name) = a thing in M; point(name) = (xy, direction) in the
  base frame for the hand; see(what), fresh(name), forget(name); fix_at_lid(), lid_shows_closed();
  hold_axis(axis), settle(flag).
"""
import numpy as np
from manip import hold, drive, follow, move_to, pose2, wrap, BASE_Z, TALL_TRUNK, TUCK
from skills import interpolate_cartesian
from car_skills import (tuck, open_lid, plan_top_pinch, top_pinch, set_down, carry_box, box_carry_way, supposing, carry_racket, across_the_trunk, racket_in_hand, level_way, closing_angle, park, down, held, box_spheres, racket_spheres, camera_spheres,
                        STANCE, TRUNK_STANCE, RACKET_STANCE, LID_MIDDLE, TRUNK_FLOOR, BOX_IN_TRUNK, RACKET_IN_TRUNK, OVER_THE_SILL, TABLE_TOP, HUMP_ABOVE_TABLE, HANDLE_ABOVE_TABLE, BOX, BOX_GRIP_BELOW_RIM, SQUARE)

FOOTPRINT = (-.40, .24, .345)       # chassis: rear, front and half width in the base frame
SWEEP = .53                         # the chassis corner farthest from the base centre: the circle a turn on the spot sweeps
TURN_CLEAR = SWEEP + .03
BOX_RADIUS = .274                   # half the toy box's diagonal
SOUTH, NORTH = -np.pi / 2, np.pi / 2
# Fixed places of the scene (map frame).
LID_STAGING = (-7.75, LID_MIDDLE)   # 0.6 m behind the stance at the car: room to turn, and the lid's edge in view
DOOR_X = -3.86                      # the line through the doorway between garage and corridor (0.83 m at chassis height with the leaf open; its handle leaves 0.75 m higher up)
GARAGE_SIDE = (DOOR_X, 3.25)
CORRIDOR_SIDE = (DOOR_X, 1.52)
CORRIDOR_TURN = (-2.60, 1.54)       # in the middle of the corridor (1.36 m wide): room to turn with the box in front
ENTRY = (2.9, 2.6)                  # in the kitchen, where the living room opens
# Where the robot stands to look for the things, and the headings it looks along from there: from the way in the
# whole room lies ahead; from east of the table everything is within three metres.
LOOKOUTS = (((4.35, 3.35), (NORTH, 2.45, .70)), ((4.85, 5.3), (np.pi, 2.0, .8, -.8, -2.2)), ((4.75, 6.4), (-2.7, SOUTH, np.pi)), ((3.52, 3.8), (NORTH, 2.3, .85)))
# Where the straight move to a stance starts from, as (length, direction): the direction is counted from "straight
# behind the stance" (0), towards the left (+). Sideways for the gap between the sofa and the table, which the
# chassis enters facing the table.
STAGING = tuple((length, turn) for turn in (0., np.pi / 2, -np.pi / 2, .8, -.8) for length in (.40, .55, .75, 1.0, 1.3, 1.7, 2.1, 2.5, 2.9))
# Where a thing may lie in the base frame when the hand takes it, in order of choice: (ahead, to the side). Ahead
# in steps of 2 cm: between the sofa and the table the chassis has 3 cm to spare, and with it one distance only.
REACH = tuple(sorted((round(float(d), 2) for d in np.arange(.30, .901, .02)), key=lambda d: abs(d - .62)))
TABLE_OFFSETS = {side: tuple((d, side * lateral) for d in REACH for lateral in (.19, .08)) for side in (1., -1.)}
# ... and the box's middle when the hand works on it, facing one of its long walls (to the side as well, where
# furniture stands next to the box) or, where neither long wall can be faced, one of its short walls.
# To the side of one arm before straight ahead.
BOX_OFFSETS = tuple((d, lateral) for d in (.64, .58, .70, .52, .46) for lateral in (.12, -.12, .22, -.22, 0., .30, -.30))
BOX_OFFSETS_END = tuple((d, lateral) for d in (.72, .66, .78, .60) for lateral in (0., .12, -.12, .22, -.22))
BOX_END = BOX['long'] / 2 - .016        # from the box's middle to the middle of a short wall


def gap(base, xy, radius):
    """Gap between the chassis standing at @base = (x, y, yaw) and a disc of @radius around @xy."""
    x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = np.asarray(xy, float)[:2] - [x, y]
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    nearest = np.array([np.clip(local[0], FOOTPRINT[0], FOOTPRINT[1]), np.clip(local[1], -FOOTPRINT[2], FOOTPRINT[2])])
    return float(np.linalg.norm(local - nearest)) - radius


def straight(ctx, goal, yaw, speed=.25, tolerance=.03, max_steps=1500):
    """A slow straight move at a fixed heading (the base is holonomic: any direction)."""
    status = yield from drive(ctx, goal, yaw, speed=speed, direct=True, face_travel=False, tolerance=tolerance, yaw_tolerance=.03, min_speed=.04, max_steps=max_steps)
    short = float(np.linalg.norm(np.asarray(goal, float) - ctx.est.base()[:2]))
    if status != 'arrived':
        ctx.event('straight', ok=False, goal=np.round(goal, 2).tolist(), status=status, short=round(short, 3))
    return 'arrived' if status == 'arrived' or short < .05 else status


def settle(ctx, est, steps=24):
    """Stand still and let the estimator refine the pose from what it sees."""
    est.settle(True)
    yield from hold(ctx, steps)
    est.settle(False)


def look(ctx, est, what=('box', 'table')):
    yield from hold(ctx, 3)
    return est.see(what)


def on_floor(ctx, est):
    """The toy box as an obstacle, while it stands on the floor: [(xy, radius)]."""
    if 'box' in getattr(ctx, 'holding', {}).values() or est.where('box') is None:
        return []
    return [(est.where('box'), BOX_RADIUS)]


def round_the_box(ctx, est, here, goal, extra, obstacles=None):
    """Discs the base centre stays out of on a way from @here to @goal, for the toy box on the floor with @extra
    room round it: [(xy, radius)]. Where the way starts or ends nearer to the box than that (the base has just
    worked at it, or is going to), the base gets no nearer to it than its start and its goal are. (With no disc at
    all for such a goal, a way to a point behind the box led straight over it: the box was shoved 0.35 m.)"""
    here = np.asarray(here, float)[:2]; goal = np.asarray(goal, float)[:2]
    return [(xy, max(0., min(radius + extra, float(np.linalg.norm(here - xy)) - .03, float(np.linalg.norm(goal - xy)) - .03)))
            for xy, radius in (on_floor(ctx, est) if obstacles is None else obstacles)]


def route(ctx, est, goal, yaw, turning=False, **options):
    """drive() round the toy box: with room to turn anywhere on the way if there is such a way, else keeping the
    chassis's width from it. The way is planned again when the box comes into view on the way, or turns out to
    stand elsewhere (the camera looks along the way, and the box is often first seen from a metre or two).
    turning: the base is going to turn on the spot at the goal; 'box_in_the_way' if the box turns out to stand
    where that turn would sweep."""
    goal = np.asarray(goal, float)[:2]
    for again in range(6):
        here = np.asarray(est.base()[:2], float); floor = on_floor(ctx, est); keepout = None
        # (The narrower two: the chassis's half width and what the base cuts off a bend of the way, 3 cm.)
        for extra in (TURN_CLEAR + .03, .345 + .13, .345 + .08):
            keepout = round_the_box(ctx, est, here, goal, extra, floor)
            if ctx.scene.path(here, goal, keepout=keepout) is not None:
                break
            keepout = None
        if keepout is None:
            ctx.event('route', ok=False, goal=np.round(goal, 2).tolist()); return 'no_route'
        steps = drive(ctx, goal, yaw, keepout=keepout, lookahead=.20 if keepout else .35, **options)
        while True:
            try:
                action = next(steps)
            except StopIteration as stop:
                return stop.value
            yield action
            now = on_floor(ctx, est)
            if again < 5 and (len(now) != len(floor) or any(float(np.linalg.norm(new[0] - old[0])) > .10 for new, old in zip(now, floor))):
                break
        steps.close()
        ctx.event('route', again=True, goal=np.round(goal, 2).tolist(), box=[np.round(xy, 2).tolist() for xy, _ in now])
        if turning and any(float(np.linalg.norm(goal - xy)) < TURN_CLEAR + radius + .05 for xy, radius in now):
            for _ in range(8):
                yield ctx.action(base=(0., 0., 0.))
            return 'box_in_the_way'
    return 'no_route'


# ---- the trunk ------------------------------------------------------------------------------------------
def to_lid(ctx, est, stance):
    """To the stance behind the car, the map pose taken from the trunk lid's edge on the way (while the lid is shut):
    from the staging point 0.6 m back, then again at the stance. Between the two the position is kept by dead
    reckoning, so that the scan match does not pull it back to where the walls put it."""
    status = yield from route(ctx, est, LID_STAGING, 0., tolerance=.05)
    if status != 'arrived':
        return status
    yield from settle(ctx, est, 12)
    shift = est.fix_at_lid()
    ctx.event('lid_fix', where='staging', shift=None if shift is None else np.round(shift, 4).tolist())
    est.hold_axis('xy')
    status = yield from straight(ctx, stance[:2], stance[2], speed=.2, tolerance=.012)
    yield from hold(ctx, 8)
    shift = est.fix_at_lid()
    ctx.event('lid_fix', where='stance', shift=None if shift is None else np.round(shift, 4).tolist(), base=np.round(est.base(), 4).tolist())
    return status


def open_trunk(ctx, est, attempts=3):
    """Open the trunk lid; True when the base has stood behind the car and the lid no longer shows where it lies
    shut. (Not seeing the shut lid from elsewhere says nothing: a start that was localised at the wrong end of
    the garage never got behind the car, did not see the lid from where it stalled, and went on as if the trunk
    were open. The localiser follows the rival start poses and had put the pose right by then: the next attempt
    starts from there.)"""
    for attempt in range(attempts):
        status = yield from to_lid(ctx, est, STANCE)
        result = (yield from open_lid(ctx, est)) if status == 'arrived' else status
        for again in range(2):
            if result not in ('edge_missed', 'edge_lost', 'lower_end_missed', 'lower_end_lost'):
                break
            # A hand closed on nothing or lost the lid: the lid is shut again and the base has not moved.
            yield from hold(ctx, 30)
            result = yield from open_lid(ctx, est)
        back = yield from straight(ctx, LID_STAGING, 0., speed=.3)
        yield from hold(ctx, 10)
        closed = est.lid_shows_closed(); est.hold_axis(None)
        ctx.event('trunk', attempt=attempt, result=result, shows_closed=bool(closed), behind_the_car=status == 'arrived' and back == 'arrived')
        if status == 'arrived' and back == 'arrived' and not closed:
            return True
        if status != 'arrived':
            yield from settle(ctx, est, 30)         # let the scan match (and the rival poses) settle before the next try
    return False


# ---- between the rooms ------------------------------------------------------------------------------------
def leave_garage(ctx, est):
    """From the garage through the door into the corridor. The door has no latch: the chassis pushes it open, and it
    stays where it is left. The position is kept by dead reckoning through the doorway (the leaf swings in view)."""
    for attempt in range(3):
        status = yield from route(ctx, est, GARAGE_SIDE, SOUTH)
        if status != 'arrived':
            return 'door_' + status
        yield from settle(ctx, est)
        yield from straight(ctx, GARAGE_SIDE, SOUTH, tolerance=.015)
        est.hold_axis('xy')
        status = yield from straight(ctx, CORRIDOR_SIDE, SOUTH, speed=.25, max_steps=900)
        est.hold_axis(None)
        ctx.event('door', way='out', attempt=attempt, status=status, base=np.round(est.base(), 3).tolist())
        if status == 'arrived':
            # Sideways to the middle of the corridor before turning: a turn next to the door swept the leaf with the
            # chassis's rear and shoved the base 10 cm along the corridor, where the scan match cannot tell.
            yield from straight(ctx, CORRIDOR_TURN, SOUTH, speed=.3)
            yield from settle(ctx, est)
            return 'in_corridor'
        yield from straight(ctx, GARAGE_SIDE, SOUTH)
    return 'doorway_' + status


def back_to_garage(ctx, est):
    """From the living room back through the door, facing north all the way through the corridor's west end: with
    the box in front the robot turns where the corridor is free on both sides."""
    status = yield from route(ctx, est, CORRIDOR_TURN, NORTH)
    if status != 'arrived':
        return 'corridor_' + status
    status = yield from straight(ctx, CORRIDOR_SIDE, NORTH, speed=.3)
    yield from settle(ctx, est)
    yield from straight(ctx, CORRIDOR_SIDE, NORTH, tolerance=.015)
    est.hold_axis('xy')
    status = yield from straight(ctx, GARAGE_SIDE, NORTH, speed=.25, max_steps=900)
    est.hold_axis(None)
    ctx.event('door', way='back', status=status, base=np.round(est.base(), 3).tolist())
    yield from settle(ctx, est)
    return 'in_garage' if status == 'arrived' else 'doorway_' + status


# ---- standing where the hand reaches ------------------------------------------------------------------------
def stance_options(ctx, here, target, offset, headings, obstacles=(), margins=(.025, .012), turn_clear=()):
    """Base poses from which @target (map xy) lies at @offset in the base frame, each with the point the base comes
    to first: [(cost, (x, y, yaw, staging, tight, offset))].

    The base gets to the staging point by a planned route and turns to the stance's heading there; the rest is a
    straight move without turning, and the same way back after the work. A pose counts if the chassis's rectangle
    has room at the stance and all along that move, and clears @obstacles = [(xy, radius)]. tight: the route
    planner's disc has no room at the stance (next to the table it never has). turn_clear: discs the base must
    be able to turn clear of at the staging point without being kept away from them at the stance (the box, for
    a stance at the box)."""
    target = np.asarray(target, float)[:2]; here = np.asarray(here, float)[:2]; out = []
    for yaw in headings:
        c, s = np.cos(yaw), np.sin(yaw); xy = target - np.array([c * offset[0] - s * offset[1], s * offset[0] + c * offset[1]])
        if not ctx.scene.footprint_free(xy[0], xy[1], yaw, FOOTPRINT, *margins) or any(gap((xy[0], xy[1], yaw), centre, r) < .03 for centre, r in obstacles):
            continue
        tight = not ctx.scene.base_free(xy, margin=-.02)
        for length, turn in STAGING:
            direction = yaw + np.pi + turn; staging = xy + length * np.array([np.cos(direction), np.sin(direction)])
            if not ctx.scene.base_free(staging, margin=-.02) or any(float(np.linalg.norm(staging - centre)) < TURN_CLEAR + r for centre, r in list(obstacles) + list(turn_clear)):
                continue
            swept = [xy + d * (staging - xy) / length for d in np.arange(.05, length, .05)]
            if not all(ctx.scene.footprint_free(q[0], q[1], yaw, FOOTPRINT, .02, .008) for q in swept) or any(gap((q[0], q[1], yaw), centre, r) < .03 for q in swept for centre, r in obstacles):
                continue
            out.append((float(np.linalg.norm(staging - here)) + length + .3 * abs(turn), (float(xy[0]), float(xy[1]), float(yaw), staging, tight, offset)))
            break
    return out


def stand_for(ctx, est, name, offsets, feasible, headings=None, what=('box', 'table')):
    """Stand where the active arm reaches @name: plan a stance, drive to its staging point, look, move in, look
    again. Returns dict(pose, staging, tight, offset) or None.

    offsets: where the thing may lie in the base frame, in order of choice. feasible(pose, offset, tight): the arm
    has a way to the thing from there (checked with the arm planner; it costs a second or two per pose, so poses
    are tried nearest first). headings: the headings to try (a function of the estimator), sixteen round the
    thing if not given."""
    failed = []
    for attempt in range(4):
        target = est.where(name)
        if target is None:
            return None
        here = np.asarray(est.base()[:2], float); toward = float(np.arctan2(target[1] - here[1], target[0] - here[0]))
        # Sixteen headings round the thing, and the four along the map's axes: between the sofa and the table the
        # chassis fits only square to them.
        tries = headings(est) if headings is not None else [wrap(toward + (k + 1) // 2 * (1 if k % 2 else -1) * 2 * np.pi / 16) for k in range(16)] + [0., np.pi / 2, np.pi, -np.pi / 2]
        floor = on_floor(ctx, est); obstacles = [] if name == 'box' else floor
        options = []
        for yaw in tries:
            group = offsets(est, yaw) if callable(offsets) else offsets
            for rank, offset in enumerate(group):
                if offset is not None:              # (padding: a group that is to rank behind the others)
                    options += [(cost + .04 * rank, pose) for cost, pose in stance_options(ctx, here, target, offset, [yaw], obstacles, turn_clear=floor if name == 'box' else ())]
        options.sort(key=lambda item: item[0])
        chosen = None; checked = 0; unreachable = []
        for cost, (x, y, yaw, staging, tight, offset) in options:
            if any(np.hypot(x - f[0], y - f[1]) < .10 and abs(wrap(yaw - f[2])) < .3 for f in failed) or any(float(np.linalg.norm(staging - u)) < .12 for u in unreachable):
                continue
            # A staging point that is free but that no route leads to (behind the table, between furniture) is no use.
            if ctx.scene.path(here, staging, keepout=round_the_box(ctx, est, here, staging, .345 + .08, floor)) is None:
                unreachable.append(staging); continue
            checked += 1
            if feasible((x, y, yaw), offset, tight):
                chosen = (x, y, yaw, staging, tight, offset); break
            if checked >= 40:
                break
        if chosen is None:
            ctx.event('stance', ok=False, thing=name, options=len(options), failed=len(failed)); return None
        x, y, yaw, staging, tight, offset = chosen
        ctx.event('stance', thing=name, pose=np.round([x, y, yaw], 3).tolist(), staging=np.round(staging, 3).tolist(), tight=bool(tight), offset=list(offset), options=len(options))
        status = yield from route(ctx, est, staging, yaw, tolerance=.04)
        if status != 'arrived':
            failed.append((x, y, yaw)); continue
        yield from settle(ctx, est, 8)
        yield from look(ctx, est, what)
        now = est.where(name)
        if now is None:
            return None
        if float(np.linalg.norm(now - target)) > .04:
            continue                                    # seen better from here: plan again for where it is
        status = yield from straight(ctx, (x, y), yaw, speed=.2, tolerance=.02)
        if status != 'arrived':
            yield from straight(ctx, staging, yaw, speed=.2)
            failed.append((x, y, yaw)); continue
        yield from look(ctx, est, what)
        return dict(pose=(x, y, yaw), staging=staging, tight=tight, offset=offset)
    return None


def step_back(ctx, est, at):
    yield from straight(ctx, at['staging'], at['pose'][2], speed=.2)


# ---- the things -------------------------------------------------------------------------------------------
# ---- whether the arm has a way to a thing from a stance (the same checks serve dev/study_stances.py) ----------------
# pose: the stance (x, y, yaw) in the map; offset: where the thing then lies in the base frame; tight: the chassis
# stands between furniture there. Directions are in the map frame.
def can_take_camera(ctx, closing, pose, offset, tight):
    return plan_top_pinch(ctx, np.r_[offset, TABLE_TOP + HUMP_ABOVE_TABLE - BASE_Z], closing - pose[2], base=pose2(*pose), relax=tight) is not None


def can_lower_into_box(ctx, axis, pose, offset, tight):
    return plan_top_pinch(ctx, np.r_[offset, BOX['floor'] + .02 + HUMP_ABOVE_TABLE - BASE_Z], axis - pose[2], base=pose2(*pose), relax=tight, approach=.12) is not None


def can_take_box(ctx, axis, pose, offset, tight):
    """The pinch of the wall that faces the stance (with the arm on whose side the box lies, which becomes the
    active one) and, from a long wall, the level way from there to the carrying posture as well."""
    arm = ctx.arm = 'left' if offset[1] >= 0. else 'right'; axis = axis - pose[2]; z = BOX['rim'] - BOX_GRIP_BELOW_RIM - BASE_Z
    if not facing_long_wall(axis, 0.):
        return plan_top_pinch(ctx, np.r_[offset[0] - BOX_END, offset[1], z], axis, base=pose2(*pose), relax=tight) is not None
    normal, closing = long_wall_grip(arm, axis)
    plan = plan_top_pinch(ctx, np.r_[np.asarray(offset) - BOX['wall'] * normal, z], closing, base=pose2(*pose), relax=tight, both_ways=False, locked=SQUARE)
    if plan is None:
        return False
    with supposing(ctx, arm, plan['pre']):
        ctx.attach(arm, *box_spheres(offset, axis, lift=ctx.tip()[2, 3] - z))
        return box_carry_way(ctx, ctx.validator(base=pose2(*pose), relax_from=ctx.q)) is not None


def can_take_racket(ctx, toward_head, pose, offset, tight):
    return plan_top_pinch(ctx, np.r_[offset, TABLE_TOP + HANDLE_ABOVE_TABLE - BASE_Z], toward_head - pose[2] + np.pi / 2, base=pose2(*pose), relax=tight) is not None


def take_camera(ctx, est, arm='left'):
    """Take the camera from the table by the hump on its top. Returns True when the hand holds it."""
    side = 1. if arm == 'left' else -1.; z = TABLE_TOP + HUMP_ABOVE_TABLE - BASE_Z
    for attempt in range(3):
        ctx.arm = arm
        if est.where('camera') is None:
            return False

        at = yield from stand_for(ctx, est, 'camera', TABLE_OFFSETS[side], lambda pose, offset, tight: can_take_camera(ctx, est.direction('camera'), pose, offset, tight))
        if at is None:
            return False
        point, closing = est.point('camera')
        ok = yield from top_pinch(ctx, np.r_[point, z], closing, relax=at['tight'], label='camera', look=lambda tip, angle: est.hand_view(arm, 'camera', tip, angle))
        if ok:
            tip = ctx.tip()[:3, 3]; ctx.attach(arm, *camera_spheres(tip[:2], closing, table=tip[2] + BASE_Z - HUMP_ABOVE_TABLE)); ctx.holding[arm] = 'camera'; est.taken('camera')
            # In front of the chest, clear of the table, before the base moves.
            yield from park(ctx, [(.32, side * .14, 1.02), (.30, side * .20, 1.05), (.36, side * .10, 1.0)], [down(closing + turn) for turn in (0., np.pi, np.pi / 2, -np.pi / 2)], label='carry_camera')
            yield from step_back(ctx, est, at)
            return True
        yield from tuck(ctx, (arm,))
        yield from step_back(ctx, est, at)
        if ok is False:
            est.forget('camera')                # the hand closed on nothing: it is not where it was believed
            yield from find(ctx, est, ('camera',))
    return False


def box_headings(est):
    """Facing a wall of the box, from any of its four sides."""
    axis = est.direction('box')
    return [wrap(axis + np.pi / 2), wrap(axis - np.pi / 2), wrap(axis), wrap(axis + np.pi)]


def long_wall_headings(est):
    """Facing one of the box's long walls."""
    return box_headings(est)[:2]


def facing_long_wall(axis, yaw):
    """True if a base heading @yaw looks at a long wall of a box whose long side runs along @axis (both in one frame)."""
    return abs(np.cos(axis - yaw)) < .5


def long_wall_grip(arm, axis):
    """How a hand takes the long wall that faces the robot, for a box whose long side runs along @axis in the base
    frame: (the direction across that wall, away from the robot; the direction the fingers close along).

    The hand is put on one way round only, the box on the outer side of the left hand's fingers and of the right
    hand's (seen from the wrist), and the trunk is not turned for the reach: from the other way round, and from a
    turned trunk, the arm has no way to the carrying posture that keeps the box level (the camera fell out of a
    box that was brought there by a planned joint path instead)."""
    normal = np.array([-np.sin(axis), np.cos(axis)]); normal = normal * np.sign(float(normal[0])); across = float(np.arctan2(normal[1], normal[0]))
    return normal, across + (0. if arm == 'left' else np.pi)


def box_offsets(est, yaw):
    """Where the box's middle may lie in the base frame for a stance heading @yaw."""
    return BOX_OFFSETS if facing_long_wall(est.direction('box'), yaw) else BOX_OFFSETS_END


def camera_into_box(ctx, est, arm='left'):
    """Lower the camera into the box, its long side along the box's, and let go. Returns True when it was let go
    over the box."""
    ctx.arm = arm; z = BOX['floor'] + .02 + HUMP_ABOVE_TABLE - BASE_Z
    for attempt in range(3):
        if est.where('box') is None:
            return False
        side = 1. if arm == 'left' else -1.
        at = yield from stand_for(ctx, est, 'box', lambda est_, yaw: tuple(o if o is None or side * o[1] > -.15 else None for o in box_offsets(est_, yaw)),
                                  lambda pose, offset, tight: can_lower_into_box(ctx, est.direction('box'), pose, offset, tight), headings=box_headings, what=('box',))
        if at is None:
            return False
        point, axis = est.point('box')
        ok = yield from set_down(ctx, np.r_[point, z], axis, label='camera_in_box', relax=at['tight'])
        if ok:
            ctx.holding.pop(arm, None)
            yield from tuck(ctx, (arm,))
            yield from step_back(ctx, est, at)
            return True
        yield from step_back(ctx, est, at)
    return False


def take_box(ctx, est):
    """Take the box by the middle of the wall that faces the robot, with the arm on whose side it lies, and hold
    it in front of the chest. Returns True when the hand holds it by a long wall.

    The box is carried by a long wall: it then reaches 0.3 m ahead of the hand and is no wider than the chassis.
    Where furniture leaves no stance in front of a long wall (the box next to the table, in a corner, between the
    sofa and the table), it is taken by a short wall, carried back to the staging point of that stance, which is
    open floor, set down there and taken again."""
    z = BOX['rim'] - BOX_GRIP_BELOW_RIM - BASE_Z
    for attempt in range(4):
        if est.where('box') is None:
            return False

        def feasible(pose, offset, tight):
            return can_take_box(ctx, est.direction('box'), pose, offset, tight)
        # In front of a long wall if there is such a stance; a short wall only if there is none (the box is then
        # moved to open floor and taken again: from there a long wall can be faced).
        at = yield from stand_for(ctx, est, 'box', box_offsets, feasible, headings=long_wall_headings, what=('box',))
        if at is None and est.where('box') is not None:
            at = yield from stand_for(ctx, est, 'box', box_offsets, feasible, headings=box_headings, what=('box',))
        if at is None:
            return False
        arm = ctx.arm = 'left' if at['offset'][1] >= 0. else 'right'
        point, axis = est.point('box'); long_wall = facing_long_wall(axis, 0.)
        if long_wall:
            normal, closing = long_wall_grip(arm, axis); wall = point - BOX['wall'] * normal
            ok = yield from top_pinch(ctx, np.r_[wall, z], closing, relax=at['tight'], label='box', look=lambda tip, angle: est.hand_view(arm, 'box', tip, angle), both_ways=False, locked=SQUARE)
        else:
            normal = np.array([np.cos(axis), np.sin(axis)]); normal = normal * np.sign(float(normal[0])); wall = point - BOX_END * normal      # across the short wall that faces the robot
            ok = yield from top_pinch(ctx, np.r_[wall, z], float(np.arctan2(normal[1], normal[0])), relax=at['tight'], label='box', look=lambda tip, angle: est.hand_view(arm, 'box', tip, angle))
        if ok and long_wall:
            tip = ctx.tip()[:3, 3]; ctx.attach(arm, *box_spheres(point + (tip[:2] - wall), axis, lift=tip[2] + BASE_Z - (BOX['rim'] - BOX_GRIP_BELOW_RIM))); ctx.holding[arm] = 'box'; est.taken('box')
            status = yield from carry_box(ctx)
            ctx.event('box_carried', status=status)
            yield from step_back(ctx, est, at)
            return True
        if ok:
            # Held by a short wall: back to open floor with it, down, and again.
            est.taken('box')
            yield from step_back(ctx, est, at)
            here = ctx.tip()[:3, 3].copy(); lower = interpolate_cartesian(ctx.ik, ctx.q, here, here - [0., 0., max(0., here[2] - z - .012)], ctx.tip()[:3, :3], step=.02)
            if lower is not None:
                yield from follow(ctx, lower, rate=.015, max_steps=200)
            ctx.set_grip(1.)
            yield from hold(ctx, 20)
            if lower is not None:
                yield from follow(ctx, lower[::-1][1:], rate=.02, max_steps=200)
            ctx.event('box_moved', lowered=lower is not None)
            yield from tuck(ctx, (arm,))
            yield from drive(ctx, np.asarray(est.base()[:2]) - .45 * np.array([np.cos(est.base()[2]), np.sin(est.base()[2])]), est.base()[2], direct=True, speed=.25, face_travel=False, max_steps=200)
            yield from look(ctx, est, ('box',))
            continue
        yield from tuck(ctx, (arm,))
        yield from step_back(ctx, est, at)
        if ok is False:
            # The hand closed on nothing: the box is not where it was believed (it has been pushed). It cannot be
            # far: a look from further back, where the camera sees the floor in front of the base (from the staging
            # point of a stance the nearest 0.8 m are below its view), before the lookouts are gone round again
            # without knowing where it stands.
            est.forget('box')
            x, y, yaw = est.base(); back = np.array([np.cos(yaw), np.sin(yaw)])
            for distance in (.6, .4):
                spot = np.array([x, y]) - distance * back
                if ctx.scene.base_free(spot, margin=.03):
                    yield from straight(ctx, spot, yaw, speed=.25, tolerance=.05, max_steps=300)
                    break
            yield from settle(ctx, est, 6)
            yield from look(ctx, est, ('box',))
            ctx.event('box_sought', found=est.where('box') is not None)
            if est.where('box') is None:
                yield from find(ctx, est, ('box',))
    return False


def take_racket(ctx, est, arm='right'):
    """Take the racket from the table by its handle and stand it upright beside the shoulder. Returns True when
    the hand holds it."""
    side = 1. if arm == 'left' else -1.; z = TABLE_TOP + HANDLE_ABOVE_TABLE - BASE_Z
    for attempt in range(3):
        ctx.arm = arm
        if est.where('racket') is None:
            return False
        at = yield from stand_for(ctx, est, 'racket', TABLE_OFFSETS[side], lambda pose, offset, tight: can_take_racket(ctx, est.direction('racket'), pose, offset, tight))
        if at is None:
            return False
        point, toward_head = est.point('racket')
        ok = yield from top_pinch(ctx, np.r_[point, z], toward_head + np.pi / 2, relax=at['tight'], label='racket', look=lambda tip, angle: est.hand_view(arm, 'racket', tip, angle))
        if ok:
            tip = ctx.tip()[:3, 3]; ctx.attach(arm, *racket_spheres(tip[:2], toward_head, tip[2] + BASE_Z - HANDLE_ABOVE_TABLE + .018)); ctx.holding[arm] = 'racket'; est.taken('racket')
            # Back from the table first: the racket is turned for carrying where there is room for it.
            yield from step_back(ctx, est, at)
            status = yield from carry_racket(ctx)
            ctx.event('racket_carried', status=status)
            return True
        yield from tuck(ctx, (arm,))
        yield from step_back(ctx, est, at)
        if ok is False:
            est.forget('racket')
            yield from find(ctx, est, ('racket',))
    return False


def clear_of_box(ctx, est, spot, reach=.9):
    """@spot, or the place nearest to it where the base can turn on the spot without touching the toy box (the box
    stands anywhere on the floor, at a lookout as well: a turn there shoved it 0.7 m away, and the hand then
    closed on nothing where it had been seen). None if there is no such place within @reach."""
    spot = np.asarray(spot, float); obstacles = on_floor(ctx, est)

    def away(xy):
        return all(float(np.linalg.norm(xy - centre)) > TURN_CLEAR + radius + .05 for centre, radius in obstacles)
    if away(spot):
        return spot
    for distance in np.arange(.15, reach + .01, .15):
        ring = [spot + distance * np.array([np.cos(angle), np.sin(angle)]) for angle in np.linspace(0., 2 * np.pi, 16, endpoint=False)]
        ring = [xy for xy in ring if away(xy) and ctx.scene.clearance(xy) > TURN_CLEAR]
        if ring:
            return max(ring, key=lambda xy: min(float(np.linalg.norm(xy - centre)) for centre, _ in obstacles))
    return None


def find(ctx, est, names):
    """Go round the lookouts of the living room until all of @names are known. Returns True when they are."""
    for spot, headings in LOOKOUTS:
        if all(est.where(name) is not None for name in names):
            break
        for again in range(3):
            # (Again where the box came into view on the way and stands where the base was going to turn.)
            at = clear_of_box(ctx, est, spot)
            status = 'no_place' if at is None else (yield from route(ctx, est, at, headings[0], turning=True, tolerance=.08))
            if status != 'box_in_the_way':
                break
        if status != 'arrived':
            continue
        for yaw in headings:
            if all(est.where(name) is not None for name in names):
                break
            yield from drive(ctx, est.base()[:2], yaw, direct=True, tolerance=.2, face_travel=False, max_steps=200)      # turn on the spot
            yield from settle(ctx, est, 6)
            seen = yield from look(ctx, est)
            ctx.event('lookout', spot=np.round(at, 2).tolist(), yaw=round(float(yaw), 2), seen=seen, known=[name for name in ('box', 'camera', 'racket') if est.where(name) is not None])
    return all(est.where(name) is not None for name in names)


# ---- into the trunk -----------------------------------------------------------------------------------------
def to_trunk(ctx, est, name):
    """To the stance behind the car, the lid open."""
    status = yield from route(ctx, est, LID_STAGING, 0., tolerance=.05)
    if status != 'arrived':
        return 'trunk_' + status
    yield from settle(ctx, est, 16)
    # The map pose from the bumper (the walls are far behind the car, and the car itself is no landmark for the
    # scan match); kept by dead reckoning from here to the stance and back.
    shift = est.fix_at_bumper()
    ctx.event('bumper_fix', shift=None if shift is None else np.round(shift, 4).tolist())
    est.hold_axis('xy')
    if name == 'box':
        # Up, before the base moves in: carried as on the way, the box would meet the car's rear deck.
        yield from carry_box(ctx, height=OVER_THE_SILL)
    stance = TRUNK_STANCE if name == 'box' else RACKET_STANCE
    status = yield from straight(ctx, stance[:2], 0., speed=.2, tolerance=.02)
    yield from hold(ctx, 6)
    return status


def box_into_trunk(ctx, est, arm):
    """The box over the sill and down onto the trunk's floor, level all the way; the hand leaves it the way it
    came. Returns True when the box was let go there."""
    base = est.base(); inverse = np.linalg.inv(pose2(*base)); ctx.arm = arm; check = ctx.validator(relax_from=ctx.q)
    here = ctx.tip()[:3, 3].copy(); closing = closing_angle(ctx); start = ctx.chain.copy(); way = None
    # The hand holds the wall nearest to the robot: the box's middle lies BOX['wall'] further in.
    for lift in (.03, .05):
        for dx in (0., .04, -.04):
            for dy in (0., .06, -.06):
                spot = (BOX_IN_TRUNK - BOX['wall'] + dx, base[1] + dy, TRUNK_FLOOR + lift + BOX['rim'] - BOX_GRIP_BELOW_RIM)
                target = (inverse @ np.r_[spot[0], spot[1], 0., 1.])[:3]; target[2] = spot[2] - BASE_Z
                way = level_way(ctx, [np.array([target[0], target[1], here[2]]), target], closing, check)
                if way is not None:
                    break
            if way is not None:
                break
        if way is not None:
            break
    if way is None:
        ctx.event('into_trunk', ok=False, thing='box', reason='no level way over the sill'); return False
    status = yield from follow(ctx, way, rate=.02, max_steps=900)
    yield from hold(ctx, 6)
    if status != 'done':
        # Held up on the way: back to where the hand was, the box still in it.
        yield from follow(ctx, way[::-1][1:] + [start], rate=.02, max_steps=900)
        ctx.event('into_trunk', ok=False, thing='box', reason=status); return False
    ctx.set_grip(1.); ctx.detach(arm); ctx.holding.pop(arm, None)
    yield from hold(ctx, 20)
    yield from follow(ctx, way[::-1][1:] + [start], rate=.02, max_steps=900)      # the same way back: straight up out of the box first
    ctx.event('into_trunk', ok=True, thing='box', spot=np.round(spot, 3).tolist())
    yield from tuck(ctx, (arm,))
    return True


def racket_into_trunk(ctx, est, arm):
    """The racket flat over the box, across the trunk. The way there is planned with the racket as part of the
    robot. Returns True when it was let go there.

    (The map is a height field: under the parcel shelf, further in than x = -6.33, it has no room. So the racket
    lies across the trunk, which is 1.26 m wide, and not along it.)"""
    base = est.base(); inverse = np.linalg.inv(pose2(*base)); ctx.arm = arm; check = ctx.validator(relax_from=ctx.q); goal = None
    side = 1. if arm == 'left' else -1.; rotation = across_the_trunk(side, racket_in_hand(ctx))
    # The head towards the far side of the trunk from the hand: the racket's middle then lies near the trunk's.
    for dz in (0., .03, -.03):
        for dx in (0., .06, -.05):
            for dy in (0., .06, -.06):
                spot = (RACKET_IN_TRUNK[0] + dx, LID_MIDDLE + side * .33 + dy, RACKET_IN_TRUNK[1] + dz)
                target = (inverse @ np.r_[spot[0], spot[1], 0., 1.])[:3]; target[2] = spot[2] - BASE_Z
                goal = ctx.ik.solve(ctx.q, target, rotation, count=40, validator=check)
                if goal is not None:
                    break
            if goal is not None:
                break
        if goal is not None:
            break
    if goal is None:
        ctx.event('into_trunk', ok=False, thing='racket', reason='no pose over the trunk'); return False
    status = yield from move_to(ctx, goal, rate=.03, label='into_trunk', max_steps=700)
    yield from hold(ctx, 6)
    if status != 'done':
        ctx.event('into_trunk', ok=False, thing='racket', reason=status); return False
    ctx.set_grip(1.); ctx.detach(arm); ctx.holding.pop(arm, None)
    yield from hold(ctx, 20)
    up = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], ctx.tip()[:3, 3] + [0., 0., .10], rotation, step=.03)
    if up is not None:
        yield from follow(ctx, up, rate=.02, max_steps=200)
    ctx.event('into_trunk', ok=True, thing='racket', spot=np.round(spot, 3).tolist())
    yield from tuck(ctx, (arm,))
    return True


def deliver(ctx, est, arm, name):
    """From the living room to the car with @name in @arm's hand, and into the trunk."""
    status = yield from back_to_garage(ctx, est)
    if status == 'in_garage':
        status = yield from to_trunk(ctx, est, name)
    ctx.event('stage', done='return', thing=name, status=status)
    if status != 'arrived':
        return False
    for attempt in range(2):
        done = yield from (box_into_trunk(ctx, est, arm) if name == 'box' else racket_into_trunk(ctx, est, arm))
        if done:
            break
    yield from straight(ctx, LID_STAGING, 0., speed=.3)
    est.hold_axis(None)
    return done


def full_task(ctx, est):
    ctx.holding = {}
    yield from tuck(ctx)
    opened = yield from open_trunk(ctx, est)
    ctx.event('stage', done='trunk', ok=bool(opened))
    status = yield from leave_garage(ctx, est)
    ctx.event('stage', done='door', status=status)
    if status != 'in_corridor':
        return status
    # ---- first trip: the camera into the box, the box into the trunk ----
    status = yield from route(ctx, est, ENTRY, NORTH, tolerance=.10)
    known = yield from find(ctx, est, ('box', 'camera', 'racket'))
    ctx.event('stage', done='survey', status=status, known={name: None if est.where(name) is None else est.where(name).round(3).tolist() for name in ('box', 'camera', 'racket')})
    got = yield from take_camera(ctx, est)
    if got:
        got = yield from camera_into_box(ctx, est)
    ctx.event('stage', done='camera', ok=bool(got))
    box = yield from take_box(ctx, est)
    ctx.event('stage', done='box', ok=bool(box))
    if box:
        box = yield from deliver(ctx, est, next(arm for arm, name in ctx.holding.items() if name == 'box'), 'box')
        ctx.event('stage', done='box_delivered', ok=bool(box))
        status = yield from leave_garage(ctx, est)
        if status != 'in_corridor':
            return status
        status = yield from route(ctx, est, ENTRY, NORTH, tolerance=.10)
    # ---- second trip: the racket ----
    if est.where('racket') is None:
        yield from find(ctx, est, ('racket',))
    racket = yield from take_racket(ctx, est)
    ctx.event('stage', done='racket', ok=bool(racket))
    if racket:
        racket = yield from deliver(ctx, est, 'right', 'racket')
        ctx.event('stage', done='racket_delivered', ok=bool(racket))
    return 'loaded' if box and racket else 'box_only' if box else 'racket_only' if racket else 'nothing_loaded'
