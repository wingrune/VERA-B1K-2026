"""The battery task as a sequence of skills, driven by an estimator.

The estimator supplies base() = (x, y, yaw) in M, battery(k) = centre of battery k in M (or None), can() =
centre of the trash can's footprint in M (or None), and, for the moment of contact, battery_point(k, surface)
and can_point() in the base frame, plus optionally wrist_point(arm, point) and mark_taken(k). With simulator
truth behind the estimator this script is a diagnostic of the manipulation chain; the scored controller puts
its camera-based estimator there. Battery 3 stands on the cabinet, 1 and 2 on the desk.
"""
import json
import numpy as np
from manip import hold, drive, pose2, wrap
from battery_skills import tuck, tucked, plan_surface_stance, plan_reach, grasp, can_stance, drop, can_walls, plan_can_lift, lift_can, lower_can, guard_boxes, at_edge, CAN_TOP

CAN_KEEPOUT = (.64, .58)        # the base centre stays this far from the can's centre while travelling; the smaller
                                # value is tried when the larger leaves no route (the copy room is 1.7 m deep).
                                # Anything less and the chassis shoves the can along - once out of the door.
CAN_RADIUS = .27           # circle around the can's footprint
CAN_SQUEEZE = (.50, .44)   # see route()
FOOTPRINT = (-.40, .24, .345)   # chassis: rear, front and half width in the base frame
PRESTANCE = .15            # the route ends this far behind a stance; the rest is a straight, slow move
HEAVY_LOAD = 1.05          # static load accepted when no stance is found within the usual limit


def surfaces_of(scene):
    """Work surfaces from the map: footprint, height and the sides the robot can stand at."""
    scope = scene.meta['scope']; out = {}
    for key, name, sides in (('cabinet.n.01_1', 'cabinet', ('south',)), ('desk.n.01_1', 'desk', ('south', 'north'))):
        box = scope[key]; out[name] = dict(lo=box['aabb_lo'][:2], hi=box['aabb_hi'][:2], height=float(box['aabb_hi'][2]), sides=sides)
    return out


def can_gap(base, can_xy):
    """Gap between the chassis standing at @base = (x, y, yaw) and the circle around the can."""
    x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = np.asarray(can_xy, float)[:2] - [x, y]
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    nearest = np.array([np.clip(local[0], FOOTPRINT[0], FOOTPRINT[1]), np.clip(local[1], -FOOTPRINT[2], FOOTPRINT[2])])
    return float(np.linalg.norm(local - nearest)) - CAN_RADIUS


def route(ctx, goal_xy, goal_yaw, can=None, static_only=False, squeeze=False, **options):
    """drive() that keeps away from the can, by less when the room leaves no other way.

    squeeze: as a last resort pass the can at a distance at which the chassis may shove it. Only for leaving
    the room when the can could not be carried out of the way: without that nothing more can be done at all.
    """
    status = 'no_route'
    for radius in ((CAN_KEEPOUT + (CAN_SQUEEZE if squeeze else ())) if can is not None else (None,)):
        keepout = [] if radius is None else [(np.asarray(can, float)[:2], radius)]
        if ctx.scene.path(ctx.est.base()[:2], goal_xy, static_only=static_only, keepout=keepout) is None:
            continue
        status = yield from drive(ctx, goal_xy, goal_yaw, keepout=keepout, static_only=static_only, **options)
        return status
    ctx.event('route', ok=False, goal=np.round(goal_xy, 2).tolist(), can=None if can is None else np.round(can[:2], 2).tolist())
    return status


def back_off(ctx, distance=.07):
    """After pushing into something that did not give way: retreat a little so nothing pushes the base back."""
    x, y, yaw = ctx.est.base(); x0, y0 = x, y
    for _ in range(60):
        x, y, _ = ctx.est.base()
        if np.hypot(x - x0, y - y0) >= distance:
            break
        yield ctx.action(base=(-.12, 0., 0.))
    yield from hold(ctx, 12)


def to_stance(ctx, base, can=None, static_only=False):
    """Route to a point a little behind the stance, then straight in at the stance heading.

    Where chairs are pushed aside on the way (static_only) the robot moves slowly, and when one stops it short
    of the stance it retreats a little and reports 'blocked': the battery may be within reach from there.
    """
    x, y, yaw = base; back = np.array([x, y]) - PRESTANCE * np.array([np.cos(yaw), np.sin(yaw)])
    options = dict(speed=.25) if static_only else {}
    status = yield from route(ctx, back, yaw, can=can, static_only=static_only, **options)
    if status == 'arrived':
        yield from look(ctx, ctx.est)
        status = yield from drive(ctx, (x, y), yaw, speed=.2, direct=True, tolerance=.015, min_speed=.04, face_travel=False)
        short = float(np.linalg.norm(np.array([x, y]) - ctx.est.base()[:2]))
        if status == 'arrived' or short < .03:
            return 'arrived'
    if status == 'stalled' and static_only:
        yield from back_off(ctx)
        return 'blocked'
    return status


def fetch(ctx, est, k, surface, can=None, arms=('left', 'right'), static_only=False):
    """Stand at the surface, pinch battery @k and fold the arm again. Returns the arm that holds it, or None.

    can: centre of the trash can if it stands in this room; stances the chassis could not take without
    pushing it are left out, and so are stances that turned out to be out of reach.
    """
    refused = []
    for attempt in range(4):
        target = est.battery(k)
        if target is None:
            return None
        def usable(side, x, y, yaw):
            if any(abs(x - r[0]) < .05 and abs(y - r[1]) < .05 for r in refused):
                return False
            return can is None or (can_gap((x, y, yaw), can) > .03 and can_gap((x - PRESTANCE * np.cos(yaw), y - PRESTANCE * np.sin(yaw), yaw), can) > .03)
        # From where it stands, if that works: after an approach that a chair cut short it often does.
        arm = reach_from_here(ctx, est, target, surface, arms, static_only) if attempt else None
        if arm is None:
            stance = plan_surface_stance(ctx, target, surface, usable=usable, arms=arms, static_only=static_only)
            if stance is None:      # nothing within the usual load limit: accept a heavier lean rather than give up
                stance = plan_surface_stance(ctx, target, surface, usable=usable, arms=arms, static_only=static_only, max_load=HEAVY_LOAD)
            if stance is None:
                return None
            status = yield from to_stance(ctx, stance['base'], can=can, static_only=static_only)
            if status == 'blocked':
                refused.append(stance['base']); ctx.event('fetch', ok=False, battery=k, reason='stance blocked'); continue
            if status != 'arrived':
                refused.append(stance['base']); ctx.event('fetch', ok=False, battery=k, reason='stance ' + status); continue
            arm = stance['arm']
        ctx.arm = arm
        yield from hold(ctx, 6)
        point = est.battery_point(k, surface)
        if point is None:
            return None
        ctx.event('target', battery=k, point=np.round(point, 3).tolist(), source=getattr(est, 'source', None))
        ctx.guard = standing(est, surface)
        held = yield from grasp(ctx, point, static_only=static_only, refine=wrist(est, arm), square=faces_edge(est, target, surface))
        yield from tuck(ctx, arms=(ctx.arm,), static_only=static_only)
        ctx.guard = np.zeros((0, 6))
        if held:
            taken(est, k); return ctx.arm
    return None


def standing(est, surface):
    """Boxes around the batteries known to stand on @surface, the one about to be taken included (see guard_boxes)."""
    lo, hi = np.asarray(surface['lo'], float) - .05, np.asarray(surface['hi'], float) + .05; points = []
    for j in (1, 2, 3):
        b = est.battery(j)
        if b is not None and np.all(b[:2] >= lo) and np.all(b[:2] <= hi) and abs(b[2] - surface['height'] - .031) < .03:
            points.append(b[:2])
    return guard_boxes(points, surface['height'])


def faces_edge(est, target, surface):
    """Whether the battery at @target stands at the edge of @surface on the side the robot is on (see plan_reach)."""
    return at_edge(surface, target, 'south' if np.sin(est.base()[2]) > 0 else 'north')


def sturdy(ctx, stance, static_only=True):
    """How many of four 3 cm displacements of the target the reach from @stance survives (0-4). The robot stops a
    centimetre or two from where it planned to stand; a reach at the limit of the arm is then gone."""
    world = pose2(*stance['base']); previous = ctx.arm; ctx.arm = stance['arm']; nominal = tucked(ctx); count = 0
    try:
        for dx, dy in ((.03, 0.), (-.03, 0.), (0., .03), (0., -.03)):
            count += plan_reach(ctx, np.asarray(stance['point'], float) + [dx, dy, 0.], base=world, q=nominal, count=8, static_only=static_only, square=stance.get('square', False)) is not None
    finally:
        ctx.arm = previous
    return count


def wrist(est, arm):
    """The estimator's close-up correction for @arm, if it has one."""
    look = getattr(est, 'wrist_point', None)
    return None if look is None else (lambda point: look(arm, point))


def taken(est, k):
    mark = getattr(est, 'mark_taken', None)
    if mark is not None:
        mark(k)


def look(ctx, est):
    """Let the estimator take in what the cameras show now (a metre before a battery it is seen best)."""
    see = getattr(est, 'see', None)
    if see is not None:
        yield from hold(ctx, 3)
        see()


def reach_from_here(ctx, est, target, surface, arms, static_only):
    """The arm that reaches the battery at @target from the current base pose, or None."""
    world = pose2(*est.base()); previous = ctx.arm
    point = (np.linalg.inv(world) @ np.r_[np.asarray(target, float)[:2], surface['height'] + .033, 1.])[:3]
    if not .15 < point[0] < 1.0:
        return None
    try:
        for arm in (('left', 'right') if point[1] >= 0 else ('right', 'left')):
            if arm not in arms:
                continue
            ctx.arm = arm
            if plan_reach(ctx, point, count=12, static_only=static_only) is not None:
                ctx.event('reach_from_here', arm=arm, point=point.round(3).tolist()); return arm
    finally:
        ctx.arm = previous
    return None


def deliver(ctx, est, arms):
    """Drive to the can and let the batteries in @arms fall into it. Returns the number of drops made."""
    done = 0
    for arm in arms:
        can = est.can()
        if can is None:
            return done
        ctx.arm = arm; dropped = False
        for x, y, yaw in can_stance(ctx, can[:2], est.base()[:2]):
            status = yield from route(ctx, (x, y), yaw, can=can, tolerance=.03)
            if status != 'arrived':
                continue
            yield from hold(ctx, 4)
            point = est.can_point()
            if point is None:
                continue
            dropped = yield from drop(ctx, point)
            yield from tuck(ctx, arms=(arm,))
            if dropped:
                break
        done += int(dropped)
    return done


# ---- the can in the way ---------------------------------------------------------------------------
# The can starts anywhere on the copy-room floor: in one start out of six it stands in front of the doorway,
# and sometimes where the robot has to stand for the cabinet battery. It is then carried to the south side of
# the room: pinched by a wall, lifted, and moved by translating the base without turning.
CAN_REACH = .50                 # distance of the pinched wall ahead of the base
CAN_ZONE = ((8.75, 10.05), (-10.72, -9.90))     # where the can may be put: clear of the doorway (east) and of the cabinet front (north)
CAN_PASSAGE = .70           # a place for the can is good if the robot could still pass it at this distance (more than it will keep)


def can_blocks_exit(ctx, est):
    can = est.can()
    return can is not None and ctx.scene.path(ctx.est.base()[:2], (CORRIDOR_X, COPY_DOOR_Y), keepout=[(can[:2], CAN_KEEPOUT[-1])]) is None


def can_spots(ctx, can_xy, stand_xy, clear_x=None, limit=6):
    """Places the can could be carried to by a robot that holds it from @stand_xy and translates without turning.

    A place is good if, with the can there, the robot can still drive from where it will then stand to the corridor
    without coming near the can, and - while the cabinet battery is still to be fetched (@clear_x is its x) - the
    can is not where the robot must stand for it. Returns [(cost, spot, robot goal)], best first.
    """
    offset = np.asarray(can_xy, float)[:2] - np.asarray(stand_xy, float); rows = []
    for sx in np.arange(CAN_ZONE[0][0], CAN_ZONE[0][1] + .01, .15):
        for sy in np.arange(CAN_ZONE[1][0], CAN_ZONE[1][1] + .01, .08):
            spot = np.array([sx, sy]); goal = spot - offset
            if not (ctx.scene.base_free(goal, margin=-.02) and float(ctx.scene.clearance(spot)) >= .30 and ctx.scene.segment_free(stand_xy, goal)):
                continue
            if clear_x is not None and abs(sx - clear_x) < .80 and sy > -10.50:
                continue
            rows.append((float(np.linalg.norm(goal - stand_xy)), spot, goal))
    out = []
    for cost, spot, goal in sorted(rows, key=lambda r: r[0]):
        if ctx.scene.path(goal, (CORRIDOR_X, COPY_DOOR_Y), keepout=[(spot, CAN_PASSAGE)]) is not None:
            out.append((cost, spot, goal))
            if len(out) >= limit:
                break
    return out


def move_can(ctx, est, clear_x=None):
    """Carry the can to a place where it is in nobody's way. Returns True if it was put down somewhere new.

    clear_x: x of a battery on the cabinet that is still to be fetched; the can is not put where the robot will
    have to stand for it.
    """
    can = est.can()
    if can is None:
        return False
    if getattr(est, 'see', None) is not None and getattr(est, 'forget_can', None) is not None:
        # Its heading shows only from close by: go and look at it from a metre away first, from another side
        # if the first view does not show enough of its rim.
        views = 0; known = np.array(can[:2])
        for x, y, heading in can_stance(ctx, known, est.base()[:2], distance=1.0):
            status = yield from route(ctx, (x, y), heading, can=known, tolerance=.05)
            if status != 'arrived':
                continue
            est.forget_can(); views += 1
            for _ in range(3):
                yield from look(ctx, est)
            if (est.can() is not None and est.can_yaw() is not None) or views >= 3:
                break
        can = est.can()
        if can is None:
            ctx.event('move_can', ok=False, reason='can not seen from a metre away'); return False
    yaw = est.can_yaw() if hasattr(est, 'can_yaw') else None
    if yaw is None:
        ctx.event('move_can', ok=False, reason='heading of the can unknown'); return False
    robot = np.array(est.base()[:2]); options = []
    for wall, normal in can_walls(can[:2], yaw):
        stand = wall + CAN_REACH * normal                       # the robot stands outside this wall, facing the can
        if not ctx.scene.base_free(stand, margin=-.04):
            continue
        # The can may stand between the robot and that side with no room to pass it properly (the room is
        # 1.7 m deep): then the robot squeezes past, and looks at the can again from the stand.
        keepout = next((r for r in CAN_KEEPOUT + CAN_SQUEEZE if ctx.scene.path(robot, stand, keepout=[(can[:2], r)]) is not None), None)
        if keepout is None:
            continue
        if not can_spots(ctx, can[:2], stand, clear_x, limit=1):    # held from this side, is there anywhere good to take it?
            continue
        # ... and is there room at that place for the lean down to the can?
        heading = float(np.arctan2(-normal[1], -normal[0])); world = pose2(stand[0], stand[1], heading); inverse = np.linalg.inv(world)
        point = (inverse @ np.r_[wall, 0., 1.])[:2]
        if plan_can_lift(ctx, point, inverse[:2, :2] @ normal, base=world, q=tucked(ctx)) is not None:
            options.append((keepout < CAN_KEEPOUT[-1], float(np.linalg.norm(stand - robot)), wall, normal, stand))
    if not options:
        ctx.event('move_can', ok=False, reason='no side to hold the can from leads to a good place')
    for _, _, wall, normal, stand in sorted(options, key=lambda o: o[:2]):
        heading = float(np.arctan2(-normal[1], -normal[0]))
        status = yield from route(ctx, stand, heading, can=can, tolerance=.02, squeeze=True)
        if status != 'arrived':
            continue
        yield from hold(ctx, 5)
        # Half a metre in front of it the can is seen best: take its pose from this view alone.
        if getattr(est, 'forget_can', None) is not None and getattr(est, 'see', None) is not None:
            est.forget_can()
            for _ in range(3):
                yield from look(ctx, est)
        can = est.can(); yaw = est.can_yaw()
        if can is None or yaw is None:
            ctx.event('move_can', ok=False, reason='can not seen from the stance'); continue
        wall, normal = min(can_walls(can[:2], yaw), key=lambda w: float(np.linalg.norm(w[0] + CAN_REACH * w[1] - np.array(est.base()[:2]))))
        world = pose2(*est.base()); inverse = np.linalg.inv(world)
        point = (inverse @ np.r_[wall, 0., 1.])[:2]; direction = inverse[:2, :2] @ normal
        lifted = yield from lift_can(ctx, point, direction)       # chooses the arm
        if lifted is None:
            yield from tuck(ctx); continue
        x, y, heading = est.base(); moved = False
        for _, spot, goal in can_spots(ctx, can[:2], np.array([x, y]), clear_x, limit=3):
            status = yield from drive(ctx, goal, heading, speed=.25, direct=True, face_travel=False, tolerance=.04)
            ctx.event('move_can', spot=spot.round(2).tolist(), status=status)
            if status == 'arrived':
                moved = True; break
        yield from lower_can(ctx, lifted)
        yield from tuck(ctx, arms=(ctx.arm,))
        forget = getattr(est, 'forget_can', None)
        if forget is not None:
            forget()
        return moved
    return False


def cabinet_stage(ctx, est):
    """The battery on the copy-room cabinet into the can: a quarter of the score without leaving the room."""
    surfaces = surfaces_of(ctx.scene)
    status = yield from tuck(ctx)
    ctx.event('tuck', status=status)
    can = est.can()
    arm = yield from fetch(ctx, est, 3, surfaces['cabinet'], can=can)
    if arm is None:
        return 'cabinet_fetch_failed'
    done = yield from deliver(ctx, est, [arm])
    yield from hold(ctx, 15)
    return 'cabinet_done' if done else 'cabinet_drop_failed'


# ---- between the rooms -----------------------------------------------------------------------
# Fixed places of the scene (map frame). The doorway of the copy room is 0.95 m wide; the office door is
# pushed open into the office and then leaves 0.73 m of its 0.84 m doorway; the lane past the east end of
# the desk is the only way to the desk's far side: the chassis runs under the desk top and the folded trunk
# passes 0.03 m from its edge, with 0.04 m left to the east wall.
CORRIDOR_X = 12.75
COPY_DOOR_Y = -9.925
OFFICE_DOOR_Y = 2.61            # facing west (0.68 m wide): centre of the open doorway
OFFICE_DOOR_Y_SIDEWAYS = 2.69   # facing north (0.64 m long, origin 0.08 m ahead of its middle)
OFFICE_ENTRY_X = 10.95
DOOR_VIEW = (13.05, 2.50)       # from here, facing the door, the end of the wall it is in shows 0.6 m to the left
DOOR_WALL_END_Y = 1.905         # map y of that end (the south end of the office's east wall)
TURN_SPOT = (11.00, 2.86)       # room to turn between the door, the wall and the desk
SLOT_X = 11.33
SLOT_SOUTH_Y = 3.05
SLOT_NORTH_Y = 5.10
NORTH = np.pi / 2


def straight(ctx, goal, yaw, speed=.25, tolerance=.03, max_steps=1500):
    """A slow straight move at a fixed heading: for doorways and the lane beside the desk."""
    status = yield from drive(ctx, goal, yaw, speed=speed, direct=True, face_travel=False, tolerance=tolerance, yaw_tolerance=.03, min_speed=.04, max_steps=max_steps)
    short = float(np.linalg.norm(np.asarray(goal, float) - ctx.est.base()[:2]))
    if status != 'arrived':
        ctx.event('straight', ok=False, goal=np.round(goal, 2).tolist(), status=status, short=round(short, 3))
    return 'arrived' if status == 'arrived' or short < .06 else status


def hold_axis(est, axis):
    """Ask the estimator to keep one map coordinate by dead reckoning ('x', 'y'; None ends it)."""
    hook = getattr(est, 'hold_axis', None)
    if hook is not None:
        hook(axis)


EDGE_INSET = .015


def follow_edge(ctx, est, side, steps=10):
    """Ask the estimator to take the pose from an edge of the desk top ('north', 'south', 'east'; None ends it)
    and give it a moment to do so. Returns what the estimator last read, or None."""
    hook = getattr(est, 'follow_edge', None)
    if hook is None:
        return None
    if side is None:
        hook(None); return None
    desk = surfaces_of(ctx.scene)['desk']; lo, hi = desk['lo'], desk['hi']
    point, normal = {'north': ((lo[0], hi[1]), (0., 1.)), 'south': ((lo[0], lo[1]), (0., -1.)), 'east': ((hi[0], lo[1]), (1., 0.))}[side]
    # The top as the camera sees it ends EDGE_INSET inside the box the map has for it (measured from a metre away
    # on its north and south sides: 1.5 and 1.6 cm, heading exact; dev/jobs/capture_views.py).
    point = (point[0] - EDGE_INSET * normal[0], point[1] - EDGE_INSET * normal[1])
    # The ends of the edge to the robot's right and left as it faces the desk: where one is in view it gives the
    # coordinate along the edge too.
    west, east = lo[0] + EDGE_INSET, hi[0] - EDGE_INSET
    ends = {'north': ((west, hi[1]), (east, hi[1])), 'south': ((east, lo[1]), (west, lo[1])), 'east': (None, None)}[side]
    hook(dict(point=point, normal=normal, height=desk['height'], ends=ends))
    if hasattr(est, 'settle'):
        est.settle(True)
    yield from hold(ctx, steps)
    if hasattr(est, 'settle'):
        est.settle(False)
    read = getattr(est, 'edge_last', None)
    if steps:
        slide = getattr(est, 'edge_slide', None)
        ctx.event('edge', side=side, reads=getattr(est, 'edge_reads', None), last=None if read is None else [round(float(read[0]), 3), round(float(np.degrees(read[1])), 2)],
                  slides=getattr(est, 'edge_slides', None), slide=None if slide is None else round(slide, 3))
    return read


def settle(ctx, est, steps=24):
    """Stand still and let the estimator refine the pose from what it sees."""
    hook = getattr(est, 'settle', None)
    if hook is not None:
        hook(True)
    yield from hold(ctx, steps)
    if hook is not None:
        hook(False)


def to_office(ctx, est, can=None):
    """From the copy room up the corridor and through the office door, which the chassis pushes open.

    The robot goes through sideways, facing north: across the doorway it then measures 0.64 m instead of
    0.68 m, and it does not have to turn inside.
    """
    status = yield from route(ctx, (CORRIDOR_X, COPY_DOOR_Y), NORTH, can=can, squeeze=True)
    if status != 'arrived':
        return 'corridor_' + status
    # Five centimetres to either side is all the doorway leaves, and the view along the corridor does not fix
    # the position along it that well. In front of the door the wall it is in ends on the left: that end is
    # measured directly and the coordinate is then carried by dead reckoning while the robot turns and slides in.
    status = yield from drive(ctx, DOOR_VIEW, np.pi)
    if status != 'arrived':
        return 'door_' + status
    yield from settle(ctx, est)
    align = getattr(est, 'align_to_wall_end', None)
    if align is not None:
        yield from hold(ctx, 4)
        ctx.event('door_align', shift=align(DOOR_WALL_END_Y))
    hold_axis(est, 'xy')
    yield from straight(ctx, (CORRIDOR_X, OFFICE_DOOR_Y_SIDEWAYS), np.pi, tolerance=.012)
    yield from straight(ctx, (CORRIDOR_X, OFFICE_DOOR_Y_SIDEWAYS), NORTH, tolerance=.012)
    status = yield from straight(ctx, (OFFICE_ENTRY_X, OFFICE_DOOR_Y_SIDEWAYS), NORTH, speed=.3, max_steps=600)
    if status == 'arrived':
        status = yield from straight(ctx, TURN_SPOT, NORTH)
    hold_axis(est, None)
    yield from in_office(ctx, est)
    return 'in_office' if status == 'arrived' else 'entry_' + status


def in_office(ctx, est):
    """Facing the desk from the door side: from here on the pose is taken from the desk's edges (follow_edge) and
    dead reckoning, until the robot is back in the corridor. In the office the scan match is unreliable in places
    (chairs get pushed about, the door leaf stands open, behind the desk little of the map is in view): on the
    way out it was 10 cm off in the doorway, which leaves 5 cm to either side. If the edge does not show, the
    scan match it is."""
    read = yield from follow_edge(ctx, est, 'south')
    if read is None and getattr(est, 'follow_edge', None) is not None:
        yield from follow_edge(ctx, est, None)
        yield from settle(ctx, est)


def to_copy_room(ctx, est):
    """Out of the office sideways (the doorway is wider than the robot is long), down the corridor, into the copy room."""
    status = yield from straight(ctx, (TURN_SPOT[0], OFFICE_DOOR_Y_SIDEWAYS), NORTH, tolerance=.015)
    if status != 'arrived':
        return 'exit_' + status
    yield from settle(ctx, est)
    yield from straight(ctx, (TURN_SPOT[0], OFFICE_DOOR_Y_SIDEWAYS), NORTH, tolerance=.012)
    hold_axis(est, 'xy')
    status = yield from straight(ctx, (CORRIDOR_X, OFFICE_DOOR_Y_SIDEWAYS), NORTH, speed=.3)
    hold_axis(est, None)
    yield from follow_edge(ctx, est, None)          # in the corridor the map is the reference again
    if status != 'arrived':
        return 'doorway_' + status
    status = yield from drive(ctx, (CORRIDOR_X, COPY_DOOR_Y), np.pi)
    if status != 'arrived':
        return 'corridor_' + status
    status = yield from drive(ctx, (11.2, COPY_DOOR_Y), np.pi)
    return 'in_copy_room' if status == 'arrived' else 'copy_room_' + status


def lane(ctx, north):
    """Past the east end of the desk, front first: facing north on the way to its far side, south on the way back.

    Two centimetres to either side is what the passage leaves. The end of the desk top runs along it 12 cm
    from the robot's middle and is in view ahead on the way in: the estimator takes the cross coordinate and
    the heading from that edge (see follow_edge); once the edge has passed under the camera, half a metre is
    dead reckoned.
    """
    ends = [(SLOT_X, SLOT_SOUTH_Y), (SLOT_X, SLOT_NORTH_Y)]
    start, goal = ends if north else ends[::-1]; heading = NORTH if north else SOUTH
    status = yield from straight(ctx, start, heading, tolerance=.02)
    if status != 'arrived':
        return 'lane_start_' + status
    if getattr(ctx.est, 'follow_edge', None) is not None:
        read = yield from follow_edge(ctx, ctx.est, 'east')
        if read is None:              # the edge is not in view: as before, by the map and dead reckoning
            yield from follow_edge(ctx, ctx.est, None)
            yield from settle(ctx, ctx.est)
            hold_axis(ctx.est, 'xy')
    status = yield from straight(ctx, start, heading, tolerance=.012)
    status = yield from straight(ctx, goal, heading, tolerance=.03)
    hold_axis(ctx.est, None)
    # The scan match stays off: the pose is carried by dead reckoning until the robot faces the desk again and
    # the edge on that side takes over (behind_desk, front_of_desk).
    yield from follow_edge(ctx, ctx.est, ('north' if north else 'south') if status == 'arrived' else None, steps=0)
    return 'arrived' if status == 'arrived' else 'lane_' + status


def behind_desk(ctx, est):
    """From the north end of the passage to the turning place behind the desk, facing the desk from then on.

    Behind the desk the robot always faces it, and the pose comes from the edge of the desk top in front of it
    (follow_edge): across the edge and the heading from the edge, along it by dead reckoning from the passage,
    where the desk's end gave that coordinate. First away from the passage, then the half turn: beside the
    wall there is no room for it.
    """
    yield from straight(ctx, TURN_NORTH, NORTH)
    yield from straight(ctx, TURN_NORTH, SOUTH)
    yield from follow_edge(ctx, est, 'north')


def front_of_desk(ctx, est):
    """From the south end of the passage (arrived facing south, dead reckoned since the desk's end went out of
    view) to the turning place by the door, facing north again: there the desk's south edge and its east
    corner give the whole pose before the way out of the door, which is taken sideways, facing north."""
    yield from straight(ctx, TURN_SPOT, SOUTH)
    yield from straight(ctx, TURN_SPOT, NORTH)
    yield from in_office(ctx, est)


def travel_test(ctx, est):
    """Diagnostic: fold, go to the office, round the desk and come back, without touching a battery."""
    status = yield from tuck(ctx)
    result = yield from to_office(ctx, est, can=est.can())
    ctx.event('travel', leg='to_office', result=result)
    if result != 'in_office':
        return result
    result = yield from lane(ctx, north=True)
    ctx.event('travel', leg='lane_north', result=result)
    if result != 'arrived':
        return result
    yield from behind_desk(ctx, est)
    yield from hold(ctx, 10)
    result = yield from lane(ctx, north=False)
    ctx.event('travel', leg='lane_south', result=result)
    if result != 'arrived':
        return result
    yield from front_of_desk(ctx, est)
    result = yield from to_copy_room(ctx, est)
    ctx.event('travel', leg='to_copy_room', result=result)
    return result


# ---- the desk ---------------------------------------------------------------------------------
SOUTH_REACH = .40          # batteries no deeper than this behind the desk's south edge are taken from the south side
# Behind the desk (its north side) the strip to the wall is 1.47 m deep and the office chair stands in it.
# Pushed towards the desk the chair goes into the knee hole and leaves a passage along the wall; the robot then
# stands at the desk on either side of it.
CHAIR_X = 9.96             # line of the push into the knee hole. The chair's hub is at 9.91; pushed along 9.84 or 9.90 the chair
                           # ends up leaning 10 and 6 deg on the desk, along 9.96 upright and 9 cm deeper (chair_west_trial.py)
BEHIND_Y = 5.58            # line along the wall on which the robot moves behind the desk (facing the desk)
PARK_Y = 4.78              # how far the robot follows the chair into the knee hole
TURN_NORTH = (10.90, 5.42) # where it turns between the lane and the strip
# Two ways of getting the chair out of the way; each leaves a stretch of the desk it stands in front of.
#   'park': into the knee hole. The robot then stands on either side of it.
#   'west': along the desk to the west wall. The robot stands east of it.
# box: where the chair then is, for the arms to keep clear of; zones: base x from which the chassis is clear of it.
# Where the chair ends up was read from the simulator in development runs: pushed into the knee hole (coming
# along the wall line from the east, then down x = 9.96) its hub ended at x = 9.66-9.72, y = 4.44-4.60 in 18
# runs of 18, the chair leaning 8-11 deg on the desk with its back to the robot. Its feet reach 0.35 m from the
# hub at floor level (the zones keep the chassis off them); its back and seat about 0.28 m (the box). Pushed
# west, the hub ended at x = 8.66-8.91, y = 5.16-5.45 in 20 of 20.
CHAIR = {'park': dict(box=(9.38, 4.60, 0., 9.98, 4.98, 1.10), zones=((8.63, 8.95), (10.46, 11.30))),
         'west': dict(box=(8.30, 4.62, 0., 9.08, 5.60, 1.22), zones=((9.42, 11.30),))}
# The chair is pushed west with the SIDE of the chassis, the robot facing the desk and moving sideways along it.
# The wheels are under the two front corners and under the middle of the rear. Pushed with the front, the feet
# of the chair's star base get under the front wheels: the chair tips 9-16 deg, the robot leans up to 6 deg,
# and when it drives away the chair comes along. The side has no wheel behind the front corner, and on this
# line the front wheel passes south of the foot that points east. Measured (dev/jobs/chair_west_trial.py):
# on y = 4.88 the chair rolls to the wall upright and stays; on y = 4.98 the wheel climbs the foot and the
# robot leans 28 deg. The desk's edge (the trunk is 0.2 m from it) bounds the line on the other side.
WEST_PUSH_Y = 4.86
CORNER_VIEW_X = 10.70      # on the line behind the desk: from here the desk's east end is in view half a metre to the left
WEST_PUSH_X = 9.28         # the push ends here: the chair then stands a hand's width from the west wall, not squeezed against it
SOUTH = -np.pi / 2


def clear_chair(ctx, mode):
    """Push the office chair out of the way ('park' or 'west'). Ends on the line along the wall, facing the desk."""
    if mode == 'park':
        yield from straight(ctx, TURN_NORTH, SOUTH)
        yield from straight(ctx, (CHAIR_X, BEHIND_Y), SOUTH)
        status = yield from straight(ctx, (CHAIR_X, PARK_Y), SOUTH, speed=.2)
        x, y, yaw = ctx.est.base()
        yield from straight(ctx, (CHAIR_X, BEHIND_Y), SOUTH)
    else:
        yield from straight(ctx, TURN_NORTH, SOUTH)
        yield from settle(ctx, ctx.est)             # the line matters to a few centimetres: fit the pose standing still
        yield from straight(ctx, (TURN_NORTH[0], WEST_PUSH_Y), SOUTH, speed=.2, tolerance=.012)
        yield from settle(ctx, ctx.est)
        yield from straight(ctx, (TURN_NORTH[0], WEST_PUSH_Y), SOUTH, speed=.2, tolerance=.012)
        status = yield from straight(ctx, (WEST_PUSH_X, WEST_PUSH_Y), SOUTH, speed=.15, tolerance=.02)
        x, y, yaw = ctx.est.base()
        yield from straight(ctx, (x + .55, WEST_PUSH_Y), SOUTH)
        yield from straight(ctx, (x + .55, BEHIND_Y), SOUTH)
        # Pushing the chair, the base moved up to 20 cm less than it reported. Back to where the east end of the
        # desk's edge is in view: from it the estimator reads the position along the desk again.
        yield from straight(ctx, (CORNER_VIEW_X, BEHIND_Y), SOUTH)
        yield from follow_edge(ctx, ctx.est, 'north', steps=12)
    ctx.event('clear_chair', mode=mode, status=status, reached=[round(x, 3), round(y, 3)])
    # From now on the chair is where it was pushed to; the map's height field still has it where it stood and
    # is not used for it (static_only). Arms reaching past it must go round it.
    ctx.scene.boxes = np.vstack([ctx.scene.boxes, np.array(CHAIR[mode]['box'], float)[None]])


def fetch_north(ctx, est, k, desk, arms, zones):
    """Battery @k from the desk's north side, from a stance in @zones (base x ranges clear of the chair). Returns the arm or None."""
    refused = []
    for attempt in range(4):
        target = est.battery(k)
        if target is None:
            return None
        def usable(side, x, y, yaw):
            return any(lo <= x <= hi for lo, hi in zones) and not any(abs(x - r[0]) < .06 and abs(y - r[1]) < .05 for r in refused)
        stance = plan_surface_stance(ctx, target, dict(desk, sides=('north',)), usable=usable, arms=arms, static_only=True)
        if stance is None:
            stance = plan_surface_stance(ctx, target, dict(desk, sides=('north',)), usable=usable, arms=arms, static_only=True, max_load=HEAVY_LOAD)
        if stance is None:
            return None
        x, y, yaw = stance['base']
        yield from straight(ctx, (x, BEHIND_Y), SOUTH)
        yield from look(ctx, est)
        status = yield from straight(ctx, (x, y), SOUTH, speed=.2, tolerance=.015)
        if status != 'arrived':
            refused.append(stance['base']); ctx.event('fetch', ok=False, battery=k, reason='stance ' + status)
            yield from back_off(ctx)
            arm = reach_from_here(ctx, est, est.battery(k), desk, arms, True)
            if arm is None:
                yield from straight(ctx, (ctx.est.base()[0], BEHIND_Y), SOUTH); continue
        else:
            arm = stance['arm']
        ctx.arm = arm
        yield from hold(ctx, 6)
        point = est.battery_point(k, desk)
        if point is None:
            return None
        ctx.event('target', battery=k, point=np.round(point, 3).tolist(), source=getattr(est, 'source', None))
        ctx.guard = standing(est, desk)
        held = yield from grasp(ctx, point, static_only=True, refine=wrist(est, arm), square=faces_edge(est, target, desk))
        if not held and ctx.events and ctx.events[-1].get('event') == 'grasp' and ctx.events[-1].get('reason'):
            refused.append(tuple(est.base()))        # the hand could not even get there from this stance: another one next time
        yield from tuck(ctx, arms=(arm,), static_only=True)
        ctx.guard = np.zeros((0, 6))
        yield from straight(ctx, (ctx.est.base()[0], BEHIND_Y), SOUTH)
        if held:
            taken(est, k); return arm
    return None


def desk_stage(ctx, est, desk):
    """Both desk batteries, one per hand. Near ones from the south side; for the others the robot drives past the
    east end of the desk and takes them from its far side. Returns the arms that hold a battery, or None if it
    could not get back from behind the desk."""
    sides = {}
    for k in (1, 2):
        b = est.battery(k)
        if b is not None:
            sides[k] = 'south' if b[1] - desk['lo'][1] <= SOUTH_REACH else 'north'
    held = []
    for k in [k for k, s in sides.items() if s == 'south']:
        arm = yield from fetch(ctx, est, k, dict(desk, sides=('south',)), arms=[a for a in ('left', 'right') if a not in held], static_only=True)
        ctx.event('desk_battery', battery=k, side='south', arm=arm)
        if arm is not None:
            held.append(arm)
    north = [k for k, s in sides.items() if s == 'north']
    if north:
        status = yield from route(ctx, TURN_SPOT, NORTH, static_only=True, speed=.25)
        status = yield from lane(ctx, north=True)
        ctx.event('lane', north=True, status=status)
        if status != 'arrived':
            return held
        yield from behind_desk(ctx, est)
        # Where the chair goes depends on where the batteries stand: each way leaves a different stretch of the desk
        # free, and from the stances that remain a battery may be within reach of one arm only. For each way the
        # arms are assigned so that as many batteries as possible get one; the knee hole if both ways give the same.
        free = [a for a in ('left', 'right') if a not in held]

        def assign(mode):
            """({battery: arm}, how sturdy) for this way of clearing the chair."""
            boxes = ctx.scene.boxes; zones = CHAIR[mode]['zones']; surface = dict(desk, sides=('north',)); able = {}
            ctx.scene.boxes = np.vstack([boxes, np.array(CHAIR[mode]['box'], float)[None]])
            try:
                for k in north:
                    target = est.battery(k); able[k] = {}
                    usable = lambda side, x, y, yaw: any(lo <= x <= hi for lo, hi in zones)
                    for a in (free if target is not None else ()):
                        stance = plan_surface_stance(ctx, target, surface, usable=usable, arms=[a], static_only=True)
                        if stance is not None:
                            # A reach across the body to the far side (the only one left beside the chair in the
                            # knee hole for a battery behind it) was planned and then refused at the desk twice.
                            side = float(stance['point'][1]) * (1. if a == 'left' else -1.)
                            able[k][a] = 1 + sturdy(ctx, stance) - (3 if side < -.15 else 0) - (2 if abs(stance['point'][1]) >= .55 else 0)
            finally:
                ctx.scene.boxes = boxes
            options = [({k: a}, able[k][a]) for k in north for a in able[k]]
            if len(north) == 2:
                both = [({north[0]: a, north[1]: b}, able[north[0]][a] + able[north[1]][b]) for a in able[north[0]] for b in able[north[1]] if a != b]
                options = both or options
            return max(options, key=lambda o: o[1]) if options else ({}, 0)
        plans = {m: assign(m) for m in ('park', 'west')}
        mode = max(plans, key=lambda m: (len(plans[m][0]), plans[m][1], m == 'park')); plan = plans[mode][0]
        ctx.event('chair_mode', mode=mode, plans={m: dict(arms={str(k): a for k, a in v[0].items()}, sturdy=v[1]) for m, v in plans.items()})
        yield from clear_chair(ctx, mode)
        for k in sorted(north, key=lambda k: k not in plan):       # those with an arm of their own first
            arms = [plan[k]] if k in plan and plan[k] not in held else [a for a in ('left', 'right') if a not in held and a not in [plan[j] for j in plan if j != k]]
            arm = (yield from fetch_north(ctx, est, k, desk, arms, CHAIR[mode]['zones'])) if arms else None
            ctx.event('desk_battery', battery=k, side='north', arm=arm)
            if arm is not None:
                held.append(arm)
        for attempt in range(3):
            yield from follow_edge(ctx, est, 'north', steps=2)
            yield from straight(ctx, (ctx.est.base()[0], BEHIND_Y), SOUTH)
            yield from straight(ctx, (TURN_NORTH[0], BEHIND_Y), SOUTH)
            yield from straight(ctx, TURN_NORTH, SOUTH)
            status = yield from lane(ctx, north=False)
            ctx.event('lane', north=False, status=status)
            if status == 'arrived':
                break
        else:
            return None
        yield from front_of_desk(ctx, est)
    return held


def full_task(ctx, est, survey=None):
    """Cabinet battery into the can, then both desk batteries: fetch, carry back, drop.

    survey: optional generator function(where) that looks around ('copy_room': the can and the cabinet
    battery; 'desk': the two desk batteries) so that the estimator knows them.
    """
    surfaces = surfaces_of(ctx.scene)
    ctx.scene.boxes = np.array([box for box in ctx.scene.boxes if not any(np.allclose(box, c['box']) for c in CHAIR.values())]).reshape(-1, 6)   # a new episode: the chair is back
    status = yield from tuck(ctx)
    if survey is not None:
        yield from survey('copy_room')
    if est.can() is None:
        return 'no_can'
    done = 0; arm = None
    # The cabinet battery first, with the can where it stands if that works; the can is carried away only when
    # it is in the way of that, and before leaving if it is in the way of the door.
    if est.battery(3) is not None:
        arm = yield from fetch(ctx, est, 3, surfaces['cabinet'], can=est.can())
        if arm is None and est.battery(3) is not None:
            # The can may stand where the robot has to: carry it away once and try again.
            moved = yield from move_can(ctx, est, clear_x=float(est.battery(3)[0]))
            ctx.event('stage', stage='can_moved', moved=moved)
            if survey is not None:
                yield from survey('can')
            if moved:
                arm = yield from fetch(ctx, est, 3, surfaces['cabinet'], can=est.can())
        if arm is not None:
            done = yield from deliver(ctx, est, [arm])
    ctx.event('stage', stage='cabinet', fetched=arm is not None, dropped=done)
    if can_blocks_exit(ctx, est):
        moved = yield from move_can(ctx, est)
        ctx.event('stage', stage='can_moved', moved=moved)
        if survey is not None:
            yield from survey('can')
    result = yield from to_office(ctx, est, can=est.can())
    ctx.event('stage', stage='to_office', result=result)
    if result != 'in_office':
        return result
    if survey is not None:
        yield from survey('desk')
    held = yield from desk_stage(ctx, est, surfaces['desk'])
    ctx.event('stage', stage='desk', held=held)
    if held is None:
        return 'stuck_behind_desk'
    result = yield from to_copy_room(ctx, est)
    ctx.event('stage', stage='to_copy_room', result=result)
    if result != 'in_copy_room':
        return result
    done += (yield from deliver(ctx, est, held))
    yield from hold(ctx, 20)
    return 'done_%d' % done


def desk_test(ctx, est):
    """Diagnostic: straight to the office, both desk batteries, back, into the can (the cabinet battery is left alone)."""
    surfaces = surfaces_of(ctx.scene)
    ctx.scene.boxes = np.array([box for box in ctx.scene.boxes if not any(np.allclose(box, c['box']) for c in CHAIR.values())]).reshape(-1, 6)
    status = yield from tuck(ctx)
    result = yield from to_office(ctx, est, can=est.can())
    ctx.event('stage', stage='to_office', result=result)
    if result != 'in_office':
        return result
    held = yield from desk_stage(ctx, est, surfaces['desk'])
    ctx.event('stage', stage='desk', held=held)
    if held is None:
        return 'stuck_behind_desk'
    result = yield from to_copy_room(ctx, est)
    ctx.event('stage', stage='to_copy_room', result=result)
    if result != 'in_copy_room':
        return result
    done = yield from deliver(ctx, est, held)
    yield from hold(ctx, 20)
    return 'done_%d' % done
