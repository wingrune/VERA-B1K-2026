"""The trash task as a sequence of skills, driven by an estimator.

Three soda cans stand on the living-room floor; they have to end up inside the trash can, which stands on the
kitchen floor. The robot has two hands: it takes two cans, lets them fall into the trash can, then the third.

The estimator supplies
  base() = (x, y, yaw) in M;
  soda(k) = xy of can k in M (None once it is taken or not known), bin() = xy of the trash can in M (or None);
  soda_point(k), bin_point() = the same in the base frame, for the moment of contact;
and optionally
  known() = the numbers of the cans it knows (with simulator truth they are 1, 2, 3);
  see() (take in what the cameras show now), mark_taken(k);
  lying(k), soda_axis(k) = for a can that has fallen over, the direction of its axis in M / in the base frame;
  gap(ignore) = the two sides of a gap between furniture straight ahead, as the head camera shows them.
With simulator truth behind the estimator this script is a diagnostic of the motion; the scored controller puts
its camera-based estimator there.
"""
import itertools
import numpy as np
from manip import hold, drive, pose2, wrap
import drawer
from trash_skills import tuck, plan_floor_pinch, pinch_from_floor, plan_drop, drop_into_bin, plan_lay, lay_trap, drop_on_floor, SODA_WIDTH, BIN_WIDTH, BIN_TOP

FOOTPRINT = (-.40, .24, .345)       # chassis: rear, front and half width in the base frame
PINCH_AT = ((.55, .10), (.62, .08), (.48, .12), (.40, .10), (.70, .0), (.70, .10), (.76, .0), (.34, .14), (.82, .0), (.86, .0))   # where a glass is wanted in the base frame when it is pinched (ahead, to the side of
                                    # the arm), best first: the base centre stays 0.40 m from the sink's front, the glasses stand 0.1-0.3 m behind it
GLASS_BOX = (.76, 1.01)             # the other bag is a box this high (m) for the validator while a hand works near it
TABLE_BOX = (-.79, -1.84, 0., -.16, -.60, .40)   # the coffee table (coffee_table_fqluyq_0) in the map, up to its top
SIDE_BY_SIDE = .05                  # (the drawer's open part is 18 x 27 cm) each hand lets its battery go this far to its own side of the place
DROP_AT = (.66, 0.)                 # where the place on the coffee table is wanted in the base frame when a bag is set down
                                    # box (0.38 m across, up to the rim) in the validator no arm pose is found nearer than 0.55 m
DROP_ALSO = ((.60, 0.), (.74, 0.))
# Where the straight move to a stance starts from, as (distance, direction): the direction is counted from
# "straight behind the stance" (0), towards the left (+). Straight behind first: from 0.45 m the head camera still
# sees a can on the floor in front. The others are for where furniture or another can leaves no room behind.
STAGING = ((.45, 0.), (.30, 0.), (.65, 0.), (.85, 0.), (.55, .8), (.55, -.8), (.80, .8), (.80, -.8), (.55, 1.57), (.55, -1.57), (.80, 1.57), (.80, -1.57))
TIGHT_APPROACH = (1.05, .85, 1.25, .65, .45)    # ... and into a gap between furniture the straight move starts outside it, a metre back if there is room: from there the head camera sees both sides of the gap
TIGHT_MARGIN = (.001, -.02)         # what the map must show for such a gap: no fixed thing under the chassis, and movable furniture at most 0.02 m under its edge (the map's grid is 0.02 m; the way in is then centred by what the camera sees, and a table that is touched gives way)
TIGHT_LATERALS = (-.06, 0., .06)      # sideways offsets of the battery tried there (m; 29 of them made a stance search for one battery take minutes)
SWEEP = .53                         # the chassis corner farthest from the base centre: the circle a turn on the spot sweeps
TURN_CLEAR = SWEEP + .03            # the base turns only where every can and the trash can are farther than this from it (from their surface to its centre)
SODA_KEEPOUT = TURN_CLEAR + SODA_WIDTH / 2      # the base centre stays this far from a can it is not about to take: it may turn anywhere on its way
BIN_KEEPOUT = TURN_CLEAR + BIN_WIDTH / 2        # ... and from the trash can while travelling
SQUEEZE = .05                       # room between the chassis and a can or the trash can that it passes without turning (see route())
HEADINGS = 16
CENTRE_IN_GAP = False               # (a tight stance at the coffee table has furniture on one side only: the camera's gap, made for a can between the sofa and the table, came out 0.4-0.7 m wide and skewed 20 degrees, and every such stance was given up, 309)
MAX_CHECKS = 160                    # stances tested with the arm planner at most, per search (about a second each)
KITCHEN_Y = 99.                    # no stance north of this: the bar's battery is taken from the living room or not at all (on the way into the kitchen, through the passage at its east end, the map pose went 0.2 m and 14 degrees off, and then 1.5 m, 303)


def gap(base, xy, radius):
    """Gap between the chassis standing at @base = (x, y, yaw) and a disc of @radius around @xy."""
    x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = np.asarray(xy, float)[:2] - [x, y]
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    nearest = np.array([np.clip(local[0], FOOTPRINT[0], FOOTPRINT[1]), np.clip(local[1], -FOOTPRINT[2], FOOTPRINT[2])])
    return float(np.linalg.norm(local - nearest)) - radius


DUPLICATE = .08     # two things nearer than this (m) are one piece of glass


def grip_of(est, k):
    """Height (m above the floor) at which the fingertips close on battery @k: from its sighting, or (with simulator
    truth, which gives no sighting) 2 cm under the top of a battery standing on the coffee table."""
    grip = getattr(est, 'soda_grip', None)
    grip = grip(k) if grip is not None else None
    return grip if grip is not None else .437


def known(est):
    """Numbers of the cans the estimator knows to stand somewhere. (With simulator truth behind it they are 1, 2, 3;
    the camera-based estimator numbers what it finds.)"""
    numbers = getattr(est, 'known', None)
    found = numbers() if numbers is not None else [k for k in (1, 2, 3) if est.soda(k) is not None]
    # A piece of glass up to 13 cm long can show as two things a few centimetres apart: the second is then in
    # the way of the hand that pinches the first (start 303). Only the first of such a pair is kept.
    kept = []
    for k in found:
        if all(float(np.linalg.norm(np.asarray(est.soda(k), float)[:2] - np.asarray(est.soda(j), float)[:2])) > DUPLICATE for j in kept):
            kept.append(k)
    return kept


def look(ctx, est):
    see = getattr(est, 'see', None)
    if see is not None:
        yield from hold(ctx, 2)
        see()


def others(est, k=None, with_bin=True):
    """Keep-out discs for the things on the floor the robot is not about to handle."""
    return []           # the mousetraps lie on the cabinet top and under the sink, out of the chassis's way


def loose(est):
    """What stands loose on the floor as far as the estimator knows: [(xy, radius)]."""
    things = [(np.asarray(est.soda(k), float)[:2], SODA_WIDTH / 2) for k in known(est)]
    return things


def leave(ctx, est, margin=.02):
    """Step straight back from a can or the trash can that stands inside the circle the chassis sweeps when it
    turns. (After a pinch that was refused the robot turned to drive on, and a front corner knocked the can over.)"""
    for _ in range(3):
        x, y, yaw = ctx.est.base(); here = np.array([x, y])
        near = [(float(np.linalg.norm(here - centre)) - radius, centre) for centre, radius in loose(est)]
        near = [item for item in near if item[0] < SWEEP + margin]
        if not near:
            return
        room, centre = min(near, key=lambda item: item[0]); away = (here - centre) / max(float(np.linalg.norm(here - centre)), 1e-6)
        for share in (1., .6, .3):
            goal = here + away * (SWEEP + margin + .03 - room) * share
            if ctx.scene.base_free(goal, margin=-.04):
                yield from drive(ctx, goal, yaw, speed=.3, direct=True, tolerance=.03, face_travel=False, max_steps=150)
                break
        else:
            return


def nose_first_room(path, things):
    """The least room between the chassis and @things = [(xy, radius)] when it is driven along @path nose first:
    along each stretch with the heading of that stretch, and turning on the spot at each corner."""
    room = 1.; headings = [float(np.arctan2(q[1] - p[1], q[0] - p[0])) for p, q in zip(path, path[1:])]
    for (p, q), yaw in zip(zip(path, path[1:]), headings):
        count = max(2, int(np.linalg.norm(np.asarray(q) - np.asarray(p)) / .05) + 1)
        for k in range(count):
            s = np.asarray(p, float) + (np.asarray(q, float) - np.asarray(p, float)) * k / (count - 1)
            room = min([room] + [gap((s[0], s[1], yaw), centre, radius) for centre, radius in things])
    for corner, before, after in zip(path[1:-1], headings, headings[1:]):
        for share in np.linspace(0., 1., 9):
            yaw = before + share * wrap(after - before)
            room = min([room] + [gap((corner[0], corner[1], yaw), centre, radius) for centre, radius in things])
    return room


def route(ctx, goal_xy, goal_yaw, keepout=(), **options):
    """drive() around the cans and the trash can.

    keepout: discs [(xy, radius)] the base centre stays out of, or a function that returns them: it is then
    asked again on the way, and the route is planned anew when what is known has changed (a can that came into
    view may stand on the way). With these discs the base may turn anywhere on its way.

    Where they leave no way (the trash can 1 m from a chair shuts off half the kitchen; two cans half a metre
    apart shut a passage), the base passes nose first: the chassis is 0.69 m wide whichever way it is driven,
    but only along its heading does it stay that narrow. The way is planned with the cans and the trash can as
    wide as that needs, and taken only if the chassis's rectangle, driven along it nose first, keeps SQUEEZE
    from each. The base turns to the way's first stretch where it stands and to @goal_yaw at the goal, and it
    has room to turn at both. (A route that merely came closer to the cans while the base turned as it liked
    brushed cans and knocked them over on two of twenty public starts.)
    """
    est = options.pop('est', None)
    if est is not None:
        yield from leave(ctx, est)
    for attempt in range(6):
        planned = list(keepout() if callable(keepout) else keepout); here = np.asarray(ctx.est.base()[:2], float); squeeze = False
        if ctx.scene.path(here, goal_xy, keepout=planned) is not None:
            steps = drive(ctx, goal_xy, goal_yaw, keepout=planned, **options)
        else:
            things = loose(est) if est is not None else []
            clear = all(float(np.linalg.norm(np.asarray(spot, float) - centre)) - radius >= TURN_CLEAR - .02 for spot in (here, goal_xy) for centre, radius in things)
            # With the usual room to furniture first; then with 0.38 m from the base centre to furniture, which
            # leaves the chassis 3 cm on a side (the last way out of a corner of the kitchen that the trash
            # can, the table and its chairs shut: four of 279 starts).
            room = None; narrow = []
            for margin, extra in ((.02, .04), (-.02, .02)):
                narrow = [(centre, FOOTPRINT[2] + radius + SQUEEZE + extra) for centre, radius in things]
                path = ctx.scene.path(here, goal_xy, margin=margin, keepout=narrow) if things else None
                room = nose_first_room(path, things) if path is not None else None
                if room is not None and room >= SQUEEZE:
                    break
            if room is None or not clear or room < SQUEEZE:
                ctx.event('route', ok=False, goal=np.round(goal_xy, 2).tolist(), discs=len(planned), room=None if room is None else round(room, 3), room_to_turn=bool(clear))
                return 'no_route'
            ctx.event('route', squeeze=True, room=round(room, 3), goal=np.round(goal_xy, 2).tolist())
            squeeze = True
            first = float(np.arctan2(path[1][1] - path[0][1], path[1][0] - path[0][0]))
            yield from drive(ctx, here, first, direct=True, tolerance=.15, face_travel=False, max_steps=200)          # turn on the spot
            steps = drive(ctx, goal_xy, goal_yaw, keepout=narrow, **dict(options, nose_first=True, route=path))
        count = 0; status = None
        while True:
            try:
                action = next(steps)
            except StopIteration as stop:
                status = stop.value; break
            yield action
            count += 1
            if callable(keepout) and count % 12 == 0:
                now = keepout()
                if len(now) != len(planned) or any(float(np.linalg.norm(np.asarray(p) - np.asarray(q))) > .15 for (p, _), (q, _) in zip(now, planned)):
                    for _ in range(5):                  # let the base come to rest, then plan again
                        yield ctx.action(base=(0., 0., 0.))
                    break
        if status is not None:
            if status == 'arrived' and squeeze:
                status = yield from drive(ctx, goal_xy, goal_yaw, direct=True, tolerance=max(.06, options.get('tolerance', .035)), face_travel=False, max_steps=250)
            return status
    return 'no_route'


def stance_options(ctx, here, target, offset, obstacles, radius, reasons=None, laterals=(), count=HEADINGS):
    """Base poses from which @target (map xy) lies at @offset in the base frame, each with the point the base
    comes to first: [(x, y, yaw, staging, tight, offset)], nearest to @here first.

    The base gets to the staging point by a planned route and turns to the stance's heading there; the rest is
    a straight move without turning, and the same way back after the work. So a pose counts if
      * the chassis has room at the stance and clears @obstacles = [(xy, radius)] (the other cans, the trash can);
      * some staging point (STAGING) has room for the planner's disc, lies TURN_CLEAR from every obstacle and
        from the target (@radius), and the chassis, moved straight from it to the stance, clears the obstacles.
    A turn next to a can sweeps it with a chassis corner, and so does a route that passes between two cans
    standing half a metre apart: both knocked cans over before turning was kept to the staging points.
    reasons: optional dict that counts what ruled the other headings out. count: how many headings are tried,
    evenly round the target, the first towards it from @here.

    Room is first asked of the disc the route planner takes the chassis for. Where that has none (one public
    start has a can between the sofa and the coffee table, 0.72 m apart), the chassis's own rectangle may still
    fit: such a "tight" pose is entered straight from outside, along its own heading, and left the same way.
    In a gap the chassis has a few centimetres to either side, so the target cannot be asked to lie at the usual
    offset: @laterals are the other sideways offsets tried for a tight pose, and the headings along the map's
    axes are tried besides those counted from the robot's bearing (furniture stands along the walls).
    """
    target = np.asarray(target, float)[:2]; here = np.asarray(here, float)[:2]; out = []
    toward = float(np.arctan2(target[1] - here[1], target[0] - here[0]))
    headings = [wrap(toward + (index + 1) // 2 * (1 if index % 2 else -1) * 2 * np.pi / count) for index in range(count)]
    square = [yaw for yaw in (0., np.pi / 2, np.pi, -np.pi / 2) if laterals and all(abs(wrap(yaw - h)) > .05 for h in headings)]
    for yaw in headings + square:
        c, s = np.cos(yaw), np.sin(yaw); why = None
        place = lambda off: target - np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])
        choices = [(offset, False)] if ctx.scene.base_free(place(offset), margin=-.02) and yaw not in square else []
        if not choices and not ctx.scene.base_free(place(offset), margin=-.02):
            # Of the sideways offsets that fit, the middle one: the chassis then has the same room on both sides.
            choices = [((offset[0], lateral), True) for lateral in laterals]
            choices = [(off, flag) for off, flag in choices if ctx.scene.footprint_free(*place(off), yaw, FOOTPRINT, *TIGHT_MARGIN)]
            choices = choices[len(choices) // 2:len(choices) // 2 + 1]
        if not choices:
            why = 'no room at the stance'
        else:
            (off, tight), = choices; xy = place(off)
            if any(gap((xy[0], xy[1], yaw), centre, r) < .03 for centre, r in obstacles):
                why = 'another can or the trash can at the stance'
            else:
                why = 'no room for the approach'
                for length, turn in ([(length, 0.) for length in TIGHT_APPROACH] if tight else STAGING):
                    direction = yaw + np.pi + turn; staging = xy + length * np.array([np.cos(direction), np.sin(direction)])
                    if not ctx.scene.base_free(staging, margin=-.02):
                        continue
                    along = np.arange(.05, length, .05); swept = [xy + d * (staging - xy) / length for d in along] + [staging]
                    if tight:
                        if not all(ctx.scene.footprint_free(q[0], q[1], yaw, FOOTPRINT, *TIGHT_MARGIN) for q in swept[:-1]):
                            continue
                    elif not ctx.scene.segment_free(staging, xy):
                        continue
                    if any(float(np.linalg.norm(staging - centre)) < TURN_CLEAR + r for centre, r in list(obstacles) + [(target, radius)]):
                        why = 'another can or the trash can where the base would turn'; continue
                    if any(gap((q[0], q[1], yaw), centre, r) < .03 for q in swept for centre, r in obstacles):
                        why = 'another can or the trash can on the approach'; continue
                    # A tight pose is the last resort: it counts as a metre and a half farther than it is.
                    out.append((float(np.linalg.norm(staging - here)) + (1.5 if tight else 0.) + .3 * abs(turn), (float(xy[0]), float(xy[1]), float(yaw), staging, tight, off))); why = None
                    break
        if why is not None and reasons is not None:
            reasons[why] = reasons.get(why, 0) + 1
    return [pose for _, pose in sorted(out, key=lambda item: item[0])]


def stances(ctx, est, target, offsets, obstacles, radius, feasible=None, laterals=()):
    """Yields the stance_options() for which @feasible((x, y, yaw), offset) holds: the arm has a way to the
    target from there (next to the fireplace it has none from some sides). That test costs a second or two, so
    it is made only when the caller asks for the next pose.

    offsets: where the target may lie in the base frame, the usual place first. Sixteen headings are tried for
    each, then forty-eight: where furniture leaves one or two stances (a trash can between the fridge and a
    chair), sixteen headings counted from where the robot happens to stand can miss them all.
    """
    seen = []; checks = 0
    for count in (HEADINGS, 3 * HEADINGS):
        for offset in offsets:
            for x, y, yaw, staging, tight, off in stance_options(ctx, est.base()[:2], target, offset, obstacles, radius, laterals=laterals, count=count):
                if any(np.hypot(x - q[0], y - q[1]) < .03 and abs(wrap(yaw - q[2])) < .03 for q in seen):
                    continue
                seen.append((x, y, yaw))
                if feasible is not None:
                    checks += 1
                    if checks > MAX_CHECKS:
                        # (a battery out of reach cost minutes of computation per search; the scored episodes are
                        # stopped after two hours of wall time, without a metrics file)
                        ctx.event('stances', ok=False, reason='search cut', checks=checks - 1); return
                if feasible is None or feasible((x, y, yaw), off):
                    yield x, y, yaw, staging, tight, off


def centre_in_gap(ctx, est, k, goal, yaw):
    """In front of a gap between furniture that the chassis just fits: line the base up with what the head camera
    shows of the gap's two sides, on the middle line between them and along it. Returns (goal, yaw, here): the
    stance moved onto that line, the heading along it and where the base now stands (the way out); or None if the
    gap shows too narrow.

    The gap between the sofa and the coffee table leaves the chassis 2 cm on each side. The map and the map pose
    are each good to about that, and a heading 1.7 degrees off costs as much again over the chassis's length:
    the way in is taken from the camera, not from the map.
    """
    here = np.asarray(ctx.est.base()[:2], float)
    for attempt in range(4):
        yield from hold(ctx, 4)
        sides = est.gap(ignore=[est.soda_point(k)]); x, y, heading = ctx.est.base(); here = np.array([x, y])
        if sides[0] is None or sides[1] is None:
            ctx.event('gap', ok=None, sides=[None if side is None else np.round(side, 3).tolist() for side in sides], attempt=attempt)
            return goal, yaw, here                   # one side does not show: by the map
        a, b = (sides[0][0] + sides[1][0]) / 2, (sides[0][1] + sides[1][1]) / 2; skew = float(np.arctan(b))
        ahead = np.array([np.cos(yaw), np.sin(yaw)]); distance = float(np.dot(goal - here, ahead))
        width = (sides[0][0] - sides[1][0]) + (sides[0][1] - sides[1][1]) * distance
        ctx.event('gap', attempt=attempt, middle=round(a, 3), skew_deg=round(float(np.degrees(skew)), 2), width=round(width, 3))
        if width * np.cos(skew) < 2 * FOOTPRINT[2] + .005:
            ctx.event('gap', ok=False, width=round(width, 3)); return None
        # The middle line in the map: through the point @a to the left of the base, heading @skew off the base's.
        c, s = np.cos(heading), np.sin(heading); yaw = heading + skew; ahead = np.array([np.cos(yaw), np.sin(yaw)])
        foot_x = -a * b / (1. + b * b); foot = here + np.array([c * foot_x - s * (a + b * foot_x), s * foot_x + c * (a + b * foot_x)])
        goal = foot + distance * ahead
        if abs(a) < .008 and abs(skew) < .006:
            break
        yield from drive(ctx, foot, yaw, speed=.15, direct=True, tolerance=.006, yaw_tolerance=.005, min_speed=.03, min_turn=.05, face_travel=False, max_steps=250)
        here = np.asarray(ctx.est.base()[:2], float)
    return goal, yaw, here


def fetch(ctx, est, k, arm):
    """Stand in front of can @k, pinch it with @arm and fold the arm again. Returns how the hand holds it:
    'standing' (pinched from above, the can along the hand), 'lying' (a can that had fallen over, pinched across),
    or False."""
    side = 1. if arm == 'left' else -1.; offsets = [(a, side * b) for a, b in PINCH_AT]; failed = []
    fallen = getattr(est, 'lying', lambda k: None)
    for attempt in range(3):
        target = est.soda(k)
        if target is None:
            return False
        obstacles = [(c, SODA_WIDTH / 2) for c, _ in others(est, k, with_bin=False)] + ([(np.asarray(est.bin(), float)[:2], BIN_WIDTH / 2)] if est.bin() is not None else [])
        reached = False; ctx.arm = arm
        tried = lambda pose: any(np.hypot(pose[0] - f[0], pose[1] - f[1]) < .15 and abs(wrap(pose[2] - f[2])) < .35 for f in failed)
        axis = fallen(k)
        # The other cans are boxes for the validator, as they will be when the hand goes down (without them a
        # stance was chosen from which the arm then had no way past a neighbour).
        boxes = np.array([[c[0] - SODA_WIDTH / 2, c[1] - SODA_WIDTH / 2, z - .05, c[0] + SODA_WIDTH / 2, c[1] + SODA_WIDTH / 2, z + .04]
                          for c, z in [(np.asarray(est.soda(j), float)[:2], grip_of(est, j)) for j in known(est) if j != k]]).reshape(-1, 6)
        height = grip_of(est, k)

        def reachable(pose, off):
            if tried(pose) or pose[1] > KITCHEN_Y:
                return False
            ctx.guard = boxes
            plan = plan_floor_pinch(ctx, off, base=pose2(*pose), seeds=1, tight=not ctx.scene.base_free(pose[:2], margin=-.02), lying=None if axis is None else axis - pose[2], height=height,
                                    neighbours=[np.linalg.inv(pose2(*pose))[:2, :2] @ (c - np.asarray(pose[:2])) for c, _ in others(est, k, with_bin=False)])
            ctx.guard = np.zeros((0, 6))
            return plan is not None
        options = stances(ctx, est, target, offsets, obstacles, SODA_WIDTH / 2, feasible=reachable, laterals=TIGHT_LATERALS)
        for x, y, yaw, staging, tight, off in itertools.islice(options, 3):
            def around(staging=staging):
                # The other cans and the trash can at their full distance; the can itself as far as the staging point allows.
                spot = est.soda(k) if est.soda(k) is not None else target
                return others(est, k) + [(np.asarray(spot, float)[:2], min(SODA_KEEPOUT, float(np.linalg.norm(staging - np.asarray(spot, float)[:2])) - .03))]
            status = yield from route(ctx, staging, yaw, keepout=around, tolerance=.04, est=est)
            if status != 'arrived':
                continue
            yield from look(ctx, est)
            if est.soda(k) is None:                 # it is not there (taken for a can by mistake, or knocked away)
                return False
            fresh = getattr(est, 'fresh', None)
            if fresh is not None and not fresh(k):
                # What is known of the can's place may be old or from far off (after a long way with turns it was
                # 3 cm off, and the hand knocked the can over), and from here it does not show well: either it is
                # not in the image with the stance's heading (a staging point to the side), or it is so near that
                # the image's lower edge cuts it (a staging point 0.3 m behind the stance). The base may turn
                # here: it faces the can, and if the can is nearer than 0.8 m it steps straight back until it is
                # 0.85 m off, where there is room for that; it looks, and comes back.
                here = np.asarray(ctx.est.base()[:2], float); spot = np.asarray(est.soda(k), float)[:2]; apart = float(np.linalg.norm(spot - here))
                facing = float(np.arctan2(spot[1] - here[1], spot[0] - here[0])); view = here
                yield from drive(ctx, here, facing, direct=True, tolerance=.2, yaw_tolerance=.06, face_travel=False, max_steps=200)
                if apart < .80:
                    back = here - (.85 - apart) * np.array([np.cos(facing), np.sin(facing)])
                    clear = all(gap((back[0], back[1], facing), centre, radius) > .05 for centre, radius in loose(est))
                    if ctx.scene.base_free(back, margin=-.02) and clear:
                        view = back
                        yield from drive(ctx, view, facing, speed=.25, direct=True, tolerance=.04, face_travel=False, max_steps=200)
                yield from hold(ctx, 4)
                yield from look(ctx, est)
                if view is not here:
                    yield from drive(ctx, here, facing, speed=.25, direct=True, tolerance=.03, face_travel=False, max_steps=200)
                yield from drive(ctx, here, yaw, direct=True, tolerance=.2, yaw_tolerance=.03, face_travel=False, max_steps=200)
                ctx.event('look', soda=k, apart=round(apart, 2), stepped_back=view is not here, fresh=bool(fresh(k)))
                if est.soda(k) is None:
                    return False
            target = est.soda(k)
            # Straight in, to where the can now lies at the wanted offset (the last look may have moved it a little).
            c, s = np.cos(yaw), np.sin(yaw); goal = np.asarray(target, float)[:2] - np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])
            if tight and CENTRE_IN_GAP and getattr(est, 'gap', None) is not None:
                lined_up = yield from centre_in_gap(ctx, est, k, goal, yaw)
                if lined_up is None:
                    continue
                goal, yaw, staging = lined_up
            status = yield from drive(ctx, goal, yaw, speed=.25, direct=True, tolerance=.02, min_speed=.04, face_travel=False, max_steps=400)
            reached = status == 'arrived' or float(np.linalg.norm(goal - ctx.est.base()[:2])) < .05
            if reached:
                break
            # Held up on the way in: back to where the base may turn, before anything else.
            yield from drive(ctx, staging, yaw, speed=.3, direct=True, tolerance=.05, face_travel=False, max_steps=400)
        if not reached:
            ctx.event('fetch', ok=False, soda=k, reason='no stance reached'); return False
        ctx.arm = arm
        yield from hold(ctx, 4)
        point = est.soda_point(k)
        if point is None:
            return False
        # The other cans: none of the robot may pass through them on the way down (boxes for the validator),
        # and one that stands right next to this can decides which way the fingers close.
        others_k = [j for j in known(est) if j != k]; beside = [est.soda_point(j) for j in others_k]; world = pose2(*ctx.est.base())
        spots = [(world[:2, :2] @ np.asarray(b, float)[:2] + world[:2, 3], grip_of(est, j)) for b, j in zip(beside, others_k) if b is not None]
        ctx.guard = np.array([[x - SODA_WIDTH / 2, y - SODA_WIDTH / 2, z - .05, x + SODA_WIDTH / 2, y + SODA_WIDTH / 2, z + .04] for (x, y), z in spots]).reshape(-1, 6)
        across = est.soda_axis(k) if fallen(k) is not None else None
        ctx.event('target', soda=k, arm=arm, point=np.round(point, 3).tolist(), source=getattr(est, 'source', None), lying=None if across is None else round(float(np.degrees(across))),
                  beside=[np.round(b, 3).tolist() for b in beside if b is not None and np.linalg.norm(np.asarray(b)[:2] - point[:2]) < .3])
        held = yield from pinch_from_floor(ctx, point, tight=tight, neighbours=beside, lying=across, height=grip_of(est, k))
        status = yield from tuck(ctx, arms=(arm,))          # the other cans stay in the validator while the arm folds
        ctx.guard = np.zeros((0, 6))
        if status != 'done':
            yield from tuck(ctx, arms=(arm,))
        # Back the way it came, to where the base may turn.
        yield from drive(ctx, staging, yaw, speed=.3, direct=True, tolerance=.05, face_travel=False, max_steps=400)
        if held:
            mark = getattr(est, 'mark_taken', None)
            if mark is not None:
                mark(k)
            return 'standing' if across is None else 'lying'
        failed.append((x, y, yaw))                 # another side next time
        if held is False and getattr(est, 'forget', None) is not None:
            est.forget(k)                           # the hand went down and closed on nothing: the can is not where it was believed
        yield from look(ctx, est)
    return False


LYING = {'left': np.radians(60.), 'right': np.radians(-60.)}     # the way each hand holds a can lying most easily: its bottom pointing this far off straight ahead


def lying_directions(arm, yaw, used):
    """The azimuths (base frame, for a base heading @yaw) along which the can in @arm may be let go lying, best
    first: the hand's own direction, straight ahead and the other hand's direction, ordered by the angle at
    which each crosses the cans already in the trash can (@used: their directions in the map), widest first;
    the hand's own first where it makes no difference. Two cans from one stance cross at 60 degrees, a third
    lies across both."""
    def apart(azimuth):
        return min([abs((azimuth + yaw - u + np.pi / 2) % np.pi - np.pi / 2) for u in used] or [np.pi / 2])
    options = [LYING[arm], 0., -LYING[arm]]
    return sorted(options, key=lambda azimuth: (-round(apart(azimuth), 2), options.index(azimuth)))


# ---- laying the mousetraps under the sink (v01) ---------------------------------------------------------------------
# A mousetrap counts as under the sink when the vertical line through the middle of its box meets the vanity
# (furniture_sink_ojjqku_0: x -2.42..-1.905, y 1.62..2.83, 0.35-0.85 m up, free floor under it): middle at
# x <= -1.905 (oracle job under301: -1.900 yes, -1.897 no). Next to the sink cannot hold for a trap on the floor
# (its box is 0.33 m under the vanity's, and next-to allows 0.15 m). So every trap is laid with its middle 4 cm
# in under the vanity's front, 0.15 m apart along it; one that cannot be laid there is let go on the floor in front.
# Reaching the floor right in front of the vanity, the trunk bends over it and hits its top (offline plans): a trap
# is laid from the side, the base standing parallel to the vanity's front 0.45-0.52 m from it, one trap per stance:
# the left hand facing south (places in the south half), the right hand facing north (oracle job trial301: both
# laid under, flat, on the floor).
LAY_X = -1.945
LAY_SPOTS = {'left': ((LAY_X, 1.95), (LAY_X, 2.10), (LAY_X, 1.80), (LAY_X, 2.25)), 'right': ((LAY_X, 2.70), (LAY_X, 2.55), (LAY_X, 2.40))}
LAY_ARM_AT = {'left': ((.55, -.45), (.45, -.45), (.65, -.45), (.55, -.52), (.40, -.52)), 'right': ((.55, .45), (.45, .45), (.65, .45), (.55, .52), (.40, .52))}
LAY_TURNS = (0., .35, -.35)         # the trap's axis this far off pointing straight in (rad)
# Where a trap goes when it cannot be laid under the sink: the open floor between the bathtub and the shower,
# 0.15 m in front of the bathtub's side, out of the robot's ways (it still counts as on the floor).
FLOOR_SPOTS = ((-1.40, 2.93), (-1.10, 2.93), (-.80, 2.93), (-.50, 2.93), (-.20, 2.93), (.10, 2.93))
FLOOR_TURNS = (0., .5, -.5, 1.57, -1.57)
FLOOR_AT = ((.60, 0.), (.55, 0.), (.70, 0.), (.50, 0.))


def lay_spots(ctx, mode, arm=None):
    used = ctx.__dict__.setdefault('laid', [])
    spots = list(LAY_SPOTS[arm]) if mode == 'under' else list(FLOOR_SPOTS)
    return [s for s in spots if all(np.hypot(s[0] - u[0], s[1] - u[1]) > .1 for u in used)]


def deliver(ctx, est, arms, used):
    """Drive to the sink and lay the mousetraps in @arms = [(arm, how)] under it; those that cannot be laid there
    go on the open floor. Returns the entries let go."""
    gone = []
    for mode in ('under', 'floor'):
        groups = [[item] for item in arms] if mode == 'under' else ([list(arms)] if len(arms) < 2 else [list(arms)] + [[item] for item in arms])
        for group in groups:
            group = [item for item in group if item not in gone]
            if group:
                gone += yield from deliver_from_one_stance(ctx, est, group, mode, ctx.__dict__.setdefault('drop_failures', []))
        if len(gone) == len(arms):
            break
    if len(gone) < len(arms):
        ctx.event('deliver', ok=False, reason='no stance to lay a trap', left=[arm for arm, _ in arms if (arm, _) not in gone])
        # Last resort: let it fall on the floor in front of the base, wherever it stands (it still counts as on the floor).
        for item in [item for item in arms if item not in gone]:
            ctx.arm = item[0]
            if (yield from drop_on_floor(ctx, ctx.scene, ctx.base_matrix())):
                gone.append(item)
            yield from tuck(ctx, arms=(item[0],))
    return gone


def assign(pose, spots, arms):
    """The places (map, base frame) of @arms at a stance @pose: the left arm takes the place furthest to the left."""
    inverse = np.linalg.inv(pose2(*pose)); local = [(inverse @ np.r_[s, 0., 1.])[:2] for s in spots]
    order = sorted(range(len(spots)), key=lambda i: -local[i][1])
    if len(arms) == 1:
        return {arms[0][0]: (spots[order[0]], local[order[0]])}
    return {'left': (spots[order[0]], local[order[0]]), 'right': (spots[order[-1]], local[order[-1]])}


def lay_azimuths(mode, yaw):
    return [wrap(np.pi - yaw + turn) for turn in LAY_TURNS] if mode == 'under' else [wrap(-yaw + turn) for turn in FLOOR_TURNS]


def deliver_from_one_stance(ctx, est, arms, mode, failed):
    free = lay_spots(ctx, mode, arms[0][0])
    if len(free) < len(arms):
        return []
    # Neighbouring places, the first free ones first (two arms: 0.30 m apart, one each side of the base's line).
    spots = [free[0]] if len(arms) == 1 else [free[0], min(free[1:], key=lambda s: abs(float(np.hypot(s[0] - free[0][0], s[1] - free[0][1])) - .30))]
    target = np.mean(np.array(spots), axis=0)

    def reachable(pose, off):
        if any(mode == f[4] and arm == f[3] and np.hypot(pose[0] - f[0], pose[1] - f[1]) < .15 and abs(wrap(pose[2] - f[2])) < .35 for f in failed for arm, _ in arms):
            return False
        places = assign(pose, spots, arms)
        for arm, _ in arms:
            ctx.arm = arm
            if not any(plan_lay(ctx, places[arm][1], azimuth, base=pose2(*pose)) is not None for azimuth in lay_azimuths(mode, pose[2])[:3]):
                return False
        return True

    gone = []
    for x, y, yaw, staging, tight, off in itertools.islice(stances(ctx, est, target, LAY_ARM_AT[arms[0][0]] if mode == 'under' else FLOOR_AT, [], .10, feasible=reachable), 3):
        status = yield from route(ctx, staging, yaw, keepout=lambda: others(est, None, with_bin=False), tolerance=.04, est=est)
        if status != 'arrived':
            continue
        c, s = np.cos(yaw), np.sin(yaw); goal = target - np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])
        status = yield from drive(ctx, goal, yaw, speed=.25, direct=True, tolerance=.025, min_speed=.04, face_travel=False, max_steps=400)
        if status != 'arrived' and float(np.linalg.norm(goal - ctx.est.base()[:2])) > .06:
            yield from drive(ctx, staging, yaw, speed=.3, direct=True, tolerance=.05, face_travel=False, max_steps=400)
            continue
        here = tuple(ctx.est.base()); places = assign(here, spots, arms)
        for arm, how in arms:
            ctx.arm = arm
            yield from hold(ctx, 3)
            spot, centre = places[arm]
            # (in the vanity's frame: up to 1.5 cm further in or out, 3 cm along it)
            c_, s_ = np.cos(here[2]), np.sin(here[2]); to_base = lambda dx, dy: (c_ * dx + s_ * dy, -s_ * dx + c_ * dy)
            shifts = [(0., 0.)] + ([to_base(dx, dy) for dx, dy in ((-.015, 0.), (0., .03), (0., -.03), (-.015, .03), (-.015, -.03), (.01, 0.))] if mode == 'under' else [(0., .05), (0., -.05), (.05, 0.)])
            azimuth = yield from lay_trap(ctx, centre, lay_azimuths(mode, here[2]), shifts=shifts)
            if azimuth is not None:
                ctx.laid.append(spot); ctx.event('laid', arm=arm, mode=mode, spot=list(spot))
            status = yield from tuck(ctx, arms=(arm,))
            if status != 'done':
                yield from drive(ctx, staging, yaw, speed=.2, direct=True, tolerance=.05, face_travel=False, max_steps=400)
                yield from tuck(ctx, arms=(arm,))
            if azimuth is not None:
                gone.append((arm, how))
            else:
                failed.append((x, y, yaw, arm, mode))
        yield from drive(ctx, staging, yaw, speed=.3, direct=True, tolerance=.05, face_travel=False, max_steps=400)
        return gone
    return gone


def plan_tour(ctx, est, cans=None, hands=2):
    """Which of @cans (default: all that are known) go first, one for each of the @hands that are free, and in which
    order: [[(k, arm), ...], [(k, arm), ...]] = the first trip and what is left, by the length of the whole way.

    A way is measured with the route planner, around the cans that still stand at that point of the order: a can
    in a passage is taken before the cans behind it (driving between two cans half a metre apart knocked one
    over). Where the planner finds no way the order counts as 30 m longer.
    """
    cans = known(est) if cans is None else list(cans); bin_xy = est.bin(); here = np.asarray(est.base()[:2], float)
    if not cans:
        return []
    # Not more than four at a time, the nearest (every order of them is tried).
    cans = sorted(cans, key=lambda k: float(np.linalg.norm(np.asarray(est.soda(k), float)[:2] - here)))[:4]; known_lengths = {}
    spots = {k: np.asarray(est.soda(k), float)[:2] for k in cans}; home = np.asarray(bin_xy, float)[:2] if bin_xy is not None else here

    def length(a, b, standing):
        key = (tuple(np.round(a, 2)), tuple(np.round(b, 2)), tuple(sorted(standing)))
        if key not in known_lengths:
            # A can next to an end of the way keeps the base only as far off as that end is from it.
            discs = [(spots[j], min(SODA_KEEPOUT, max(.25, min(float(np.linalg.norm(spots[j] - a)), float(np.linalg.norm(spots[j] - b))) - .12))) for j in standing]
            path = ctx.scene.path(a, b, keepout=discs)
            known_lengths[key] = 30. + float(np.linalg.norm(b - a)) if path is None else float(sum(np.linalg.norm(np.asarray(q) - np.asarray(p)) for p, q in zip(path, path[1:])))
        return known_lengths[key]

    best = None
    for order in itertools.permutations(cans):
        first, rest = order[:hands], order[hands:]; standing = set(order) - {first[0]}
        total = length(here, spots[first[0]], standing)
        if len(first) > 1:
            standing = standing - {first[1]}; total += length(spots[first[0]], spots[first[1]], standing)
        total += length(spots[first[-1]], home, standing)
        if rest:
            total += length(home, spots[rest[0]], set(rest) - {rest[0]}) + length(spots[rest[0]], home, set(rest) - {rest[0]})
        if best is None or total < best[0] - 1e-6:
            best = (total, first, rest)
    _, first, rest = best
    return [list(first), list(rest)]


# ---- the bar's battery from the kitchen side (x02) -----------------------------------------------------------------
# Behind the chairs the bar's battery has no stance from the living room (x01: 301, 304, 306, 310). From the kitchen
# side it has, east of the oven; but on the way in, through the passage at the bar's east end, the map pose went 0.2 m
# and 14 degrees off once (303 in development; right to 1.5 cm on 304 and 310 of round k2). So the robot goes there
# last, when nothing else is left: whatever happens there, the batteries already in the drawer stay in.
BAR_Y = 1.30                        # a battery north of this stands on the bar
KITCHEN_SPOT = (-.60, 2.20)         # where the base goes first in the kitchen (0.58 m from anything in the map)
LIVING_SPOT = (0., .20)             # ... and back in the living room (0.77 m)
KITCHEN_WEST = -1.20                # ... and only for a battery east of this: further west the oven keeps the base away on the kitchen side too (301, 306), and on the way to one at x -1.28 the robot fell over in the passage (304, round k3)
KITCHEN_BEFORE = -1               # the trip is begun only before this step (it takes some 2500)


def from_the_kitchen(ctx, est, survey, k, used):
    """Take bar battery @k from the kitchen side and let it fall into the drawer. Returns how many were let go."""
    global KITCHEN_Y
    ctx.event('kitchen', start=True, soda=k)
    status = yield from route(ctx, np.asarray(KITCHEN_SPOT, float), -np.pi / 2, keepout=lambda: others(est, None, with_bin=False), tolerance=.12, est=est)
    if status != 'arrived':
        ctx.event('kitchen', ok=False, reason='no way in', status=status); return 0
    # (no fresh global fix here: in the kitchen the scans match several places, and the global localiser took a pose
    # 2 m and 35 degrees off where the tracked one was right to 1.5 cm, 304 and 310 of round k2)
    yield from look(ctx, est)
    saved = KITCHEN_Y; KITCHEN_Y = 9.; how = False; arm = 'left'
    try:
        for arm in ('left', 'right'):
            if est.soda(k) is None:
                break
            how = yield from fetch(ctx, est, k, arm)
            if how:
                break
    finally:
        KITCHEN_Y = saved
    status = yield from route(ctx, np.asarray(LIVING_SPOT, float), 0., keepout=lambda: others(est, None, with_bin=False), tolerance=.12, est=est)
    ctx.event('kitchen', ok=bool(how), back=status)
    if not how:
        return 0
    gone = yield from deliver(ctx, est, [(arm, how)], used)
    return len(gone)


PIECES = 4          # the four mousetraps
SEARCHES = 2        # rounds of the lookouts after nothing is left that is known (all four traps show from any lookout)


def full_task(ctx, est, survey=None):
    """Fold, look around, then trips to the cans: two at a time, one in each hand, into the trash can.

    survey: optional generator function(where) that looks around and tells the estimator: 'start' where the
    robot stands, 'bin' for the trash can if the start did not show it, 'cans' when no can is known.
    """
    status = yield from tuck(ctx)
    ctx.event('tuck', status=status)
    if survey is not None:
        yield from survey('start')
        if est.bin() is None:
            yield from survey('bin')
    if est.bin() is None:
        return 'no_trash_can'
    done = 0; searches = 0; lying = []; misses = []; held = []; stuck = 0
    # A can that both hands have failed to take twice is left where it is (the tries cost 300 steps each and
    # the other cans are worth more).
    left_alone = lambda k: sum(float(np.linalg.norm(np.asarray(est.soda(k), float)[:2] - spot)) < .15 for spot, _ in misses) >= 4
    # A trap one hand has missed twice is left to the other hand (302: four misses of one trap, all by the right hand).
    arm_missed = lambda k, arm: sum(float(np.linalg.norm(np.asarray(est.soda(k), float)[:2] - spot)) < .15 for spot, a in misses if a == arm) >= 2
    wanted = lambda: [k for k in known(est) if not left_alone(k)]
    kitchen_tried = False
    for trip in range(12):                      # further rounds take what an earlier one dropped or missed
        if not held and not wanted() and not kitchen_tried and survey is not None and ctx.steps < KITCHEN_BEFORE:
            bar = [k for k in known(est) if est.soda(k) is not None and float(est.soda(k)[1]) > BAR_Y and float(est.soda(k)[0]) > KITCHEN_WEST]
            if bar:
                kitchen_tried = True
                got = yield from from_the_kitchen(ctx, est, survey, bar[0], lying)
                done += got
                ctx.event('delivered', trip=trip, total=done, still_held=[], kitchen=True)
                continue
        if not held and not wanted():
            if survey is None or searches >= SEARCHES or done >= PIECES:
                break
            searches += 1
            yield from survey('cans')
            if not wanted():
                continue                        # the next round of lookouts (a round stops at the first that shows a piece)
        free = [arm for arm in ('left', 'right') if arm not in [a for a, _ in held]]
        if free and wanted():
            tour = plan_tour(ctx, est, wanted(), hands=len(free)); order = [k for leg in tour for k in leg]
            ctx.event('tour', trip=trip, plan=tour, hands=free)
            for arm in free:
                # The cans in the order of the plan, then any that came into view on the way. A can this arm
                # cannot have (no stance: cans standing close together are in each other's way until the outer
                # one is gone) is left for the other arm or for the next round.
                missed = set()
                while True:
                    rest = [k for k in order if k in wanted() and k not in missed and not arm_missed(k, arm)] + [k for k in wanted() if k not in order and k not in missed and not arm_missed(k, arm)]
                    if not rest:
                        break
                    spot = np.asarray(est.soda(rest[0]), float)[:2].copy()
                    ok = yield from fetch(ctx, est, rest[0], arm)
                    ctx.event('fetched', soda=rest[0], arm=arm, ok=bool(ok), how=ok or None)
                    if ok:
                        held.append((arm, ok)); break
                    missed.add(rest[0]); misses.append((spot, arm))
        if not held:
            continue
        gone = yield from deliver(ctx, est, held, lying)
        done += len(gone); held = [item for item in held if item not in gone]
        ctx.event('delivered', trip=trip, total=done, still_held=[arm for arm, _ in held])
        if held and not gone:
            stuck += 1
            if stuck >= 2:                          # the trash can cannot be reached from anywhere
                break
    yield from hold(ctx, 20)
    return 'done_%d' % done
