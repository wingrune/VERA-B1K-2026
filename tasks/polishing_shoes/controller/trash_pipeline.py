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
from trash_skills import tuck, plan_floor_pinch, pinch_from_floor, plan_drop, drop_into_bin, SODA_WIDTH, BIN_WIDTH, BIN_TOP

FOOTPRINT = (-.40, .24, .345)       # chassis: rear, front and half width in the base frame
PINCH_AT = ((.37, .13), (.45, .10), (.52, .06))   # where a shoe is wanted in the base frame when it is pinched (ahead, to the side of the arm)
                                    # the arm), best first: the base centre stays 0.40 m from the sink's front, the glasses stand 0.1-0.3 m behind it
GLASS_BOX = (0., .11)              # the other shoe is a box this high for the validator while a hand works near it
SIDE_BY_SIDE = .10                  # each hand sets its shoe down this far to its own side of the place (0.29 m apart the shoes did not count as next to each other)
DROP_AT = (.45, 0.)                 # where the place by the footstool is wanted in the base frame when a shoe is set down
                                    # box (0.38 m across, up to the rim) in the validator no arm pose is found nearer than 0.55 m
DROP_ALSO = ((.40, 0.), (.52, 0.))
# Where the straight move to a stance starts from, as (distance, direction): the direction is counted from
# "straight behind the stance" (0), towards the left (+). Straight behind first: from 0.45 m the head camera still
# sees a can on the floor in front. The others are for where furniture or another can leaves no room behind.
STAGING = ((.45, 0.), (.30, 0.), (.65, 0.), (.85, 0.), (.55, .8), (.55, -.8), (.80, .8), (.80, -.8), (.55, 1.57), (.55, -1.57), (.80, 1.57), (.80, -1.57))
TIGHT_APPROACH = (1.05, .85, 1.25, .65, .45)    # ... and into a gap between furniture the straight move starts outside it, a metre back if there is room: from there the head camera sees both sides of the gap
TIGHT_MARGIN = (.001, -.02)         # what the map must show for such a gap: no fixed thing under the chassis, and movable furniture at most 0.02 m under its edge (the map's grid is 0.02 m; the way in is then centred by what the camera sees, and a table that is touched gives way)
TIGHT_LATERALS = tuple(round(v, 2) for v in np.arange(-.14, .145, .01))     # sideways offsets of the can tried there (m)
SWEEP = .53                         # the chassis corner farthest from the base centre: the circle a turn on the spot sweeps
TURN_CLEAR = SWEEP + .03            # the base turns only where every can and the trash can are farther than this from it (from their surface to its centre)
SODA_KEEPOUT = TURN_CLEAR + SODA_WIDTH / 2      # the base centre stays this far from a can it is not about to take: it may turn anywhere on its way
BIN_KEEPOUT = TURN_CLEAR + BIN_WIDTH / 2        # ... and from the trash can while travelling
SQUEEZE = .05                       # room between the chassis and a can or the trash can that it passes without turning (see route())
HEADINGS = 16


def gap(base, xy, radius):
    """Gap between the chassis standing at @base = (x, y, yaw) and a disc of @radius around @xy."""
    x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = np.asarray(xy, float)[:2] - [x, y]
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    nearest = np.array([np.clip(local[0], FOOTPRINT[0], FOOTPRINT[1]), np.clip(local[1], -FOOTPRINT[2], FOOTPRINT[2])])
    return float(np.linalg.norm(local - nearest)) - radius


DUPLICATE = .08     # two things nearer than this (m) are one piece of glass


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
    discs = []          # the glasses stand on the sink, out of the chassis's way
    if with_bin and est.bin() is not None:
        discs.append((np.asarray(est.bin(), float)[:2], BIN_KEEPOUT))
    return discs


def loose(est):
    """What stands loose on the floor as far as the estimator knows: [(xy, radius)]."""
    things = [(np.asarray(est.soda(k), float)[:2], SODA_WIDTH / 2) for k in known(est)]
    if est.bin() is not None:
        things.append((np.asarray(est.bin(), float)[:2], BIN_WIDTH / 2))
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
    seen = []
    for count in (HEADINGS, 3 * HEADINGS):
        for offset in offsets:
            for x, y, yaw, staging, tight, off in stance_options(ctx, est.base()[:2], target, offset, obstacles, radius, laterals=laterals, count=count):
                if any(np.hypot(x - q[0], y - q[1]) < .03 and abs(wrap(yaw - q[2])) < .03 for q in seen):
                    continue
                seen.append((x, y, yaw))
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
        boxes = np.array([[c[0] - SODA_WIDTH / 2, c[1] - SODA_WIDTH / 2, GLASS_BOX[0], c[0] + SODA_WIDTH / 2, c[1] + SODA_WIDTH / 2, GLASS_BOX[1]]
                          for c in [np.asarray(est.soda(j), float)[:2] for j in known(est) if j != k]]).reshape(-1, 6)

        def reachable(pose, off):
            if tried(pose):
                return False
            ctx.guard = boxes
            plan = plan_floor_pinch(ctx, off, base=pose2(*pose), seeds=1, tight=not ctx.scene.base_free(pose[:2], margin=-.02), lying=None if axis is None else axis - pose[2],
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
            if tight and getattr(est, 'gap', None) is not None:
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
        beside = [est.soda_point(j) for j in known(est) if j != k]; world = pose2(*ctx.est.base())
        spots = [world[:2, :2] @ np.asarray(b, float)[:2] + world[:2, 3] for b in beside if b is not None]
        ctx.guard = np.array([[x - SODA_WIDTH / 2, y - SODA_WIDTH / 2, GLASS_BOX[0], x + SODA_WIDTH / 2, y + SODA_WIDTH / 2, GLASS_BOX[1]] for x, y in spots]).reshape(-1, 6)
        across = est.soda_axis(k) if fallen(k) is not None else None
        ctx.event('target', soda=k, arm=arm, point=np.round(point, 3).tolist(), source=getattr(est, 'source', None), lying=None if across is None else round(float(np.degrees(across))),
                  beside=[np.round(b, 3).tolist() for b in beside if b is not None and np.linalg.norm(np.asarray(b)[:2] - point[:2]) < .3])
        held = yield from pinch_from_floor(ctx, point, tight=tight, neighbours=beside, lying=across)
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


def deliver(ctx, est, arms, used):
    """Drive to the trash can and let the cans in @arms = [(arm, how it holds its can: 'standing' or 'lying')]
    fall into it, lying. Returns the entries of @arms whose cans were let go.
    used: the directions (map frame) of the cans let go into it before; those of this delivery are added.

    Both hands from one stance if there is one; else each hand from a stance of its own (with the trash can
    between the fridge and a chair, one training start had a stance for the left hand only)."""
    gone = []; failed = ctx.__dict__.setdefault('drop_failures', [])       # stances from which an arm was held up
    for group in ([list(arms)] if len(arms) < 2 else [list(arms)] + [[item] for item in arms]):
        group = [item for item in group if item not in gone]
        if group:
            gone += yield from deliver_from_one_stance(ctx, est, group, used, failed)
    if not gone:
        ctx.event('deliver', ok=False, reason='no stance at the trash can')
    return gone


def deliver_from_one_stance(ctx, est, arms, used, failed):
    target = est.bin(); gone = []
    if target is None or not arms:
        return gone
    obstacles = [(c, SODA_WIDTH / 2) for c, _ in others(est, None, with_bin=False)]

    def reachable(pose, off):
        if any(arm == f[3] and np.hypot(pose[0] - f[0], pose[1] - f[1]) < .15 and abs(wrap(pose[2] - f[2])) < .35 for f in failed for arm, _ in arms):
            return False
        before = list(used)
        for arm, how in arms:
            ctx.arm = arm; azimuth = lying_directions(arm, pose[2], before)[0]; before.append(azimuth + pose[2])
            if not plan_drop(ctx, off, azimuth, base=pose2(*pose), across=how == 'lying')[1]:
                return False
        return True

    for x, y, yaw, staging, tight, off in itertools.islice(stances(ctx, est, target, (DROP_AT,) + DROP_ALSO, obstacles, BIN_WIDTH / 2, feasible=reachable), 4):
        def around(staging=staging):
            spot = est.bin() if est.bin() is not None else target
            return others(est, None, with_bin=False) + [(np.asarray(spot, float)[:2], min(BIN_KEEPOUT, float(np.linalg.norm(staging - np.asarray(spot, float)[:2])) - .03))]
        status = yield from route(ctx, staging, yaw, keepout=around, tolerance=.04, est=est)
        if status != 'arrived':
            continue
        yield from look(ctx, est)
        target = est.bin() if est.bin() is not None else target
        c, s = np.cos(yaw), np.sin(yaw); goal = np.asarray(target, float)[:2] - np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])
        status = yield from drive(ctx, goal, yaw, speed=.25, direct=True, tolerance=.025, min_speed=.04, face_travel=False, max_steps=400)
        if status != 'arrived' and float(np.linalg.norm(goal - ctx.est.base()[:2])) > .06:
            yield from drive(ctx, staging, yaw, speed=.3, direct=True, tolerance=.05, face_travel=False, max_steps=400)
            continue
        for arm, how in arms:
            ctx.arm = arm
            yield from hold(ctx, 3)
            point = est.bin_point()
            if point is None:
                continue
            # The trash can is not in the map: a box for the validator, so that no arm passes through it.
            world = pose2(*ctx.est.base()); spot = world[:2, :2] @ np.asarray(point, float)[:2] + world[:2, 3]
            ctx.guard = np.array([[spot[0] - BIN_WIDTH / 2, spot[1] - BIN_WIDTH / 2, 0., spot[0] + BIN_WIDTH / 2, spot[1] + BIN_WIDTH / 2, BIN_TOP]])
            heading = ctx.est.base()[2]
            # The two shoes side by side: each hand sets its shoe down 0.13 m to its own side of the place.
            place = np.asarray(point, float)[:2] + [0., SIDE_BY_SIDE if arm == 'left' else -SIDE_BY_SIDE]
            azimuth = yield from drop_into_bin(ctx, place, lying_directions(arm, heading, used), across=how == 'lying')
            # The arm folds with the trash can still in the validator: on one public start it folded through
            # the rim and tipped the trash can over, with two cans in it.
            status = yield from tuck(ctx, arms=(arm,))
            ctx.guard = np.zeros((0, 6))
            if status != 'done':
                # No way to fold next to it: move back first, then fold.
                yield from drive(ctx, staging, yaw, speed=.2, direct=True, tolerance=.05, face_travel=False, max_steps=400)
                yield from tuck(ctx, arms=(arm,))
            if azimuth is not None:
                gone.append((arm, how)); used.append(azimuth + heading)
            else:
                failed.append((x, y, yaw, arm))
        # Back the way it came, to where the base may turn.
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


PIECES = 2          # the two shoes
SEARCHES = 6        # rounds of the lookouts after nothing is left that is known (the pieces are 4.5 cm high: seen from 2.5 m at most)


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
    left_alone = lambda k: sum(float(np.linalg.norm(np.asarray(est.soda(k), float)[:2] - spot)) < .15 for spot in misses) >= 4
    wanted = lambda: [k for k in known(est) if not left_alone(k)]
    for trip in range(12):                      # further rounds take what an earlier one dropped or missed
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
                    rest = [k for k in order if k in wanted() and k not in missed] + [k for k in wanted() if k not in order and k not in missed]
                    if not rest:
                        break
                    spot = np.asarray(est.soda(rest[0]), float)[:2].copy()
                    ok = yield from fetch(ctx, est, rest[0], arm)
                    ctx.event('fetched', soda=rest[0], arm=arm, ok=bool(ok), how=ok or None)
                    if ok:
                        held.append((arm, ok)); break
                    missed.add(rest[0]); misses.append(spot)
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
