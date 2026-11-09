"""The toolbox, the pruner and the shears: their geometry and the skills that take them.

Points and rotations are in the base frame B (x ahead, y left, z up from the base_link origin, which rides BASE_Z
above the floor) unless a name says "map" or "own" (a thing's own frame). Heights called "above the floor" are
map z.

The toolbox's own frame: x across it (its lid's hinge on the -x long side, its latches on the +x one: the front),
y along it, origin in the middle of its footprint. From the probes (dev/jobs/probe_static.py, probe_kit.py):
  - 0.243 across (what the camera shows of its long faces: at own x +0.120 and -0.123), 0.62 along, the lid's top
    plate 0.283 above the floor;
  - on the lid a handle: a bar along own y, own x -0.005..0.017, y -0.067..0.073, 0.323-0.353 above the floor, a gap
    of 4 cm between it and the lid;
  - the lid turns about a line along own y at own x -0.123, 0.207 above the floor, from 0 (shut) to 2.967 rad; it
    counts as open from 0.148 rad; let go below about 0.9 rad it falls shut, from about 1.5 rad on it falls open.
"""
import numpy as np
from manip import hold, follow, move_to, BASE_Z
from skills import interpolate_cartesian
from car_skills import frame, held, CLOSE_STEPS

HINGE = np.array([-.123, .207])         # own x and height above the floor of the lid's hinge line
BAR = np.array([.006, .338])            # ... of the middle of the handle's bar (lid shut)
BAR_HALF = .065                         # half the bar's length that a hand can take (own y)
BAR_GRIP = -.022                        # the fingers' ends go this far below the bar's middle: 0.7 cm under its lower face, 3 cm above the lid
# The hand's middle line goes to the hinge side of the bar's middle by AIM: the finger on the front side then brings the
# bar up against the other one, which has closed all the way (the bar's middle is then 1.2 cm from the hand's), and
# the lid follows the hand (142 leads of 142 in trials and runs). With the bar in the middle between the fingers both
# stop on it (0.7-1.1 cm from closed), and then the lid stayed shut and the whole box came up with the hand, the
# hand stalled on the way or the lid was left open (6 times of 8; the hand was 0.1 to 1.2 cm to the front side of
# there). FIRM: the closed-all-the-way finger further open than this means such a grip; the hand lets go and takes
# the bar again AIM_AGAIN further to the hinge side. (Over a sweep of the hand's place across the bar, with the arc
# worked out from the grip: from 1.2 cm to the front side to 2.0 cm to the hinge side 35 leads of 36 went to the
# end and no lid was left open; at 3.2 cm to the hinge side the closing finger drags the box 5 cm, and the lid was
# left open 3 times of 8.)
AIM = -.012
AIM_AGAIN = -.010
FIRM = .004
FACES = (.120, -.123)                   # own x of the long faces as the camera shows them: front, hinge side
HALF_LENGTH = .31
LID_TOP = .283
HELD_AT = 1.35                          # lid angle at which the hand holds the lid while a tool goes in (rad)
LET_GO_AT = 0.                          # ... and at which it lets go on the way back: shut. (Let go at 0.10 rad the lid fell onto a pruner that
                                        # stood high in the box and came to rest at 0.135 rad, and with the shears in as well at 0.163: open.)
LET_GO_EARLY = .20                      # ... while a tool is still to come: the lid comes down the rest of the way by itself. (Shut by the hand
                                        # on shears that stood on edge at an end wall, it was stuck on them: at the next visit it stayed shut,
                                        # and the whole box came up with the hand.)
OPEN_ENOUGH = .95                       # a lid led this far (rad) has room for a tool between its edge and the box's front wall (15 cm)
LID_STEP = .06                          # the hand follows the lid's arc in steps of this (rad)
DROP = np.array([.045, .50])            # own x and height above the floor at which a tool is let go over the open box
DROP_ALONG = (.15, .19)                 # ... and how far along own y from the hand that holds the lid, in order of choice
DROP_WITHIN = .20                       # ... but no further than this from the box's middle (its end walls are at 0.29)
OVER_FIRST = .12                        # the hand with the tool comes over the box this much higher first, then straight down
BOX_ROUND = (.04, .03)                  # radius of the spheres that stand for the open box in the way of the arm with the tool, in order of choice


def turn_z(angle):
    c, s = np.cos(angle), np.sin(angle); return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def box_in_base(est_base, place):
    """4x4 of the toolbox's own frame in the base frame. est_base: (x, y, yaw) of the base in M; place: (x, y, yaw)
    of the box in M (yaw: of its own x axis, which points out of its front). The frame's origin is on the floor."""
    x, y, yaw = est_base; c, s = np.cos(yaw), np.sin(yaw); out = np.eye(4)
    out[:3, :3] = turn_z(place[2] - yaw); d = np.array([place[0] - x, place[1] - y])
    out[:3, 3] = [c * d[0] + s * d[1], -s * d[0] + c * d[1], -BASE_Z]
    return out


def bar_point(angle, along=0., aim=AIM):
    """Own-frame point (x, y, height above the floor) where the fingers' ends hold the handle, lid at @angle. aim:
    how far the hand's middle is from the bar's, across the bar in the lid's plane (to the hinge side: negative)."""
    c, s = np.cos(angle), np.sin(angle); r = BAR + [aim, BAR_GRIP] - HINGE         # the lid opens by turning its front up and over to -x
    return np.array([HINGE[0] + c * r[0] - s * r[1], along, HINGE[1] + s * r[0] + c * r[1]])


def bar_rotation(angle, flip=False):
    """Own-frame tool rotation of the hand on the handle, lid at @angle: the fingers point down into the lid's
    plane and close across the bar (along own x when the lid is shut); @flip: the hand turned half a turn about
    its fingers (either hand can come either way round)."""
    c, s = np.cos(angle), np.sin(angle); about = np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]])       # the lid's turn: +x towards +z
    approach = about @ np.array([0., 0., -1.]); closing = about @ np.array([-1. if flip else 1., 0., 0.])
    return frame(approach, closing)


def plan_lid(ctx, box, start=0., end=HELD_AT, along=0., flip=False, base=None, q=None, locked=None, sweep=True, aim=AIM):
    """The active hand on the handle with the lid at @start, and the way that leads the lid to @end: dict(grip (the
    posture at @start), pre (10 cm above it along the fingers), way (postures along the arc)), or None.
    box: the toolbox's frame in the base frame (box_in_base) for the base pose the plan is for.
    sweep: the lid as it turns is in the collision model at every step of the way, for all of the robot but the
    gripper that holds its handle. (From an end of the box a left arm that came from the hinge side lay where the lid
    swings: the lid stayed shut and the box came up with the hand.)"""
    full = ctx.q if q is None else np.asarray(q, float); things = dict(ctx.obstacles)
    turn, origin = box[:3, :3], box[:3, 3]; world = ctx.base_matrix() if base is None else base
    gripper = np.array([name.startswith(f'{ctx.arm}_gripper') for name in ctx.collision.names])

    def checker(angle):
        if sweep:
            points, radii = box_spheres(box, angle); ctx.obstacles = dict(things, toolbox=(points @ world[:3, :3].T + world[:3, 3], radii, gripper))
        made = ctx.validator(base=base); ctx.obstacles = things
        return made
    check = checker(start)

    def pose(angle):
        return origin + turn @ bar_point(angle, along, aim), turn @ bar_rotation(angle, flip)
    point, rotation = pose(start)
    for grip in ctx.ik.solve(full, point, rotation, count=24, validator=check, all_solutions=True, locked=locked)[:6]:
        at = full.copy(); at[ctx.ik.columns] = grip
        pre = ctx.ik.solve(at, point + .10 * rotation[:, 2], rotation, count=2, validator=check, locked=locked, stay=.05)
        if pre is None:
            continue
        way = []; ok = True
        for angle in np.linspace(start, end, max(2, int(np.ceil(abs(end - start) / LID_STEP)) + 1))[1:]:
            p, r = pose(angle); check = checker(angle)
            step = ctx.ik.solve(at, p, r, count=1, validator=check, locked=locked, stay=.05)
            if step is None or np.max(np.abs(step - at[ctx.ik.columns])) > .5:
                step = ctx.ik.solve(at, p, r, count=4, validator=check, locked=locked)
            if step is None or np.max(np.abs(step - at[ctx.ik.columns])) > .5:
                ok = False; break
            at[ctx.ik.columns] = step; way.append(step)
        if ok:
            return dict(grip=grip, pre=pre, way=way, point=point, rotation=rotation, flip=flip, along=along, end=end, aim=aim)
    return None


def take_handle(ctx, plan, label='handle'):
    """Close the active hand on the handle as @plan (plan_lid) has it. Returns True if the fingers stayed apart on
    it the way that opens the lid (see AIM); if both stopped on the bar, the hand lets go again, plan['firm'] is set
    and False is returned."""
    ctx.set_grip(1.)
    status = yield from move_to(ctx, plan['pre'], rate=.04, label=label)
    if status != 'done':
        ctx.event(label, ok=False, reason='no way to the handle', arm=ctx.arm, status=status); return None
    yield from hold(ctx, 4)
    status = yield from follow(ctx, [plan['grip']], rate=.015, max_steps=300)
    reached = ctx.tip()[:3, 3].copy()
    ctx.set_grip(-1.)
    yield from hold(ctx, CLOSE_STEPS)
    ok = held(ctx); plan['firm'] = bool(ok and float(ctx.fingers()[1]) > FIRM and not plan.get('flip'))
    ctx.event(label, ok=ok and not plan['firm'], firm=plan['firm'], aim=plan.get('aim'), arm=ctx.arm, descend=status, fingers=ctx.fingers().round(4).tolist(), short=(reached - plan['point']).round(3).tolist(), point=np.round(plan['point'], 3).tolist())
    ok = ok and not plan['firm']
    if not ok:
        ctx.set_grip(1.)
        yield from hold(ctx, 6)
        yield from follow(ctx, [plan['pre']], rate=.02, max_steps=200)
    return ok


def arc_from_grip(ctx, box, end, base=None):
    """The way that leads the lid from 0 to @end for the hand that has just closed on the handle, worked out from
    where the hand is and how its fingers stopped, not from where the box was seen: postures along the arc, or None.
    The bar's middle is between the fingers, the hinge line HINGE - BAR from it across the box, at its known height;
    the hand's pose is turned about that line. (Led along the arc the plan had from the look, the hand stalled before
    the end on 2 leads of 12 in a sweep of its place across the bar; let go there, the lid fell open.)
    box: the toolbox's frame in the base frame as the look had it (its heading is what is used)."""
    tip = ctx.tip(); fingers = ctx.fingers(); turn = box[:3, :3]; things = dict(ctx.obstacles); world = ctx.base_matrix() if base is None else base
    bar = tip[:3, 3] + .5 * float(fingers[0] - fingers[1]) * tip[:3, 1]
    hinge = bar + turn @ np.array([HINGE[0] - BAR[0], 0., 0.]); hinge[2] = box[2, 3] + HINGE[1]
    # the box as the hand has it: its frame moved so that the bar is where the fingers hold it
    moved = box.copy(); moved[:3, 3] += turn @ np.array([float((bar - (box[:3, 3] + turn @ np.array([BAR[0], 0., 0.]))) @ turn[:, 0]), 0., 0.])
    gripper = np.array([name.startswith(f'{ctx.arm}_gripper') for name in ctx.collision.names]); at = ctx.q.copy(); way = []
    for angle in np.linspace(0., end, max(2, int(np.ceil(end / LID_STEP)) + 1))[1:]:
        c, s = np.cos(angle), np.sin(angle); about = turn @ np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]]) @ turn.T
        points, radii = box_spheres(moved, angle); ctx.obstacles = dict(things, toolbox=(points @ world[:3, :3].T + world[:3, 3], radii, gripper))
        check = ctx.validator(base=base); ctx.obstacles = things
        p = hinge + about @ (tip[:3, 3] - hinge); r = about @ tip[:3, :3]
        step = ctx.ik.solve(at, p, r, count=1, validator=check, stay=.05)
        if step is None or np.max(np.abs(step - at[ctx.ik.columns])) > .5:
            step = ctx.ik.solve(at, p, r, count=4, validator=check)
        if step is None or np.max(np.abs(step - at[ctx.ik.columns])) > .5:
            return None
        at[ctx.ik.columns] = step; way.append(step)
    return way


def lead_lid(ctx, way, label='lid', rate=.012):
    """Lead the lid along @way (postures along its arc) with the hand that holds the handle. Returns the status."""
    status = yield from follow(ctx, way, rate=rate, max_steps=900)
    yield from hold(ctx, 6)
    ctx.event(label, ok=status == 'done', arm=ctx.arm, status=status, fingers=ctx.fingers().round(4).tolist())
    return status


def release_handle(ctx, lift=.08):
    """Open the hand on the handle and draw it back along its fingers."""
    ctx.set_grip(1.)
    yield from hold(ctx, 20)
    tip = ctx.tip(); goal = ctx.ik.solve(ctx.q, tip[:3, 3] + lift * tip[:3, 2], tip[:3, :3], count=2, stay=.05)
    if goal is not None:
        yield from follow(ctx, [goal], rate=.02, max_steps=200)


# ---- the tools -----------------------------------------------------------------------------------------------
# The pruner stands on the ends of its two handles, blades up: 0.226 high; the handles side by side, together 3 cm
# by 6.7 cm, up to 0.134 above the floor, the blades above them. A hand from straight above takes the pair of
# handles under the blades, whichever way it is turned (at most 6.9 cm across for fingers that open 10 cm).
PRUNER_GRIP = .078                      # height above the floor of the fingers' ends on the pruner's handles
PRUNER_MIDDLE = np.array([0., -.0125])  # of the pair of handles, in the pruner's own x, y
# The shears lie flat: 13 cm long, 5.1 cm across at the handles (own x 0..0.049, y -0.029..0.022), 2.5 cm high.
SHEARS_GRIP = .008                      # height above the floor of the fingers' ends on the shears
SHEARS_HANDLES = np.array([.024, -.0035])       # the middle of the handles, in the shears' own x, y
SHEARS_BLADES = np.array([-.040, -.007])        # ... of the blades (2 cm across)


def tool_point(place, own_xy, height):
    """Map-frame point above own-frame @own_xy of a tool at @place = (x, y, yaw) in M, @height above the floor."""
    c, s = np.cos(place[2]), np.sin(place[2])
    return np.array([place[0] + c * own_xy[0] - s * own_xy[1], place[1] + s * own_xy[0] + c * own_xy[1], height])


def in_base(est_base, point):
    """A map-frame point (x, y, height above the floor) in the base frame."""
    x, y, yaw = est_base; c, s = np.cos(yaw), np.sin(yaw); dx, dy = point[0] - x, point[1] - y
    return np.array([c * dx + s * dy, -s * dx + c * dy, point[2] - BASE_Z])


def plan_slant_pinch(ctx, point, pitch, heading=0., base=None, q=None, way_in=.10, locked=None):
    """A pinch with the fingers pointing along @heading (rad, base frame) and down by @pitch, closing level and
    across: dict(grip, pre, inward (postures along the way in), point, rotation) or None. The hand comes in along
    its fingers from @way_in. For a thing that stands taller than the fingers are long (the pruner): a hand from
    straight above meets its top with the palm before the fingers are down at its handles."""
    approach = np.array([np.cos(heading) * np.cos(pitch), np.sin(heading) * np.cos(pitch), -np.sin(pitch)]); closing = np.array([-np.sin(heading), np.cos(heading), 0.])
    rotation = frame(approach, closing); full = ctx.q if q is None else np.asarray(q, float); check = ctx.validator(base=base); point = np.asarray(point, float)
    for grip in ctx.ik.solve(full, point, rotation, count=24, validator=check, all_solutions=True, locked=locked)[:6]:
        at = full.copy(); at[ctx.ik.columns] = grip
        out = interpolate_cartesian(ctx.ik, at, point, point - way_in * approach, rotation, validator=check, step=.025, locked=locked)
        if out is not None:
            return dict(grip=grip, pre=out[-1], inward=out[::-1][1:] + [grip], point=point, rotation=rotation)
    return None


def slant_pinch(ctx, plan, label='pinch', gap=.012, lift=.10):
    """Close the active hand as @plan (plan_slant_pinch) has it and lift straight up by @lift. Returns True if the
    fingers stayed further apart than @gap (sum of both, m), False if they closed on nothing, None if the hand
    did not get there."""
    ctx.set_grip(1.)
    status = yield from move_to(ctx, plan['pre'], rate=.04, label=label)
    if status != 'done':
        ctx.event(label, ok=False, reason='no way there', arm=ctx.arm, status=status); return None
    yield from hold(ctx, 4)
    status = yield from follow(ctx, plan['inward'], rate=.015, max_steps=300)
    reached = ctx.tip()[:3, 3].copy()
    ctx.set_grip(-1.)
    yield from hold(ctx, CLOSE_STEPS)
    fingers = ctx.fingers().copy()
    up = interpolate_cartesian(ctx.ik, ctx.q, reached, reached + [0., 0., lift], plan['rotation'], step=.025)
    if up is not None:
        yield from follow(ctx, up, rate=.015, max_steps=300)
    yield from hold(ctx, 4)
    ok = float(np.sum(ctx.fingers())) > gap
    ctx.event(label, ok=ok, arm=ctx.arm, way_in=status, lifted=up is not None, fingers=fingers.round(4).tolist(), after_lift=ctx.fingers().round(4).tolist(), short=(reached - plan['point']).round(3).tolist(), point=np.round(plan['point'], 3).tolist())
    if not ok:
        ctx.set_grip(1.)
    return ok


# ---- a tool into the box ---------------------------------------------------------------------------------------
def pitched(pitch, heading=0.):
    """Tool rotation: fingers pointing along @heading (rad, base frame) and down by @pitch, closing level."""
    approach = np.array([np.cos(heading) * np.cos(pitch), np.sin(heading) * np.cos(pitch), -np.sin(pitch)])
    return frame(approach, np.array([-np.sin(heading), np.cos(heading), 0.]))


def box_spheres(box, angle, radius=.03):
    """The toolbox with its lid at @angle as spheres, in the frame @box is given in (N x 3, radii): the rim of its
    base and the lid's plate, for an arm that works over the open box."""
    rim = [(x, y, .185) for x in (-.115, .115) for y in np.arange(-.30, .301, .06)] + [(x, y, .185) for y in (-.30, .30) for x in np.arange(-.06, .061, .06)]
    c, s = np.cos(angle), np.sin(angle); plate = []
    for r in np.arange(.03, .245, .045):                 # out from the hinge along the lid, 7.5 cm above the hinge line when shut
        for y in np.arange(-.28, .281, .07):
            plate.append((HINGE[0] + c * r - s * .06, y, HINGE[1] + s * r + c * .06))
    points = np.array(rim + plate); return box[:3, 3] + points @ box[:3, :3].T, np.full(len(points), radius)


def plan_drop(ctx, box, full, locked, along, base=None, quick=False, across=None):
    """A posture of the active arm (the one that holds a tool) with its fingers' ends over the open half of the box,
    the trunk held as @locked has it: (the 11-vector, what was chosen) or None. full: the whole robot's posture to
    start from (the other hand on the lid). base: the base pose the plan is for (4x4 in M), if not the present one.
    quick: only the attitudes of first choice (where there is a posture at all it is nearly always one of them).
    across: the box's own x at which the tool is let go, in order of choice (None: DROP's, for a lid held at
    HELD_AT).

    Where along the box, and how the hand is turned: next to the hand on the lid, never nearer than 9 cm to an end
    wall, and so that what stands out of the hand points to the box's middle. The pruner stands out of the hand that
    took it by 11 cm, 45 degrees off the fingers and square to the way they close. Let go 0.24 m from the lid's hand
    it came down on the end wall and lay outside (twice); let go across the box towards the front, it went over
    the front wall (twice)."""
    check = ctx.validator(base=base); length = box[:2, 1] / max(float(np.linalg.norm(box[:2, 1])), 1e-9)       # the box's own y, in the base frame
    lengthwise = abs(length[0]) > abs(length[1])
    if lengthwise:                              # the base stands at an end and looks along the box: the near side of the lid's hand first (the shorter reach)
        sides = (-1., 1.) if length[0] > 0 else (1., -1.)
    else:                                       # at the hinge side or the front: the side of the arm that holds the tool first
        sides = (1., -1.) if (1. if ctx.arm == 'left' else -1.) * length[1] > 0 else (-1., 1.)
    # A tool that is in the box already lies where it was let go: the next one goes into the other half. (Let go onto
    # the shears, the pruner threw them out of the box: they lay 0.4 m beyond its end.)
    lies = getattr(ctx, 'in_box', [])
    if lies:
        sides = (-1., 1.) if lies[-1] > 0 else (1., -1.)
        if quick:
            sides = sides[:1]                   # (the same half only where no stance has the other one: let go onto the pruner, the shears flew out too)

    def attitudes(side):
        """The hand's attitudes for a drop on @side of the lid's hand: first those that have what stands out of the
        hand along the box and towards its middle (fingers down; the hand slanted); then, second choice, across
        the box towards the lid (which stops it), the hand slanted ahead of a base that looks along the box, and
        along the box away from the middle. Never across the box towards the front wall: so let go, the pruner
        came down on that wall and went over it (twice)."""
        toward = -side * length; closing = float(np.arctan2(toward[0], -toward[1])); heading = float(np.arctan2(toward[1], toward[0]))
        first = [down_rotation(closing), pitched(np.radians(60.), heading), pitched(np.radians(35.), heading)]
        to_lid = next(c for c in (closing + np.pi / 2, closing - np.pi / 2) if np.array([np.sin(c), -np.cos(c)]) @ box[:2, 0] < 0)
        second = [down_rotation(to_lid)] + ([pitched(np.radians(60.)), pitched(np.radians(35.))] if lengthwise else []) + [down_rotation(closing + np.pi)]
        return first, second

    direct = None
    for choice in (0,) if quick else (0, 1):
        for side in sides:
            rotations = attitudes(side)[choice]
            for step in DROP_ALONG:
                if abs(along + side * step) > DROP_WITHIN:
                    continue
                for dx in (DROP[0], 0.) if across is None else across:
                    for height in (.47, .54):
                        point = box[:3, 3] + box[:3, :3] @ np.array([dx, along + side * step, height])
                        for index, rotation in enumerate(rotations):
                            goal = ctx.ik.solve(full, point, rotation, count=6, validator=check, locked=locked)
                            if goal is not None:
                                # ... and the same hand OVER_FIRST higher, from where it comes straight down: it crosses the lid's
                                # upper edge on its way, and came straight from where it carried the tool it met the edge with its
                                # wrist camera and stopped there, on about one visit in seven (the arm hangs 1-2 cm lower than
                                # planned when it reaches far).
                                # A place where the hand has no posture that much higher is taken only if no other has one: from
                                # there it would go straight over the lid (2 of 4 such drops threw the lid open or shoved the box,
                                # and the tool lay on the floor).
                                at = np.asarray(full, float).copy(); at[ctx.ik.columns] = goal
                                above = ctx.ik.solve(at, point + [0., 0., OVER_FIRST], rotation, count=2, validator=check, locked=locked, stay=.05)
                                found = goal, dict(along=round(float(along + side * step), 3), across=dx, height=height, attitude=f'{choice}.{index}', over_first=above is not None), above
                                if above is not None:
                                    return found
                                direct = direct or found
    return direct


def down_rotation(closing):
    return frame(np.array([0., 0., -1.]), np.array([np.cos(closing), np.sin(closing), 0.]))


def plan_visit(ctx, box, lid_arm, base=None, held_at=HELD_AT, quick=False, across=None, aim=AIM):
    """One hand opens the lid and holds it, the other lets a tool go over the open box: dict(lid (plan_lid), drop
    (posture of the other arm), lid_arm, tool_arm) for the first way the arm planner has, or None.
    box: the toolbox's frame in the base frame for the base pose the plan is for (@base: that pose, 4x4 in M, if it
    is not the present one). quick: the hand on the middle of the handle only and the tool let go next to it: a
    stance that has a plan has it there in a second or two, and one that has none says so in four instead of
    thirty."""
    tool_arm = 'right' if lid_arm == 'left' else 'left'; before = ctx.arm; found = None; things = dict(ctx.obstacles)
    world = ctx.base_matrix() if base is None else base
    # The hand one way round only (bar_rotation's flip False). Turned half a turn about its fingers it took the handle
    # just the same, but the lid then stayed shut and the whole box came up with the hand: 17 times of 18 over every
    # side and either hand, against 18 of 18 opened the other way round (dev/jobs/kit_trial.py).
    for flip in (False,):
        for along in (0.,) if quick else (0., .04, -.04, .06, -.06):
            ctx.obstacles = things; ctx.arm = lid_arm; plan = plan_lid(ctx, box, 0., held_at, along=along, flip=flip, base=base, aim=aim)
            if plan is None:
                continue
            full = ctx.q.copy(); full[ctx.ik.columns] = plan['way'][-1]; locked = {k: float(plan['way'][-1][k]) for k in range(4)}
            # The open box is in the way of the arm that brings the tool; the hand on the handle may touch it. Round
            # generously first; as it is, where that leaves no posture (a box in a corner, the tool already in the hand).
            for radius in BOX_ROUND:
                points, radii = box_spheres(box, held_at, radius); in_map = points @ world[:3, :3].T + world[:3, 3]
                ctx.obstacles = dict(things, toolbox=(in_map, radii, lid_arm)); ctx.arm = tool_arm; drop = plan_drop(ctx, box, full, locked, along, base=base, quick=quick, across=across)
                if drop is not None:
                    found = dict(lid=plan, drop=drop[0], let_go=dict(drop[1], round=radius), above=drop[2], lid_arm=lid_arm, tool_arm=tool_arm, box=(points, radii), frame=box, across=across); break
            if found:
                break
        if found:
            break
    ctx.arm = before; ctx.obstacles = things
    return found


def put_in(ctx, visit, label='put_in', gap=None, shut=True):
    """With a tool in visit['tool_arm']'s hand: the other hand takes the handle and leads the lid up, the tool is let
    go over the open box, the lid is led back and let go. Returns 'in' when all of that was done, else what stopped
    it ('handle': the handle was not taken; 'grip': it was taken the way that does not open the lid, and let go
    again; 'lid': the lid did not follow; 'drop': no way for the tool's hand; 'let_go': the tool's hand had shut
    by the time it was over the box (gap: fingers further apart than this, sum of both, still hold the tool; None:
    not asked) and was opened there all the same, so the tool is in the box if it still hung by the grasp, and
    gone before if not): the lid is led back all the same. shut: the hand shuts the lid on the tool (the last
    one); else it lets the lid go LET_GO_EARLY from shut, as it does whenever the tool did not go in."""
    lid, lid_arm, tool_arm = visit['lid'], visit['lid_arm'], visit['tool_arm']
    ctx.arm = lid_arm
    ok = yield from take_handle(ctx, lid)
    if not ok:
        return 'grip' if lid.get('firm') else 'handle'
    before = float(np.sum(ctx.fingers())); planned = lid['way'][-1]
    way = arc_from_grip(ctx, visit['frame'], lid['end'])
    ctx.event('arc', from_grip=way is not None, fingers=ctx.fingers().round(4).tolist(), off=round(float(.5 * (ctx.fingers()[0] - ctx.fingers()[1]) + lid.get('aim', 0.)), 4))
    if way is not None:
        lid = dict(lid, grip=ctx.chain.copy(), way=way)
    status = yield from lead_lid(ctx, lid['way'])
    result = 'in'
    # A lid that follows the hand leaves the fingers as they closed on the handle (2.4 cm apart, to a tenth of a
    # millimetre); where it stayed shut and the box came up instead, the bar had turned between them (3.4 cm).
    # A hand that stalled on the way (about one lead in nine, at 1.1 to 1.3 rad) holds the lid where it is: far
    # enough open from OPEN_ENOUGH on.
    arc = [lid['grip']] + list(lid['way']); led = lid['end'] * int(np.argmin([float(np.max(np.abs(np.asarray(step) - ctx.chain))) for step in arc])) / len(lid['way'])
    if abs(float(np.sum(ctx.fingers())) - before) > .004 or (status != 'done' and led < OPEN_ENOUGH):
        ctx.event(label, ok=False, reason='the lid did not follow', fingers_before=round(before, 4), fingers=ctx.fingers().round(4).tolist(), status=status, led=round(float(led), 2))
        result = 'lid'
    else:
        ctx.arm = tool_arm; back = ctx.chain.copy(); things = dict(ctx.obstacles); world = ctx.base_matrix(); drop = visit['drop']; above = visit.get('above')
        ctx.obstacles = dict(things, toolbox=(visit['box'][0] @ world[:3, :3].T + world[:3, 3], visit['box'][1], lid_arm))
        if float(np.max(np.abs(ctx.chain[:4] - planned[:4]))) > .02:
            # The trunk is not where the plan had it for the lid held open: the tool's hand is planned for where it is.
            again = plan_drop(ctx, visit['frame'], ctx.q.copy(), {k: float(ctx.chain[k]) for k in range(4)}, lid['along'], across=visit.get('across'))
            if again is not None:
                drop = again[0]; above = again[2]; visit = dict(visit, let_go=again[1])
        if above is not None:
            status = yield from move_to(ctx, np.r_[ctx.chain[:4], above[4:]], rate=.03, label=label, still_trunk=True)
            if status == 'done':
                status = yield from follow(ctx, [np.r_[ctx.chain[:4], drop[4:]]], rate=.02, max_steps=200)
        else:
            status = yield from move_to(ctx, np.r_[ctx.chain[:4], drop[4:]], rate=.03, label=label, still_trunk=True)
        # Where the fingers' ends are over the box (its own x, y, height): a hand that was stopped on its way lets go
        # all the same if it is where it was to let go, to 3.5 cm across the box (stopped 5 cm nearer the front wall
        # and 5 cm nearer an end, it let the pruner go onto the floor).
        frame_ = visit['frame']; over = frame_[:3, :3].T @ (ctx.tip()[:3, 3] - frame_[:3, 3]); planned_x = (visit.get('let_go') or {}).get('across', DROP[0])
        there = bool(abs(over[0] - planned_x) < .035 and abs(over[1]) < DROP_WITHIN and over[2] > .35)
        if status != 'done':
            ctx.event(label, ok=there, reason='stopped on the way over the box', over=np.round(over, 3).tolist(), status=status)
        # A hand whose fingers have shut is opened over the box all the same: a tool that has turned out from between
        # them hangs on by the grasp. (A pruner taken from the end its blades lean to did, after it had touched the
        # chassis while the trunk bent for the lid: it hung over the open box, the hand was taken back shut, and
        # opened at the staging point it let the pruner fall flat on the floor.)
        shut_hand = gap is not None and float(np.sum(ctx.fingers())) < gap
        if shut_hand:
            ctx.event(label, ok=False, reason='the tool is not between the fingers', fingers=ctx.fingers().round(4).tolist(), over=np.round(over, 3).tolist(), there=bool(status == 'done' or there))
        if status == 'done' or there:
            yield from hold(ctx, 6)
            ctx.set_grip(1.); ctx.detach(tool_arm); ctx.let_go_at = float(over[1])
            yield from hold(ctx, 30)
            if shut_hand:
                result = 'let_go'
        else:
            result = 'drop'
        yield from move_to(ctx, np.r_[ctx.chain[:4], back[4:]], rate=.04, label=label, still_trunk=True)
        ctx.obstacles = things; ctx.arm = lid_arm
    # Back along the arc from where the hand is on it (where the way up stalled, the way back from its end led the lid
    # further open, and let go there it fell open and stayed so), down to the angle at which the lid is let go: the
    # hand shuts the lid, on what is in the box if that stands high.
    arc = [lid['grip']] + list(lid['way']); reached = int(np.argmin([float(np.max(np.abs(np.asarray(step) - ctx.chain))) for step in arc]))
    let_go_at = LET_GO_AT if shut and result in ('in', 'let_go') else LET_GO_EARLY; first = int(np.ceil(let_go_at / lid['end'] * len(lid['way'])))
    way_back = arc[first:reached][::-1]
    if way_back:
        status = yield from lead_lid(ctx, way_back, label='lid_back')
        if status != 'done':
            # The way back stalls on about one visit in nine (18 of 161), a step from the top; why is not known (no
            # contact of the arm shows there). Let go up there, the lid stays where it is or goes whichever way the
            # hand's withdrawal sends it: 16 times it fell shut, once open, and the box was left so. The hand is opened
            # instead and goes on along the arc: the finger behind the bar takes the lid along, and the bar has room
            # between the open fingers if the arc is not the one it can go.
            reached = int(np.argmin([float(np.max(np.abs(np.asarray(step) - ctx.chain))) for step in arc])); rest = arc[first:reached][::-1]
            off = (np.asarray(rest[0]) - ctx.chain) if rest else np.zeros(1)
            ctx.set_grip(1.)
            yield from hold(ctx, 12)
            status = (yield from follow(ctx, rest, rate=.012, max_steps=600)) if rest else 'done'
            ctx.event('lid_pushed', ok=status == 'done', arm=ctx.arm, status=status, left=len(rest), stuck=np.round(off, 3).tolist())
    yield from release_handle(ctx)
    ctx.event(label, ok=result == 'in', result=result, lid_arm=lid_arm, tool_arm=tool_arm, let_go=visit.get('let_go'), lid_let_go_at=let_go_at)
    return result
