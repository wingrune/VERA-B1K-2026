"""Skills of the car task: open the trunk lid, take things from a table and from the floor from above, put them down.

All points are in the base frame B (x ahead, y left, z up from the base_link origin, which rides BASE_Z above the
floor) unless a name says "map". The skills are generators that yield one 23-D action per step (see manip.py).

Grasping is the simulator's assisted grasp: when the fingers close with both of them touching a thing that lies
between them, a joint holds it to the hand (a ball joint for a link of an articulated thing such as the trunk
lid) until the gripper is commanded fully open. A command a little short of fully open parts the fingers and
keeps the joint: the trunk lid is led through the second half of its swing that way, with the hand's attitude
free.
"""
from contextlib import contextmanager
import numpy as np
from manip import hold, follow, move_to, ARMS, TALL_TRUNK, TUCK, BASE_Z
from skills import interpolate_cartesian
from observer import LID_APEX, LID_SLOPE, LID_ROUND

CLOSE_STEPS = 45            # steps the fingers stay closed before the hand moves: the grasp joint takes a moment to form
HELD_GAP = .012             # fingers further apart than this (sum of both, m) after closing hold something
PARTED = .8                 # gripper command that parts the fingers without letting go of the grasp joint
UPRIGHT = {k: float(v) for k, v in enumerate(TALL_TRUNK)}        # trunk joints as they are for travelling, for IK with the trunk held

# ---- the car's trunk lid (the car is fixed: map frame) ---------------------------------------------------
HINGE = np.array([-6.164, 1.143])       # x and height of the lid's hinge line, which runs along y
LID_MIDDLE = 6.878                      # y of the lid's middle
EDGE_SIDE = .20                         # the left hand takes the lid's upper rear edge this far left of its middle (m)
EDGE_HEIGHT = 1.107                     # height of the fingers' ends there
EDGE_BEHIND = .003                      # ... and how far behind the edge's rearmost line as the camera shows it
EDGE_SLANT = np.radians(50.)            # the hand comes onto the edge slanting down by this much, the edge between its fingers
LIP = np.array([-6.737, .722])          # the lower end of the lid's rear face (x, height; lid closed), middle of its thickness
LIP_ALONG = np.array([.168, .986])      # direction from there up the rear face (x, z; lid closed)
LIP_IN = .025                           # the fingers' ends go this far up the face from its lower end
LIP_INWARD = .015                       # ... and the hand's middle line this far inside the rear face: the middle of the lid's thickness there (on the
                                        # face itself one finger closed on air, and the grasp joint formed in one try of three)
LIP_SIDE = -.17                         # the right hand takes it this far from the lid's middle (m; negative: right)
HANDOVER = .50                          # lid angle at which the right hand takes over (rad)
OPEN = 2.10                             # ... and to which it leads the lid: beyond 2.0 the lid's weight keeps it open
LID_STEP = .05                          # the hands follow the lid's arc in steps of this (rad)
STANCE = (-7.13, LID_MIDDLE, 0.)        # where the base stands to open the lid: its front 4 cm from the bumper
TRUNK_STANCE = (-7.17, LID_MIDDLE, 0.)  # ... and to put things in
RACKET_STANCE = (-7.32, LID_MIDDLE, 0.) # ... the racket: further back, so that a racket that hangs in front of the chassis stays clear of the bumper
TRUNK_FLOOR = .77                       # height of the trunk's floor at its rear (it falls to 0.735 further in)
BOX_IN_TRUNK = -6.50                    # x at which the box's middle is put down: it counts as inside from -6.60 on, and further in than -6.33 the
                                        # map, a height field, has the parcel shelf's height and no room under it
RACKET_IN_TRUNK = (-6.52, 1.02)         # x and height at which the hand holds the racket, lying across the trunk, over the box (the box's rim is at 0.96 there;
                                        # what counts as inside the trunk ends at 1.07)
OVER_THE_SILL = .97                     # height of the fingers' ends that carries the box clear of the car's rear deck (0.77 m; the box hangs 0.125 below them)

# ---- the things ------------------------------------------------------------------------------------------
TABLE_TOP = .423
HUMP_ABOVE_TABLE = .139                 # fingers' ends on the camera's hump: this far above what the camera stands on
HANDLE_ABOVE_TABLE = .012               # ... on the racket's handle (it is 0.037 thick)
BOX = dict(long=.455, short=.305, rim=.155, wall=.1395, floor=.02)      # the toy box; wall: from its middle line to the middle of a long wall
BOX_GRIP_BELOW_RIM = .03
SQUARE = {3: 0.}                        # chain joints held for taking the box: the trunk not turned about the vertical
RACKET = dict(hold=.37, head=-.328, end=.501)       # along the racket from its frame's origin: where the hand holds the handle, the head's far end, the handle's end
CAMERA_SIZE = (.113, .198, .152)        # the camera: across, along (the way the fingers close on its hump), height


def about_y(angle):
    c, s = np.cos(angle), np.sin(angle); return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def frame(approach, closing):
    """Tool rotation for fingers pointing along @approach and closing along @closing (unit vectors, at right angles)."""
    return np.column_stack([np.cross(closing, -approach), closing, -approach])


def down(closing):
    """Tool rotation: fingers straight down, closing along the direction @closing (rad, base frame)."""
    return frame(np.array([0., 0., -1.]), np.array([np.cos(closing), np.sin(closing), 0.]))


def held(ctx, arm=None):
    return float(np.sum(ctx.fingers(arm))) > HELD_GAP


def tuck(ctx, arms=('left', 'right'), rate=.04):
    """Bring the trunk upright and fold the given arms in front of the chest."""
    previous = ctx.arm; result = 'done'
    for arm in arms:
        ctx.arm = arm
        status = yield from move_to(ctx, np.r_[TALL_TRUNK, TUCK[arm]], rate=rate, label='tuck')
        if status != 'done':
            result = status
    ctx.arm = previous
    return result


# ---- shapes of the things a hand holds, for the collision model ---------------------------------------------
def box_spheres(centre, axis, lift=0.):
    """Spheres for the toy box standing with its middle at @centre (base frame xy) and its long side along @axis
    (rad), its floor @lift above the ground: (points, radii)."""
    c, s = np.cos(axis), np.sin(axis); along = np.array([c, s, 0.]); across = np.array([-s, c, 0.]); r = .022; points = []
    for u in np.linspace(-BOX['long'] / 2 + r, BOX['long'] / 2 - r, 5):
        for v in np.linspace(-BOX['short'] / 2 + r, BOX['short'] / 2 - r, 4):
            points.append(np.r_[centre, 0.] + u * along + v * across + [0., 0., lift + r - BASE_Z])
            if abs(u) > BOX['long'] / 2 - 2 * r or abs(v) > BOX['short'] / 2 - 2 * r:
                points.append(np.r_[centre, 0.] + u * along + v * across + [0., 0., lift + BOX['rim'] - r - BASE_Z])
    return np.array(points), r


def racket_spheres(hold_point, toward_head, height):
    """Spheres for the racket lying flat: @hold_point (base frame xy) is where the hand holds its handle,
    @toward_head (rad) the direction from there to its head, @height its middle above the floor."""
    c, s = np.cos(toward_head), np.sin(toward_head); along = np.array([c, s, 0.]); across = np.array([-s, c, 0.]); r = .028
    origin = np.r_[hold_point, height - BASE_Z] + RACKET['hold'] * along      # the racket frame's origin: its x runs from the head to the handle
    points = [origin - x * along for x in (.49, .43, .37, .31, .25)] + [origin - .17 * along + v * across for v in (-.04, .04)]
    for angle in np.linspace(0., 2 * np.pi, 12, endpoint=False):
        points.append(origin + (.11 + .20 * np.cos(angle)) * along + .15 * np.sin(angle) * across)
    points += [origin + .11 * along, origin + .02 * along, origin + .20 * along]
    return np.array(points), r


def camera_spheres(hump, closing, table=TABLE_TOP):
    """Spheres for the camera standing on a surface @table high, the middle of its hump at @hump (base frame xy),
    the fingers closing along @closing (rad)."""
    c, s = np.cos(closing), np.sin(closing); along = np.array([c, s, 0.]); across = np.array([-s, c, 0.]); r = .04; points = []
    for u in (-.06, .06):
        for v in (-.02, .02):
            for z in (.04, .11):
                points.append(np.r_[hump, table - BASE_Z] + u * along + v * across + [0., 0., z])
    return np.array(points), r


# ---- a pinch from straight above ------------------------------------------------------------------------------
def plan_top_pinch(ctx, point, closing, base=None, q=None, approach=.10, relax=False, seeds=1, both_ways=True, locked=None):
    """A pinch from straight above with the active arm: the fingers' ends at @point (base frame xyz), closing along
    @closing (rad; either way round unless @both_ways is False). Returns dict(low, up, pre, rotation, load, point)
    or None. The hand comes straight down @approach and leaves the same way.
    base: the base pose (4x4 in M) the plan is for, if not the present one (a stance that is being considered).
    relax: the present posture is taken as it is where it is already nearer to something than the margins (a stance
    with the chassis between furniture). locked: {index in the chain: value} of joints that are held there."""
    full = ctx.q if q is None else np.asarray(q, float); point = np.asarray(point, float); found = []
    check = ctx.validator(base=base, relax_from=full if relax else None)
    for flip in ((0., np.pi) if both_ways else (0.,)):
        rotation = down(closing + flip)
        for seed in range(seeds):
            for low in ctx.ik.solve(full, point, rotation, count=24, validator=check, all_solutions=True, seed=seed, locked=locked):
                config = full.copy(); config[ctx.ik.columns] = low
                found.append((ctx.load(config)[0], rotation, low))
    found.sort(key=lambda item: item[0])
    # The least loaded postures first, one of each kind (many of the solutions are the same posture): the
    # straight way up does not exist from every one of them.
    distinct = []
    for item in found:
        if all(float(np.max(np.abs(item[2] - other[2]))) > .25 for other in distinct):
            distinct.append(item)
    for load, rotation, low in distinct[:10]:
        start = full.copy(); start[ctx.ik.columns] = low
        up = interpolate_cartesian(ctx.ik, start, point, point + [0., 0., approach], rotation, validator=check, step=.025, locked=locked)
        if up is not None:
            return dict(low=low, up=up, pre=up[-1], rotation=rotation, load=float(load), solutions=len(found), point=point)
    return None


def top_pinch(ctx, point, closing, relax=False, label='pinch', approach=.10, look=None, both_ways=True, locked=None):
    """Pinch what is at @point from above with the active arm and lift it @approach. Returns True if the fingers
    stayed apart after closing, False if they closed on nothing, None if the hand did not go down.
    look(tip, closing) -> base-frame xy or None: where the thing is as the wrist camera shows it once the hand
    hangs above it; the hand is put right by that before it goes down (what the head camera saw from the way in
    is 1-3 cm off after a long way in, and the fingers have 1 cm to spare on the camera's hump).
    both_ways, locked: as for plan_top_pinch."""
    point = np.asarray(point, float)
    plan = plan_top_pinch(ctx, point, closing, relax=relax, approach=approach, both_ways=both_ways, locked=locked)
    if plan is None:
        ctx.event(label, ok=False, reason='no way down', arm=ctx.arm, point=np.round(point, 3).tolist()); return None
    ctx.set_grip(1.)
    status = yield from move_to(ctx, plan['pre'], rate=.04, label=label)
    if status != 'done':
        return None
    yield from hold(ctx, 5)
    if look is not None:
        for again in range(2):
            seen = look(ctx.tip()[:3, 3], closing_angle(ctx))
            shift = None if seen is None else np.asarray(seen, float)[:2] - plan['point'][:2]
            ctx.event(label + '_look', arm=ctx.arm, shift=None if shift is None else np.round(shift, 4).tolist())
            if shift is None or float(np.linalg.norm(shift)) < .004 or float(np.linalg.norm(shift)) > .06:
                break
            better = plan_top_pinch(ctx, np.r_[plan['point'][:2] + shift, point[2]], closing, relax=relax, approach=approach, both_ways=both_ways, locked=locked)
            if better is None:
                break
            status = yield from move_to(ctx, better['pre'], rate=.03, label=label)
            if status != 'done':
                return None
            plan = better
            yield from hold(ctx, 5)
    status = yield from follow(ctx, plan['up'][::-1][1:] + [plan['low']], rate=.015, max_steps=300)
    reached = ctx.tip()[:3, 3].copy()
    ctx.set_grip(-1.)
    yield from hold(ctx, CLOSE_STEPS)
    fingers = ctx.fingers().copy()
    yield from follow(ctx, plan['up'], rate=.015, max_steps=300)
    yield from hold(ctx, 4)
    ok = held(ctx)
    ctx.event(label, ok=ok, arm=ctx.arm, descend=status, load=round(plan['load'], 2), solutions=plan['solutions'], fingers=fingers.round(4).tolist(),
              after_lift=ctx.fingers().round(4).tolist(), short=(reached - plan['point']).round(3).tolist(), point=np.round(plan['point'], 3).tolist())
    if not ok:
        ctx.set_grip(1.)
    return ok


def set_down(ctx, point, closing, label='set_down', approach=.12, relax=False, both_ways=True):
    """Bring what the active hand holds straight down until the fingers' ends are at @point, let go, and lift the
    hand again. Returns True when it was let go there."""
    plan = plan_top_pinch(ctx, point, closing, relax=relax, approach=approach, both_ways=both_ways)
    if plan is None:
        ctx.event(label, ok=False, reason='no way down', arm=ctx.arm, point=np.round(point, 3).tolist()); return False
    status = yield from move_to(ctx, plan['pre'], rate=.04, label=label)
    if status != 'done':
        return False
    status = yield from follow(ctx, plan['up'][::-1][1:] + [plan['low']], rate=.015, max_steps=300)
    reached = ctx.tip()[:3, 3].copy()
    ctx.set_grip(1.); ctx.detach(ctx.arm)
    yield from hold(ctx, 20)
    yield from follow(ctx, plan['up'], rate=.02, max_steps=300)
    ctx.event(label, ok=True, arm=ctx.arm, descend=status, short=(reached - plan['point']).round(3).tolist(), point=np.round(plan['point'], 3).tolist())
    return True


HALF_WIDTH = .345           # of the chassis: arms and what they hold stay within it for travelling (the doorway leaves 4 cm more on a side)


def park(ctx, spots, rotations, label='carry', rate=.03, narrow=True):
    """Move the active hand (with what it holds) to the first of @spots x @rotations that the arm reaches with the
    trunk upright and, if @narrow, with nothing wider than the chassis (an elbow that stuck out brushed the door
    frame). Returns the follow status, or 'no_pose'."""
    free = ctx.validator(relax_from=ctx.q); arm = ctx.arm; upper = ~ctx.base_part; radii = ctx.collision.radii[upper]

    def check(full, poses):
        if not free(full, poses):
            return False
        if not narrow:
            return True
        centres = ctx.collision.centers(poses)[upper]
        if float(np.max(np.abs(centres[:, 1]) + radii)) > HALF_WIDTH + .02:
            return False
        if arm in ctx.payload:
            points, size = ctx.carried(poses, arm)
            return float(np.max(np.abs(points[:, 1]) + size)) <= HALF_WIDTH + .02
        return True
    for rotation in rotations:
        for spot in spots:
            goal = ctx.ik.solve(ctx.q, np.asarray(spot, float) - [0., 0., BASE_Z], rotation, count=40, validator=check, locked=UPRIGHT)
            if goal is not None:
                status = yield from move_to(ctx, goal, rate=rate, label=label)
                if status == 'done':
                    return status
    # No such pose: at least the trunk upright, the arm as it is (the localiser corrects the pose only with the
    # trunk upright, and a hand left low in front is in the way of everything).
    ctx.event(label, ok=False, reason='no pose', arm=ctx.arm)
    status = yield from move_to(ctx, np.r_[TALL_TRUNK, ctx.chain[4:]], rate=rate, label=label)
    return 'no_pose'


def side_of_arm(ctx):
    return 1. if ctx.arm == 'left' else -1.


def closing_angle(ctx, arm=None):
    """The direction along which the hand's fingers close, in the base frame (rad)."""
    rotation = ctx.tip(arm)[:3, :3]
    return float(np.arctan2(rotation[1, 1], rotation[0, 1]))


def level_way(ctx, points, closing, check=None, step=.03, upright=None):
    """Joint waypoints that take the active hand's fingers' ends through @points (base frame, straight from one to
    the next, starting where they are) with the fingers pointing straight down all the way, turning evenly to
    close along @closing by the end: what the hand holds stays level. None if a stretch has no such pose.
    (A planned joint path keeps the hand's attitude only at its ends: on one, the box tipped and the camera
    fell out of it.)
    upright: if given (1 or 2), the trunk is brought upright along the way, in proportion to the way done (1) or
    to its square (2), and the arm alone solves each step: the way then ends in the posture for travelling."""
    full = ctx.q.copy(); start = ctx.tip()[:3, 3].copy(); first = closing_angle(ctx); turn = float((closing - first + np.pi) % (2 * np.pi) - np.pi)
    points = [np.asarray(point, float) for point in points]; total = sum(float(np.linalg.norm(b - a)) for a, b in zip([start] + points, points)); done = 0.; way = []
    trunk = ctx.chain[:4].copy()
    for target in points:
        length = float(np.linalg.norm(target - start)); count = max(1, int(np.ceil(length / step)))
        for k in range(1, count + 1):
            share = (done + length * k / count) / max(total, 1e-6)
            point = start + (target - start) * k / count; rotation = down(first + turn * share)
            locked = None if upright is None else {index: float(value) for index, value in enumerate(trunk + (TALL_TRUNK - trunk) * share ** upright)}
            solution = ctx.ik.solve(full, point, rotation, count=1, validator=check, stay=.05, locked=locked)       # from the pose before, staying near it
            if solution is None or np.max(np.abs(solution - full[ctx.ik.columns])) > .6:
                solution = ctx.ik.solve(full, point, rotation, count=6, validator=check, locked=locked)
            if solution is None or np.max(np.abs(solution - full[ctx.ik.columns])) > .6:
                return None
            full = full.copy(); full[ctx.ik.columns] = solution; way.append(solution)
        done += length; start = target
    return way


def straighten(ctx, check=None, steps=8):
    """Joint waypoints that bring the trunk upright while the active hand's fingers' ends stay where they are and
    point as they do: the arm makes up for the trunk. None if some step has no such pose.
    (The localiser corrects the pose only with the trunk upright: a box carried with the trunk as the reach had
    left it was driven into the stairs, a quarter of a metre off.)"""
    full = ctx.q.copy(); tip = ctx.tip(); trunk = ctx.chain[:4].copy(); way = []
    for k in range(1, steps + 1):
        locked = {index: float(value) for index, value in enumerate(trunk + (TALL_TRUNK - trunk) * k / steps)}
        solution = ctx.ik.solve(full, tip[:3, 3], tip[:3, :3], count=1, validator=check, locked=locked, stay=.05)
        if solution is None or np.max(np.abs(solution - full[ctx.ik.columns])) > .6:
            solution = ctx.ik.solve(full, tip[:3, 3], tip[:3, :3], count=8, validator=check, locked=locked)
        if solution is None or np.max(np.abs(solution - full[ctx.ik.columns])) > .6:
            return None
        full = full.copy(); full[ctx.ik.columns] = solution; way.append(solution)
    return way


@contextmanager
def supposing(ctx, arm, chain):
    """Plan as if the trunk and @arm stood at @chain (trunk 4 + arm 7): for a stance that is being considered.
    What is attached to the hands meanwhile is dropped at the end."""
    s, payload, active = ctx.s, dict(ctx.payload), ctx.arm
    ctx.s = s.copy(); ctx.s[53:57] = chain[:4]; ctx.s[ARMS[arm]['joints']] = chain[4:]; ctx.arm = arm
    try:
        yield
    finally:
        ctx.s, ctx.payload, ctx.arm = s, payload, active


def box_carry_way(ctx, check, height=None):
    """Joint waypoints from where the active hand holds the box to the posture it is carried in: the box in front
    of the chest, the wall the hand holds 0.27 m ahead of the base centre, its rim 0.89 m up, the trunk upright.
    The box then reaches little further ahead than the chassis's rear corners reach back, and the base turns as
    freely as without it. height: of the fingers' ends, if not that (OVER_THE_SILL in front of the car). The box
    stays level all the way. None if there is no such way."""
    points, _ = ctx.payload[ctx.arm]; side = 1. if float(np.mean(points[:, 1])) > 0 else -1.        # the box lies to this side of the fingers, along the way they close
    closing = 0. if side > 0 else np.pi; here = ctx.tip()[:3, 3].copy(); own = side_of_arm(ctx)
    for z in ((height,) if height else (.86, .90)):
        # (To the arm's own side of the middle first: at the middle line the arm's second joint is at its limit.)
        for x, y in ((.27, .10 * own), (.30, .10 * own), (.33, .12 * own), (.27, 0.), (.36, .10 * own)):
            point = np.array([x, y, z - BASE_Z])
            # Straight there, the trunk coming upright on the way; or straight up first.
            for via, schedule in (([], 1), ([], 2), ([np.array([here[0], here[1], point[2]])], 1)):
                way = level_way(ctx, via + [point], closing, check, upright=schedule)
                if way is not None:
                    return way
    return None


def carry_box(ctx, height=None):
    """Bring the box the active hand holds to the carrying posture (box_carry_way), keeping it level."""
    way = box_carry_way(ctx, ctx.validator(relax_from=ctx.q), height)
    if way is not None:
        status = yield from follow(ctx, way, rate=.02, max_steps=900)
        if status != 'done':
            ctx.event('carry_box', ok=False, reason=status)
        return status
    ctx.event('carry_box', ok=False, reason='no level way', arm=ctx.arm)
    points, _ = ctx.payload[ctx.arm]; closing = 0. if float(np.mean(points[:, 1])) > 0 else np.pi
    spots = [(x, y, z) for z in ((height,) if height else (.86, .83, .90)) for x in (.27, .30, .24) for y in (0., .04, -.04)]
    return (yield from park(ctx, spots, [down(closing)], label='carry_box'))


def racket_in_hand(ctx):
    """Which way the racket's head lies from the hand along the tool's x axis (+1 or -1): the racket is held by its
    handle from above, its long side along that axis."""
    points, _ = ctx.payload[ctx.arm]; toward_head = np.mean(points[-3:], axis=0) - np.mean(points[:5], axis=0)
    return 1. if toward_head[0] > 0 else -1.


def carry_racket(ctx):
    """The racket where it is no wider than the chassis, does not reach ahead and is not in the camera's view:
    flat above the head, its head pointing back; or, where the wrist does not turn that way (it depends on which
    way round the hand took the handle), hanging in front of the chassis, head down; or upright in front of the
    shoulder. (Upright beside the shoulder with its flat side along the way ahead, the arm has no pose at all;
    with its flat side across, it is half a metre wide.)"""
    side = -1. if ctx.arm == 'right' else 1.; s = racket_in_hand(ctx)

    def rotation(fingers, head):
        """Tool rotation for fingers pointing along @fingers and the racket's head along @head."""
        z = -np.asarray(fingers, float); x = s * np.asarray(head, float)
        return np.column_stack([x, np.cross(z, x), z])
    ways = [([(.25, side * .15, 1.75), (.28, side * .15, 1.75), (.22, side * .17, 1.72)], rotation((0., 0., 1.), (-1., 0., 0.))),
            # (0.10 to the arm's side: nearer to the middle line the forearm lies against the trunk.)
            ([(.36, side * .10, 1.05), (.40, side * .10, 1.05), (.36, side * .13, 1.05), (.40, side * .05, 1.05), (.44, side * .08, 1.10)], rotation((1., 0., 0.), (0., 0., -1.))),
            ([(.30, side * .185, 1.0), (.30, side * .15, .95)], rotation((1., 0., 0.), (0., 0., 1.)))]
    for spots, turn in ways:
        check = ctx.validator(relax_from=ctx.q)
        if any(ctx.ik.solve(ctx.q, np.asarray(spot, float) - [0., 0., BASE_Z], turn, count=24, validator=check, locked=UPRIGHT) is not None for spot in spots):
            status = yield from park(ctx, spots, [turn], label='carry_racket')
            if status == 'done':
                return status
    return (yield from park(ctx, ways[0][0], [ways[0][1]], label='carry_racket'))


def across_the_trunk(side, s):
    """Tool rotation: fingers down, the racket's head pointing to the far side of the trunk from the hand that holds it."""
    return down(np.pi if (s > 0) == (side < 0) else 0.)


# ---- the trunk lid ------------------------------------------------------------------------------------------
def lid_points(base, angle, side):
    """Where a point of the lid is in the base frame of a base standing at @base (x, y, yaw in M), the lid opened
    by @angle: returns a function (x, z of the point with the lid closed) -> base-frame xyz, for the line across
    the lid @side metres from its middle."""
    x0, y0, yaw = base; c, s = np.cos(yaw), np.sin(yaw); turn = about_y(angle)

    def to_base(xz):
        local = turn @ np.array([xz[0] - HINGE[0], 0., xz[1] - HINGE[1]])
        world = np.array([HINGE[0] + local[0], LID_MIDDLE + side, HINGE[1] + local[2]])
        dx, dy = world[0] - x0, world[1] - y0
        return np.array([c * dx + s * dy, -s * dx + c * dy, world[2] - BASE_Z])
    return to_base


def edge_x(side):
    """x (map) of the lid's upper rear edge as the camera shows it, @side metres from the lid's middle."""
    return LID_APEX[0] + LID_SLOPE * (np.sqrt(side * side + LID_ROUND ** 2) - LID_ROUND)


def open_lid(ctx, est):
    """Open the trunk lid from the stance behind the car (STANCE); the base pose must have been taken from the
    lid's edge just before (Observer.lid_pose), so that the hand finds the edge to a few millimetres.
    Returns 'open', or what stopped it.

      1. left hand: onto that edge, slanting down, the edge between the fingers; closes; follows the edge's arc
         round the hinge to HANDOVER, the hand turning with the lid;
      2. right arm alone (the trunk joints stay put, the left hand holds the lid): takes the lid's lower end from
         below and behind;
      3. left hand lets go and folds;
      4. right hand: fingers parted (the grasp joint stays), follows the lower end's arc to OPEN, lets go.
    Beyond 2.0 rad the lid's weight keeps it open; below it falls shut.
    """
    base = est.base(); yaw = base[2]; c, s = np.cos(yaw), np.sin(yaw)
    # The car's axes in the base frame (the base may stand a degree or two askew).
    ahead = np.array([c, -s, 0.]); up = np.array([0., 0., 1.]); turn_axes = np.column_stack([ahead, np.cross(up, ahead), up])
    locked_turn = {3: 0.}
    # ---- 1. left hand on the edge ----
    ctx.arm = 'left'; check = ctx.validator()
    approach = turn_axes @ np.array([np.cos(EDGE_SLANT), 0., -np.sin(EDGE_SLANT)]); closing = turn_axes @ np.array([np.sin(EDGE_SLANT), 0., np.cos(EDGE_SLANT)])
    rotation = frame(approach, closing)
    grip_point = lid_points(base, 0., EDGE_SIDE)(np.array([edge_x(EDGE_SIDE) - EDGE_BEHIND, EDGE_HEIGHT])); before = grip_point - approach * .10
    pre = ctx.ik.solve(ctx.q, before, rotation, validator=check, count=32, locked=locked_turn)
    if pre is None:
        ctx.event('lid', ok=False, stage='edge', reason='no pose before the edge'); return 'no_pose'
    ctx.set_grip(1.)
    status = yield from move_to(ctx, pre, rate=.04, label='lid')
    if status != 'done':
        return 'edge_' + status
    inward = interpolate_cartesian(ctx.ik, ctx.q, before, grip_point, rotation, step=.02, locked=locked_turn)
    if inward is None:
        ctx.event('lid', ok=False, stage='edge', reason='no way onto the edge'); return 'no_way_in'
    yield from follow(ctx, inward, rate=.015, max_steps=300)
    ctx.set_grip(-1.)
    yield from hold(ctx, CLOSE_STEPS)
    if not held(ctx):
        ctx.event('lid', ok=False, stage='edge', reason='closed on nothing', fingers=ctx.fingers().round(4).tolist())
        ctx.set_grip(1.); yield from hold(ctx, 6)
        yield from follow(ctx, inward[::-1][1:] + [pre], rate=.02, max_steps=200)
        return 'edge_missed'
    ctx.event('lid', stage='edge taken', fingers=ctx.fingers().round(4).tolist(), base=np.round(base, 4).tolist(), short=(ctx.tip()[:3, 3] - grip_point).round(4).tolist())
    # The hinge line as the hand has it: the edge it holds is a known way from it.
    tip0 = ctx.tip()[:3, 3].copy(); hinge = np.array([tip0[0], tip0[1], 0.]) + turn_axes @ np.array([HINGE[0] - (edge_x(EDGE_SIDE) - EDGE_BEHIND), 0., HINGE[1] - BASE_Z])
    current = ctx.q.copy(); theta = 0.

    def turned(angle):
        return turn_axes @ about_y(angle) @ turn_axes.T

    while theta < HANDOVER - 1e-6:
        theta += LID_STEP; point = hinge + turned(theta) @ (tip0 - hinge)
        solution = ctx.ik.solve(current, point, turned(theta) @ rotation, count=6, validator=check, locked=locked_turn)
        if solution is None or np.max(np.abs(solution - current[ctx.ik.columns])) > .7:
            solution = ctx.ik.solve(current, point, turned(theta) @ rotation, count=24, validator=check, free_axis=np.array([1., 0., 0.]), locked=locked_turn)
        if solution is None or np.max(np.abs(solution - current[ctx.ik.columns])) > .7:
            theta -= LID_STEP; break
        yield from follow(ctx, [solution], rate=.02, max_steps=60)
        current = ctx.q.copy()
    lifted = theta
    if lifted < .3 or not held(ctx):
        ctx.event('lid', ok=False, stage='lift', reason='the edge does not come up', angle=round(lifted, 2), fingers=ctx.fingers().round(4).tolist())
        ctx.set_grip(1.); yield from hold(ctx, 20)
        yield from tuck(ctx, ('left',))
        return 'edge_lost'
    # ---- 2. right arm alone to the lid's lower end ----
    v0 = tip0 - hinge; v1 = ctx.tip('left')[:3, 3] - hinge; local0 = turn_axes.T @ v0; local1 = turn_axes.T @ v1
    angle = float((np.arctan2(local0[2], local0[0]) - np.arctan2(local1[2], local1[0])) % (2 * np.pi))       # the lid's angle as the left hand has it
    trunk = ctx.s[53:57].copy(); locked = {k: float(trunk[k]) for k in range(4)}; ctx.arm = 'right'
    along = turned(angle) @ turn_axes @ np.array([LIP_ALONG[0], 0., LIP_ALONG[1]]); across = turned(angle) @ turn_axes @ np.array([-LIP_ALONG[1], 0., LIP_ALONG[0]])
    sideways = turn_axes @ np.array([0., LIP_SIDE - EDGE_SIDE, 0.])
    lip_closed = hinge + sideways + turn_axes @ np.array([LIP[0] - HINGE[0] + LIP_ALONG[0] * LIP_IN, 0., LIP[1] - HINGE[1] + LIP_ALONG[1] * LIP_IN])
    across0 = turn_axes @ np.array([-LIP_ALONG[1], 0., LIP_ALONG[0]])      # out of the rear face, lid shut
    lip_tip = hinge + sideways + turned(angle) @ (lip_closed - hinge - sideways) - across * LIP_INWARD
    take = None
    for offset in (0., -20., 20., -35., 35.):
        rot = turned(np.radians(offset)) @ frame(along, across); back = -rot[:, 2]
        low = ctx.ik.solve(ctx.q, lip_tip - back * .10, rot, count=32, validator=check, locked=locked)
        if low is None:
            continue
        full = ctx.q.copy(); full[ctx.ik.columns] = low
        way = interpolate_cartesian(ctx.ik, full, lip_tip - back * .10, lip_tip, rot, step=.02, locked=locked)
        if way is not None:
            take = (offset, low, way); break
    if take is None:
        ctx.event('lid', ok=False, stage='lower end', reason='the right arm does not reach it', angle=round(angle, 3))
        ctx.arm = 'left'; ctx.set_grip(1.); yield from hold(ctx, 20)
        yield from tuck(ctx)
        return 'no_reach'
    offset, low, way = take
    ctx.event('lid', stage='lifted', angle=round(angle, 3), fingers=ctx.fingers('left').round(4).tolist(), offset=offset)
    ctx.set_grip(1.)
    status = yield from move_to(ctx, low, rate=.04, still_trunk=True, label='lid')
    yield from follow(ctx, way, rate=.015, max_steps=300)
    ctx.set_grip(-1.)
    yield from hold(ctx, CLOSE_STEPS)
    if status != 'done' or not held(ctx):
        ctx.event('lid', ok=False, stage='lower end', reason='closed on nothing' if status == 'done' else status, fingers=ctx.fingers().round(4).tolist())
        ctx.set_grip(1.); yield from hold(ctx, 8)
        yield from move_to(ctx, np.r_[trunk, TUCK['right']], rate=.04, still_trunk=True, label='lid')
        ctx.arm = 'left'; ctx.set_grip(1.); yield from hold(ctx, 20)
        yield from tuck(ctx)
        return 'lower_end_missed'
    ctx.event('lid', stage='lower end taken', fingers=ctx.fingers().round(4).tolist(), short=(ctx.tip()[:3, 3] - lip_tip).round(4).tolist())
    # ---- 3. left hand lets go and folds (trunk still) ----
    ctx.arm = 'left'; ctx.set_grip(1.)
    yield from hold(ctx, 15)
    away = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], ctx.tip()[:3, 3] + ctx.tip()[:3, 2] * .10, ctx.tip()[:3, :3], step=.03, locked=locked)
    if away is not None:
        yield from follow(ctx, away, rate=.02, max_steps=150)
    yield from move_to(ctx, np.r_[trunk, TUCK['left']], rate=.04, still_trunk=True, label='lid')
    # ---- 4. right hand follows the lower end's arc, its attitude free ----
    ctx.arm = 'right'; sagged = ctx.tip()[:3, 3] - lip_tip
    ctx.event('lid', stage='left away', fingers=ctx.fingers().round(4).tolist(), sagged=sagged.round(3).tolist())
    if float(np.sum(ctx.fingers())) < .004 and sagged[2] > -.03:
        # The lid has fallen out of the right hand's fingers: they have closed all the way, and the hand has not
        # sunk under the lid's weight (with the lid on it, it sinks 14 cm): the grasp joint had not formed.
        ctx.event('lid', ok=False, stage='lower end', reason='the lid fell when the left hand let go')
        ctx.set_grip(1.); yield from hold(ctx, 6)
        yield from tuck(ctx, ('right', 'left'))
        return 'lower_end_lost'
    ctx.set_grip(PARTED)
    yield from hold(ctx, 20)
    ctx.event('lid', stage='parted', fingers=ctx.fingers().round(4).tolist())
    # The lid's angle from now on is read off the hand: the turn of the held point round the hinge line, counted
    # from where that point is with the lid shut. (Not from the angle at the hand-over: with the left hand gone
    # the lid's weight pulls the right arm down, and the lid with it, by 0.2 rad; led by an angle that did not
    # know of that, the lid was let go at 1.9 rad, short of where it stays open.)
    pivot = hinge + sideways; shut = turn_axes.T @ (lip_closed - across0 * LIP_INWARD - pivot); current = ctx.q.copy(); check = ctx.validator(relax_from=ctx.q)

    def lid_angle():
        now = turn_axes.T @ (ctx.tip()[:3, 3] - pivot)
        return float((np.arctan2(shut[2], shut[0]) - np.arctan2(now[2], now[0]) + np.pi) % (2 * np.pi) - np.pi)

    radius = float(np.linalg.norm(shut)); tip = ctx.tip()[:3, 3].copy(); theta = reached = lid_angle(); after_handover = reached; stuck = 0

    def on_arc(angle_):
        direction = float(np.arctan2(shut[2], shut[0])) - angle_
        return pivot + turn_axes @ np.array([radius * np.cos(direction), 0., radius * np.sin(direction)])

    while reached < OPEN - 1e-6 and stuck < 6:
        # Where the lid is now, by the hand; the next point three steps ahead of that.
        reached = lid_angle()
        target = min(OPEN + .08, max(theta, reached + 3 * LID_STEP)); point = on_arc(target)
        solution = ctx.ik.solve(current, point, None, count=3, validator=check, stay=.06)
        if solution is None or np.max(np.abs(solution - current[ctx.ik.columns])) > .7:
            solution = ctx.ik.solve(current, point, None, count=24, validator=check, stay=.02)
        if solution is None or np.max(np.abs(solution - current[ctx.ik.columns])) > .7:
            ctx.event('lid', ok=False, stage='swing', reason='no pose further along the arc', angle=round(reached, 3)); break
        before = reached
        yield from follow(ctx, [solution], rate=.03, max_steps=45)
        current = ctx.q.copy(); theta = target
        reached = lid_angle()
        stuck = stuck + 1 if reached < before + .01 else 0
    ctx.set_grip(1.)
    yield from hold(ctx, 25)
    back = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], ctx.tip()[:3, 3] + turn_axes @ np.array([-.12, 0., -.05]), None, step=.03)
    if back is not None:
        yield from follow(ctx, back, rate=.02, max_steps=150)
    ctx.event('lid', ok=reached >= 2.0, stage='swing', handover=round(angle, 3), after_handover=round(after_handover, 3), led_to=round(reached, 3), offset=offset)
    yield from tuck(ctx, ('right', 'left'))
    return 'open' if reached >= 2.0 else 'short'
