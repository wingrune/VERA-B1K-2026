"""Skills for the two bags of unloading_the_car: taking a box-like thing by a top edge with a slanted hand.

All points are in the base frame B (x ahead, y left, z up from the base_link origin, which rides BASE_Z above the
floor) unless a name says "map".

A bag is wider than the fingers open, so it cannot be pinched from above across its body. It is taken the way the
trunk lid's edge is: the fingers point forward and down, one finger on the bag's top face and the other on the side
face that looks at the robot, the edge between them. The simulator's assisted grasp then holds the bag to the hand
(both fingers touch it, and it lies between them) until the gripper is commanded fully open.
"""
import numpy as np
from manip import hold, follow, move_to, BASE_Z
from skills import interpolate_cartesian
from car_skills import frame, held, CLOSE_STEPS

SLANT = np.radians(50.)             # the fingers point down from the horizontal by this much
# Where the middle between the fingers' ends goes, seen across the edge: this far below the top face and this far
# outside the side face (negative: inside the thing). With the fingers 10 cm apart and slanted by 50 degrees, the
# upper finger's end is then 1 cm above the top face and the lower one 2 cm outside the side face: both come in
# clear of the thing and close onto the two faces, the edge between them. (4.5 cm below the edge, as on the trunk
# lid, puts the upper finger's end inside the bag: the hand shoved the bag and closed on nothing in 9 tries of 12.)
BELOW_EDGE = .02
BEHIND_EDGE = -.015
WAY_IN = .10                        # the hand comes onto the edge along its fingers from this far
BLOCKED = .03                       # a hand that stops this far short of the edge has met something on its way
# Sizes of the bags as boxes (m): along their own x, y, and their height (for the collision model of a bag in the hand).
SIZE = dict(briefcase=(.256, .299, .087), satchel=(.413, .489, .217))


def turn_z(angle):
    c, s = np.cos(angle), np.sin(angle); return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


UP = np.array([0., 0., 1.])


def pinch_frame(edge, top, side, slant=SLANT, below=BELOW_EDGE, behind=BEHIND_EDGE, skew=0.):
    """Where the fingers' ends go and how the hand is turned to take the edge between two faces of a thing: @edge is
    a point of the edge, @top and @side the outward normals (unit vectors, at right angles) of the face the upper
    finger lies on and of the face the lower finger lies on. The fingers point into the corner, @slant away from
    the plane of the top face. skew (rad): the hand turned about the line its fingers close along, so that it comes
    onto the edge askew (the fingers close across the edge as before). Returns (point, approach, rotation)."""
    top = np.asarray(top, float); side = np.asarray(side, float)
    approach = -np.cos(slant) * side - np.sin(slant) * top; closing = -np.sin(slant) * side + np.cos(slant) * top
    if skew:
        approach = np.cos(skew) * approach + np.sin(skew) * np.cross(closing, approach)
    return np.asarray(edge, float) + behind * side - below * top, approach, frame(approach, closing)


def level_side(inward):
    """Outward normal of an upright face whose direction into the thing is @inward (rad)."""
    return -np.array([np.cos(inward), np.sin(inward), 0.])


def plan_pinch(ctx, edge, top, side, slant=SLANT, below=BELOW_EDGE, behind=BEHIND_EDGE, base=None, relax=False, way_in=WAY_IN, skew=0.):
    """The slanted pinch of an edge with the active arm: dict(grip, pre, inward (waypoints), point, approach, rotation)
    or None. base: the base pose (4x4 in M) the plan is for, if not the present one."""
    point, approach, rotation = pinch_frame(edge, top, side, slant, below, behind, skew); full = ctx.q
    check = ctx.validator(base=base, relax_from=full if relax else None)
    for grip in ctx.ik.solve(full, point, rotation, count=24, validator=check, all_solutions=True)[:6]:
        at_grip = full.copy(); at_grip[ctx.ik.columns] = grip
        way_out = interpolate_cartesian(ctx.ik, at_grip, point, point - way_in * approach, rotation, validator=check, step=.025)
        if way_out is not None:
            return dict(grip=grip, pre=way_out[-1], inward=way_out[::-1][1:] + [grip], point=point, approach=approach, rotation=rotation)
    return None


def plan_edge_pinch(ctx, edge, inward, **options):
    """plan_pinch() for the top edge of an upright face of something that lies level (@inward: the direction into it)."""
    return plan_pinch(ctx, edge, UP, level_side(inward), **options)


def pinch(ctx, edge, top, side, label='edge', relax=False, **options):
    """Take the thing by the edge at @edge between its faces with outward normals @top and @side. Returns True if
    the fingers stayed apart after closing, False if they closed on nothing, None if the hand found no way there."""
    plan = plan_pinch(ctx, edge, top, side, relax=relax, **options)
    if plan is None:
        ctx.event(label, ok=False, reason='no way to the edge', arm=ctx.arm, edge=np.round(edge, 3).tolist()); return None
    ctx.set_grip(1.)
    status = yield from move_to(ctx, plan['pre'], rate=.04, label=label)
    if status != 'done':
        return None
    yield from hold(ctx, 4)
    status = yield from follow(ctx, plan['inward'], rate=.015, max_steps=300)
    reached = ctx.tip()[:3, 3].copy()
    if float(np.linalg.norm(reached - plan['point'])) > BLOCKED:
        # Something the collision model does not have stood in the hand's way (a right hand that came askew onto
        # the satchel's end by the trunk's wall stopped 6 and 8 cm short, twice of four): no closing on what is not
        # between the fingers; out the way it came.
        ctx.event(label, ok=False, reason='way in blocked', arm=ctx.arm, way_in=status, short=(reached - plan['point']).round(3).tolist(), point=np.round(plan['point'], 3).tolist())
        yield from follow(ctx, plan['inward'][::-1][1:] + [plan['pre']], rate=.02, max_steps=200)
        return None
    ctx.set_grip(-1.)
    yield from hold(ctx, CLOSE_STEPS)
    ok = held(ctx)
    ctx.event(label, ok=ok, arm=ctx.arm, way_in=status, fingers=ctx.fingers().round(4).tolist(), short=(reached - plan['point']).round(3).tolist(), point=np.round(plan['point'], 3).tolist())
    if not ok:
        ctx.set_grip(1.)
        yield from hold(ctx, 6)
        yield from follow(ctx, plan['inward'][::-1][1:] + [plan['pre']], rate=.02, max_steps=200)
    return ok


def edge_pinch(ctx, edge, inward, label='edge', relax=False, **options):
    """pinch() for the top edge of an upright face of something that lies level."""
    return (yield from pinch(ctx, edge, UP, level_side(inward), label=label, relax=relax, **options))


# Edges of the bags that a slanted hand can take, in each bag's own frame (from its collision shape: see
# dev/jobs/probe_static.py and the picture study/bag_shapes.png): (a point on the edge, xyz from the bag's origin;
# the direction across the edge into the bag, rad; half the length of the edge that can be used).
# The briefcase is a slab with a small handle on its +x side: the four edges of its top face (the +x edge beside
# the handle). The satchel is a wedge: a gently sloping front (-x), a flat top 9 cm wide, a steep back (+x) and
# upright ends (+-y): its back ridge and the top edges of its two ends, where the top is highest.
# The satchel's frame is its asset's. The briefcase is described from its middle (of its body, half-way through its
# thickness), its x along the short side, y along the long side and z out of the broad face that looks up: it is the
# same slab whichever of its broad faces is up.
SLAB = dict(short=.226, long=.298, thick=.089)
LEANING = np.radians(20.)           # a briefcase tilted by more than this leans on something or stands on a side: see edges()
# On the rim of a leaning briefcase the fingers' ends go further inside the broad face than on a bag that lies flat:
# the lower finger then starts 0.5-1 cm from that face and touches it first, pressing the briefcase against what it
# leans on, before the upper finger comes down on the rim. With BEHIND_EDGE (the lower finger 2.3 cm away) the upper
# finger came down first and pushed the briefcase along its own plane, down and away: the hand closed on nothing in
# 5 tries of 5 (starts 103, 221, 46); 3 cm inside the true face it held in 2 tries of 2 on the first two, and 2.7 cm
# inside what the camera has of the face (which lies 0.3-0.9 cm inside the true one) held on the third.
BEHIND_RIM = -.026
EDGES = dict(
    # (Measured on the collision shape: the flat top, 0.10-0.105 up, ends at x = 0.134, where the back falls away at 71
    # degrees; at the ends the upright faces stand at y = 0.194 and -0.192. The top beside them is highest for x
    # round 0.1 and comes down to the front as a slope: 0.05 up at x = -0.08, which is the part of an end that a hand
    # reaches when the satchel's low front looks at the robot and its back lies under the parcel shelf. On the -y
    # end a strap ring stands out 4.5 cm, 8 cm below the top, for x 0.04-0.12: the hand goes on beside it and less
    # deep, or its lower finger lands on the ring and shoves the bag (the hand then closed on air, 6 tries of 8).
    # With the edges put 1-2 cm further out and lower than they are, the upper finger came down on the top and
    # the lower one closed on air: 9 of 22 tries held.)
    satchel=(dict(point=(.134, 0., .105), inward=np.pi, half=.15), dict(point=(.09, .194, .100), inward=-np.pi / 2, half=.05),
             dict(point=(.122, -.192, .100), inward=np.pi / 2, half=.02, below=.008),
             dict(point=(-.08, .194, .052), inward=-np.pi / 2, half=.03), dict(point=(-.08, -.192, .052), inward=np.pi / 2, half=.03)),
    # The four rims of the briefcase's upper face.
    briefcase=(dict(point=(-SLAB['short'] / 2, 0., SLAB['thick'] / 2), inward=0., half=.10), dict(point=(SLAB['short'] / 2, 0., SLAB['thick'] / 2), inward=np.pi, half=.10),
               dict(point=(0., -SLAB['long'] / 2, SLAB['thick'] / 2), inward=np.pi / 2, half=.07), dict(point=(0., SLAB['long'] / 2, SLAB['thick'] / 2), inward=-np.pi / 2, half=.07)))
# A satchel that lies with its low front to the robot and an end near the trunk's wall (3 starts in 100) leaves no
# edge for a hand that comes squarely: the back is under the parcel shelf, one end has the wall beside it, and the
# other end looks away from the robot. That end is taken with the hand askew: turned by SKEW about the line the
# fingers close along, so that it points more ahead and less across.
SKEW = np.radians(35.); SKEWED = (1, 2, 3, 4)
MIDDLE = dict(briefcase=np.zeros(3), satchel=np.array([-.005, -.002, -.002]))         # of the box round a bag, in the bag's frame


def edges(name, origin, turn):
    """The edges of a bag that a hand can take, nearest to a robot that looks along +x first. origin, turn: where the
    bag's frame is (xyz and 3x3, both in the frame the answer is wanted in). Each: dict(point, top, side (outward
    normals of the face for the upper finger and of the one for the lower finger), inward (rad: the level direction
    the hand points along), half (half the length of the edge that can be used), below (how far below the edge the
    fingers' ends go), behind (how far outside the side face they go; negative: inside), deeper (how much further
    inside on a second try), id (which of the bag's edges it is, whatever the order), skew."""
    origin = np.asarray(origin, float); turn = np.asarray(turn, float); out = []
    if name == 'briefcase' and turn[2, 2] < np.cos(LEANING):
        # It leans on something or stands on a side: every edge of the slab that has a face looking up, for the
        # upper finger, and beside it a face that looks neither straight up nor down, for the lower one. The upper
        # finger then presses the slab onto what carries it, as on a bag that lies flat.
        #   - it leans on the satchel: its upper rim from above, the lower finger on the broad face that looks up
        #     (id like '+y+z': the upper finger's face, then the lower one's, as the slab's own axes);
        #   - it stands on a long side against the trunk's wall (start 31): the end of the narrow face on top that
        #     looks at the robot, the lower finger on the end face ('+x-y'). The rim beside a broad face has the
        #     wall on one side and wants a hand that points across the trunk on the other: no stance, four visits.
        size = np.array([SLAB['short'], SLAB['long'], SLAB['thick']]) / 2
        for i in range(3):
            for sign_i in (1., -1.):
                top = sign_i * turn[:, i]
                if top[2] < .3:
                    continue
                for j in range(3):
                    for sign_j in (1., -1.) if j != i else ():
                        side = sign_j * turn[:, j]
                        if not -.3 <= side[2] <= .8:
                            continue
                        broad = j == 2                   # the lower finger on a broad face: see BEHIND_RIM
                        out.append(dict(point=origin + top * size[i] + side * size[j], top=top, side=side, half=max(float(size[3 - i - j]) - .04, .02), below=BELOW_EDGE,
                                        behind=BEHIND_RIM if broad else BEHIND_EDGE, deeper=.006 if broad else .012, id=f"{'+-'[sign_i < 0]}{'xyz'[i]}{'+-'[sign_j < 0]}{'xyz'[j]}", skew=0.))
    else:
        for number, edge in enumerate(EDGES[name]):
            out.append(dict(point=origin + turn @ np.asarray(edge['point']), top=turn[:, 2].copy(), side=turn @ level_side(edge['inward']), half=edge['half'], below=edge.get('below', BELOW_EDGE), behind=BEHIND_EDGE, deeper=.012, id=number, skew=0.))
            # An end of the satchel also with the hand askew: see SKEW.
            for skew in (SKEW, -SKEW) if name == 'satchel' and number in SKEWED else ():
                out.append(dict(out[-1], id=f"{number}{'+' if skew > 0 else '-'}", skew=skew))
    for found in out:
        # (A hand that comes down almost straight, as onto a leaning briefcase's rim, points no way in particular.)
        approach = pinch_frame(found['point'], found['top'], found['side'], skew=found.get('skew', 0.))[1]; found['inward'] = float(np.arctan2(approach[1], approach[0])) if np.hypot(approach[0], approach[1]) > .35 else 0.
    # An edge the hand comes onto from the robot's side before one it would have to reach round for; among those,
    # one it comes onto squarely before one it comes onto askew; among those, the nearest.
    return sorted(out, key=lambda item: (abs(item['inward']) > 1.9, bool(item.get('skew')), item['point'][0] + .25 * abs(item['inward'])))


def flat(place):
    """(origin, turn) of a bag that lies level at @place = (x, y, z, yaw)."""
    x, y, z, yaw = place
    return np.array([x, y, z]), turn_z(yaw)


def spheres(name, origin, turn, r=.03):
    """Spheres that fill the box round a bag whose frame is at @origin, @turn: (points, radius), for the collision
    model of a bag in the hand."""
    lx, ly, lz = SIZE[name]; r = min(r, lz / 2); points = []
    for u in np.linspace(-lx / 2 + r, lx / 2 - r, max(2, int(np.ceil(lx / (1.6 * r))))):
        for v in np.linspace(-ly / 2 + r, ly / 2 - r, max(2, int(np.ceil(ly / (1.6 * r))))):
            for w in np.linspace(-lz / 2 + r, lz / 2 - r, max(1, int(np.ceil(lz / (2.2 * r))))):
                if abs(u) > lx / 2 - 2.2 * r or abs(v) > ly / 2 - 2.2 * r or lz < 4.5 * r:           # the shell is enough
                    points.append(np.asarray(origin, float) + np.asarray(turn, float) @ (MIDDLE[name] + [u, v, w]))
    return np.array(points), r


def arms_for(inward):
    """The arms to try for an edge whose direction into the bag is @inward (rad from straight ahead), in order."""
    return ('left', 'right') if inward < -.12 else ('right', 'left') if inward > .12 else ('left', 'right')


# ---- out of the trunk and to the travelling posture ------------------------------------------------------------
# The hand's attitude at a pinch square to the base, turned down about the base's y axis (the bag then hangs ahead of
# and below the hand) and spun about the vertical, in order of choice; and where the fingers' ends are then held:
# (ahead, to the arm's side, height). Which of them the arm planner accepts depends on the bag and on the edge it
# is held by; the first accepted one is taken. The satchel held by its back ridge hangs 0.45 m below the hand.
CARRY_ATTITUDES = ((60., 0.), (60., -30.), (60., 30.), (45., 0.), (45., -30.), (45., 30.), (30., 0.), (75., 0.))      # (turned down, spun towards the arm's side) in degrees
CARRY_SPOTS = ((.34, 0., .95), (.34, 0., 1.05), (.40, 0., 1.05), (.34, .06, .95), (.34, .10, .95), (.34, .10, 1.05), (.40, .10, 1.05), (.34, .10, 1.15), (.28, .10, 1.25))    # (in the middle first: the bag is then narrowest for the doorway)


def carry_rotation(turn, spin=0., slant=SLANT):
    """The hand's attitude for travelling with a bag that was taken by an edge: as at a pinch square to the base,
    spun about the vertical by @spin and turned down by @turn (rad). Whatever way the edge lay in the trunk, the
    bag then hangs the same way from the hand."""
    approach = np.array([np.cos(slant), 0., -np.sin(slant)]); closing = np.array([np.sin(slant), 0., np.cos(slant)])
    c, s = np.cos(turn), np.sin(turn)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]]) @ turn_z(spin) @ frame(approach, closing)


def low_end(ctx, arm):
    """Height above the floor of the lowest point of what @arm's hand holds, as the collision model has it."""
    points, radii = ctx.carried(ctx.body.fk(ctx.q)[0], arm)
    return float(np.min(points[:, 2] - radii)) + BASE_Z


def lift_out(ctx, check=None, moves=((0., 0., .11), (-.22, 0., 0.), (0., 0., .10))):
    """Joint waypoints that take the hand, with the bag it holds, up, back towards the robot (out from under the
    parcel shelf) and up over the sill, its attitude kept. None if a stretch has no such poses.
    (Up by 11 cm first: the satchel then passes over a briefcase that lies between it and the sill, and is still
    2 cm under the parcel shelf. Lifted by 6 cm only, it swept the briefcase out of the trunk and under the car.)"""
    tip = ctx.tip(); here = tip[:3, 3].copy(); full = ctx.q.copy(); way = []
    for delta in moves:
        leg = interpolate_cartesian(ctx.ik, full, here, here + np.asarray(delta), tip[:3, :3], validator=check, step=.02)
        if leg is None:
            return None
        way += leg; full = full.copy(); full[ctx.ik.columns] = leg[-1]; here = here + np.asarray(delta)
    return way
