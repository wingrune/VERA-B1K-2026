"""The pantry by the kitchen's east end (bottom_cabinet_fancyy_0, cabinet.n.01_3): its east door opened with the hinge
skill, and the food set down on a shelf inside.

The pantry stands 8.11-9.34 (x) by -2.20 to -1.55 (y), 2.09 m high, its doors on the north face. The east bay
(x 8.74-9.34) has one full-height door (j_door2, hinged at its east edge, x 9.328) with a handle bar 2.5 cm high and
6 cm proud at 0.795-0.82 m (x 8.82-9.25); shelves at 0.48, 0.849, 1.162, 1.479 and 1.792 m. In the map the pantry is
kept as boxes (one per collision part), not in the height field, so that the arm may reach between its shelves.

A thing in the pantry on a shelf is inside the cabinet for the goal (to be confirmed by an oracle dev job).
"""
import numpy as np
from manip import hold, follow, move_to, drive, pose2, BASE_Z, TALL_TRUNK, TUCK
from skills import interpolate_cartesian
from trash_skills import tuck, least_loaded, reach_rotation
import hinge

HINGE = np.array([9.328, -1.60, 0.])          # the east door's hinge line (vertical)
AXIS = np.array([0., 0., -1.])                 # opening turns the free (west) edge out to the north
GRIP = np.array([8.87, -1.585, .82])          # on the handle bar, 5 cm in from its west end, 1.2 cm over its middle
STANCE = (8.516, -1.161, -.785)                 # the base for the grasp and the lead (23-54 deg there, dt2, dt5, r2; from (8.50, -1.05) the door did not follow the hand, dt3, dt4)
STAGING = (8.50, -.945)                         # where the base comes from: straight in from here (the aisle's middle; (8.45, -0.70) is not free in the map)
OPEN_TO = np.radians(80.)
DOOR_BOX = (8.70, -1.62, 0., 9.36, -1.53, 2.10)   # the door's closed place (its boxes in the map are dropped once it is open)
BAY = (8.76, 9.32, -2.17, -1.63)               # x0 x1 y0 y1 of the east bay's inside
SHELF = .849                                   # the shelf the food is set down on (the next one up is at 1.144)
PLACE_AT = np.array([9.04, -1.86])             # where on it (the middle of the bay)
CLEAR = .015                                   # a thing is let go this far above the shelf


def drop_door_boxes(scene):
    """Forget the door's closed place in the map (it stands open now)."""
    x0, y0, z0, x1, y1, z1 = DOOR_BOX
    keep = [b for b in scene.boxes if not (b[0] > x0 - .01 and b[3] < x1 + .01 and b[1] > y0 - .01 and b[4] < y1 + .01)]
    scene.boxes = np.array(keep).reshape(-1, 6)


def open_door(ctx, est, go=None, arm='left'):
    """Pinch the east door's handle and lead it open. Returns the angle the door was led to (rad)."""
    yield from tuck(ctx)
    if go is not None:
        status = yield from go(STAGING, STANCE[2])
        if status != 'arrived':
            ctx.event('pantry', ok=False, reason='no way to the door', status=status); return 0.
    yield from drive(ctx, STANCE[:2], STANCE[2], speed=.15, direct=True, tolerance=.012, yaw_tolerance=.02, min_speed=.03, min_turn=.05, face_travel=False, max_steps=400)
    yield from hold(ctx, 4)
    saved = hinge.HAND_INTO; hinge.HAND_INTO = -.2
    try:
        held, _ = yield from hinge.pinch(ctx, est, arm, GRIP, np.array([0., -1., 0.]), np.array([0., 0., 1.]))
        got = 0.
        if held:
            got = yield from hinge.lead(ctx, est, HINGE, AXIS, OPEN_TO)
        yield from hold(ctx, 30)            # (let go only once the door has come to rest: drawn back at once, the hand took it back with it)
        yield from hinge.let_go(ctx)
    finally:
        hinge.HAND_INTO = saved
    if got > np.radians(40.):
        drop_door_boxes(ctx.scene)
    ctx.event('pantry', ok=got > np.radians(40.), opened=round(float(np.degrees(got)), 1))
    est.pantry = got
    return got


def place(ctx, est, arm, hang):
    """Set the thing held in @arm (it reaches @hang m below the fingertips' centre) down on the shelf, the hand
    fingers down over PLACE_AT, then draw the hand out of the bay. Returns True if it was let go there."""
    ctx.arm = arm; inv = np.linalg.inv(pose2(*est.base()))
    tip = SHELF + CLEAR + hang
    target = (inv @ np.r_[PLACE_AT[0], PLACE_AT[1], tip - BASE_Z, 1.])[:3]
    outward = inv[:3, :3] @ np.array([0., 1., 0.])            # out of the bay, in the base frame
    found = least_loaded(ctx, ctx.q, target, [reach_rotation(np.arctan2(outward[1], outward[0])), reach_rotation(np.arctan2(outward[1], outward[0]) + np.pi / 2)],
                         ctx.validator(), seeds=2)
    for load, rotation, goal in found[:4]:
        status = yield from move_to(ctx, goal, rate=.03, label='place', max_steps=400)
        if status != 'done':
            continue
        yield from hold(ctx, 6)
        reached = ctx.tip()[:3, 3].copy()
        ctx.set_grip(1.)
        yield from hold(ctx, 20)
        # Up 3 cm (off the thing), then straight out of the bay.
        here = ctx.tip()[:3, 3]
        way = interpolate_cartesian(ctx.ik, ctx.q, here, here + [0., 0., .03], rotation, step=.015)
        if way is not None:
            yield from follow(ctx, way, rate=.02, max_steps=60)
        here = ctx.tip()[:3, 3]
        way = interpolate_cartesian(ctx.ik, ctx.q, here, here + .40 * outward, rotation, step=.03)
        if way is not None:
            yield from follow(ctx, way, rate=.03, max_steps=200)
        yield from tuck(ctx, (arm,))
        ctx.event('place', ok=True, arm=arm, load=round(float(load), 2), tip=np.round(reached, 3).tolist())
        return True
    ctx.event('place', ok=False, arm=arm, solutions=len(found))
    return False


# ---- stage 2: the leaf pushed open from inside -------------------------------------------------------------------
# The hand on the handle leads the door only to 23-54 deg (dish11 dt2, dt4, dt5, r2): the leaf's free end, beyond the
# handle, comes against the forearm that holds it. So the door is then pushed on from inside: the closed hand goes
# into the gap between the leaf and the bay's front from the west (fingers along the leaf towards the hinge, 3.5 cm
# off its inner face, PUSH_FROM deg), and is led round the hinge line to PUSH_TO deg: wherever the leaf stands, the
# hand meets its inner face and pushes it ahead; the forearm stays outside the leaf's sweep.
PUSH_STANCE = (8.55, -.85, -1.047)  # offline (push_search.py, push_plan.py): the way into the gap and the lead to 92 deg are found from here
PUSH_ARM = 'right'                  # (the left arm, on the leaf's side, knocked the door from 54 to 14 deg on its way in, dt6)
PUSH_FROM, PUSH_TO = np.radians(16.), np.radians(90.)
PUSH_RADIUS, PUSH_HEIGHT, PUSH_OFF = .50, 1.05, .035


def push_start(theta=PUSH_FROM):
    u = np.array([-np.cos(theta), np.sin(theta), 0.]); n = np.array([-np.sin(theta), -np.cos(theta), 0.])
    p = HINGE + PUSH_RADIUS * u + PUSH_OFF * n; p[2] = PUSH_HEIGHT
    return p, hinge.hand_frame(-u, np.array([0., 0., 1.]))


LEAF = .62


def leaf_boxes(scene, angles, tag):
    """Put the leaf, standing at each of @angles (rad), into the map as small boxes (replacing those of @tag)."""
    keep = [b for b, t in zip(scene.boxes, getattr(scene, 'box_tags', [None] * len(scene.boxes))) if t != tag]
    tags = [t for t in getattr(scene, 'box_tags', [None] * len(scene.boxes)) if t != tag]
    for th in angles:
        u = np.array([-np.cos(th), np.sin(th)])
        for k in range(6):
            a = HINGE[:2] + u * (LEAF * k / 6); b = HINGE[:2] + u * (LEAF * (k + 1) / 6)
            lo = np.minimum(a, b) - .025; hi = np.maximum(a, b) + .025
            keep.append(np.array([lo[0], lo[1], .12, hi[0], hi[1], 2.09])); tags.append(tag)
    scene.boxes = np.array(keep).reshape(-1, 6); scene.box_tags = tags


def push_open(ctx, est, go=None, led=None):
    """Push the leaf open from inside (see above). @led: the angle the hand led it to before (rad): the leaf is kept
    out of the way in to the gap. Returns the angle the hand was led to (rad)."""
    yield from tuck(ctx)
    if go is not None:
        status = yield from go(STAGING, PUSH_STANCE[2])
        if status != 'arrived':
            ctx.event('door_push', ok=False, reason='no way', status=status); return 0.
    yield from drive(ctx, PUSH_STANCE[:2], PUSH_STANCE[2], speed=.15, direct=True, tolerance=.012, yaw_tolerance=.02, min_speed=.03, min_turn=.05, face_travel=False, max_steps=400)
    yield from hold(ctx, 4)
    drop_door_boxes(ctx.scene)
    if led is not None and led > PUSH_FROM + np.radians(12.):
        leaf_boxes(ctx.scene, np.radians(np.arange(-14., 6., 4.)) + led, 'leaf')
    ctx.arm = PUSH_ARM; p, rot_map = push_start()
    pb, rb = hinge.to_base(est, p, rot_map); ahead = -rb[:, 2]
    check = ctx.validator(hand_margin=-.03); plan = None
    for load, rotation, low in least_loaded(ctx, ctx.q, pb, [rb], check, seeds=2)[:12]:
        start = ctx.q.copy(); start[ctx.ik.columns] = low
        back = interpolate_cartesian(ctx.ik, start, pb, pb - .10 * ahead, rotation, validator=check, step=.02)
        if back is not None:
            plan = dict(pre=back[-1], way=back[::-1][1:] + [low]); break
    if plan is None:
        ctx.event('door_push', ok=False, reason='no plan'); return 0.
    ctx.set_grip(-1.)
    status = yield from move_to(ctx, plan['pre'], rate=.04, label='push_pre')
    if status != 'done':
        ctx.event('door_push', ok=False, reason='no way to the gap', status=status); return 0.
    yield from follow(ctx, plan['way'], rate=.012, max_steps=250)
    saved = hinge.HELD; hinge.HELD = -1.          # (the closed hand holds nothing: the lead goes on regardless)
    try:
        got = yield from hinge.lead(ctx, est, HINGE, AXIS, PUSH_TO - PUSH_FROM)
    finally:
        hinge.HELD = saved
    yield from hold(ctx, 20)
    # out of the gap the way the hand came in (back along the leaf), then fold the arm
    tip = ctx.tip()[:3, 3]; rotation = ctx.tip()[:3, :3]
    way = interpolate_cartesian(ctx.ik, ctx.q, tip, tip + .18 * rotation[:, 2], rotation, step=.025)
    if way is not None:
        yield from follow(ctx, way, rate=.03, max_steps=150)
    ctx.set_grip(1.)
    yield from tuck(ctx, (PUSH_ARM,))
    reached = PUSH_FROM + got
    if reached > np.radians(60.):
        leaf_boxes(ctx.scene, [reached - np.radians(6.), reached], 'leaf')
    else:
        leaf_boxes(ctx.scene, [], 'leaf')
    ctx.event('door_push', ok=reached > np.radians(70.), led_to=round(float(np.degrees(reached)), 1))
    est.pantry = max(getattr(est, 'pantry', 0.), reached)
    return reached
