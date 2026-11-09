"""The plates of putting_dishes_away_after_cleaning: taken by the rim from the bar or the countertop, carried level and
set down on a shelf of the pantry's east bay (bottom_cabinet_fancyy_0, cabinet.n.01_2).

Facts (dev probe 301, house_single_floor):
- 8 plates, 0.24-0.31 m across, 2.3-3.4 cm high; their rim rises from ~1.5 cm (at 0.8 R) to the full height at the edge.
  Four stand on the bar (bar_udatjt_0, top 1.052, x 5.79-8.79, y -0.305..0.848; they stand along its south side), four
  on the countertop by the south wall (countertop_kelzer_0, top 0.888, x 4.48-7.53, y -2.36..-1.585).
- Inside (oracle, dev job inside_test): a plate on the east bay's shelves at 0.849, 1.162 or 1.479 is inside the pantry,
  also several in a jumbled pile; one whose middle is at the bay's front edge (y -1.62) is not.
Frames: map (M) = the scene's; base frame = the robot base's.
"""
import numpy as np
from manip import hold, follow, move_to, pose2, BASE_Z, TALL_TRUNK
from skills import interpolate_cartesian
from trash_skills import tuck

BAR = dict(x=(5.80, 8.78), y=(-.305, .848), top=1.052, face=-.305, yaw=np.pi / 2)        # base faces north at the bar
COUNTER = dict(x=(4.49, 7.52), y=(-2.36, -1.585), top=.888, face=-1.585, yaw=-np.pi / 2)  # ... and south at the countertop
SURFACES = dict(bar=BAR, counter=COUNTER)
GRIP = (.018, .010)         # fingertips' centre this high above the surface, this far in from the plate's edge
PITCHES = (90., 78.)        # fingers down (deg from level); the rim between them, closing along the way to the plate's middle
CLOSE_STEPS = 45
LIFT = .08
HELD = .002                 # fingers further apart than this (sum, m) hold a plate

# The pantry's east bay (inside x 8.76-9.32, y -2.17..-1.63); shelves at 0.849, 1.162, 1.479.
TIP_X = (8.98, 8.93)        # where the fingers hold the plate's rim when it is let go (the bay's west half: the open door
TIP_Y = -1.56               # stands before its east half), 4 cm in front of the bay (the plate's middle 0.11-0.145 m in)
HALF_HEIGHT = .015
SHELVES = (1.162, .849)
CLEAR = .03                 # a plate's bottom this far above the shelf (or the pile on it) when it is let go
PILE = .018                 # a pile grows by about this per plate


def frame(approach, closing):
    return np.column_stack([np.cross(closing, -approach), closing, -approach])


def edge_rotation(heading, pitch):
    """Fingers pointing along @heading and down by @pitch (rad), closing in the upright plane through @heading."""
    approach = np.array([np.cos(heading) * np.cos(pitch), np.sin(heading) * np.cos(pitch), -np.sin(pitch)])
    closing = np.array([np.cos(heading) * np.sin(pitch), np.sin(heading) * np.sin(pitch), np.cos(pitch)])
    return frame(approach, closing)


def in_base(base, xy):
    x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = np.asarray(xy, float)[:2] - [x, y]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


def to_map(base, local):
    x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw)
    return np.array([x + c * local[0] - s * local[1], y + s * local[0] + c * local[1]])


def rim_grip(base, centre, radius, top, pitch):
    """Fingers' ends and rotation (base frame) on the near rim of a plate whose middle is at @centre (map)."""
    local = in_base(base, centre); heading = float(np.arctan2(local[1], local[0]))
    xy = local - (radius - GRIP[1]) * np.array([np.cos(heading), np.sin(heading)])
    return np.r_[xy, top + GRIP[0] - BASE_Z], edge_rotation(heading, np.radians(pitch))


def plan_pinch(ctx, point, rotation, base=None, way_in=.10, count=24, hand_margin=-.03):
    full = ctx.q; point = np.asarray(point, float)
    check = ctx.validator(base=base, hand_margin=hand_margin)
    approach = -rotation[:, 2]
    for grip in ctx.ik.solve(full, point, rotation, count=count, validator=check, all_solutions=True)[:6]:
        at = full.copy(); at[ctx.ik.columns] = grip
        out = interpolate_cartesian(ctx.ik, at, point, point - way_in * approach, rotation, validator=check, step=.025)
        if out is not None:
            return dict(grip=grip, pre=out[-1], inward=out[::-1][1:] + [grip], point=point, rotation=rotation)
    return None


def pinch(ctx, plan, label='plate', lift=LIFT):
    """Close the active hand as @plan has it and lift straight up. Returns True if it holds, None if no way there."""
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
    up = interpolate_cartesian(ctx.ik, ctx.q, reached, reached + [0., 0., lift], plan['rotation'], step=.02)
    if up is not None:
        yield from follow(ctx, up, rate=.012, max_steps=300)
    yield from hold(ctx, 6)
    ok = float(np.sum(ctx.fingers())) > HELD
    ctx.event(label, ok=ok, arm=ctx.arm, way_in=status, lifted=up is not None, fingers=fingers.round(4).tolist(), after_lift=ctx.fingers().round(4).tolist(),
              short=(reached - plan['point']).round(3).tolist(), point=np.round(plan['point'], 3).tolist())
    if not ok:
        ctx.set_grip(1.)
    return ok


def carry(ctx, arm):
    """Draw the held plate in, level, to where it is held on the way (the rotation kept)."""
    side = 1. if arm == 'left' else -1.; ctx.arm = arm; rotation = ctx.tip()[:3, :3]
    free = ctx.validator(relax_from=ctx.q)
    for x, y, z in ((.40, .12, 1.12), (.45, .10, 1.15), (.38, .18, 1.08), (.50, .10, 1.20), (.45, .20, 1.05)):
        goal = ctx.ik.solve(ctx.q, np.array([x, side * y, z - BASE_Z]), rotation, count=30, validator=free)
        if goal is not None:
            status = yield from move_to(ctx, goal, rate=.012, label='carry')
            if status == 'done':
                return status
    ctx.event('carry', ok=False, reason='no level pose up and in: left as it is', arm=arm)
    return 'kept'


def holds(ctx, arm):
    return float(np.sum(ctx.fingers(arm))) > HELD


def place_plan(ctx, base, held, shelf_z, count=24, rot_now=None, tips=None):
    """Arm configuration that holds the plate over a shelf of the east bay with the fingers just outside the bay's
    front (TIP_Y; further in the wrist meets the shelf above) and the plate south of them, its bottom CLEAR over
    @shelf_z (the shelf or the pile on it); the hand as it holds the plate now, turned about the vertical. held: the
    plate's middle in the tool frame. Returns (configuration, rotation, tip) or None."""
    rot_now = ctx.tip()[:3, :3] if rot_now is None else rot_now; offset_now = rot_now @ held
    for tx in (tips or TIP_X):
        for into in np.radians([-90., -80., -100.]):
            want = into - base[2]; have = float(np.arctan2(offset_now[1], offset_now[0])); turn = want - have
            c, s = np.cos(turn), np.sin(turn); Rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            rot = Rz @ rot_now; off = Rz @ offset_now
            tip = np.r_[in_base(base, (tx, TIP_Y)), shelf_z + CLEAR + HALF_HEIGHT - off[2] - BASE_Z]
            sol = ctx.ik.solve(ctx.q, tip, rot, count=count, validator=ctx.validator(base=pose2(*base)))
            if sol is not None:
                return sol, rot, tip
    return None


OUTSIDE = .30               # the plate is brought in front of the bay this far north of its place, then slid in level


def place(ctx, est, arm, held, shelf_z, push=None):
    """Set the plate in @arm down on a shelf of the east bay: to a pose in front of the bay at the right height, slid
    straight in, let go, the hand lifted and drawn back out. Returns True if it was let go in its place."""
    ctx.arm = arm; base = est.base()
    found = place_plan(ctx, base, held, shelf_z)
    if found is None:
        ctx.event('place', ok=False, reason='no plan', shelf=shelf_z); return False
    sol, rot, tip = found
    outward = np.linalg.inv(pose2(*base))[:3, :3] @ np.array([0., 1., 0.])
    for out in (OUTSIDE, .22, .15):
        pre = tip + out * outward
        pre_sol = ctx.ik.solve(ctx.q, pre, rot, count=24, validator=ctx.validator())
        if pre_sol is None:
            continue
        full = ctx.q.copy(); full[ctx.ik.columns] = pre_sol
        way = interpolate_cartesian(ctx.ik, full, pre, tip, rot, validator=ctx.validator(), step=.02)
        if way is not None:
            break
    else:
        ctx.event('place', ok=False, reason='no way in', shelf=shelf_z); return False
    status = yield from move_to(ctx, pre_sol, rate=.02, label='place_pre', max_steps=500)
    if status != 'done':
        ctx.event('place', ok=False, reason='no way to the front', status=status); return False
    yield from hold(ctx, 4)
    status = yield from follow(ctx, way, rate=.008, max_steps=400)
    reached = ctx.tip()[:3, 3].copy()
    yield from hold(ctx, 6)
    ctx.set_grip(1.)
    yield from hold(ctx, 25)
    here = ctx.tip()[:3, 3]
    up = interpolate_cartesian(ctx.ik, ctx.q, here, here + [0., 0., .03], rot, step=.015)
    if up is not None:
        yield from follow(ctx, up, rate=.02, max_steps=60)
    here = ctx.tip()[:3, 3]
    back = interpolate_cartesian(ctx.ik, ctx.q, here, here + .25 * outward, rot, step=.03)
    if back is not None:
        yield from follow(ctx, back, rate=.03, max_steps=200)
    pushed = None
    if PUSH:
        pushed = yield from push_in(ctx, est, arm, shelf_z, tip, depths=push)
    yield from tuck(ctx, (arm,))
    ctx.event('place', ok=True, way_in=status, short=np.round(reached - tip, 3).tolist(), out=out, shelf=round(float(shelf_z), 3), pushed=pushed)
    return True


PUSH = True                 # after letting go, the closed hand pushes the plate further in by its near rim
PUSH_FROM, PUSH_TO = .05, -.07      # (m along map y from where the fingers held the rim) the push's start and end
PUSH_PITCH = 45.


def push_in(ctx, est, arm, shelf_z, held_at, depths=None):
    """The closed hand, fingers pointing into the bay and down by PUSH_PITCH, pushes the plate's near rim in along the
    bay's depth. @held_at: where the fingers held the rim when it was let go (base frame). Returns the status."""
    ctx.arm = arm; base = est.base(); inv = np.linalg.inv(pose2(*base))
    into = inv[:3, :3] @ np.array([0., -1., 0.]); heading = float(np.arctan2(into[1], into[0]))
    rot = edge_rotation(heading, np.radians(PUSH_PITCH))
    z = shelf_z + .012 - BASE_Z
    start = np.r_[held_at[:2] - PUSH_FROM * into[:2], z]
    check = ctx.validator(hand_margin=-.03)
    sol = ctx.ik.solve(ctx.q, start + [0., 0., .06], rot, count=24, validator=check)
    if sol is None:
        ctx.event('push', ok=False, reason='no pose'); return 'no_pose'
    full = ctx.q.copy(); full[ctx.ik.columns] = sol
    down = interpolate_cartesian(ctx.ik, full, start + [0., 0., .06], start, rot, validator=check, step=.02)
    if down is None:
        ctx.event('push', ok=False, reason='no way down'); return 'no_way'
    full[ctx.ik.columns] = down[-1]
    way = None
    for depth in (depths or (-PUSH_TO,)):
        end = np.r_[held_at[:2] + depth * into[:2], z]
        way = interpolate_cartesian(ctx.ik, full, start, end, rot, validator=check, step=.02)
        if way is not None:
            break
    if way is None:
        ctx.event('push', ok=False, reason='no way in'); return 'no_way'
    ctx.set_grip(-1.)
    status = yield from move_to(ctx, sol, rate=.03, label='push_pre', max_steps=300)
    if status != 'done':
        ctx.set_grip(1.); return status
    yield from follow(ctx, down, rate=.015, max_steps=150)
    status = yield from follow(ctx, way, rate=.008, max_steps=300)
    here = ctx.tip()[:3, 3]
    back = interpolate_cartesian(ctx.ik, ctx.q, here, here + [0., 0., .04] - .15 * into, rot, step=.03)
    if back is not None:
        yield from follow(ctx, back, rate=.03, max_steps=150)
    ctx.set_grip(1.)
    ctx.event('push', ok=True, status=status, depth=depth)
    return status
