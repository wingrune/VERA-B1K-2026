"""The top drawer of the cabinet by the living room's west wall (bottom_cabinet_bamfsz_1, cabinet.n.01_2), pulled
open so that the batteries can be let fall into it.

The goal counts a battery as inside the cabinet when it lies in one of its drawers that stands open (dev job
drawer301: a battery let go into each of three drawers opened by 0.12-0.20 m was inside). The top drawer runs out
along +x from its front at x -1.529 (map), 0.36 m at most. Its handle is a bar 0.40 m wide (y -0.655 to -0.259),
1.6 cm high (0.672-0.688 m up), standing 1.8 cm off the front. The hand comes at it level from the east, its
fingers closing up and down across the bar; the base then drives straight back and the drawer comes with the hand.
"""
import numpy as np
from manip import hold, follow, move_to, drive, pose2, BASE_Z, BASE_LINEAR_GAIN
from skills import interpolate_cartesian
from trash_skills import tuck, least_loaded

FACE_X = -1.529                       # the drawer's front, shut (map x)
HANDLE = (-1.549, -.37, .698)         # where the fingertips' centre is led to (map): 11 cm in from the bar's north end, 1.8 cm over its middle (aimed at the middle, the hand closed 1.5-2 cm under it, 301, 302), 2 cm behind the drawer's front: the hand stops at the front with the bar between the fingers (aimed at the bar's middle, from a base 2 cm further back than the map pose had it, the fingers closed in front of the bar, 302)
STANCE = (-.93, -.22, np.pi)          # the base facing west, the handle 0.59 m ahead and 0.15 m to its left (the coffee table's corner at (-0.79, -0.60) is 3 cm off the chassis)
STAGING = (-.70, -.22)                # the base comes straight in from here
WAY_IN = .10                          # the hand comes the last this far straight at the drawer
PULL = .22                            # the base drives back this far holding the handle: the drawer opens as much
HELD = .006                           # fingers further apart than this (sum of both, m) after closing hold the bar (1.6 cm high)
CENTRED = .008                        # fingers that stand closer than this to each other's opening hold the bar in the hand's middle
HAND_INTO = -.11                      # the hand may come this far into the cabinet's outline in the map (the fingers round the bar)
OPEN_PART = (-1.55, -.60, -.33)       # the drawer's open part: x from the cabinet's front frame (and up to the drawer's front less 4 cm), y between its sides less 4 cm
TOP = .725                            # the drawer's top edge (m)


def level_rotation(azimuth, up=True):
    """Tool rotation (base frame) with the fingers level, pointing along @azimuth, closing up and down."""
    z = -np.array([np.cos(azimuth), np.sin(azimuth), 0.]); y = np.array([0., 0., 1. if up else -1.])
    return np.column_stack([np.cross(y, z), y, z])


def open_part(opened):
    """The middle of the drawer's open part (map xy) when it stands @opened (m) out."""
    x1 = FACE_X + opened - .04
    return np.array([(OPEN_PART[0] + x1) / 2., (OPEN_PART[1] + OPEN_PART[2]) / 2.])


def guard_box(opened):
    """The open drawer as a box (map) for the validator: its front and sides stand out of the cabinet."""
    return np.array([[-1.58, -.69, .60, FACE_X + opened + .02, -.23, TOP + .01]])


def grasp_handle(ctx, est, arm, opened=0.):
    """The hand @arm takes the handle (the drawer standing @opened out) from where the base stands. Returns True if it
    holds it, False if the fingers closed on nothing, None if the hand did not get there."""
    ctx.arm = arm; base = est.base(); world = pose2(*base); inv = np.linalg.inv(world)
    point = (inv @ np.r_[HANDLE[0] + opened, HANDLE[1], HANDLE[2] - BASE_Z, 1.])[:3]
    west = inv[:2, :2] @ np.array([-1., 0.]); azimuth = float(np.arctan2(west[1], west[0]))     # (west, in the base frame)
    ahead = np.array([np.cos(azimuth), np.sin(azimuth), 0.])
    # (the lower finger, 5 cm below the bar, comes within the cabinet's outline, which the map's height field fills
    # up to its top: 7.2 cm under it, 301)
    check = ctx.validator(hand_margin=HAND_INTO)
    # (the pose at the handle first, then the way back out from it, as the floor pinch plans: planned the other way,
    # from a pose 10 cm out, no straight way in was found in the simulator, 301 and 302)
    found = least_loaded(ctx, ctx.q, point, [level_rotation(azimuth, True), level_rotation(azimuth, False)], check, seeds=2)
    plan = None
    for load, rotation, low in found[:12]:
        start = ctx.q.copy(); start[ctx.ik.columns] = low
        back = interpolate_cartesian(ctx.ik, start, point, point - WAY_IN * ahead, rotation, validator=check, step=.02)
        if back is not None:
            plan = dict(pre=back[-1], way=back[::-1][1:] + [low], rotation=rotation, load=float(load)); break
    if plan is None:
        ctx.event('drawer_grasp', ok=False, reason='no way to the handle', arm=arm, solutions=len(found), s=np.round(ctx.s, 4).tolist(), base=[round(float(v), 4) for v in base]); return None
    ctx.set_grip(1.)
    status = yield from move_to(ctx, plan['pre'], rate=.04, label='drawer_pre')
    if status != 'done':
        return None
    yield from hold(ctx, 3)
    status = yield from follow(ctx, plan['way'], rate=.012, max_steps=200)      # (it stalls at the drawer's front)
    reached = ctx.tip()[:3, 3].copy()
    yield ctx.action(chain=ctx.chain)         # (stopped at the front: the arm holds where it is instead of pressing on)
    ctx.set_grip(-1.)
    yield from hold(ctx, 25)
    centred = []
    for fix in range(2):
        # The fingers closed with the bar off the hand's middle (one stood 2.5 cm out, the other closed right in, the
        # hand 1.5 cm below the bar: the pull then slipped off it after 6-8 cm, 301, 302): open, move the hand by half
        # the difference toward the finger that stopped further out (finger 1 opens along the tool's y), close again.
        f = ctx.fingers().copy()
        if float(np.sum(f)) <= HELD or abs(float(f[0] - f[1])) < CENTRED:
            break
        shift = .5 * float(f[0] - f[1]) * plan['rotation'][:, 1]
        ctx.set_grip(1.)
        yield from hold(ctx, 14)
        tip = ctx.tip()[:3, 3]
        way = interpolate_cartesian(ctx.ik, ctx.q, tip, tip + shift, plan['rotation'], validator=check, step=.01)
        if way is not None:
            yield from follow(ctx, way, rate=.008, max_steps=80)
        ctx.set_grip(-1.)
        yield from hold(ctx, 25)
        centred.append([np.round(f, 4).tolist(), np.round(shift, 3).tolist(), way is not None])
    fingers = ctx.fingers().copy(); held = float(np.sum(fingers)) > HELD
    ctx.event('drawer_grasp', ok=held, arm=arm, way=status, load=round(plan['load'], 2), fingers=fingers.round(4).tolist(), short=(reached - point).round(3).tolist(),
              up=bool(plan['rotation'][2, 1] > 0), centred=centred, opened=round(float(opened), 3))
    ctx.drawer_plan = plan
    return held


def pull(ctx, est, distance=PULL, speed=.05):
    """The base backs straight away from the cabinet with the hand on the handle, until it has come @distance or the
    fingers slip off the bar (then it stops at once: the drawer stays where it was let go). Returns how far the base
    came with the fingers still on the bar (m): as far as the drawer came."""
    x, y, yaw = est.base(); start = np.array([x, y]); back = -np.array([np.cos(yaw), np.sin(yaw)])
    moved = 0.; slipped = False; command = 0.; trace = []
    for k in range(int(distance / speed * 30 * 2.5) + 30):
        moved = float(np.dot(np.asarray(est.base()[:2], float) - start, back))
        if float(np.sum(ctx.fingers())) <= HELD:
            slipped = True; break
        if moved >= distance:
            break
        command = min(speed / BASE_LINEAR_GAIN, command + .01)
        if k % 15 == 0:
            trace.append((round(moved, 3), np.round(ctx.fingers(), 4).tolist()))
        yield ctx.action(base=(-command, 0., 0.))
    for _ in range(12):
        yield ctx.action(base=(0., 0., 0.))
    ctx.event('drawer_pull', moved=round(moved, 3), slipped=slipped, fingers=ctx.fingers().round(4).tolist(), trace=trace[-8:])
    return max(0., moved)


def let_go(ctx):
    """Open the hand and draw it back out of the handle, then fold the arm."""
    ctx.set_grip(1.)
    yield from hold(ctx, 15)
    tip = ctx.tip()[:3, 3]; rotation = ctx.tip()[:3, :3]
    back = interpolate_cartesian(ctx.ik, ctx.q, tip, tip + .10 * rotation[:, 2], rotation, step=.025)
    if back is not None:
        yield from follow(ctx, back, rate=.03, max_steps=120)
    yield from tuck(ctx, (ctx.arm,))


def open_top_drawer(ctx, est, go=None, tries=4):
    """Come to the stance in front of the cabinet, take the handle with the left hand and pull the drawer out; where
    the fingers slip off the bar, take it again where it now is and pull on. go: generator function(xy, yaw) that
    brings the base to a point (the pipeline's route around what is known); without it the base drives there
    directly. Returns how far the drawer stands open (m, 0 if it did not open)."""
    opened = 0.; misses = 0
    yield from tuck(ctx)                # (with the arms as the start left them, no straight way in to the handle was found)
    for attempt in range(tries):
        if attempt == 0:
            if go is not None:
                status = yield from go(STAGING, STANCE[2])
            else:
                status = yield from drive(ctx, STAGING, STANCE[2], speed=.3, tolerance=.04)
            if status != 'arrived':
                ctx.event('drawer', ok=False, reason='no way to the cabinet', status=status); return 0.
        # The base stands as far back from the stance as the drawer is out: the handle is where it was the first time.
        stance = np.array(STANCE[:2]) + [opened, 0.]
        yield from drive(ctx, stance, STANCE[2], speed=.15, direct=True, tolerance=.012, yaw_tolerance=.02, min_speed=.03, min_turn=.05, face_travel=False, max_steps=400)
        yield from hold(ctx, 4)
        held = yield from grasp_handle(ctx, est, 'left', opened)
        got = 0.
        if held:
            got = yield from pull(ctx, est, PULL - opened)
            opened += got
        else:
            misses += 1
        yield from let_go(ctx)
        ctx.event('drawer', ok=opened > .08, attempt=attempt, held=held, got=round(got, 3), opened=round(opened, 3))
        if opened >= PULL - .03 or misses >= 2:
            break
    yield from drive(ctx, STAGING, STANCE[2], speed=.2, direct=True, tolerance=.05, face_travel=False, max_steps=300)
    est.drawer = opened
    return opened
