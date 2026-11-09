"""Skills of the battery task: reach a battery standing on a table-height surface, grasp it, carry it
folded in front of the chest, and let it fall into the trash can.

A battery is a standing cylinder (3.4 cm across, 6.1 cm tall). The gripper closes horizontally around it
and may approach anywhere between straight down and level: the far half of a deep surface is reached with
the wrist nearly level. Frames as in manip.py; points given to the skills are in the base frame unless
their name says otherwise.
"""
import numpy as np
from manip import (hold, follow, drive, move_to, pose2, wrap, ARMS, TALL_TRUNK, TUCK, MAX_LOAD)
from skills import interpolate_cartesian

GRASP_HEIGHT = .033                 # fingertip centre above the surface a battery stands on (its centre is at 0.031)
PITCHES = (0., .7, 1.2)             # wrist pitch from straight down (rad), tried in this order
APPROACH = .10                      # length of the straight final approach along the tool axis
DEEPER = .008                       # a tilted approach aims this far past the battery's axis: stopping short pushes it away
WRIST_AGREES = .025                 # a tilted approach keeps its target when the wrist camera agrees with it this well (m)
CLOSE_UP_PITCH = .3                 # the wrist camera's correction is used only for approaches steeper than this
HELD_GAP = .012                     # fingers further apart than this (sum of both, m) after closing hold something
CAN_HALF = np.array([.208, .163])   # half extents of the trash can's footprint
CAN_TOP = .452                      # height of its rim
GUARD = (.025, .065)                # half width and height of the box kept clear around a standing battery while an arm travels
DROP_HEIGHT = .15                   # fingertip centre above the rim when the battery is let go


def reach_rotation(azimuth, pitch):
    """Tool rotation (base frame) approaching along @azimuth at @pitch from straight down, closing horizontally."""
    approach = np.array([np.sin(pitch) * np.cos(azimuth), np.sin(pitch) * np.sin(azimuth), -np.cos(pitch)])
    y = np.array([-np.sin(azimuth), np.cos(azimuth), 0.]); z = -approach
    return np.column_stack([np.cross(y, z), y, z]), approach


def guard_boxes(points, height):
    """Boxes (M) around batteries standing at @points (xy) on a surface @height high. A battery weighs 75 g and
    falls over at a touch; lying down it cannot be pinched. The arm's way to and from a grasp is planned
    against the furniture only, so the batteries get boxes of their own."""
    return np.array([[x - GUARD[0], y - GUARD[0], height, x + GUARD[0], y + GUARD[0], height + GUARD[1]] for x, y in points], float).reshape(-1, 6)


def plan_reach(ctx, point, base=None, q=None, count=16, pitches=PITCHES, max_load=MAX_LOAD, approach=APPROACH, static_only=False, square=False):
    """A straight approach of the active arm to @point (base frame) for a horizontal pinch.

    The contact pose is solved first and the way back along the tool axis is followed from it, so both ends
    belong to one arm configuration. Returns a dict (pre, down, rotation, pitch, solutions, load) or None.

    The hand normally comes in along the line from the robot to the battery and closes across it. square: first
    try coming in straight ahead instead, the fingers closing along the edge of the surface the robot faces.
    A battery that stands at that edge (many do: 1 cm from it) is then pushed along the edge by the finger that
    touches first, not over it: closing at an angle, a battery 0.28 m to the side went to the floor.
    """
    full = ctx.q if q is None else np.asarray(q, float); point = np.asarray(point, float)
    check = ctx.validator(base=base, max_load=max_load, static_only=static_only)
    radial = float(np.arctan2(point[1], point[0]))
    for pitch in pitches:
        for azimuth in ((0., radial) if square and abs(radial) > .05 else (radial,)):
            rotation, direction = reach_rotation(azimuth, pitch)
            target = point + (DEEPER * np.array([np.cos(azimuth), np.sin(azimuth), 0.]) if pitch > CLOSE_UP_PITCH else 0.)
            lows = ctx.ik.solve(full, target, rotation, count=count, validator=check, all_solutions=True)
            for low in lows[:4]:
                start = full.copy(); start[ctx.ik.columns] = low
                back = interpolate_cartesian(ctx.ik, start, target, target - approach * direction, rotation, validator=check, step=.02)
                if back is None:
                    continue
                return dict(pre=back[-1], down=back[::-1][1:] + [low], rotation=rotation, pitch=float(pitch), azimuth=float(azimuth), solutions=len(lows), load=ctx.load(start)[0], target=target)
    return None


def tucked(ctx, q=None):
    """The full configuration @q (default: current) with the trunk upright and both arms folded."""
    full = (ctx.q if q is None else np.asarray(q, float)).copy()
    full[ctx.iks['left'].columns[:4]] = TALL_TRUNK
    for arm in ARMS:
        full[ctx.iks[arm].columns[4:]] = TUCK[arm]
    return full


def tuck(ctx, arms=('left', 'right'), static_only=False):
    """Bring the trunk upright and fold the given arms in front of the chest (a held battery stays in the hand)."""
    previous = ctx.arm; result = 'done'
    for arm in arms:
        ctx.arm = arm
        status = yield from move_to(ctx, np.r_[TALL_TRUNK, TUCK[arm]], label='tuck', static_only=static_only)
        if status != 'done':
            result = status
    ctx.arm = previous
    return result


def surface_stances(ctx, target, surface, standoffs=(.30, .22, .38), laterals=(0., .15, -.15, .28, -.28, .42, -.42, .55, -.55, .65, -.65)):
    """Base poses (x, y, yaw) in M from which @target (world xy) on @surface might be reached, nearest side first.

    surface: dict(lo, hi) = footprint in M, sides = accessible sides among 'south', 'north'.
    """
    lo, hi = np.asarray(surface['lo'], float), np.asarray(surface['hi'], float); out = []
    for side in surface['sides']:
        edge, normal, yaw = (lo[1], -1., np.pi / 2) if side == 'south' else (hi[1], 1., -np.pi / 2)
        for standoff in standoffs:
            for lateral in laterals:
                out.append((side, float(target[0] + lateral), float(edge + normal * standoff), yaw))
    return out


EDGE_NEAR = .07                     # a battery this close to the edge the robot faces is pinched along the edge (m)


def at_edge(surface, target, side):
    """Whether the battery at @target (map xy) stands at the edge of @surface that a robot on @side faces."""
    depth = float(target[1] - surface['lo'][1]) if side == 'south' else float(surface['hi'][1] - target[1])
    return depth < EDGE_NEAR


def plan_surface_stance(ctx, target_world, surface, usable=None, limit=72, arms=('left', 'right'), static_only=False, max_load=MAX_LOAD):
    """(side, base pose, arm, reach) for the battery at @target_world (x, y, z of its centre), or None.

    The stance is checked the way it will be used: the robot arrives folded (trunk upright), so that posture
    must fit there, and the reach is planned from it. Among the stances that work, one from which the arm
    reaches on its own side is preferred to one from which it reaches across the body (those were planned
    and then refused at the desk), then the one whose reach is found by the most arm configurations.
    """
    target = np.asarray(target_world, float); nominal = tucked(ctx); best = None; tried = 0; previous = ctx.arm

    def quality(stance):
        own_side = float(stance['point'][1]) * (1. if stance['arm'] == 'left' else -1.) > -.10
        return own_side, stance['reach']['solutions']
    try:
        for side, x, y, yaw in surface_stances(ctx, target, surface):
            if tried >= limit or (best is not None and quality(best)[0] and best['reach']['solutions'] >= 8):
                break
            if usable is not None and not usable(side, x, y, yaw):
                continue
            world = pose2(x, y, yaw); tried += 1
            if not ctx.validator(base=world, static_only=True)(nominal, ctx.body.fk(nominal)[0]):
                continue
            point = (np.linalg.inv(world) @ np.r_[target[:2], surface['height'] + GRASP_HEIGHT, 1.])[:3]
            square = at_edge(surface, target, side)
            for arm in (('left', 'right') if point[1] >= 0 else ('right', 'left')):
                if arm not in arms:
                    continue
                ctx.arm = arm
                reach = plan_reach(ctx, point, base=world, q=nominal, count=12, static_only=static_only, max_load=max_load, square=square)
                if reach is not None:
                    found = dict(side=side, base=(x, y, yaw), arm=arm, reach=reach, point=point, square=square)
                    if best is None or quality(found) > quality(best):
                        best = found
                    break
    finally:
        ctx.arm = previous
    ctx.event('stance', ok=best is not None, tried=tried, **({} if best is None else dict(side=best['side'], base=[round(v, 3) for v in best['base']], arm=best['arm'],
              point=best['point'].round(3).tolist(), pitch=round(best['reach']['pitch'], 2), solutions=best['reach']['solutions'], load=round(best['reach']['load'], 2))))
    return best


def grasp(ctx, point, static_only=False, refine=None, square=False):
    """Pinch the battery whose centre is at @point (base frame) with the active arm and lift it clear.

    refine: optional callable(point) -> corrected point or None, asked once the hand hangs in front of the
    battery (the wrist camera then looks straight at it). Returns True if the fingers stayed apart after
    closing (they close completely on nothing).
    """
    point = np.asarray(point, float)
    # ctx.guard holds a box for every battery standing here. The one to be taken is avoided on the way to the
    # point above it and of course not in the final approach; the others throughout.
    guard = ctx.guard; here = (pose2(*ctx.est.base()) @ np.r_[point[:2], 0., 1.])[:2]
    others = np.array([box for box in guard if np.linalg.norm((box[:2] + box[3:5]) / 2 - here) > .04]).reshape(-1, 6)
    reach = None
    for boxes in ((others, np.zeros((0, 6))) if len(others) else (others,)):       # with a neighbour in the way: rather touch it than not try
        ctx.guard = boxes
        reach = plan_reach(ctx, point, static_only=static_only, square=square)
        if reach is None:           # nothing within the usual load limit: a heavier lean rather than no attempt
            reach = plan_reach(ctx, point, static_only=static_only, max_load=1.05, square=square)
        if reach is not None:
            others = boxes; break
    if reach is None:
        ctx.guard = others
        ctx.event('grasp', ok=False, reason='no straight approach', arm=ctx.arm, point=point.round(3).tolist(), proprio=np.round(ctx.s, 4).tolist()); return False
    ctx.set_grip(1.)
    for boxes in ((guard, others) if len(guard) > len(others) else (others,)):
        ctx.guard = boxes
        status = yield from move_to(ctx, reach['pre'], rate=.035, label='grasp', max_load=1.15, static_only=static_only)
        if status != 'no_path':
            break
    ctx.guard = others
    if status != 'done':
        return False
    yield from hold(ctx, 3)
    down = reach['down']; shift = None
    # With the hand above (or, for a tilted approach, in front of) the battery the wrist camera looks straight at
    # it. A steep approach always takes its reading. A tilted one takes it only when it disagrees with the
    # target by more than its own error: then the target is wrong (carried by odometry past a pushed chair it
    # was 10 cm off, and the hand closed beside the battery four times).
    better = refine(point) if refine is not None else None
    if better is not None and reach['pitch'] > CLOSE_UP_PITCH and np.linalg.norm(np.asarray(better, float)[:2] - point[:2]) < WRIST_AGREES:
        better = None
    if better is not None and np.linalg.norm(np.asarray(better, float)[:2] - point[:2]) < .16:
        shift = (np.asarray(better, float)[:2] - point[:2]).round(4).tolist(); point = np.r_[np.asarray(better, float)[:2], point[2]]
        aim = point + (DEEPER * np.array([np.cos(reach['azimuth']), np.sin(reach['azimuth']), 0.]) if reach['pitch'] > CLOSE_UP_PITCH else 0.)
        down = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], aim, reach['rotation'], step=.015) or down
    status = yield from follow(ctx, down, rate=.012, max_steps=300)
    ctx.set_grip(-1.)
    yield from hold(ctx, 30)
    fingers = ctx.fingers().copy()
    yield from follow(ctx, [reach['pre']], rate=.02)
    yield from hold(ctx, 6)
    held = float(np.sum(ctx.fingers())) > HELD_GAP
    ctx.event('grasp', ok=held, arm=ctx.arm, descend=status, pitch=round(reach['pitch'], 2), fingers=fingers.round(4).tolist(), after_lift=ctx.fingers().round(4).tolist(),
              tip=ctx.tip()[:3, 3].round(3).tolist(), refined=shift)
    if not held:
        ctx.set_grip(1.)
    return held


def can_stance(ctx, can_xy, robot_xy, distance=.62):
    """Base poses (x, y, yaw) facing the can from @distance, the one on the robot's side first."""
    can_xy = np.asarray(can_xy, float); toward = np.arctan2(robot_xy[1] - can_xy[1], robot_xy[0] - can_xy[0]); out = []
    for delta in (0., .5, -.5, 1., -1., 1.6, -1.6, 2.4, -2.4, np.pi):
        angle = toward + delta; xy = can_xy + distance * np.array([np.cos(angle), np.sin(angle)])
        if ctx.scene.base_free(xy, margin=.02):
            out.append((float(xy[0]), float(xy[1]), float(wrap(angle + np.pi))))
    return out


def drop(ctx, can_point):
    """Hold the active hand above the can centre @can_point (base frame, z = rim height) and open it."""
    target = np.asarray(can_point, float) + [0., 0., DROP_HEIGHT]; goal = None
    check = ctx.validator(max_load=MAX_LOAD)
    for pitch in (0., .7, 1.2):
        rotation, _ = reach_rotation(np.arctan2(target[1], target[0]), pitch)
        goal = ctx.ik.solve(ctx.q, target, rotation, count=20, validator=check)
        if goal is not None:
            break
    if goal is None:
        ctx.event('drop', ok=False, reason='no pose above the can'); return False
    status = yield from move_to(ctx, goal, rate=.035, label='drop', max_load=1.)
    if status != 'done':
        return False
    yield from hold(ctx, 5)
    tip = ctx.tip()[:3, 3]
    ctx.set_grip(1.)
    yield from hold(ctx, 25)
    ctx.event('drop', ok=True, arm=ctx.arm, tip_minus_target=(tip - target).round(3).tolist(), fingers=ctx.fingers().round(4).tolist())
    return True


# ---- moving the trash can ----------------------------------------------------------------------
CAN_GRIP_DEPTH = .028       # fingertip centre below the rim when a wall of the can is pinched
CAN_LIFT = .12
CAN_HELD_GAP = .0015        # the wall is thin: fingers this far apart (sum, m) after closing still hold it


def can_walls(can_xy, can_yaw):
    """Midpoints of the four walls of the can (map frame) with their outward normals."""
    c, s = np.cos(can_yaw), np.sin(can_yaw); out = []
    for normal, half in ((np.array([c, s]), CAN_HALF[0]), (np.array([-c, -s]), CAN_HALF[0]), (np.array([-s, c]), CAN_HALF[1]), (np.array([s, -c]), CAN_HALF[1])):
        out.append((np.asarray(can_xy, float)[:2] + half * normal, normal))
    return out


def plan_can_lift(ctx, wall_point, normal, base=None, q=None):
    """Arm motion for pinching the can's wall at @wall_point (base frame xy) across it (@normal, base frame) and
    lifting it: dict(arm, up = waypoints from the pinch to the lifted pose, low = the pinch), or None.

    Every pose is checked against the scene, so this also tells whether the place the robot stands at leaves
    room for the lean (with the cabinet right behind it, it does not). The wall is usually straight ahead, and
    there one arm reaches it with the fingers closing one way only (the other arm the opposite way): both arms
    and both ways are tried, the arm on the wall's side first. Leaves ctx.arm as it was.
    """
    from skills import top_down_rotation
    full = ctx.q if q is None else np.asarray(q, float); previous = ctx.arm
    point = np.r_[np.asarray(wall_point, float)[:2], CAN_TOP - CAN_GRIP_DEPTH - .012]
    try:
        for arm in (('left', 'right') if point[1] >= 0 else ('right', 'left')):
            ctx.arm = arm
            # No load limit here: the static estimate rejects nearly every low reach, yet the robot holds them (the
            # fax was taken from the floor this way). Of the configurations found, the least loaded ones are tried first.
            check = ctx.validator(base=base)
            def loaded(x):
                config = full.copy(); config[ctx.ik.columns] = x; return ctx.load(config, arm)[0]
            for sign in (1., -1.):
                rotation = top_down_rotation(sign * np.asarray(normal, float)[:2])
                lows = ctx.ik.solve(full, point, rotation, count=40, validator=check, all_solutions=True)
                ordered = sorted(lows[:12], key=loaded)
                for lift in (CAN_LIFT, .07):          # a hand's width is enough to carry the can
                    for low in ordered:
                        start = full.copy(); start[ctx.ik.columns] = low
                        up = interpolate_cartesian(ctx.ik, start, point, point + [0., 0., lift], rotation, validator=check, step=.02)
                        if up is not None:
                            return dict(arm=arm, up=up, low=low)
    finally:
        ctx.arm = previous
    return None


def lift_can(ctx, wall_point, normal):
    """Pinch the can's wall and lift the can (see plan_can_lift) with the arm the plan names, which becomes
    the active arm. Returns the plan, or None if that failed."""
    plan = plan_can_lift(ctx, wall_point, normal)
    if plan is None:
        ctx.event('lift_can', ok=False, reason='no pinch and lift within reach'); return None
    up, low = plan['up'], plan['low']; ctx.arm = plan['arm']
    ctx.set_grip(1.)
    status = yield from move_to(ctx, up[-1], rate=.035, label='lift_can')
    if status != 'done':
        return None
    yield from follow(ctx, up[::-1][1:] + [low], rate=.012, max_steps=300)
    ctx.set_grip(-1.)
    yield from hold(ctx, 30)
    fingers = ctx.fingers().copy()
    yield from follow(ctx, up, rate=.012, max_steps=300)
    yield from hold(ctx, 8)
    held = float(np.sum(ctx.fingers())) > CAN_HELD_GAP
    ctx.event('lift_can', ok=held, arm=ctx.arm, fingers=fingers.round(4).tolist(), after_lift=ctx.fingers().round(4).tolist())
    if not held:
        ctx.set_grip(1.); return None
    return plan


def lower_can(ctx, lifted):
    """Put the lifted can down where the hand is and let go."""
    yield from follow(ctx, lifted['up'][::-1][1:] + [lifted['low']], rate=.012, max_steps=300)
    ctx.set_grip(1.)
    yield from hold(ctx, 20)
    yield from follow(ctx, lifted['up'], rate=.02, max_steps=200)
