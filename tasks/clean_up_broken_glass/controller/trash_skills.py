"""Skills of the trash task: fold the arms, pinch a soda can that stands on the floor, let it fall into the trash can.

All points are in the base frame B (x ahead, y left, z up from the base_link origin, which rides BASE_Z above
the floor). The skills are generators that yield one 23-D action per step (see manip.py).
"""
import numpy as np
from manip import hold, follow, move_to, TALL_TRUNK, TUCK, BASE_Z
from skills import interpolate_cartesian

SODA_GRIP = .030            # fingertip centre above the floor when a standing can is pinched (cans are 0.127 and 0.135 m tall)
SODA_WIDTH = .070           # across a can
LYING_GRIP = .042           # fingertip centre above the floor when a can that lies is pinched across (its axis is 0.038 m up)
APPROACH = .10              # the hand comes straight down this far, and lifts the can this far
HELD_GAP = .004              # fingers further apart than this (sum of both, m) after closing hold a can
BIN_TOP = .422              # height of the trash can's rim
BIN_WIDTH = .38             # across the trash can, outside
LYING_CLEAR = .08           # a piece is let go this far above the rim (the forearm needs this much room over a 0.42 m rim)
DROP_HIGHER = (0., .08)     # tried in this order when no arm pose is found above the trash can
LIFT_AFTER = .12            # the open hand goes straight up this far before the arm folds (folding from just above the rim tipped the trash can over)
GRIP_ABOVE_MIDDLE = SODA_GRIP - .0675       # the fingers hold a can this far from its middle, towards its top
NEIGHBOUR = .20             # another can this close to the one being pinched decides which way the fingers close (m, between axes)


def reach_rotation(azimuth, pitch=0.):
    """Tool rotation (base frame) approaching along @azimuth at @pitch from straight down, closing horizontally across @azimuth."""
    approach = np.array([np.sin(pitch) * np.cos(azimuth), np.sin(pitch) * np.sin(azimuth), -np.cos(pitch)])
    y = np.array([-np.sin(azimuth), np.cos(azimuth), 0.]); z = -approach
    return np.column_stack([np.cross(y, z), y, z])


def tuck(ctx, arms=('left', 'right'), rate=.04):
    """Bring the trunk upright and fold the given arms in front of the chest (a held can stays in the hand)."""
    previous = ctx.arm; result = 'done'
    for arm in arms:
        ctx.arm = arm
        status = yield from move_to(ctx, np.r_[TALL_TRUNK, TUCK[arm]], rate=rate, label='tuck')
        if status != 'done':
            result = status
    ctx.arm = previous
    return result


def least_loaded(ctx, full, point, rotations, check, seeds=2, count=24):
    """Arm-and-trunk configurations that put the tool at @point with one of @rotations, the statically least
    loaded first: [(load, rotation, configuration)]. Reaching down to the floor the first trunk joint carries
    one to two times its effort limit in most configurations; the few that fold the trunk over the base stay
    near the limit."""
    found = []
    for rotation in rotations:
        for seed in range(seeds):
            for low in ctx.ik.solve(full, point, rotation, count=count, validator=check, all_solutions=True, seed=seed):
                config = full.copy(); config[ctx.ik.columns] = low
                found.append((ctx.load(config)[0], rotation, low))
    found.sort(key=lambda item: item[0])
    return found


def plan_floor_pinch(ctx, xy, base=None, q=None, seeds=2, tight=False, neighbours=(), lying=None):
    """A top-down pinch of the can standing at @xy (base frame) with the active arm: dict(low, up, pre, rotation,
    load, solutions) or None. The hand comes straight down APPROACH and lifts the can the same way.
    base: the base pose (4x4 in M) the plan is for, if not the present one (a stance that is being considered).
    neighbours: other cans (base frame xy). With one within NEIGHBOUR the fingers come down on the two sides of
    the line through both cans, where there is room for them even if the cans touch.
    lying: for a can that has fallen over, the direction of its axis (base frame, rad): the fingers then come
    down on its two sides, at the height of its axis.
    tight: the stance is in a gap between furniture where the chassis has been checked by its own outline; the
    spheres that stand for it in the validator are wider and are let be as they are in the folded posture."""
    full = ctx.q if q is None else np.asarray(q, float); xy = np.asarray(xy, float)[:2]
    point = np.r_[xy, (SODA_GRIP if lying is None else LYING_GRIP) - BASE_Z]; check = ctx.validator(base=base, relax_from=full if tight else None); radial = float(np.arctan2(xy[1], xy[0]))
    # A can is round: the fingers may close across the line from the robot or along it.
    rotations = [reach_rotation(radial), reach_rotation(radial + np.pi / 2)]
    near = [np.asarray(n, float)[:2] - xy for n in neighbours if n is not None]
    near = [d for d in near if .01 < float(np.linalg.norm(d)) < NEIGHBOUR]
    if near:
        d = min(near, key=lambda v: float(np.linalg.norm(v))); line = float(np.arctan2(d[1], d[0]))
        rotations = [reach_rotation(line), reach_rotation(line + np.pi)]
    if lying is not None:
        rotations = [reach_rotation(lying), reach_rotation(lying + np.pi)]
    found = least_loaded(ctx, full, point, rotations, check, seeds=seeds)
    for load, rotation, low in found[:6]:
        start = full.copy(); start[ctx.ik.columns] = low
        up = interpolate_cartesian(ctx.ik, start, point, point + [0., 0., APPROACH], rotation, validator=check, step=.02)
        if up is not None:
            return dict(low=low, up=up, pre=up[-1], rotation=rotation, load=float(load), solutions=len(found), point=point)
    return None


def pinch_from_floor(ctx, xy, tight=False, neighbours=(), lying=None):
    """Pinch the can at @xy (base frame) with the active arm and lift it clear of the floor: standing, from
    above; or, with @lying = the direction of its axis, a can that has fallen over, across its middle.
    tight, neighbours: see plan_floor_pinch. Returns True if the fingers stayed apart after closing, False if
    they closed on nothing, None if the hand did not go down (no way to the can)."""
    plan = plan_floor_pinch(ctx, xy, tight=tight, neighbours=neighbours, lying=lying)
    if plan is None:
        ctx.event('pinch', ok=False, reason='no way down to the can', arm=ctx.arm, xy=np.round(xy, 3).tolist()); return None
    ctx.set_grip(1.)
    status = yield from move_to(ctx, plan['pre'], rate=.04, label='pinch')
    if status != 'done':
        return None
    yield from hold(ctx, 3)
    down = plan['up'][::-1][1:] + [plan['low']]; point = plan['point']
    status = yield from follow(ctx, down, rate=.015, max_steps=300)
    reached = ctx.tip()[:3, 3].copy()
    ctx.set_grip(-1.)
    yield from hold(ctx, 22)
    fingers = ctx.fingers().copy()
    yield from follow(ctx, [plan['pre']], rate=.02, max_steps=200)
    yield from hold(ctx, 4)
    held = float(np.sum(ctx.fingers())) > HELD_GAP
    ctx.event('pinch', ok=held, arm=ctx.arm, lying=lying is not None, descend=status, load=round(plan['load'], 2), solutions=plan['solutions'], fingers=fingers.round(4).tolist(),
              after_lift=ctx.fingers().round(4).tolist(), short=(reached - point).round(3).tolist())
    if not held:
        ctx.set_grip(1.)
    return held


def plan_drop(ctx, xy, azimuth, base=None, across=False, higher=0.):
    """Poses of the active arm that hold its can lying above @xy (base frame, the middle of the trash can), the
    can's axis along @azimuth, least loaded first: (target, [(load, rotation, configuration)]).
    base: as in plan_floor_pinch. across: the can was pinched lying, across its middle (it lies in the hand as
    it is, the hand pointing down); otherwise it was pinched standing, from above, and the hand is turned level."""
    axis = np.array([np.cos(azimuth), np.sin(azimuth)]); height = BIN_TOP + LYING_CLEAR + higher + SODA_WIDTH / 2 - BASE_Z
    if across:
        target = np.r_[np.asarray(xy, float)[:2], height]; rotations = [reach_rotation(azimuth), reach_rotation(azimuth + np.pi)]
    else:
        # The fingers hold the can GRIP_ABOVE_MIDDLE from its middle, towards its top.
        target = np.r_[np.asarray(xy, float)[:2] - GRIP_ABOVE_MIDDLE * axis, height]; rotations = [reach_rotation(azimuth, np.pi / 2)]
    return target, least_loaded(ctx, ctx.q, target, rotations, ctx.validator(base=base), seeds=1)


def drop_into_bin(ctx, xy, azimuths, across=False):
    """Hold the can in the active hand lying above the trash can at @xy (base frame), its axis along one of
    @azimuths (tried in this order), and open the hand. Returns the azimuth it was let go along, or None.

    Why lying: the simulator counts a can as inside when the middle of its box is within the trash can's
    fillable volume, which ends 0.22-0.24 m above the floor, and the trash can narrows downwards. Cans let go
    upright pile up with the top one's middle at 0.18-0.25 m (it missed in 3 of 14 trials let go over the middle,
    and in runs also when spread out); let go lying and crossed, the top one ends at 0.13-0.22 m
    (dev/jobs/drop_study.py).

    The can in the hand is not in the collision model. Where the arm is held up on its way (it happened with a
    can that had been pinched lying), it folds again and another direction is tried.
    """
    reason = 'no pose above the trash can'
    for higher, azimuth in [(h, a) for h in DROP_HIGHER for a in azimuths]:
        target, found = plan_drop(ctx, xy, azimuth, across=across, higher=higher)
        for load, rotation, goal in found[:3]:
            status = yield from move_to(ctx, goal, rate=.04, label='drop', max_steps=260)
            if status == 'done':
                yield from hold(ctx, 4)
                tip = ctx.tip()[:3, 3]
                ctx.set_grip(1.)
                yield from hold(ctx, 18)
                up = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], ctx.tip()[:3, 3] + [0., 0., LIFT_AFTER], rotation, step=.03)
                if up is not None:
                    yield from follow(ctx, up, rate=.03, max_steps=120)
                ctx.event('drop', ok=True, arm=ctx.arm, load=round(float(load), 2), azimuth=round(float(np.degrees(azimuth))), tip_minus_target=(tip - target).round(3).tolist(), fingers=ctx.fingers().round(4).tolist(), lifted=up is not None)
                return azimuth
            if status != 'no_path':
                reason = f'arm held up ({status})'
                yield from move_to(ctx, np.r_[TALL_TRUNK, TUCK[ctx.arm]], rate=.04, label='tuck', max_steps=260)
                break
    ctx.event('drop', ok=False, reason=reason, arm=ctx.arm, xy=np.round(xy, 3).tolist())
    return None
