"""Skills of the trash task: fold the arms, pinch a soda can that stands on the floor, let it fall into the trash can.

All points are in the base frame B (x ahead, y left, z up from the base_link origin, which rides BASE_Z above
the floor). The skills are generators that yield one 23-D action per step (see manip.py).
"""
import numpy as np
from manip import hold, follow, move_to, TALL_TRUNK, TUCK, BASE_Z
from skills import interpolate_cartesian

SODA_GRIP = .96             # fingertip centre (m above the floor) for the first try, by the handles 4 cm below their top (at 1.0 m)
RIM = 0.                    # the handles stand over the bag's middle (a pinch at its wall, 5.5 cm out, tipped two bags over)
RIM_TRIES = (0., .055, -.055, .055)    # where on the way to the bag's middle the pinch goes, one try after another (the handles stand over its long sides)
GRIP_TRIES = (.96, .90, .90, .93)    # ... and how high: from the second try the bag's own top edge, under the handles, one finger inside
GLASS_BASE = .76            # the breakfast table's top, where the bags stand (m)
HANG = .045                 # a battery held 2 cm under its top reaches this far below the fingertip centre
SODA_WIDTH = .08            # around a mousetrap for the validator (5.0 x 10.8 cm, 1.8 cm high; boxes of other traps only near the fingers)
LYING_GRIP = .042           # fingertip centre above the floor when a can that lies is pinched across (its axis is 0.038 m up)
APPROACH = .13              # the hand comes straight down this far onto a glass's wall, and lifts it this far
HELD_GAP = .010              # (a battery is 3.4 cm across) fingers further apart than this (sum of both, m) after closing hold a glass (its wall is a few mm thick)
BIN_TOP = .725              # the top drawer's top edge
BIN_WIDTH = .38             # across the trash can, outside
LYING_CLEAR = .035          # a battery is let go with its bottom this far above the drawer's top edge
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


def plan_floor_pinch(ctx, xy, base=None, q=None, seeds=2, tight=False, neighbours=(), lying=None, height=None):
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
    radial = float(np.arctan2(xy[1], xy[0])); toward = -xy / max(float(np.linalg.norm(xy)), 1e-9)
    # A glass is pinched by its wall on the side nearest the robot: one finger inside, one outside, closing along
    # the line from the robot (fingers 10 cm apart: 3 cm of error either way still puts one inside).
    if lying is not None and height is not None:
        return plan_trap_pinch(ctx, xy, lying, height, full, base=base, tight=tight, seeds=seeds)
    rim = RIM_TRIES[getattr(ctx, 'pinches', 0) % len(RIM_TRIES)]
    grip = GRIP_TRIES[getattr(ctx, 'pinches', 0) % len(GRIP_TRIES)]
    if height is not None:
        rim = 0.; grip = height        # a battery: over its middle, at the height its sighting gave
    point = np.r_[xy + rim * toward, (grip if lying is None else LYING_GRIP) - BASE_Z]; check = ctx.validator(base=base, relax_from=full if tight else None)
    rotations = [reach_rotation(radial + np.pi / 2), reach_rotation(radial)]      # (the handles' plane is not known: the fingers close along or across the way to the bag)
    if lying is not None:
        rotations = [reach_rotation(lying), reach_rotation(lying + np.pi)]
    found = least_loaded(ctx, full, point, rotations, check, seeds=seeds)
    for load, rotation, low in found[:6]:
        start = full.copy(); start[ctx.ik.columns] = low
        up = interpolate_cartesian(ctx.ik, start, point, point + [0., 0., APPROACH], rotation, validator=check, step=.02)
        if up is not None:
            return dict(low=low, up=up, pre=up[-1], rotation=rotation, load=float(load), solutions=len(found), point=point)
    return None


GRIP_OFF = .03              # a mousetrap is held this far from its middle along its long axis: laid under the sink, its middle is
                            # this much further in than the fingers (the vanity's front is 0.35-0.85 m up; under it is free floor)
LAY_TIP = .025              # fingertip centre above the floor when a mousetrap is let go (it is held 4 mm over its bottom)
SLIDE = .14                 # the hand comes down this far in front of the place and slides in level (the trap's far end must
                            # pass under the vanity's front panel, not through it)


def plan_trap_pinch(ctx, xy, axis, height, full, base=None, tight=False, seeds=2):
    """A top-down pinch of a mousetrap lying flat at @xy (base frame), its long axis along @axis (base frame,
    modulo half a turn): the fingers close across the axis at @height (m above the floor), GRIP_OFF from its
    middle along the axis, on the side the tool's x points away from (so the middle lies +GRIP_OFF along tool x)."""
    xy = np.asarray(xy, float)[:2]; check = ctx.validator(base=base, relax_from=full if tight else None); found = []
    for azimuth in (axis, axis + np.pi):
        point = np.r_[xy - GRIP_OFF * np.array([np.cos(azimuth), np.sin(azimuth)]), height - BASE_Z]
        found += [(load, rotation, low, point) for load, rotation, low in least_loaded(ctx, full, point, [reach_rotation(azimuth)], check, seeds=seeds)]
    found.sort(key=lambda item: item[0])
    for load, rotation, low, point in found[:6]:
        start = full.copy(); start[ctx.ik.columns] = low
        up = interpolate_cartesian(ctx.ik, start, point, point + [0., 0., APPROACH], rotation, validator=check, step=.02)
        if up is not None:
            return dict(low=low, up=up, pre=up[-1], rotation=rotation, load=float(load), solutions=len(found), point=point)
    return None


def plan_lay(ctx, centre, azimuth, base=None):
    """How the active hand lays its mousetrap with the trap's middle at @centre (base frame xy) and its long axis
    along @azimuth (pointing in under the sink): the hand GRIP_OFF short of the middle, LAY_TIP over the floor;
    it comes down SLIDE further out and 3 cm higher and slides in level. dict(low, back, pre, point, load) or None."""
    direction = np.array([np.cos(azimuth), np.sin(azimuth)]); point = np.r_[np.asarray(centre, float)[:2] - GRIP_OFF * direction, LAY_TIP - BASE_Z]
    rotation = reach_rotation(azimuth); check = ctx.validator(base=base)
    for load, _, low in least_loaded(ctx, ctx.q, point, [rotation], check, seeds=1)[:4]:
        start = ctx.q.copy(); start[ctx.ik.columns] = low
        back = interpolate_cartesian(ctx.ik, start, point, point + np.r_[-SLIDE * direction, .03], rotation, validator=check, step=.02)
        if back is not None:
            return dict(low=low, back=back, pre=back[-1], point=point, load=float(load), rotation=rotation)
    return None


def lay_trap(ctx, centre, azimuths, shifts=((0., 0.),)):
    """Lay the mousetrap in the active hand on the floor, its middle at @centre (base frame xy) moved by the first of
    @shifts (map-independent, base frame) and along the first of @azimuths that has a plan. Returns that azimuth or None.
    (The base stops up to a few centimetres off the stance the plan was found for: on 301 no plan was left there.)"""
    reason = 'no plan'; base_centre = np.asarray(centre, float)[:2]
    for shift, azimuth in [(s, a) for s in shifts for a in azimuths]:
        centre = base_centre + np.asarray(shift, float)
        plan = plan_lay(ctx, centre, azimuth)
        if plan is None:
            continue
        status = yield from move_to(ctx, plan['pre'], rate=.04, label='lay', max_steps=320)
        if status != 'done':
            reason = f'arm held up ({status})'
            if status != 'no_path':
                yield from move_to(ctx, np.r_[TALL_TRUNK, TUCK[ctx.arm]], rate=.04, label='tuck', max_steps=260)
                break
            continue
        yield from hold(ctx, 3)
        status = yield from follow(ctx, plan['back'][::-1][1:] + [plan['low']], rate=.012, max_steps=300)
        yield from hold(ctx, 4)
        tip = ctx.tip()[:3, 3].copy()
        ctx.set_grip(1.)
        yield from hold(ctx, 20)
        yield from follow(ctx, plan['back'], rate=.02, max_steps=200)
        ctx.event('lay', ok=True, arm=ctx.arm, slide=status, load=round(plan['load'], 2), azimuth=round(float(np.degrees(azimuth))), centre=np.round(centre, 3).tolist(),
                  tip_minus_target=(tip - plan['point']).round(3).tolist(), fingers=ctx.fingers().round(4).tolist())
        return azimuth
    ctx.event('lay', ok=False, reason=reason, arm=ctx.arm, centre=np.round(centre, 3).tolist())
    return None


DROP_TIP = .25              # last resort: the hand opens with its fingertips this high over free floor, the trap falls flat


def drop_on_floor(ctx, scene, world):
    """Last resort for a trap that could not be laid anywhere (306: the right hand stalled on the way down to the
    floor three times): hold it DROP_TIP over free floor in front of the base and let it fall. Returns True if let go."""
    for forward, side, azimuth in ((.55, .15, 0.), (.55, -.15, 0.), (.65, 0., .5), (.45, .25, -.5), (.45, -.25, .5)):
        side = side if ctx.arm == 'left' else -side
        xy = np.array([forward, side]); spot = world[:2, :2] @ xy + world[:2, 3]
        i, j = scene.cell(spot[None]); i, j = int(i[0]), int(j[0])
        if not scene.inside(i, j) or float(scene.top[i, j]) > .03 or float(scene.distance[i, j]) < .10:
            continue
        found = least_loaded(ctx, ctx.q, np.r_[xy, DROP_TIP - BASE_Z], [reach_rotation(azimuth), reach_rotation(azimuth + np.pi / 2)], ctx.validator(), seeds=1)
        for load, rotation, goal in found[:2]:
            status = yield from move_to(ctx, goal, rate=.04, label='drop', max_steps=300)
            if status == 'done':
                yield from hold(ctx, 4)
                ctx.set_grip(1.)
                yield from hold(ctx, 25)
                ctx.event('dropped', ok=True, arm=ctx.arm, spot=np.round(spot, 3).tolist())
                return True
            if status != 'no_path':
                break
    ctx.event('dropped', ok=False, arm=ctx.arm)
    return False


def pinch_from_floor(ctx, xy, tight=False, neighbours=(), lying=None, height=None):
    """Pinch the can at @xy (base frame) with the active arm and lift it clear of the floor: standing, from
    above; or, with @lying = the direction of its axis, a can that has fallen over, across its middle.
    tight, neighbours: see plan_floor_pinch. Returns True if the fingers stayed apart after closing, False if
    they closed on nothing, None if the hand did not go down (no way to the can)."""
    plan = plan_floor_pinch(ctx, xy, tight=tight, neighbours=neighbours, lying=lying, height=height)
    ctx.pinches = getattr(ctx, 'pinches', 0) + 1     # (the next pinch tries the next point of RIM_TRIES)
    if plan is None:
        ctx.event('pinch', ok=False, reason='no way down to the can', arm=ctx.arm, xy=np.round(xy, 3).tolist()); return None
    ctx.set_grip(1.)
    # On its way to the start of the descent the arm passes at the glass's height: the glass itself is a box for the
    # validator until then (without it the open fingers tipped glasses over before the descent, starts 301 and 302).
    world = ctx.base_matrix(); spot = world[:2, :2] @ np.asarray(xy, float)[:2] + world[:2, 3]; r = SODA_WIDTH / 2 + .01
    low = GLASS_BASE if height is None else height - .05
    others = ctx.guard; ctx.guard = np.vstack([others, [[spot[0] - r, spot[1] - r, low, spot[0] + r, spot[1] + r, low + (.25 if height is None else .10)]]])
    status = yield from move_to(ctx, plan['pre'], rate=.04, label='pinch')
    ctx.guard = others
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


def plan_drop(ctx, xy, azimuth, base=None, across=True, higher=0.):
    """Poses of the active arm that hold its can lying above @xy (base frame, the middle of the trash can), the
    can's axis along @azimuth, least loaded first: (target, [(load, rotation, configuration)]).
    base: as in plan_floor_pinch. across: the can was pinched lying, across its middle (it lies in the hand as
    it is, the hand pointing down); otherwise it was pinched standing, from above, and the hand is turned level."""
    across = True       # a glass hangs from the hand pointing down, as it was taken
    axis = np.array([np.cos(azimuth), np.sin(azimuth)]); height = BIN_TOP + LYING_CLEAR + higher + HANG - BASE_Z
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
