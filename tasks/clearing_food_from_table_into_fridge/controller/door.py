"""The fridge's right door: opened wide enough for a plate, and shut again, by a hand that takes its handle and leads it
round the hinge.

The door turns about a vertical hinge at its west edge (kitchen.HINGE) from 0 (shut) to 2.36 rad; it stays where it is
let go. The hand takes the handle bar from the front, its fingers closing across the bar, and the door follows the
hand as long as the grasp holds (in trials it followed to within a degree up to where it met the chassis). From one
stance the chassis is in the door's way somewhere on its swing, so the swing is led in stages, each from a stance
clear of the leaf over the stage's angles (dev: the stances were searched offline with the arm planner and the leaf's
outline).
"""
import numpy as np
from manip import hold, follow, move_to, BASE_Z, TALL_TRUNK
from car_skills import tuck, frame
from car_pipeline import straight, settle
import decor
import kitchen as K

R = float(np.linalg.norm(K.HANDLE - K.HINGE)); PHI = float(np.arctan2(*(K.HANDLE - K.HINGE)[::-1]))
DEPTH = .030                        # how far behind the handle's front face the fingers' ends go (the bar is 4.7 cm deep with its foot; at 1.2 cm the hand closed on its front edge and the grasp did not take, 1 trial of 2)
# Stages: (how ('handle': the hand leads the door by its handle; 'edge': by its free edge, pinched across the door's
# thickness; 'push': the closed hand pushes its inner face), stance (x, y, yaw), arm, from deg, to deg, approach): the
# base comes to the stance by a straight move from 'approach' (a point with room to turn).
# (Led by its handle from the base facing west, the right hand reaching across, the door followed to 96 degrees and
# stayed; with the left hand the idle right one, tucked in front of the chest, met the door at 38. Opened by 96 degrees
# the door's inner bins leave the west half's opening 3 cm clear of a plate's edge. Shut: the hand pinches the free
# edge across the door's thickness and leads it to 17 degrees, then takes the handle again and leads it to 3; in a
# trial the fridge was then shut (0.06 rad). Pushed on to its stop at 135 degrees the door could not be led back:
# the arm met the table and the door.)
# (Led to 96 the door stood at 85-106 degrees in whole runs, the map pose a few centimetres off; at 85 a plate met
# the door's bins. Led to 105; and shut past its stop: led to 3 degrees it stood at 9, still open.)
OPEN_STAGES = (('handle', (5.50, .60, np.pi), 'right', 0., 105., (5.95, .60)),)
WIDEN = (('push', (5.50, .60, np.pi), 'right', 40., 128., (5.95, .60)),)
# (z02) WIDEN: then, once, the door is pushed on to about 128 degrees from where a look shows it (at 96 the door's bins
# stand 2-3 cm from the west half's opening; on 301 the lead left it at 72, and a tupperware set down at x 4.86, two side
# by side on the lower shelf, met them three times, e/f 301). Whatever the push does, the door counts as open as the lead
# left it (a push stage that failed to plan made the whole door stage fail, g 301). Not shut again in z02.
SHUT_STAGES = (('edge', (5.50, .60, np.pi), 'right', 105., 17., (5.95, .60)),
               ('handle', (5.50, .60, np.pi), 'right', 17., -4., (5.95, .60)),
               ('press', (5.50, .60, np.pi), 'right', 0., 0., (5.95, .60)))
# (from 105 degrees on the free end stands 1.0 m from the edge stance's base, at the arm's reach: the hand that had
# pinched it lost it at the lead's first step, 303, 305, 313. From 0.2 m nearer it is led to 62 degrees, as far as the
# door's swing keeps off the chassis there, and from the usual stance on.)
NEAR_EDGE = ('edge', (5.30, .50, np.pi), 'right', 105., 62., (5.95, .50))
PRESS = dict(x=5.05, y=(.03, -.14), z=1.0, most=20.)     # the closed hand pushes the door's outer face shut here (map), 0.15 m in from the handle, the door at most this open (deg)
PUSH_SHUT = (('push_out', (5.50, .60, np.pi), 'right', 105., 20., (5.95, .60)),
             ('handle', (5.50, .60, np.pi), 'right', 17., -4., (5.95, .60)))
DOOR_STOP = np.radians(135.)


def handle_grasp(theta):
    """The fingers' ends (map, height) and the hand's approach and closing directions on the handle at door angle
    @theta (rad)."""
    a = PHI + theta; face = K.HINGE + R * np.array([np.cos(a), np.sin(a)])
    approach = np.array([np.sin(theta), -np.cos(theta), 0.]); closing = np.array([np.cos(theta), np.sin(theta), 0.])
    return np.r_[face + DEPTH * approach[:2], K.HANDLE_Z], approach, closing


# (the door's free end is a frame 7.1 cm thick, -0.009 to 0.062 across from the hinge's line; on its outer side the
# handle bar stands 2.5 cm off the face from 0.71 to 1.69 m up: the fingers, 10 cm apart, closed on it or beside the
# door when the pinch was centred, 305, 314, 319. Set 1.2 cm in from the middle the outer finger comes along the face.)
EDGE = dict(along=.55, across=.015, z=1.0)      # the edge pinch: 3 cm in from the free end, across the door's thickness
EDGE_FURTHER = (.03, .06)                       # a pinch that closed on nothing is tried again this much further in


def edge_grasp(theta):
    """The same for a pinch of the door's free edge across its thickness (7 cm), the hand coming along the door
    toward the hinge."""
    d = np.array([np.cos(theta), np.sin(theta)]); n = np.array([-d[1], d[0]])
    return np.r_[K.HINGE + EDGE['along'] * d + EDGE['across'] * n, EDGE['z']], np.r_[-d, 0.], np.r_[n, 0.]


def leaf(angle):
    """The door's leaf at @angle (rad) for the arm planner: balls along it and across its thickness (the handle and the
    bins included), (points N x 3 in M, radii, touch=False). (Folded back without it, the arm swept the door from 35
    degrees back to 16.)"""
    d = np.array([np.cos(angle), np.sin(angle)]); n = np.array([-d[1], d[0]]); points = []
    for a in np.arange(.04, .59, .06):
        for b in (-.035, .025, .085):
            for z in np.arange(.10, 1.95, .09):
                points.append([*(K.HINGE + a * d + b * n), z])
    return np.array(points), np.full(len(points), .04), False


def keep_off(ctx, est):
    """The open door goes into the arm planner's obstacles (and comes out once it is shut)."""
    if getattr(est, 'door', 0.) > .05:
        ctx.obstacles['door'] = leaf(est.door)
    else:
        ctx.obstacles.pop('door', None)


def to_base(base, p, vectors=()):
    x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); m = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
    return m @ (np.asarray(p, float) - [x, y, BASE_Z]), [m @ np.asarray(v, float) for v in vectors]


def idle_aside(ctx, arm):
    """The other hand out of the door's way: to its side and back, high (tucked in front of the chest, the turning
    trunk brought it against the door at 38 degrees, and the door stopped there)."""
    from car_skills import park, down
    side = 1. if arm == 'left' else -1.; ctx.arm = arm
    forward = frame(np.array([1., 0., 0.]), np.array([0., 0., 1.]))         # (fingers forward: with them down no side pose was reached)
    status = yield from park(ctx, [(.05, side * .42, 1.10), (-.05, side * .35, .95), (.25, side * .30, 1.00)], [forward, down(np.pi / 2)], label='aside', rate=.04, narrow=False)
    return status


def lead(ctx, est, arm, start, end, step=np.radians(3.), how='handle'):
    """Take the handle (how='edge': the free edge) with @arm's hand with the door at @start (rad) and lead it to @end.
    Returns the angle the door was let go at (as far as the hand got), or None if the hand did not get it."""
    yield from idle_aside(ctx, 'right' if arm == 'left' else 'left')
    ctx.arm = arm; base = est.base(); grasp = edge_grasp if how == 'edge' else handle_grasp
    # (on the way in the arm keeps off the door, the hand may touch it: going in for the edge without it, the arm
    # pushed the door from 135 to 125 degrees and the hand found nothing there)
    if start > .05:
        points, radii, _ = leaf(start); ctx.obstacles['door'] = (points, radii, True)
    else:
        ctx.obstacles.pop('door', None)
    # (the free edge where the look before showed it: the door as seen stood that far along its line)
    seen = getattr(est, 'door_seen', None); off = np.zeros(3)
    if how == 'edge' and seen is not None and est.ob.steps - seen['step'] < 400:
        # (across only: the free end's place along the door as the look gave it was 2 cm out where the hand landed
        # 4-6 cm past the end, the pose believed 3-6 cm south of where the base stood: tests t1-t4)
        off = np.r_[seen.get('across', 0.) * np.array([-np.sin(start), np.cos(start)]), 0.]
    point, (ap, cl) = to_base(base, grasp(start)[0] + off, grasp(start)[1:])
    plan = decor.plan_pinch(ctx, point, frame(ap, cl), way_in=.12 if how == 'edge' else .10, hand_margin=-.05)
    if plan is None:
        ctx.event('door', ok=False, reason='no way to the ' + how, arm=arm, at=round(float(np.degrees(start)), 1)); return None
    if 'door' in ctx.obstacles:
        # (to the start of the way in, the hand too keeps off the door; only the last straight stretch meets it)
        points, radii, _ = ctx.obstacles['door']; ctx.obstacles['door'] = (points, radii, False)
    look = None
    # (for the free edge the head camera's look before sets the target: the wrist camera's took the bins or the
    # handle for the edge's front and moved the hand 4-7 cm on, 306, 318, 319)
    if hasattr(est, 'wrist_front') and how == 'handle':
        depth = DEPTH if how == 'handle' else K.DOOR_LENGTH - EDGE['along']
        look = lambda tip: est.wrist_front(arm, tip, ap, cl, depth, width=(.015, .06) if how == 'handle' else (.05, .11))
    replan = lambda p_: decor.plan_pinch(ctx, np.r_[p_[:2], point[2]], frame(ap, cl), way_in=.12 if how == 'edge' else .10, hand_margin=-.05)
    ok = yield from decor.pinch(ctx, plan, label=how, gap=.002, lift=0., look=look, replan=replan, debug=lambda: dict(getattr(est, 'wrist_debug', {})))
    if not ok and how == 'edge':
        # The fingers closed on nothing: past the free end (where the pose put the hand 4-6 cm too far out along the
        # door). Drawn back along the fingers, the pinch is tried again further in along the door, toward the hinge.
        d = np.array([np.cos(start), np.sin(start)])
        for further in EDGE_FURTHER:
            ctx.set_grip(1.); yield from hold(ctx, 10)
            yield from decor.slide(ctx, .08 * ctx.tip()[:3, 2], rate=.02)
            point, (ap, cl) = to_base(base, grasp(start)[0] + off - np.r_[further * d, 0.], grasp(start)[1:])
            if 'door' in ctx.obstacles:
                points_, radii_, _ = ctx.obstacles['door']; ctx.obstacles['door'] = (points_, radii_, True)     # (the hand may touch it)
            plan = decor.plan_pinch(ctx, point, frame(ap, cl), way_in=.08, hand_margin=-.05)
            if plan is None:
                ctx.event('edge_again', ok=False, further=further, reason='no way'); break
            ok = yield from decor.pinch(ctx, plan, label=how, gap=.002, lift=0.)
            ctx.event('edge_again', ok=bool(ok), further=further)
            if ok:
                break
    if not ok:
        keep_off(ctx, est); yield from tuck(ctx, (arm,)); return None
    angle = start; full = ctx.q.copy(); sign = 1. if end > start else -1.; trace = []
    # Where the hand has the door now (the wrist look put it right) against where the map pose put it: the arc is
    # shifted by as much (the base stood 2-3 cm off where it believed it was, and the hand closed beside the handle).
    shift = ctx.tip()[:3, 3][:2] - point[:2]
    shift = shift if float(np.linalg.norm(shift)) < .06 else np.zeros(2)
    trace.append(('shift', np.round(shift, 3).tolist()))
    for theta in np.arange(start + sign * step, end + sign * 1e-6, sign * step):
        p, (a2, c2) = to_base(base, grasp(theta)[0] + off, grasp(theta)[1:]); p = p + np.r_[shift, 0.]
        # (the rest of the robot keeps off the door as it will stand, and off the furniture; the working hand holds it)
        points, radii, _ = leaf(theta); ctx.obstacles['door'] = (points, radii, True)
        check = ctx.validator(relax_from=full); ctx.obstacles.pop('door', None)
        sol = ctx.ik.solve(full, p, frame(a2, c2), count=1, stay=.05, validator=check)
        # (near the next pose before any other: a far solution flips the wrist while it holds the door; a check that
        # failed there is relaxed, the door's balls first, the rest after)
        for relaxed in (ctx.validator(relax_from=full), None):
            if sol is not None and np.max(np.abs(sol - full[ctx.ik.columns])) <= .6:
                break
            sol = ctx.ik.solve(full, p, frame(a2, c2), count=1, stay=.05, validator=relaxed)
        if sol is None or np.max(np.abs(sol - full[ctx.ik.columns])) > .6:
            sol = ctx.ik.solve(full, p, frame(a2, c2), count=4, validator=check)
        if sol is None or np.max(np.abs(sol - full[ctx.ik.columns])) > .6:
            trace.append((round(float(np.degrees(theta))), 'no ik' if sol is None else 'jump %.2f' % float(np.max(np.abs(sol - full[ctx.ik.columns])))))
            break
        full[ctx.ik.columns] = sol
        status = yield from follow(ctx, [sol], rate=.01, max_steps=120)
        if float(np.sum(ctx.fingers(arm))) < .002:
            trace.append((round(float(np.degrees(theta))), 'let go')); break       # (the door stays where it was last held)
        angle = float(theta); trace.append((round(float(np.degrees(theta))), status[:1]))
    ctx.set_grip(1.)
    yield from hold(ctx, 20)
    est.door = angle if how == 'handle' or end < start else max(est.door, angle)
    yield from decor.slide(ctx, .08 * ctx.tip()[:3, 2], rate=.02)        # (back along the fingers)
    keep_off(ctx, est)
    yield from tuck(ctx, (arm, 'right' if arm == 'left' else 'left'))
    ctx.event('door', ok=True, how=how, arm=arm, start=round(float(np.degrees(start)), 1), end=round(float(np.degrees(angle)), 1), off=np.round(off[:2], 3).tolist(), trace=trace)
    return angle


def push(ctx, est, arm, start, end, radius=.40, offset=.08, z=1.0, step=np.radians(5.), face='inner'):
    """The closed hand pushes the door's inner face round the hinge from @start to @end (rad), at @radius from the
    hinge and @offset in front of the face (the hand starts @offset clear of it and meets it on the way), @z up.
    (In a trial the door, let go at 61 degrees, swung on to 115 under one push.) Returns the angle the hand got to."""
    ctx.arm = arm; base = est.base(); ctx.set_grip(-1.)
    # (on the way to the start the arm keeps off the door, the hand may touch it; the push itself is not checked)
    points, radii, _ = leaf(est.door); ctx.obstacles['door'] = (points, radii, True)
    yield from hold(ctx, 12)
    sign = 1. if face == 'inner' else -1.

    def at(theta):
        d = np.array([np.cos(theta), np.sin(theta)]); n = sign * np.array([np.sin(theta), -np.cos(theta)])     # (that face's normal)
        return np.r_[K.HINGE + radius * d + offset * n, z]
    # (fingers straight down, the hand upright behind the door: with the fingers level the wrist met the east door's handle)
    rotation = decor.top_rotation(float(np.arctan2(*(at(start) - np.r_[K.HINGE, 0.])[1::-1])) - base[2])
    point, _ = to_base(base, at(start))
    plan = decor.plan_pinch(ctx, point, rotation, way_in=.0001, hand_margin=-.05)
    if plan is None:
        ctx.event('door_push', ok=False, reason='no way to the start', arm=arm); ctx.set_grip(1.); keep_off(ctx, est); return None
    status = yield from move_to(ctx, plan['pre'], rate=.03, label='door_push')
    if status != 'done':
        ctx.set_grip(1.); keep_off(ctx, est); yield from tuck(ctx, (arm,)); return None
    from skills import interpolate_cartesian
    reached = start; direction = 1. if end > start else -1.
    for theta in np.arange(start + direction * step, end + direction * 1e-6, direction * step):
        p, _ = to_base(base, at(theta))
        way = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], p, rotation, step=.02)
        if way is None:
            break
        yield from follow(ctx, way, rate=.012, max_steps=200)
        reached = float(theta)
    est.door = (DOOR_STOP if reached > DOOR_STOP else max(est.door, reached)) if direction > 0 else min(est.door, reached)
    ctx.set_grip(1.)
    yield from decor.slide(ctx, [0., 0., .12], rate=.02)
    keep_off(ctx, est)
    yield from tuck(ctx, (arm,))
    ctx.event('door_push', ok=True, arm=arm, face=face, start=round(float(np.degrees(start)), 1), end=round(float(np.degrees(reached)), 1))
    return reached


def press_shut(ctx, est, arm):
    """The closed hand, its fingers level and pointing south, pushes the door's outer face from in front of the
    shut door to behind its face: a door the handle's lead left ajar (the hand had the handle by one finger and the
    door stood at 7 degrees, still open, 307) is pushed shut. Returns the follow status."""
    ctx.arm = arm; base = est.base(); ctx.set_grip(-1.)
    yield from hold(ctx, 10)
    heading = (-np.pi / 2 - base[2] + np.pi) % (2 * np.pi) - np.pi; rotation = decor.edge_rotation(heading, 0.)
    start, _ = to_base(base, np.r_[PRESS['x'], PRESS['y'][0], PRESS['z']]); end, _ = to_base(base, np.r_[PRESS['x'], PRESS['y'][1], PRESS['z']])
    ctx.obstacles.pop('door', None)
    plan = decor.plan_pinch(ctx, start, rotation, way_in=.0001, hand_margin=-.05)
    if plan is None:
        ctx.event('door_press', ok=False, reason='no way to the start'); ctx.set_grip(1.); keep_off(ctx, est); return 'no_plan'
    status = yield from move_to(ctx, plan['pre'], rate=.03, label='door_press')
    if status == 'done':
        from skills import interpolate_cartesian
        way = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], end, rotation, step=.02)
        if way is not None:
            status = yield from follow(ctx, way, rate=.008, max_steps=300)
        yield from decor.slide(ctx, np.r_[to_base((0., 0., base[2]), np.r_[0., .12, 0.])[0][:2], 0.], rate=.02)
    est.door = 0.
    ctx.set_grip(1.); keep_off(ctx, est)
    yield from tuck(ctx, (arm,))
    ctx.event('door_press', ok=True, status=status)
    return status


def run_stages(ctx, est, stages):
    from plates_pipeline import go, turn_to
    for kind, stance, arm, a0, a1, approach in stages:
        status = yield from go(ctx, est, approach, stance[2], tolerance=.05)
        if status != 'arrived':
            ctx.event('door', ok=False, reason='no way to the stance', stance=list(stance)); return False
        yield from turn_to(ctx, est, stance[2]); yield from settle(ctx, est, 6)
        if est.door > .2 and hasattr(est, 'door_look'):
            # (where the door stands, from a look at it, before a hand goes for it)
            yield from turn_to(ctx, est, float(np.arctan2(K.HINGE[1] + .2 - est.base()[1], K.HINGE[0] + .1 - est.base()[0])))
            yield from hold(ctx, 4)
            seen = est.door_look(K.HINGE, K.DOOR_LENGTH)
            ctx.event('door_look', believed=round(float(np.degrees(est.door)), 1), seen=None if seen is None else round(float(np.degrees(seen)), 1), debug=dict(getattr(est, 'door_debug', {})))
            if seen is not None:
                est.door = seen         # (pushed on to its stop by the arm the door stood at 136 degrees, believed at 96, 301)
            yield from turn_to(ctx, est, stance[2])
        status = yield from straight(ctx, stance[:2], stance[2], speed=.12, tolerance=.015)
        if kind == 'handle' and a1 < a0 and est.door > np.radians(25.):
            # (the free edge's lead let go of the door short of where the handle is taken: the edge again, on the
            # next attempt; the handle at 36 degrees closed on nothing, 301)
            ctx.event('door', ok=False, reason='still wide for the handle', at=round(float(np.degrees(est.door)), 1)); return False
        if kind == 'press':
            if est.door > np.radians(PRESS['most']):
                continue                # (only a door nearly shut is pressed: its face further out the hand would start behind it)
            yield from press_shut(ctx, est, arm)
            yield from straight(ctx, approach, stance[2], speed=.15)
            continue
        if kind == 'push':
            got = yield from push(ctx, est, arm, max(est.door - np.radians(6.), np.radians(a0)), np.radians(a1))
        elif kind == 'push_out':
            # (the closed hand round the free end, on the door's outer face, pushes it shut along the arc: no pinch of
            # a 7 cm edge between 10 cm fingers, which closed beside it from a pose 2-5 cm off, 305, 314, 319)
            got = yield from push(ctx, est, arm, est.door + np.radians(4.), np.radians(a1), radius=.45, face='outer')
        else:
            got = yield from lead(ctx, est, arm, np.radians(a0) if kind == 'handle' and est.door < .02 else est.door, np.radians(a1), how=kind)
        if got is None and kind == 'handle' and a1 < a0 and est.door <= np.radians(PRESS['most']):
            # (the hand closed on nothing at the handle, the door 17 degrees open five times over, test t13: pressed
            # shut instead)
            yield from press_shut(ctx, est, arm); got = 0.
        yield from straight(ctx, approach, stance[2], speed=.15)
        if got is None:
            return False
    return True


EDGE_WIDEN = ('edge', (5.30, .50, np.pi), 'right', 0., 105., (5.95, .50))
SHORT_OF = np.radians(88.)


def open_wide(ctx, est):
    ok = yield from run_stages(ctx, est, OPEN_STAGES)
    ok = ok and est.door > np.radians(85)
    if ok and WIDEN:
        believed = est.door
        if hasattr(est, 'door_look'):
            # (z02) where the door really stands: on 301 the lead left it at 70 degrees (believed 96) every time, and
            # its bins shut the west spot; then its free edge is led on to 105 first (z01's edge lead), then pushed
            from plates_pipeline import go, turn_to
            status = yield from go(ctx, est, (5.95, .60), np.pi, tolerance=.05)
            yield from turn_to(ctx, est, float(np.arctan2(K.HINGE[1] + .2 - est.base()[1], K.HINGE[0] + .1 - est.base()[0])))
            yield from hold(ctx, 4)
            seen = est.door_look(K.HINGE, K.DOOR_LENGTH)
            ctx.event('door_look', believed=round(float(np.degrees(est.door)), 1), seen=None if seen is None else round(float(np.degrees(seen)), 1), debug=dict(getattr(est, 'door_debug', {})))
            if seen is not None and seen < SHORT_OF:
                est.door = seen
                led = yield from run_stages(ctx, est, (EDGE_WIDEN,))
                ctx.event('door_edge_widen', ok=bool(led), door=round(float(np.degrees(est.door)), 1))
        widened = yield from run_stages(ctx, est, WIDEN)
        ctx.event('door_widen', ok=bool(widened), door=round(float(np.degrees(est.door)), 1), believed=round(float(np.degrees(believed)), 1))
    return ok


def shut(ctx, est, stages=SHUT_STAGES):
    # (the free edge's stage only while the door stands wide: from 25 degrees in the handle leads it)
    stages = stages if est.door > np.radians(25.) else tuple(st for st in stages if st[0] in ('handle', 'press'))
    if stages is SHUT_STAGES and est.door > np.radians(80.):
        stages = (NEAR_EDGE,) + stages
    ok = yield from run_stages(ctx, est, stages)
    return ok and est.door < K.DOOR_OPEN_FROM
