"""A hinged leaf led open by the hand: a cabinet's or a fridge's door (vertical axis), a bin's or a box's lid
(horizontal axis).

The hand pinches the leaf (its handle or its free edge), then is led along the arc that the pinched point makes about
the hinge's axis, a few degrees at a time, the hand's frame turning with the leaf. This is the fridge-door skill of
`cleaning_up_plates_and_food` f01 (`door.py`, vertical axis only) made general, in the controller family of
`store_batteries` x01 (whose drawer skill gives the pinch: a level hand led a little behind the leaf's face, so that it
stops at the face with the edge between the fingers, then centred on it by the fingers' difference).

The hinge's line (a point and a direction in the map) comes from a probe of the scene: dev job `hinge_probe.py` reads
the child link's pose at several joint angles; the axis is the line those poses rotate about.
"""
import numpy as np
from manip import hold, follow, move_to, pose2, BASE_Z
from skills import interpolate_cartesian
from trash_skills import tuck, least_loaded

HELD = .006             # fingers further apart than this (sum, m) hold the leaf
CENTRED = .008          # fingers closer than this to each other's opening hold it in the hand's middle
HAND_INTO = -.11        # the hand may come this far into the leaf's outline in the map (the fingers round its edge)


def rotation_about(axis, angle):
    """Rotation matrix by @angle (rad) about unit vector @axis (right-hand rule)."""
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    k = np.array([[0., -a[2], a[1]], [a[2], 0., -a[0]], [-a[1], a[0], 0.]])
    return np.eye(3) + np.sin(angle) * k + (1. - np.cos(angle)) * (k @ k)


def hand_frame(approach, closing):
    """Tool rotation (columns x, y = closing, z = back along the fingers) for a hand coming along @approach."""
    z = -np.asarray(approach, float); z = z / np.linalg.norm(z)
    y = np.asarray(closing, float); y = y - np.dot(y, z) * z; y = y / np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


def to_base(est, point_map, rotation_map=None):
    inv = np.linalg.inv(pose2(*est.base()))
    p = (inv @ np.r_[point_map[0], point_map[1], point_map[2] - BASE_Z, 1.])[:3]
    return p if rotation_map is None else (p, inv[:3, :3] @ rotation_map)


def pinch(ctx, est, arm, point, approach, closing, way_in=.10):
    """Pinch the leaf at @point (map) with @arm, the hand coming along @approach (map) for the last @way_in, the
    fingers closing along @closing (map). Returns (held, rotation in the map) or (None, None) if no way there."""
    ctx.arm = arm; rot_map = hand_frame(approach, closing)
    p, rot = to_base(est, np.asarray(point, float), rot_map); ahead = -rot[:, 2]
    check = ctx.validator(hand_margin=HAND_INTO)
    plan = None
    for load, rotation, low in least_loaded(ctx, ctx.q, p, [rot], check, seeds=2)[:12]:
        start = ctx.q.copy(); start[ctx.ik.columns] = low
        back = interpolate_cartesian(ctx.ik, start, p, p - way_in * ahead, rotation, validator=check, step=.02)
        if back is not None:
            plan = dict(pre=back[-1], way=back[::-1][1:] + [low], rotation=rotation, load=float(load)); break
    if plan is None:
        ctx.event('hinge_pinch', ok=False, reason='no way', arm=arm); return None, None
    ctx.set_grip(1.)
    status = yield from move_to(ctx, plan['pre'], rate=.04, label='hinge_pre')
    if status != 'done':
        return None, None
    yield from follow(ctx, plan['way'], rate=.012, max_steps=200)       # (it stalls at the leaf's face)
    yield ctx.action(chain=ctx.chain)
    ctx.set_grip(-1.)
    yield from hold(ctx, 25)
    for fix in range(2):
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
    held = float(np.sum(ctx.fingers())) > HELD
    ctx.event('hinge_pinch', ok=held, arm=arm, fingers=ctx.fingers().round(4).tolist(), load=round(plan['load'], 2))
    return held, rot_map


def lead(ctx, est, hinge, axis, angle, step=np.radians(4.)):
    """With the active hand on the leaf, lead it by @angle (rad, right-hand rule about @axis) about the line through
    @hinge (map) along @axis. The hand's present pose is the arc's start. Returns the angle the leaf was led to (as far
    as the hand got with the fingers still on it)."""
    world = pose2(*est.base()); tip = ctx.tip(); start_p = world[:3, :3] @ tip[:3, 3] + world[:3, 3] + [0., 0., BASE_Z]
    start_r = world[:3, :3] @ tip[:3, :3]; hinge = np.asarray(hinge, float); full = ctx.q.copy(); reached = 0.; trace = []
    for k in range(1, int(abs(angle) / step) + 1):
        a = np.sign(angle) * k * step; R = rotation_about(axis, a)
        p, rot = to_base(est, hinge + R @ (start_p - hinge), R @ start_r)
        sol = None
        for check in (ctx.validator(relax_from=full), None):
            sol = ctx.ik.solve(full, p, rot, count=4, validator=check)
            if sol is not None and np.max(np.abs(sol - full[ctx.ik.columns])) <= .6:
                break
            sol = None
        if sol is None:
            trace.append((round(float(np.degrees(a))), 'no ik')); break
        full[ctx.ik.columns] = sol
        yield from follow(ctx, [sol], rate=.012, max_steps=120)
        if float(np.sum(ctx.fingers())) <= HELD:
            trace.append((round(float(np.degrees(a))), 'let go')); break
        reached = float(a)
    ctx.event('hinge_lead', reached=round(float(np.degrees(reached)), 1), wanted=round(float(np.degrees(angle)), 1), trace=trace[-4:])
    return reached


def let_go(ctx, back=.10):
    """Open the hand, draw it back along the fingers and fold the arm."""
    ctx.set_grip(1.)
    yield from hold(ctx, 15)
    tip = ctx.tip()[:3, 3]; rotation = ctx.tip()[:3, :3]
    way = interpolate_cartesian(ctx.ik, ctx.q, tip, tip + back * rotation[:, 2], rotation, step=.025)
    if way is not None:
        yield from follow(ctx, way, rate=.03, max_steps=120)
    yield from tuck(ctx, (ctx.arm,))
