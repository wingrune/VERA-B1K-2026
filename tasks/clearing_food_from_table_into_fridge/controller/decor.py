"""The skills of putting_away_Halloween_decorations that are new: the pinches of the pumpkin's stem, the candle's wick,
the caldron's rim and the drawer's lip, and the drawer pulled out and pushed shut.

Frames: 'base frame' is the robot base's frame; map coordinates (M) are the scene's. Every pinch is planned for the
fingers' ends (manip.TIP) at a point, the hand coming in along its fingers.
"""
import numpy as np
from manip import hold, follow, move_to
from skills import interpolate_cartesian
from car_skills import frame, CLOSE_STEPS

# ---- the things ---------------------------------------------------------------------------------------------
# Pumpkin: 15 x 15 cm, 12.6 cm high; its body ends 9.7-9.9 cm up, its stem (2.8 x 1.8 cm) stands on the middle of it
# up to 12.5 cm. Wider than the fingers open (10 cm): taken by the stem.
# Candle (pillar): 10.1 cm across, its body 11.7 cm high, a 5 mm wick on the middle of its top up to 14 cm.
# Caldron: 29.4 cm across, 25.2 cm high, open at the top; its wall is 4-6 cm thick for the collision model.
SIZE = dict(pumpkin=(.152, .126), candle=(.101, .140), caldron=(.294, .252))
STEM_TOP, BODY_TOP = dict(pumpkin=.125, candle=.140), dict(pumpkin=.098, candle=.117)
# Where the stem (the wick) is, at the height the fingers take it, in the thing's own frame (its origin 5.2 cm up for
# the pumpkin, 5.5 cm for the candle). The pumpkin's stem leans: 6 mm from its middle at its foot, 1.2 cm at its top.
GRIP_OWN = dict(pumpkin=(.0015, .008, .060), candle=(0., 0., .073))
RIM_TOP = .252

# ---- the cabinet --------------------------------------------------------------------------------------------
# bottom_cabinet_rhdbzv_0 against the living room's east wall: its front at map x 6.015 (the drawers' lip), 0.413 high;
# two drawers side by side that slide out along -x by up to 0.297 m. Along the top of each drawer's front a lip
# 2 cm deep and 8 mm thick (map x 6.015-6.035, height 0.371-0.379) is the handle.
CABINET_FRONT = 6.015
LIP = dict(depth=.020, height=.375, margin=-.08)        # margin: how far the hand may come into the cabinet as the map has it (the map's cabinet reaches 6 cm in front of the lip)
DRAWERS = dict(left=dict(joint='j_left_drawer', y=(4.588, 5.649), inside=(4.626, 5.410)),
               right=dict(joint='j_right_drawer', y=(3.331, 4.568), inside=(3.373, 4.070)))
DRAWER_OUT = .297                       # all the way out
DRAWER_FLOOR = .083                     # height of a drawer's floor; its walls end at 0.324
OPEN_FROM = .0155                       # a drawer further out than this counts as open (5 % of its travel)


def top_rotation(closing):
    """Fingers straight down, closing along map/base direction @closing (rad)."""
    return frame(np.array([0., 0., -1.]), np.array([np.cos(closing), np.sin(closing), 0.]))


def edge_rotation(heading, pitch):
    """Fingers pointing along @heading and down by @pitch, closing in the upright plane through @heading: one finger on
    the top face of what is there, the other on its side (the slanted pinch of an edge)."""
    approach = np.array([np.cos(heading) * np.cos(pitch), np.sin(heading) * np.cos(pitch), -np.sin(pitch)])
    closing = np.array([np.cos(heading) * np.sin(pitch), np.sin(heading) * np.sin(pitch), np.cos(pitch)])
    return frame(approach, closing)


def lip_rotation(heading):
    """Fingers level along @heading, closing up and down (across the drawer's lip)."""
    return frame(np.array([np.cos(heading), np.sin(heading), 0.]), np.array([0., 0., 1.]))


def plan_pinch(ctx, point, rotation, base=None, q=None, way_in=.10, locked=None, count=24, hand_margin=None):
    """The active hand's fingers' ends at @point (base frame) with tool rotation @rotation, coming in along its fingers
    from @way_in: dict(grip, pre, inward, point, rotation) or None. hand_margin: how far the working hand may come
    into what the map has (None: manip.HAND_MARGIN; the drawer's lip is part of the cabinet for the map)."""
    full = ctx.q if q is None else np.asarray(q, float); point = np.asarray(point, float)
    check = ctx.validator(base=base) if hand_margin is None else ctx.validator(base=base, hand_margin=hand_margin)
    approach = -rotation[:, 2]
    for grip in ctx.ik.solve(full, point, rotation, count=count, validator=check, all_solutions=True, locked=locked)[:6]:
        at = full.copy(); at[ctx.ik.columns] = grip
        out = interpolate_cartesian(ctx.ik, at, point, point - way_in * approach, rotation, validator=check, step=.025, locked=locked)
        if out is not None:
            return dict(grip=grip, pre=out[-1], inward=out[::-1][1:] + [grip], point=point, rotation=rotation)
    return None


def pinch(ctx, plan, label='pinch', gap=.004, lift=.10, most=None, look=None, replan=None, debug=None, looks=2):
    """Close the active hand as @plan has it and lift straight up by @lift. Returns True if the fingers stayed further
    apart than @gap (sum of both, m) and, if @most is given, closer than that, False if they did not, None if the
    hand did not get there.
    look(tip) -> base-frame xy or None: where the thing is as the wrist camera shows it once the hand hangs at the
    plan's start; replan(point) -> a new plan for the fingers' ends at @point. The hand is put right by what the
    look shows before it goes in (a 5 mm wick and a 2 cm stem leave a few millimetres to spare across the fingers)."""
    ctx.set_grip(1.)
    status = yield from move_to(ctx, plan['pre'], rate=.04, label=label)
    if status != 'done':
        ctx.event(label, ok=False, reason='no way there', arm=ctx.arm, status=status); return None
    yield from hold(ctx, 4)
    shifts = []
    if look is not None and replan is not None:
        for again in range(looks):
            yield from hold(ctx, 3)
            seen = look(ctx.tip()[:3, 3])
            shift = None if seen is None else np.asarray(seen, float)[:2] - plan['point'][:2]
            shifts.append(None if shift is None else np.round(shift, 4).tolist())
            if seen is None and debug is not None:
                shifts.append(debug())
            if shift is None or float(np.linalg.norm(shift)) < .002 or float(np.linalg.norm(shift)) > .05:
                break
            better = replan(np.r_[plan['point'][:2] + shift, plan['point'][2]])
            if better is None:
                break
            status = yield from move_to(ctx, better['pre'], rate=.03, label=label)
            if status != 'done':
                return None
            plan = better
    status = yield from follow(ctx, plan['inward'], rate=.015, max_steps=300)
    reached = ctx.tip()[:3, 3].copy()
    ctx.set_grip(-1.)
    yield from hold(ctx, CLOSE_STEPS)
    fingers = ctx.fingers().copy()
    up = None
    if lift:
        up = interpolate_cartesian(ctx.ik, ctx.q, reached, reached + [0., 0., lift], plan['rotation'], step=.025)
        if up is not None:
            yield from follow(ctx, up, rate=.015, max_steps=300)
    yield from hold(ctx, 4)
    ok = float(np.sum(ctx.fingers())) > gap and (most is None or float(np.sum(ctx.fingers())) < most)
    ctx.event(label, ok=ok, arm=ctx.arm, way_in=status, lifted=up is not None, fingers=fingers.round(4).tolist(), after_lift=ctx.fingers().round(4).tolist(),
              short=(reached - plan['point']).round(3).tolist(), point=np.round(plan['point'], 3).tolist(), looked=shifts or None)
    if not ok:
        ctx.set_grip(1.)
    return ok


def slide(ctx, shift, rotation=None, rate=.008, max_steps=900):
    """Move the active hand by @shift (base frame) in a straight line, its rotation kept. Returns the status."""
    tip = ctx.tip(); rotation = tip[:3, :3] if rotation is None else rotation
    way = interpolate_cartesian(ctx.ik, ctx.q, tip[:3, 3], tip[:3, 3] + np.asarray(shift, float), rotation, step=.02)
    if way is None:
        return 'no_way'
    status = yield from follow(ctx, way, rate=rate, max_steps=max_steps)
    return status


def let_go(ctx, back=.06):
    """Open the active hand and draw it back along its fingers."""
    ctx.set_grip(1.)
    yield from hold(ctx, 15)
    tip = ctx.tip()
    status = yield from slide(ctx, back * tip[:3, 2], rate=.02)
    return status
