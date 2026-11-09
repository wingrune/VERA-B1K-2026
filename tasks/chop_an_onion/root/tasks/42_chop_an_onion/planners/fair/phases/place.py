"""Carrying the onion into the bowl: carry (the base moves by place - object at the same yaw, the arm keeps its joints), lower (resolved-rate),
release, then look at the bowl. Moving the base without turning keeps the held object on a straight line over the counter."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import params as P
from ..world import World

CORRIDOR_MARGIN_M = 0.035          # the onion sweeps over the counter clutter with this much room
ONION_SPAN_M = 2 * P.ONION_R_M     # tip to the onion bottom when the fingers pinch near its top


BOWL_NEAR_SINK_M = 0.8           # the bowl stands on the counter beside the sink (task setup), not down the counter

CARRY_STALL_STEPS = 150          # controls of commanded drive that move the base less than CARRY_STALL_MOVE_M: a jam
CARRY_STALL_MOVE_M = 0.015
CARRY_STALL_SHIFT_M = 0.22       # lateral offset of a jammed translate target before it is re-chosen
CARRY_MAX_JAMS = 3


def bowl_plausible(w: World) -> bool:
    """Sanity of the bowl estimate before the held onion is released over it.
    place_point is frozen from an earlier look; a corrupted sighting leaves it
    0.4-0.9 m off (measured on 301/305/306), the onion is let go beside the real
    bowl and the harness never latches onion_in_bowl. A bowl next to the sink
    cannot lie inside the basin box, outside the counter rectangle or far down
    the counter from the basin centre."""
    bowl = w.items.get('bowl')
    if bowl is None or bowl.centre is None or w.counter is None:
        return True
    d = np.asarray(bowl.centre[:2], float) - w.counter.centre[:2]
    if abs(float(d @ w.counter.u)) > 0.5 * float(w.counter.length) + 0.15 or \
            abs(float(d @ w.counter.n)) > 0.5 * float(w.counter.depth) + 0.06:
        return False
    basin = getattr(w, 'basin', None)
    if basin is None or getattr(basin, 'centre', None) is None:
        return True
    db = d - (np.asarray(basin.centre[:2], float) - w.counter.centre[:2])
    if float(np.linalg.norm(db)) > BOWL_NEAR_SINK_M:
        return False
    return not (abs(float(db @ w.counter.u)) < 0.5 * float(basin.size[0]) + 0.02 and
                abs(float(db @ w.counter.n)) < 0.5 * float(basin.size[1]) + 0.02)


def place_point(c, w: World) -> np.ndarray:
    bowl = w.items['bowl']
    return np.array([bowl.centre[0], bowl.centre[1], float(bowl.top) + P.BOWL_NEAR_RISE_M])


def translate_target(c, w: World, place: np.ndarray, hand: np.ndarray) -> np.ndarray:
    """Base target of the straight translate carry. After a jam the target is shifted sideways: the straight route from the old base is
    what was blocked, a parallel one a little to the side may not be."""
    xy = w.base[:2] + (place[:2] - hand[:2])
    k = int(getattr(c, 'carry_shift', 0) or 0)
    if k:
        d = np.asarray(place[:2], float) - np.asarray(w.base[:2], float)
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            xy = xy + np.array([-d[1], d[0]]) / n * (CARRY_STALL_SHIFT_M * (1 if k % 2 else -1))
    return xy


def carry_stand(c, w: World, q: np.ndarray) -> Optional[tuple]:
    """Stand that carries the hand to the place.
    The base moves by (place - hand) at its present yaw when that cell is free in the tight footprint and has a route; else the nearest
    stand that reaches the place point. A jammed carry shifts the translate target sideways (carry_shift) before the fallback."""
    c.closer_stand = False
    place = place_point(c, w)
    hand = c.perc.tips_mid(q, c.side)
    xy = translate_target(c, w, place, hand)
    tight = c.raw_grid.inflated(P.TIGHT_RADIUS) if c.raw_grid is not None else None
    if tight is not None and not tight.blocked[tight.cell(xy)] and tight.path(w.base[:2], xy) is not None:
        c.history.append((c.step, 'carry', f'translate stand {np.round(xy, 3).tolist()}'))
        return xy, float(w.base[2])
    c.history.append((c.step, 'carry', 'translate stand blocked: nearest stand'))
    rot = c.hand_rot if c.hand_rot is not None else G.tool_rotation(-G.UP)
    stand = c.pick_stand(w, place, lambda: [(place, rot)], c.side)
    if stand is None:                         # tight footprint: the base fits 0.30 m from the counter and from the clutter beside it
        c.retighten(w, P.TIGHT_RADIUS)
        stand = c.pick_stand(w, place, lambda: [(place, rot)], c.side)
        if stand is None:
            c.retighten(w, P.ROBOT_RADIUS)
    return stand


def carry_closer(c, w: World) -> Optional[tuple]:
    """No stand reaches the place at all: walk along the base-place line to the nearest free, routable point and re-choose the stand from
    there. Moving the base is the changed input; a point right at the base is accepted only as the last stop, from where the lower servo
    tries anyway."""
    place = place_point(c, w)
    g = c.raw_grid.inflated(P.ROBOT_RADIUS) if c.raw_grid is not None else None
    if g is None:
        return None
    a = np.asarray(w.base[:2], float)
    b = np.asarray(place[:2], float)
    for t in (0.85, 0.7, 0.55, 0.4, 0.25):
        xy = a + t * (b - a)
        if not g.blocked[g.cell(xy)] and g.path(a, xy) is not None:
            c.closer_stand = bool(np.linalg.norm(xy - a) >= 0.1)
            c.history.append((c.step, 'carry', f'no stand at the place: closer stand {np.round(xy, 3).tolist()}'))
            return xy, float(w.base[2])
    return None


def corridor_z_need(c, w: World, q: np.ndarray) -> float:
    """Tip height that keeps the held onion above everything it sweeps over.
    The object path of the translate carry is the straight segment hand -> place: the counter top and every fitted item that stands in that
    strip set the floor, the tip must ride above it by the onion span plus a margin."""
    place = place_point(c, w)
    hand = c.perc.tips_mid(q, c.side)
    a, b = hand[:2], place[:2]
    ab = b - a
    l2 = float(ab @ ab)
    top = float(w.counter.top) if w.counter is not None else 0.0
    for role, it in w.items.items():
        if role in ('bowl', 'onion') or it is None or it.centre is None:
            continue
        it_top = getattr(it, 'top', None)
        if it_top is None:
            continue
        p = np.asarray(it.centre, float)[:2]
        t = 0.0 if l2 < 1e-9 else float(np.clip(((p - a) @ ab) / l2, 0.0, 1.0))
        if float(np.linalg.norm(p - (a + t * ab))) < P.ONION_R_M + CORRIDOR_MARGIN_M:
            top = max(top, float(it_top))
    return top + ONION_SPAN_M + CORRIDOR_MARGIN_M


def step_carry_lift(c, action, q, w) -> None:
    """Resolved-rate raise of the held onion until its bottom clears the corridor, before the base walks. Pure vertical move: the xy is
    fixed at the phase start, the rotation stays the held one, the base does not move."""
    hand = c.perc.tips_mid(q, c.side)
    rot = c.hand_rot if c.hand_rot is not None else G.tool_rotation(-G.UP)
    c.servo(action, q, w, np.array([c.carry_xy[0], c.carry_xy[1], float(c.carry_z)]), rot, plan=False, stepwise=True)
    if not c.holds(q, c.side):
        c.carry_raised = False
        c.grip[c.side] = K.GRIPPER_OPEN
        c.counts['drops'] += 1
        c.perc.held_cut = None
        c.carry_shift = 0
        c.carry_jams = 0
        c.grasp_failed(w, f'{c.role} dropped in the raise')
        return
    if c.err[0] < 0.03 or c.age > 150:
        c.carry_raised = True
        if not bowl_plausible(w) and getattr(c, 'bowl_relooks', 0) < 2:
            c.bowl_relooks = getattr(c, 'bowl_relooks', 0) + 1
            c.after_look = 'carry_lift'
            c.change('look', f'bowl estimate implausible ({c.bowl_relooks}): re-look before the release')
            return
        c.bowl_relooks = 0
        c.history.append((c.step, 'carry_lift', f'tip {hand[2]:.3f} of {c.carry_z:.3f} err={c.err[0]:.3f}'))
        c.change('carry', 'onion above the corridor: walk')


def step_carry(c, action, q, w) -> None:
    if c.stand is None:
        if not getattr(c, 'carry_raised', False):
            need = corridor_z_need(c, w, q)
            hand = c.perc.tips_mid(q, c.side)
            if hand[2] < need - 0.01:
                c.carry_z = float(need)
                c.carry_xy = np.asarray(hand[:2], float).copy()
                c.change('carry_lift', f'held at {hand[2]:.3f}, the corridor needs {need:.3f}')
                return
        c.build_grid(w, [place_point(c, w), w.base[:2]], P.ROBOT_RADIUS)
        goal = carry_stand(c, w, q)
        if goal is None:
            goal = carry_closer(c, w)
        if goal is None:
            c.debug[f'carry_fail_{c.step}'] = dict(base=np.round(w.base, 2).tolist(), stand=np.round(place_point(c, w)[:2], 2).tolist(),
                                                   rows=c.ascii_grid(w, place_point(c, w), half=3.0, res=0.1, marks=[xy for xy, ok in c.last_cands if ok][:40]))
            c.change('plan', f'no stand at the place {c.pick_stats}')
            return
        c.stand = goal
    dist, yaw = c.drive(action, q, w, c.stand[0], c.stand[1], hold=True)
    if c.age in (200, 800):
        c.debug[f'carry_grid_{c.step}'] = dict(base=np.round(w.base, 2).tolist(), stand=np.round(c.stand[0], 2).tolist(), rows=c.ascii_grid(w, c.stand[0]))
    if not c.holds(q, c.side):
        c.carry_raised = False
        c.carry_shift = 0
        c.carry_jams = 0
        c.carry_base = None
        c.grip[c.side] = K.GRIPPER_OPEN
        c.counts['drops'] += 1
        c.perc.held_cut = None
        c.grasp_failed(w, f'{c.role} dropped')
        return
    base2 = np.asarray(w.base[:2], float)
    last = getattr(c, 'carry_base', None)
    if dist > 0.1 and last is not None and float(np.linalg.norm(base2 - last)) < CARRY_STALL_MOVE_M:
        c.carry_stall = getattr(c, 'carry_stall', 0) + 1
    else:
        c.carry_stall = 0
    c.carry_base = base2
    if c.carry_stall >= CARRY_STALL_STEPS:
        c.carry_stall = 0
        c.carry_jams = getattr(c, 'carry_jams', 0) + 1
        c.debug[f'carry_jam_{c.step}'] = dict(base=np.round(base2, 2).tolist(), stand=np.round(c.stand[0], 2).tolist())
        c.stand = None
        if c.carry_jams <= CARRY_MAX_JAMS:
            c.carry_shift = getattr(c, 'carry_shift', 0) + 1
            c.history.append((c.step, 'carry', f'jam {c.carry_jams}: the base does not move, another stand'))
            c.change('carry', 'carry jam: another stand')
        return
    if c.arrived_at(dist, yaw):
        c.carry_stall = 0
        if getattr(c, 'closer_stand', False):
            c.closer_stand = False
            c.stand = None
            c.change('carry', 'at the closer stand: re-choose')
        else:
            c.change('lower', 'at the place')
    elif c.age > P.PHASE_TIMEOUT['carry']:
        c.carry_raised = False
        c.carry_shift = 0
        c.carry_jams = 0
        c.change('plan', 'carry timed out')


def step_lower(c, action, q, w) -> None:
    point = place_point(c, w)
    rot = c.hand_rot if c.hand_rot is not None else G.tool_rotation(-G.UP)
    c.servo(action, q, w, point, rot, plan=False, stepwise=True)
    if c.err[0] < 0.045 or c.age > P.PHASE_TIMEOUT['lower']:
        c.dwell = 0
        c.grip[c.side] = K.GRIPPER_OPEN
        c.change('release', f'{c.role} down')


def step_release(c, action, q, w) -> None:
    c.grip[c.side] = K.GRIPPER_OPEN
    up = place_point(c, w) + np.array([0., 0., 0.12])
    rot = c.hand_rot if c.hand_rot is not None else G.tool_rotation(-G.UP)
    c.servo(action, q, w, up, rot, plan=False, stepwise=True)
    c.dwell += 1
    if c.dwell >= P.OPEN_DWELL:
        c.carry_raised = False
        c.carry_shift = 0
        c.carry_jams = 0
        c.hold_q = None
        c.perc.held_cut = None
        c.perc.frozen.discard(c.role)
        c.perc.forget('onion')
        c.after_look = 'bowl_onion'
        c.change('look_bowl', 'onion released')


PHASES = {'carry': step_carry, 'carry_lift': step_carry_lift, 'lower': step_lower, 'release': step_release}
