"""Cutting the onion in the bowl with the knife in the right hand: the onion went into the bowl, so the halves and the dice are born inside the
bowl. cut_start / chop_go (a stand from which the right hand reaches the cut point), chop (the knife only touches: a slow descent onto the top of
the onion, the native rules cut on contact), cut_up, look_bowl (the bowl seen again: whole onion, halves, nothing), wait_slicer (the slicer needs
SLICER_WAIT controls without contact), dice (the same on a half, with a sideways sweep). The knife stays in the hand until the sink."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import params as P
from ..world import World


def rot_hold(c) -> np.ndarray:
    return c.hand_rot if c.hand_rot is not None else G.tool_rotation(-G.UP)


def whole_blob(c, w) -> Optional[dict]:
    blobs = c.perc.pieces.get('blobs') or []
    return max(blobs, key=lambda b: (b['top'], b['cells'])) if blobs else None


def target_of(c, w, blob: dict) -> np.ndarray:
    """Blob top, odometry frame."""
    return np.array([blob['xy'][0], blob['xy'][1], w.counter.top + float(blob['top'])])


def step_cut_start(c, action, q, w) -> None:
    c.hold_arm(action, q, 'right')
    c.rest_one(action, q, 'left')
    blob = whole_blob(c, w)
    if blob is None:
        bowl = w.items['bowl']
        c.cut_target = np.array([bowl.centre[0], bowl.centre[1], bowl.top + 0.03])
        c.history.append((c.step, 'cut_start', 'no blob seen in the bowl: the bowl centre'))
    else:
        c.cut_target = target_of(c, w, blob)
    c.stand = None
    c.cut_mode = 'chop'
    c.change('chop_go', f'cut target {np.round(c.cut_target, 3).tolist()}')


def step_chop_go(c, action, q, w) -> None:
    if c.hold_q is None:
        c.hold_q = q.copy()
    rot = rot_hold(c)
    point = c.cut_target + np.array([0., 0., 0.05])
    if c.stand is None:
        c.build_grid(w, [c.cut_target, w.base[:2]], P.ROBOT_RADIUS)
        goals = lambda: [(point, rot)]  # noqa: E731
        here = c.reach_mask('right', goals(), np.array([[w.base[0], w.base[1], w.base[2]]]), w)
        if bool(here[0]):
            c.stand = (w.base[:2].copy(), float(w.base[2]))
            c.change(c.cut_mode, 'the knife reaches the cut point from here')
            return
        stand = c.pick_stand(w, c.cut_target, goals, 'right', yaw_near=float(w.base[2]), max_ring=1.0)
        if stand is None:
            stand = c.pick_stand(w, c.cut_target, goals, 'right', max_ring=1.0)
        if stand is None:
            c.change(c.cut_mode, 'no cut stand: cut from here')
            return
        c.stand = stand
    dist, yaw = c.drive(action, q, w, c.stand[0], c.stand[1], hold=True)
    if c.arrived_at(dist, yaw):
        c.change(c.cut_mode, 'at the cut stand')
    elif c.age > P.PHASE_TIMEOUT['chop_go']:
        c.change(c.cut_mode, 'cut stand walk timed out')


def step_chop(c, action, q, w) -> None:
    c.side = 'right'
    c.grip['right'] = K.GRIPPER_CLOSED
    t = c.cut_target
    target = np.array([t[0], t[1], t[2] + 0.02 - 0.0002 * min(c.age, 250) - 0.0002 * 125 * c.cut_rounds])
    c.servo(action, q, w, target, rot_hold(c), plan=False, stepwise=True)
    if c.age >= P.CHOP_STEPS:
        c.counts['chop_rounds'] += 1
        c.change('cut_up', f'chop done err={c.err[0]:.3f}')


def step_cut_up(c, action, q, w) -> None:
    c.side = 'right'
    c.grip['right'] = K.GRIPPER_CLOSED
    t = c.cut_target
    up = np.array([t[0], t[1], t[2] + 0.28])
    c.servo(action, q, w, up, rot_hold(c), plan=False, stepwise=True)
    if c.age >= 40 or c.err[0] < 0.03:
        c.hold_q = q.copy()
        c.after_look = 'cut_check' if c.cut_mode == 'chop' else 'dice_check'
        c.bowl_before = dict(c.perc.pieces)
        c.change('look_bowl', 'knife up')


def step_look_bowl(c, action, q, w) -> None:
    if c.after_look in ('cut_check', 'dice_check'):
        c.hold_arm(action, q, 'right')
        c.rest_one(action, q, 'left')
        c.grip['right'] = K.GRIPPER_CLOSED
    else:
        c.to_rest(action, q)
    if c.age < P.LOOK_STEPS:
        return
    pieces = c.perc.pieces
    blobs = pieces.get('blobs') or []
    top = float(pieces.get('top', 0.0))
    kind = c.after_look
    if kind == 'bowl_onion':
        c.stages['onion'] = True
        c.history.append((c.step, 'look_bowl', f'bowl holds n={pieces.get("n")} top={top:.3f} blobs={len(blobs)}'))
        c.change('choose', 'onion in the bowl' if pieces.get('n') else 'no onion seen in the bowl (go on)')
    elif kind == 'cut_check':
        cut = len(blobs) >= 2 or (blobs and top < P.WHOLE_TOP_M) or not blobs
        if cut or c.cut_rounds >= 2:
            c.cut_mode, c.cut_rounds = 'dice', 0
            c.change('wait_slicer', f'onion cut: {len(blobs)} blobs, top {top:.3f}' if cut else 'whole onion after rounds: go on')
        else:
            c.cut_rounds += 1
            c.change('chop_go', f'the onion is whole (top {top:.3f}): chop deeper round {c.cut_rounds}')
    elif kind == 'dice_check':
        before = c.bowl_before.get('blobs') or []
        diced = (len(blobs) < len(before)) or (blobs and top < c.bowl_before.get('top', 1.0) - 0.012) or not blobs
        if diced or c.cut_rounds >= 2:
            c.stages['parer_cut'] = True
            c.change('choose', 'diced' if diced else 'half unchanged after rounds: go on')
        else:
            c.cut_rounds += 1
            c.change('wait_slicer', f'half not diced yet: round {c.cut_rounds}')


def step_wait_slicer(c, action, q, w) -> None:
    c.side = 'right'
    c.hold_arm(action, q, 'right')
    c.rest_one(action, q, 'left')
    c.grip['right'] = K.GRIPPER_CLOSED
    if c.age >= P.SLICER_WAIT:
        blobs = c.perc.pieces.get('blobs') or []
        if not blobs:
            c.stages['parer_cut'] = True
            c.change('choose', 'nothing left to dice')
            return
        half = max(blobs, key=lambda b: b['cells'])
        c.cut_target = target_of(c, w, half)
        c.cut_mode = 'dice'
        c.stand = None
        c.change('chop_go', f'dice the half at {np.round(c.cut_target, 3).tolist()}')


def step_dice(c, action, q, w) -> None:
    c.side = 'right'
    c.grip['right'] = K.GRIPPER_CLOSED
    t = c.cut_target
    target = np.array([t[0], t[1], t[2] + 0.02 - 0.0002 * min(c.age, 250) - 0.0002 * 125 * c.cut_rounds])
    d = t[:2] - w.base[:2]
    perp = np.array([-d[1], d[0]]) / max(float(np.linalg.norm(d)), 1e-6)
    target[:2] = target[:2] + perp * P.DICE_SWEEP_M * math.sin(2 * math.pi * c.age / P.DICE_SWEEP_PERIOD)
    c.servo(action, q, w, target, rot_hold(c), plan=False, stepwise=True)
    if c.age >= P.DICE_STEPS:
        c.counts['dice_rounds'] += 1
        c.change('cut_up', f'dice done err={c.err[0]:.3f}')


PHASES = {'cut_start': step_cut_start, 'chop_go': step_chop_go, 'chop': step_chop, 'cut_up': step_cut_up, 'look_bowl': step_look_bowl,
          'wait_slicer': step_wait_slicer, 'dice': step_dice}
