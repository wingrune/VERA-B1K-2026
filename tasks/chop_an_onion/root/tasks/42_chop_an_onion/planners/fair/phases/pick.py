"""Choosing the next step and picking an item up: choose, plan (a stand by the batched GPU reach scan), go, reach (pregrasp), grasp (descend,
close: the native assisted grasp holds what lies between the fingers, read from the finger gap), lift (resolved-rate, no IK branch change).
Order of the task (as the oracle): the onion (left) into the bowl, the knife (right) into the hand, cut the onion and a half, the knife into the
sink, the board (right) into the sink."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import params as P
from ..world import ItemEst, World

SIDE = {'onion': 'left', 'parer': 'right', 'board': 'right'}
GRASP_DZ = {'onion': -0.015, 'parer': -0.004, 'board': -0.004}


def grasp_pose(role, item, flip=False, tilt=0.0, n=None, jaw=0.0
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pregrasp, grasp, rotation of the finger midpoint.
    (pregrasp, grasp, gripper rotation) of the finger midpoint, odometry frame. The knife is pinched across its handle (jaw across the blade
    axis), the board across the strip between its handle slot and its end (jaw along the board's long axis), the onion from above."""
    if role in ('parer', 'board') and item.grip is not None and item.axis is not None:
        yaw = math.atan2(float(item.axis[1]), float(item.axis[0])) - (math.pi / 2 if role == 'board' else 0.0) + (math.pi if flip else 0.0)
        grasp = item.grip.copy()
        return grasp + np.array([0., 0., G.PRE_GAP_M]), grasp, G.top_rotation(yaw)
    pre, grasp, rot = G.grasp_points(item.centre, item.size, 'top', np.array([1., 0., 0.]))
    if jaw != 0.0:                          # the jaw turned about the vertical: the fingers straddle the onion along another direction
        rot = G.rot_yaw(jaw) @ rot
    if tilt > 0. and n is not None:       # fingertips tilted towards the robot: the palm goes back over the basin, off the front rim
        axis = -math.cos(tilt) * G.UP + math.sin(tilt) * G.horiz(np.r_[n[:2], 0.])
        return grasp - axis * G.PRE_GAP_M, grasp, G.tool_rotation(axis)
    return pre, grasp, rot


def best_jaws(item: ItemEst, w: World, pts: np.ndarray) -> List[float]:
    """Jaw yaws ordered by room.
    Jaw yaws (rad, about the vertical) ordered by the room the two open fingers have on their way down: they straddle the onion 0.058 m either
    side along the jaw; a finger footprint (a square, half side h) is blocked when the cloud has a point there that the finger would hit
    (the rim at the counter height, or a wall face between the floor and the rim). The largest h with both footprints empty is the room; an onion
    in a corner of the basin (5-7 cm from two walls) is taken along the diagonal."""
    cands = [k * math.pi / 8 for k in range(8)]
    ce, ba = w.counter, w.basin
    if ce is None or ba is None or len(pts) == 0:
        return [0.0] + cands[1:]
    c = item.centre[:2]
    floor = ba.floor
    hit = pts[(pts[:, 2] > floor + P.FINGER_BLOCK_ABOVE_M)]
    out = []
    for psi in cands:
        jw = np.array([-math.sin(psi), math.cos(psi)])
        room = 0.0
        for h in P.FINGER_HALVES_M:
            ok = True
            for sgn in (1., -1.):
                p = c + sgn * jw * P.FINGER_OFFSET_M
                d = hit[:, :2] - p
                if np.any((np.abs(d[:, 0]) < h) & (np.abs(d[:, 1]) < h)):
                    ok = False
                    break
            if not ok:
                break
            room = h
        out.append((room, psi))
    out.sort(key=lambda r: (-r[0], abs(r[1])))
    return [r[1] for r in out]


def begin(c, role: str, side: str, kind: str, reason: str) -> None:
    c.role, c.side, c.place_kind, c.stand = role, side, kind, None
    c.change('plan', reason)


def step_choose(c, action, q, w) -> None:
    c.to_rest(action, q)
    st = c.stages
    if not st['onion']:
        begin(c, 'onion', 'left', 'in_bowl', 'put the onion into the bowl')
    elif not st['knife']:
        if not c.holds(q, 'right'):
            begin(c, 'parer', 'right', 'hold', 'take the knife')
        elif not st['parer_cut']:
            c.side, c.role = 'right', 'parer'
            c.change('cut_start', 'cut the onion')
        else:
            c.side, c.role, c.stand = 'right', 'parer', None
            c.sink_try = c.tries.get('sink_parer', 0)
            c.change('sink_go', 'carry the knife to the sink')
    elif not st['board']:
        if not c.holds(q, 'right'):
            begin(c, 'board', 'right', 'hold_board', 'take the board')
        else:
            c.side, c.role, c.stand = 'right', 'board', None
            c.change('sink_go', 'carry the board to the sink')
    else:
        c.change('done', 'all stages done')


def step_plan(c, action, q, w) -> None:
    c.to_rest(action, q)
    role = c.role
    c.tries[role] = c.tries.get(role, 0) + 1
    limit = P.MAX_TRIES if c.place_kind == 'in_bowl' else P.MAX_TRIES + 3
    item = w.items.get(role)
    if c.tries[role] > limit or item is None or (role in ('board', 'parer') and item.fit <= 0.):
        c.counts['plan_fail'] += 1
        if c.tries[role] > limit:
            c.stages_fail.append(role)
            c.change('failed', f'{role}: tries used')
        else:
            c.shift_k += 1
            c.after_look = 'choose'
            c.change('approach', f'{role} not in view: another vantage point')
        return
    c.build_grid(w, [item.centre, w.base[:2]], P.ROBOT_RADIUS)
    flip = (c.tries[role] % 2 == 0)
    stand, tilt, tlog = None, 0.0, []
    jaws = best_jaws(item, w, c.perc.basin_pts)[:P.JAW_TRIES] if role == 'onion' else [0.0]
    for jaw in jaws:
        pre, grasp, rot = grasp_pose(role, item, flip, 0.0, w.counter.n if w.counter is not None else None, jaw)
        goals = lambda: [(grasp + np.array([0., 0., GRASP_DZ[role]]), rot), (pre, rot)]  # noqa: E731
        stand = c.pick_stand(w, grasp, goals, c.side, max_ring=1.25)
        if stand is None:                    # tight footprint: the base fits 0.30 m from the counter
            c.retighten(w, P.TIGHT_RADIUS)
            stand = c.pick_stand(w, grasp, goals, c.side, max_ring=1.25)
            if stand is None:
                c.retighten(w, P.ROBOT_RADIUS)
        tlog.append((round(jaw, 2), dict(c.pick_stats.get('ik') or {})))
        if stand is not None:
            break
    c.jaw = jaw
    c.tilt = tilt
    if c.pick_stats.get('cands', 99) < 6 and len(c.debug) < 6:
        c.debug[f'plan_grid_{c.step}'] = dict(base=np.round(w.base, 2).tolist(), stand=np.round(item.centre[:2], 2).tolist(), rows=c.ascii_grid(w, item.centre, half=3.0, marks=[xy for xy, ok in c.last_cands if ok][:40]))
    if stand is None:
        c.counts['plan_fail'] += 1
        c.shift_k += 1
        c.after_look = 'choose'
        c.change('approach', f'no stand for the {role} ({c.pick_stats} jaws {tlog}): another vantage point')
        return
    c.stand, c.flip = stand, flip
    c.pre_q = None if c.stand_q is None else c.stand_q[1].copy()      # the scan's solution of the pregrasp from this stand: a joint waypoint for the servo
    c.change('go', f'{role} stand {c.tries[role]} at {np.round(stand[0], 2).tolist()} yaw {stand[1]:.2f} jaw {jaw:.2f} {c.pick_stats}')


def follow_stand(c, w) -> Tuple[np.ndarray, float]:
    """Stand along the counter from the target.
    The stand in the counter seen now: along the counter from the target (the item estimate), at the planned distance from the front edge
    measured now, facing the counter. The odometry drifts by tenths of a radian over a few metres; the counter in front of the head does not."""
    ce = w.counter
    if ce is None or c.stand_rel is None or c.stand_target is None or w.step - ce.seen_step > P.FOLLOW_FRESH_STEPS:
        return c.stand
    da, d, dyaw = c.stand_rel
    a_t = float(c.stand_target @ ce.u)
    front = float(ce.n @ ce.centre) + 0.5 * ce.depth
    xy = (a_t + da) * ce.u + (front + d) * ce.n
    return xy, nav_grid.wrap(ce.yaw_facing + dyaw)


def step_go(c, action, q, w) -> None:
    goal = follow_stand(c, w)
    if c.age % 20 == 1:
        c.history.append((c.step, 'go', f'goal {np.round(goal[0], 3).tolist()} planned {np.round(c.stand[0], 3).tolist()}')) if c.age % 100 == 1 else None
    dist, yaw = c.drive(action, q, w, goal[0], goal[1])
    if c.arrived_at(dist, yaw):
        c.after_look = 'reach'
        c.change('look', f'at the {c.role} stand')
    elif c.age > P.NAV_TIMEOUT:
        c.change('plan', 'walk timed out')


def current_pose(c, w) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return grasp_pose(c.role, c.item, getattr(c, 'flip', False), 0.0, w.counter.n if w.counter is not None else None, getattr(c, 'jaw', 0.0))


def step_reach(c, action, q, w) -> None:
    """Hand to the pregrasp.
    The hand to the pregrasp: solve_near first (the seed is the present joints); when it has not converged by REACH_SWITCH_AGE (its solution lies
    on a branch the arm cannot walk to) the joints walk together to the stand scan's solution of the same pregrasp, then resolved-rate."""
    pre, _, rot = current_pose(c, w)
    sv = c.servos[c.side]
    if c.age == 1:
        c.reach_mode = 'solve'
    if c.reach_mode == 'solve' and c.age >= P.REACH_SWITCH_AGE and (c.err[0] > P.REACH_SWITCH_ERR or sv.plan_failed) and c.pre_q is not None:
        c.reach_mode = 'walk'
        sv.sync, sv.target = True, np.asarray(c.pre_q, float)
        sv.max_step = P.WALK_MAX_STEP
        c.walk_age = 0
        c.history.append((c.step, 'reach', f'solve_near has not converged (err {c.err[0]:.3f}): joint walk to the scan solution'))
    if c.reach_mode == 'walk' and sv.sync and sv.target is not None:
        c.walk_age += 1
        goal, rotation = c.goal_robot(w, pre, rot)
        c.err = sv.step(action, q, goal, rotation, plan=False)
        other = 'left' if c.side == 'right' else 'right'
        action[K.ARM_ACTION[other]] = q[list(K.ARM_DOFS[other])]
        if sv.target is None or c.walk_age > P.REACH_WAYPOINT_STEPS:
            sv.sync, sv.target, sv.max_step = False, None, 0.016
        return
    sv.sync = False
    c.servo(action, q, w, pre, rot, plan=(c.reach_mode == 'solve'), stepwise=(c.reach_mode == 'walk'))
    if c.age % 60 == 30:
        c.history.append((c.step, 'reach', f'age {c.age} ({c.reach_mode}): err {c.err[0]:.3f}/{c.err[1]:.2f} plan_failed {sv.plan_failed} pre {np.round(pre, 3).tolist()}'))
    if c.err[0] < P.REACH_POS and c.err[1] < P.REACH_ORI:
        c.dwell = 0
        c.change('refine', 'hand at the pregrasp')
    elif c.age > P.PHASE_TIMEOUT['reach']:
        c.grasp_failed(w, 'reach timed out')


def step_refine(c, action, q, w) -> None:
    """Refine the pregrasp from the wrist camera.
    Hold the pregrasp still and look at the object with the wrist camera: the head's estimate had the rim, the odometry and the anchoring
    in it (a few cm); the hand sees the object straight from above. The onion is re-measured; the board and the knife keep their fits."""
    pre, _, rot = current_pose(c, w)
    c.servo(action, q, w, pre, rot, plan=False, stepwise=True)
    if c.age < P.REFINE_STEPS:
        return
    if c.role == 'onion':
        new = c.perc.refine_onion(c.side, c.item)
        if new is not None and float(np.linalg.norm(new.centre[:2] - c.item.centre[:2])) < P.REFINE_MAX_SHIFT_M:
            c.history.append((c.step, 'refine', f'onion moved by {np.round(new.centre - c.item.centre, 3).tolist()} (wrist view, {new.n_points} points)'))
            c.item = new
        else:
            c.history.append((c.step, 'refine', 'onion not seen from the wrist: the estimate of the head'))
    c.dwell = 0
    c.change('grasp', 'refined')


def step_grasp(c, action, q, w) -> None:
    _, grasp, rot = current_pose(c, w)
    point = grasp + np.array([0., 0., GRASP_DZ[c.role]])
    c.servo(action, q, w, point, rot, plan=False, stepwise=True)       # a short Cartesian move: resolved-rate (solve_near jumped to another branch)
    if c.age % 40 == 20:
        tm = c.perc.tips_mid(q, c.side)
        c.history.append((c.step, 'grasp', f'descent age {c.age}: err {c.err[0]:.3f}/{c.err[1]:.2f} tips-item {np.round(tm - c.item.centre, 3).tolist()} plan_failed {c.servos[c.side].plan_failed} target {c.servos[c.side].target is not None}'))
    if c.err[0] < P.GRASP_ARRIVE_M or c.age > P.GRASP_MAX_AGE:
        c.grip[c.side] = K.GRIPPER_CLOSED
        c.hand_rot = rot
        c.dwell += 1
        if c.dwell in (1, P.CLOSE_DWELL):
            tm = c.perc.tips_mid(q, c.side)
            c.history.append((c.step, 'grasp', f'close {c.dwell}: tips mid - item {np.round(tm - c.item.centre, 3).tolist()} gap {c.finger_gap(q, c.side):.3f} err {c.err[0]:.3f}'))
        if c.dwell >= P.CLOSE_DWELL and (c.holds(q, c.side) or c.dwell >= P.CLOSE_DWELL + 8):
            if not c.holds(q, c.side):
                c.grasp_failed(w, f'fingers closed on nothing (gap {c.finger_gap(q, c.side):.3f})')
                return
            c.hold_q = q.copy()
            c.held_offset = c.item.centre - c.perc.tips_mid(q, c.side)
            c.change('lift', f'fingers closed, gap {c.finger_gap(q, c.side):.3f}')
    elif c.age > P.PHASE_TIMEOUT['grasp']:
        c.grasp_failed(w, f'grasp timed out err={c.err[0]:.3f} {c.err[1]:.2f}')


def step_lift(c, action, q, w) -> None:
    _, grasp, rot = current_pose(c, w)
    if c.lift_goal is None or c.age <= 1:
        rise = P.HOLD_RISE_M if c.place_kind == 'hold' else P.LIFT_RISE_M
        c.lift_goal = grasp + np.array([0., 0., rise])
    c.servo(action, q, w, c.lift_goal, rot, plan=False, stepwise=True)
    held = c.holds(q, c.side)
    if c.age > 30 and not held:
        c.grasp_failed(w, 'the object slipped out of the hand')
        return
    if held and (c.err[0] < 0.06 or c.age > 150):
        c.hold_q = q.copy()
        c.perc.held_cut = (c.side, 0.12)
        if c.place_kind in ('hold', 'hold_board'):
            c.change('choose', f'{c.role} in hand')
        else:
            c.stand = None
            c.change('carry', f'{c.role} lifted')
    elif c.age > P.PHASE_TIMEOUT['lift']:
        c.grasp_failed(w, f'{c.role} did not lift')


def grasp_failed(c, w, reason: str) -> None:
    c.counts['grasp_fail'] += 1
    c.grip[c.side] = K.GRIPPER_OPEN
    c.hold_q = None
    c.perc.frozen.discard(c.role)
    c.perc.held_cut = None
    c.perc.forget(c.role)
    c.item = None
    c.after_look = 'choose'
    c.shift_k = 0
    if c.stand is not None:
        c.bad_stands.append(np.asarray(c.stand[0], float).copy())
    c.stand = None
    c.change('approach', f'{c.role}: {reason}; look again')


PHASES = {'choose': step_choose, 'plan': step_plan, 'go': step_go, 'reach': step_reach, 'refine': step_refine, 'grasp': step_grasp, 'lift': step_lift}
