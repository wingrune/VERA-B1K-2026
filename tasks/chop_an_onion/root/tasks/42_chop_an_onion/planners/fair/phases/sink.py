"""Putting the knife and then the board into the sink basin: sink_go (a stand from which the right hand reaches the release point over the basin,
carrying with the arm held), sink_lower (over the basin, the hand turned about the vertical so that the long axis of the knife / board lies
along the basin's long side, then down by resolved-rate to just below the counter top and open), retreat, look_sink (what lies in the basin)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import params as P
from ..world import World
from .board_hunt import board_ready, step_board_hunt


def release_target(c, w: World, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hand pose to let go over the basin.
    (hand point, hand rotation) at which the held knife / board lets go over the basin: its centre over the spot, long axis along the basin."""
    ba = w.basin
    a, b = P.SINK_SPOTS[c.sink_try % len(P.SINK_SPOTS)]
    spot = ba.centre + a * np.array([math.cos(ba.yaw), math.sin(ba.yaw)]) + b * np.array([-math.sin(ba.yaw), math.cos(ba.yaw)])
    it = c.item
    rot0 = c.hand_rot if c.hand_rot is not None else G.tool_rotation(-G.UP)
    axis = it.axis if (it is not None and it.axis is not None) else np.array([1., 0.])
    want = ba.long_axis
    d = math.atan2(want[1], want[0]) - math.atan2(float(axis[1]), float(axis[0]))
    d = (d + math.pi / 2) % math.pi - math.pi / 2                     # the long axis has no direction: the smallest turn
    v = (it.centre[:2] - it.grip[:2]) if (it is not None and it.grip is not None) else np.zeros(2)
    cs, sn = math.cos(d), math.sin(d)
    v2 = np.array([[cs, -sn], [sn, cs]]) @ v
    hand = spot - v2
    z_rel = w.counter.top - P.SINK_Z_BELOW_TOP_M
    if getattr(c, 'sink_high', {}).get(c.role, False):
        z_rel = w.counter.top + 0.03          # a retry changes its input: the object falls from just above the rim
    return np.array([hand[0], hand[1], z_rel]), G.rot_yaw(d) @ rot0


def step_sink_go(c, action, q, w) -> None:
    if c.hold_q is None:
        c.hold_q = q.copy()
    if getattr(c, 'sink_base', None) is None:      # basin occupancy before the release: the divider and the empty bowl
        c.sink_base = dict(getattr(c.perc, 'contents', None) or {})
    if w.basin is None or w.counter is None:
        c.change('failed', 'no basin seen')
        return
    if c.stand is None:
        c.build_grid(w, [w.basin.centre, w.base[:2]], P.ROBOT_RADIUS)
        point, rot = release_target(c, w, q)
        c.sink_goal = (point, rot)
        up = point + np.array([0., 0., 0.10])
        goals = lambda: [(up, rot), (point, rot)]  # noqa: E731
        stand = c.pick_stand(w, point, goals, 'right', max_ring=1.1)
        if stand is None:
            c.retighten(w, P.TIGHT_RADIUS)
            stand = c.pick_stand(w, point, goals, 'right', max_ring=1.3)
        if stand is None:
            c.counts['plan_fail'] += 1
            c.change('failed', f'no stand at the sink ({c.pick_stats})')
            return
        c.stand = stand
        c.history.append((c.step, 'sink_go', f'stand {np.round(stand[0], 2).tolist()} yaw {stand[1]:.2f} hand {np.round(point, 3).tolist()}'))
    dist, yaw = c.drive(action, q, w, c.stand[0], c.stand[1], hold=True)
    if c.arrived_at(dist, yaw):
        c.sink_sub, c.dwell = 'over', 0
        c.change('sink_lower', f'{c.role} over the sink')
    elif c.age > P.PHASE_TIMEOUT['sink_go']:
        c.change('failed', 'walk to the sink timed out')


def step_sink_lower(c, action, q, w) -> None:
    c.side = 'right'
    point, rot = c.sink_goal
    if c.sink_sub == 'over':
        c.servo(action, q, w, point + np.array([0., 0., 0.10]), rot, plan=True)
        if (c.err[0] < 0.05 and c.err[1] < 0.3) or c.age > 300:
            c.sink_sub = 'down'
            c.servos['right'].target = None
            c.age_down = c.age
    else:
        c.servo(action, q, w, point, rot, plan=False, stepwise=True)
        if c.err[0] < 0.04 or c.age - c.age_down > 260:
            c.grip['right'] = K.GRIPPER_OPEN
            c.dwell += 1
        if c.dwell >= P.BASIN_DWELL:
            c.sink_sub = 'up'
            c.dwell = 0
    if c.sink_sub == 'up':
        c.grip['right'] = K.GRIPPER_OPEN
        c.servo(action, q, w, point + np.array([0., 0., 0.15]), rot, plan=False, stepwise=True)
        c.dwell += 1
        if c.dwell >= 25:
            c.hold_q = None
            c.perc.held_cut = None
            c.perc.frozen.discard(c.role)
            c.after_look = 'sink_' + c.role
            c.change('look_sink', f'{c.role} released over the basin err={c.err[0]:.3f}')
    if c.age > P.PHASE_TIMEOUT['sink_lower'] + 400:
        c.grip['right'] = K.GRIPPER_OPEN
        c.hold_q = None
        c.after_look = 'sink_' + c.role
        c.change('look_sink', 'sink lower timed out')


def _release_missed(c, w) -> Optional[str]:
    """Why the release is judged a miss, or None when it stands.
    The parent declared every release a success without looking at the result. Two
    independent RGB-D signals say the object is not in the bowl: (a) the rim zone of
    the basin gained points against the pre-release baseline (an object resting on
    the rim or across the divider - inside the basin box, which the item search masks
    out, so the counter re-find cannot see it), (b) the released role is back in the
    item band of the counter, seen fresh by this look. A stale estimate from before
    the release is not evidence, and a candidate inside the basin box belongs to
    the bowl zone, not to the counter."""
    cont = dict(getattr(c.perc, 'contents', None) or {})
    base = dict(getattr(c, 'sink_base', None) or {})
    if int(cont.get('rim', 0)) - int(base.get('rim', 0)) >= 25 and int(cont.get('cells', 0)) - int(base.get('cells', 0)) < 10:
        return 'resting on the rim'
    it = w.items.get(c.role)
    if it is None or getattr(it, 'centre', None) is None or w.counter is None or w.basin is None:
        return None
    if c.step - it.seen_step > 3 * P.LOOK_STEPS + 60 or it.n_points < 40:
        return None
    a, b = w.counter.to_local(it.centre[:2])
    ba, bb = w.counter.to_local(w.basin.centre)
    wa, wb = w.basin.size
    if abs(a - ba) < 0.5 * wa + 0.02 and abs(b - bb) < 0.5 * wb + 0.02:
        return None
    return 'back on the counter'


def step_look_sink(c, action, q, w) -> None:
    c.to_rest(action, q)
    if c.age < P.LOOK_STEPS:
        return
    cont = c.perc.contents
    c.history.append((c.step, 'look_sink', f'basin contents {cont}'))
    n = c.tries.get('sinkmiss_' + c.role, 0)
    why = _release_missed(c, w) if n < 2 else None
    c.sink_high = dict(getattr(c, 'sink_high', {}))
    c.sink_base = None
    if why is not None:
        c.tries['sinkmiss_' + c.role] = n + 1
        c.sink_high[c.role] = True          # the retry drops the object from above the rim, at the next basin spot
        c.sink_try = getattr(c, 'sink_try', 0) + 1
        c.tries[c.role] = 0                # the re-grasp gets its try budget back: the horizon is still there
        if c.role == 'board':
            c.change('board_hunt', 'board missed the basin: hunt it along the counter and re-grasp')
            return
        c.change('choose', f'{c.role} missed the basin ({why}): re-grasp and retry')
        return
    c.sink_high[c.role] = False
    if c.role == 'parer':
        c.stages['knife'] = True
        c.tries['sink_parer'] = c.tries.get('sink_parer', 0) + 1
        if not board_ready(c, w):
            c.change('board_hunt', f'board not plannable from here ({c.pick_stats}): hunt it along the counter')
            return
    else:
        c.stages['board'] = True
    c.change('choose', f'{c.role} in the sink: contents {cont}')


PHASES = {'sink_go': step_sink_go, 'sink_lower': step_sink_lower, 'look_sink': step_look_sink, 'board_hunt': step_board_hunt}
