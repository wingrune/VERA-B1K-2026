"""The board stage of the sink, rebuilt from scratch: the parent handed the board to the generic pick after the knife
release and died there (302: 'missing board' at ten looks, then a grasp at a stale false estimate - board_rel 0.6 m off -
closed on nothing and looped to the horizon; 304: board in view but 'no stand' with every ring cell dropped by the obstacle
grid, eight retries, a terminal failed at 6705 of 9599). This phase owns the whole sub-chain instead: it sweeps base spots
along the front edge of the perceived counter until two fresh sightings from DIFFERENT spots agree (false sightings that
led to an empty close are remembered and rejected), picks the stand by ONE batched fair_toolbox.ik.stand_mask over a dense
ring around the grasp point (the base disc is tested only against the perceived counter rectangle and the basin box - the
obstacle-grid prefilter that produced 'cands 0' cannot reject a stand here), drives, servos to the frozen pick.grasp_pose,
closes, verifies the hold by the finger gap, lifts resolved-rate and hands the held board to the proven sink_go/sink_lower
release. Nothing here ends the episode: every failure changes its input (next stand, next spot, next sighting) and
exhaustion falls back to the old choose path."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from core.skills_lib import r1pro_kinematics as K
from fair_toolbox import ik as TB

from .. import params as P
from ..world import World
from .pick import grasp_pose

BOARD_FRESH_STEPS = 400        # a sighting this young counts as fresh
VIEW_SETTLE = 36               # controls of standing perception at each sweep spot
SWEEP_STEP_M = 0.40
STAND_RINGS = (0.40, 0.52, 0.64, 0.78, 0.92, 1.05, 1.18, 1.30)
BASE_CLEAR_M = 0.31            # the base fits 0.34 m from the counter face (oracle measurement); the grid prefilter used more
HUNT_TIMEOUT = 2600            # then fall back to the old choose path, never a terminal failed
CONFIRM_M = 0.10               # two sightings this close, from different spots, agree


def _rot33(rot) -> np.ndarray:
    r = np.asarray(rot, float)
    return r[:3, :3] if r.shape == (4, 4) else r


def _fresh_board(c, w: World) -> Optional[object]:
    """The board estimate only when it is fresh, on the counter, outside the basin and not a rejected false sighting."""
    it = w.items.get('board')
    if it is None or getattr(it, 'centre', None) is None or w.counter is None:
        return None
    if c.step - it.seen_step > BOARD_FRESH_STEPS or int(it.n_points) < 40:
        return None
    a, b = w.counter.to_local(it.centre[:2])
    if abs(a) > 0.5 * w.counter.length - 0.03 or abs(b) > 0.5 * w.counter.depth + 0.05:
        return None
    if w.basin is not None:
        ba, bb = w.counter.to_local(w.basin.centre)
        if abs(a - ba) < 0.5 * float(w.basin.size[0]) + 0.04 and abs(b - bb) < 0.5 * float(w.basin.size[1]) + 0.04:
            return None
    if not 0.004 < float(it.top) - w.counter.top < 0.07:
        return None
    for xy, rad in getattr(c, 'board_bad', []):
        if float(np.linalg.norm(np.asarray(it.centre[:2], float) - np.asarray(xy, float))) < rad:
            return None
    return it


def _grasp_of(it) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre, grp, rot = grasp_pose('board', it)
    return np.asarray(pre, float).reshape(3), np.asarray(grp, float).reshape(3), _rot33(rot)


def _sweep_spots(w: World, base_xy) -> List[Tuple[np.ndarray, float]]:
    """Base spots along the front edge of the perceived counter, facing it, nearest first."""
    ce = w.counter
    n, u = np.asarray(ce.n, float)[:2], np.asarray(ce.u, float)[:2]
    off = 0.5 * ce.depth + 0.45
    spots = []
    for s in np.arange(-0.5 * ce.length + 0.06, 0.5 * ce.length, SWEEP_STEP_M):
        spots.append((np.asarray(ce.centre, float)[:2] + float(s) * u - off * n, math.atan2(n[1], n[0])))
    spots.sort(key=lambda sp: float(np.linalg.norm(sp[0] - np.asarray(base_xy, float))))
    return spots


def _stand_rows(w: World, grp: np.ndarray) -> List[Tuple[np.ndarray, float, float]]:
    """Dense stand ring around the grasp point; the base disc is kept out of the perceived counter rectangle and the basin
    box only - no obstacle-grid prefilter (304 died on that prefilter). Nearest stands first."""
    ce = w.counter
    rows = []
    for ring in STAND_RINGS:
        for k in range(16):
            b = 2.0 * math.pi * k / 16.0
            xy = np.asarray(grp[:2], float) + ring * np.array([math.cos(b), math.sin(b)])
            a, d = ce.to_local(xy)
            if abs(a) < 0.5 * ce.length + BASE_CLEAR_M and abs(d) < 0.5 * ce.depth + BASE_CLEAR_M:
                continue
            if w.basin is not None and float(np.linalg.norm(xy - np.asarray(w.basin.centre, float)[:2])) < 0.5 * max(w.basin.size) + BASE_CLEAR_M:
                continue
            rows.append((xy, b + math.pi, ring))
    rows.sort(key=lambda r: float(np.linalg.norm(r[0] - np.asarray(w.base[:2], float))) + 0.25 * r[2])
    return rows


def _reachable(c, grp: np.ndarray, rot: np.ndarray, rows) -> list:
    """One batched GPU stand-reachability mask over the whole ring (a mask only; the servo target stays the frozen grasp pose)."""
    if not rows:
        return []
    stands = np.array([[r[0][0], r[0][1], r[1]] for r in rows])
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = np.asarray(grp, float)
    try:
        mask = TB.stand_mask(T, stands, 'right')
    except Exception:
        return []
    return [rows[i] for i in range(len(rows)) if bool(mask[i])]


def _next_spot(c, why: str) -> None:
    c.board_spot += 1
    c.board_seen, c.board_prog = None, None
    if c.board_spot >= len(c.board_spots):
        c.grip['right'] = K.GRIPPER_OPEN
        c.board_sub = None
        c.change('choose', f'board hunt exhausted the counter sweep ({why})')
        return
    c.board_sub, c.board_watch, c.board_age = 'seek', 0, 0


def _reject_stand(c, why: str) -> None:
    c.grip['right'] = K.GRIPPER_OPEN
    c.board_stand_k += 1
    c.board_prog = None
    if c.board_stand_k < len(c.board_stands):
        c.board_sub, c.board_age = 'go', 0
    else:
        c.history.append((c.step, 'board_hunt', f'all stands rejected ({why})'))
        _next_spot(c, 'all stands rejected')


def _start_hunt(c, w: World) -> bool:
    c.board_sub, c.board_age, c.board_watch = 'seek', 0, 0
    c.board_spot, c.board_stand_k = 0, 0
    c.board_stands, c.board_item, c.board_pose = [], None, None
    c.board_seen, c.board_prog = None, None
    c.board_bad = list(getattr(c, 'board_bad', []))
    c.board_spots = _sweep_spots(w, w.base[:2])
    c.perc.frozen.discard('board')            # a frozen stale estimate never corrects itself (302)
    c.build_grid(w, [np.asarray(w.counter.centre, float)[:2], w.base[:2]], P.ROBOT_RADIUS)
    c.history.append((c.step, 'board_hunt', f'{len(c.board_spots)} sweep spots along the counter front'))
    return bool(c.board_spots)


def board_ready(c, w: World) -> bool:
    """Can the generic pick still plan the board from here? True keeps the old choose path (the seeds that already succeed)."""
    it = _fresh_board(c, w)
    if it is None:
        return False
    try:
        pre, grp, rot = _grasp_of(it)
    except Exception:
        return False
    goals = lambda: [(pre, rot), (grp, rot)]  # noqa: E731
    c.build_grid(w, [grp[:2], w.base[:2]], P.ROBOT_RADIUS)
    return c.pick_stand(w, grp, goals, 'right', max_ring=1.0) is not None


def step_board_hunt(c, action, q, w) -> None:
    c.side = 'right'
    if w.counter is None or w.basin is None:
        c.change('choose', 'board hunt: no counter or basin seen')
        return
    if getattr(c, 'board_sub', None) is None:
        if not _start_hunt(c, w):
            c.board_sub = None
            c.change('choose', 'board hunt: no sweep spots')
            return
    if c.age > HUNT_TIMEOUT:
        c.grip['right'] = K.GRIPPER_OPEN
        c.board_sub = None
        c.change('choose', 'board hunt timed out: back to the pick stage')
        return
    sub = c.board_sub
    c.board_age += 1

    if sub == 'seek':
        spot, yaw = c.board_spots[c.board_spot]
        dist, yerr = c.drive(action, q, w, spot, yaw, hold=False)
        if c.arrived_at(dist, yerr):
            c.board_sub, c.board_watch, c.board_age = 'watch', 0, 0
        else:
            prog = c.board_prog
            if prog is None or dist < prog[0] - 0.05:
                c.board_prog = (dist, c.step)
            elif c.step - prog[1] > 220:
                _next_spot(c, 'walk to the sweep spot stalled')

    elif sub == 'watch':
        c.to_rest(action, q)
        c.board_watch += 1
        it = _fresh_board(c, w)
        if it is not None:
            prev = c.board_seen
            if prev is None or prev[2] == c.board_spot or float(np.linalg.norm(it.centre[:2] - prev[0])) > CONFIRM_M:
                c.board_seen = (np.asarray(it.centre[:2], float).copy(), c.step, c.board_spot)
                c.history.append((c.step, 'board_hunt', f'board sighting {np.round(it.centre, 3).tolist()} at spot {c.board_spot}'))
            else:
                try:
                    pre, grp, rot = _grasp_of(it)
                    ok = _reachable(c, grp, rot, _stand_rows(w, grp))
                except Exception:
                    ok = []
                c.history.append((c.step, 'board_hunt', f'board confirmed {np.round(it.centre, 3).tolist()}: {len(ok)} stands reach'))
                if ok:
                    c.board_item, c.board_pose, c.board_stands, c.board_stand_k = it, (pre, grp, rot), ok, 0
                    c.board_sub, c.board_age, c.board_prog = 'go', 0, None
                    return
                c.board_seen = None      # seen but nothing reaches: parallax from another spot may correct it
        if c.board_watch >= VIEW_SETTLE:
            _next_spot(c, 'no fresh board at this spot')

    elif sub == 'go':
        xy, yaw, _ring = c.board_stands[c.board_stand_k]
        dist, yerr = c.drive(action, q, w, xy, yaw, hold=False)
        if c.arrived_at(dist, yerr):
            c.board_sub, c.board_age = 'reach', 0
        else:
            prog = c.board_prog
            if prog is None or dist < prog[0] - 0.05:
                c.board_prog = (dist, c.step)
            elif c.step - prog[1] > 200:
                _reject_stand(c, 'walk to the stand stalled')

    elif sub == 'reach':
        pre, grp, rot = c.board_pose
        c.servo(action, q, w, pre, rot, plan=True)
        if (c.err[0] < 0.05 and c.err[1] < 0.3) or c.board_age > 360:
            c.servos['right'].target = None
            c.board_sub, c.board_age = 'descend', 0

    elif sub == 'descend':
        pre, grp, rot = c.board_pose
        c.servo(action, q, w, grp, rot, plan=False, stepwise=True)
        if c.err[0] < 0.03 or c.board_age > 260:
            c.board_sub, c.board_age = 'close', 0

    elif sub == 'close':
        pre, grp, rot = c.board_pose
        c.servo(action, q, w, grp, rot, plan=False, stepwise=True)
        c.grip['right'] = K.GRIPPER_CLOSED
        if c.board_age >= P.CLOSE_DWELL + 25:
            gap = float(np.sum(q[list(K.FINGER_DOFS['right'])]))
            if gap > 0.02:
                c.history.append((c.step, 'board_hunt', f'board held, finger gap {gap:.3f}'))
                c.board_sub, c.board_age = 'lift', 0
            else:
                c.history.append((c.step, 'board_hunt', f'fingers closed on nothing (gap {gap:.3f}) at {np.round(grp, 3).tolist()}'))
                bad = list(getattr(c, 'board_bad', []))
                bad.append((np.asarray(grp[:2], float).copy(), 0.14))
                c.board_bad = bad
                c.grip['right'] = K.GRIPPER_OPEN
                _next_spot(c, 'fingers closed on nothing')   # remember the false sighting, change the input

    else:  # 'lift'
        pre, grp, rot = c.board_pose
        c.grip['right'] = K.GRIPPER_CLOSED
        c.servo(action, q, w, np.asarray(pre, float) + np.array([0., 0., 0.12]), rot, plan=False, stepwise=True)
        if (c.err[0] < 0.04 and c.board_age > 30) or c.board_age > 120:
            it = c.board_item
            ya = float(it.yaw) if getattr(it, 'yaw', None) is not None else 0.0
            it.axis = np.array([math.cos(ya), math.sin(ya)])   # the release turns the long axis to the basin
            it.grip = np.asarray(grp, float)
            c.role, c.item, c.hand_rot = 'board', it, rot
            c.stand, c.hold_q, c.sink_goal = None, None, None
            c.board_sub = None
            c.perc.held_cut = None
            c.change('sink_go', 'board in hand: carry it to the basin')
