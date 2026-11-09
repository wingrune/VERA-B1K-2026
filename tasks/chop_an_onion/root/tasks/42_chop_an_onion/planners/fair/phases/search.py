"""Phases that find the counter and look over it: settle, scan (turn in place), survey (walk to a new place when nothing was seen), approach
(drive to a vantage point in front of the counter), look (stand still, refresh the estimates). The vantage point is derived from the
perceived counter: in front of its front edge, looking at it."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from core.skills_lib import nav_grid

from .. import geometry as G
from .. import ik
from .. import params as P
from ..world import World

NEEDED = ('board', 'parer', 'bowl')


def counter_ok(w: World) -> bool:
    return w.counter is not None and w.counter.length >= P.COUNTER_MIN_LENGTH_M


def step_settle(c, action, q, w) -> None:
    c.to_rest(action, q)
    if c.age == 1:
        ik.start()                    # IK_BACKEND gpu|cpu printed once the worker is ready; raises without a GPU solver (BIMBO_IK_REQUIRE_GPU=1)
    if c.age >= P.SETTLE_STEPS:
        c.scan_yaw0, c.turned = float(w.base[2]), 0.
        c.scan_limit = P.SCAN_FULL_TURN_RAD
        c.change('scan', 'settled')


def step_scan(c, action, q, w) -> None:
    c.to_rest(action, q)
    if c.scan_yaw0 is None:
        c.scan_yaw0, c.turned = float(w.base[2]), 0.
    c.turned = abs(nav_grid.wrap(float(w.base[2]) - c.scan_yaw0)) if c.turned < math.pi else c.turned
    if c.age % (P.SCAN_TURN_CTRL + P.SCAN_PAUSE_CTRL) < P.SCAN_TURN_CTRL:     # turn a step, stand still a moment: the map and the counter are taken while standing
        action[2] = P.SCAN_RATE / nav_grid.BASE_TURN
    if counter_ok(w) and c.age > 20 and w.step - w.counter.seen_step < 10:
        c.build_grid(w, [w.counter.centre])
        c.change('approach', f'counter seen: top {w.counter.top:.3f}, {w.counter.length:.2f} x {w.counter.depth:.2f} m')
    elif c.age * P.SCAN_RATE / 30.0 * P.SCAN_TURN_CTRL / (P.SCAN_TURN_CTRL + P.SCAN_PAUSE_CTRL) > c.scan_limit:
        c.counts['surveys'] += 1
        if c.counts['surveys'] > P.MAX_SURVEYS:
            c.change('failed', 'counter not found')
        else:
            c.stand = None
            c.build_grid(w, [w.base[:2]], c.radius)
            c.change('survey', 'no counter after a full turn: explore')


def frontier_goal(c, w: World):
    """Next place to look from.
    Next place to look from: a known-free cell next to unknown floor, reachable, far from the places already visited, with the most unknown
    floor around it (the head sees the floor only a few metres ahead, so the frontier lies about that far from the base). Returns xy or None."""
    from scipy import ndimage
    perc = c.perc
    res, half = P.OBST_RES_M, P.OBST_HALF_M
    free = perc.floor_seen & (perc.obst < P.OBST_MIN_HITS)
    known = perc.floor_seen | (perc.obst >= P.OBST_MIN_HITS)
    unknown = ~known
    fr = free & ndimage.binary_dilation(unknown, iterations=P.FRONTIER_DILATE)
    frac = ndimage.uniform_filter(unknown.astype(float), size=int(P.FRONTIER_INFO_R_M / res))
    idx = np.argwhere(fr)
    if len(idx) == 0:
        return None
    xy = (idx + 0.5) * res - half
    d = np.linalg.norm(xy - w.base[:2], axis=1)
    keep = (d > P.FRONTIER_MIN_M) & (d < P.FRONTIER_MAX_M)
    for v in c.explored:
        keep &= np.linalg.norm(xy - v, axis=1) > P.FRONTIER_VISITED_M
    idx, xy, d = idx[keep], xy[keep], d[keep]
    if len(idx) == 0:
        return None
    score = d - P.FRONTIER_INFO_GAIN * frac[idx[:, 0], idx[:, 1]]
    for k in np.argsort(score)[:40]:
        if c.free_pose(xy[k]) and np.isfinite(c.route_len(w, xy[k])):
            return xy[k].copy()
    return None


def step_survey(c, action, q, w) -> None:
    """Explore to the best frontier.
    Explore: drive to the best frontier, turning the head towards the unknown on the way, then scan there. The counter, once seen in
    the head cloud (it is the one flat slab at 0.88 m with the right size), ends the exploration at once."""
    c.to_rest(action, q)
    if counter_ok(w) and w.step - w.counter.seen_step < 10 and c.age > 5:
        c.build_grid(w, [w.counter.centre, w.base[:2]], c.radius)
        c.change('approach', f'counter seen while exploring: top {w.counter.top:.3f}, {w.counter.length:.2f} x {w.counter.depth:.2f} m')
        return
    if c.stand is None:
        c.build_grid(w, [w.base[:2] + 7.0, w.base[:2] - 7.0], c.radius)      # the whole neighbourhood: the frontier may lie several metres away
        g = frontier_goal(c, w)
        if g is None:
            c.debug[f'explore_{c.step}'] = dict(base=np.round(w.base, 2).tolist(), stand=[0, 0], rows=c.ascii_grid(w, None, half=9.0, res=0.25, marks=c.explored))
            c.counts['plan_fail'] += 1
            c.scan_yaw0, c.turned = float(w.base[2]), 0.
            c.change('scan', 'no frontier: scan again')
            return
        c.stand = (g, math.atan2(g[1] - w.base[1], g[0] - w.base[0]))
        c.explored.append(g.copy())
        c.history.append((c.step, 'survey', f'frontier {np.round(g, 2).tolist()}'))
    dist, yaw = c.navigate(action, q, w, c.stand[0], c.stand[1], rest=False)
    if dist < 0.3 or c.age > 500:
        c.scan_yaw0, c.turned, c.stand = float(w.base[2]), 0., None
        c.scan_limit = P.SCAN_PARTIAL_TURN_RAD
        c.change('scan', 'at the frontier')


def vantage(c, w: World, k: int, close: bool = False) -> Optional[Tuple[np.ndarray, float]]:
    """Base pose of the k-th vantage point.
    Base pose of the k-th vantage point: in front of the counter's front edge, shifted along it by 0, +s, -s, +2s ..."""
    ce = w.counter
    shift = 0. if k == 0 else (1 if k % 2 == 1 else -1) * ((k + 1) // 2) * P.SHIFT_M
    base_d = 0.5 * ce.depth + (0.8 if close else P.VANTAGE_M)
    for extra in np.arange(0., 1.0, 0.1):
        xy = ce.from_local(shift, base_d + extra)
        if c.free_pose(xy):
            return xy, ce.yaw_facing
    return None


def step_approach(c, action, q, w) -> None:
    if not counter_ok(w):
        c.to_rest(action, q)
        c.change('scan', 'counter lost')
        return
    if c.age == 1 or c.vantage is None or c.vantage_k != (c.shift_k, c.close_view):
        c.build_grid(w, [w.counter.centre, w.base[:2]], c.radius)
        c.vantage = vantage(c, w, c.shift_k, c.close_view)
        c.vantage_k = (c.shift_k, c.close_view)
        c.vantage_t0 = c.step
    target = c.vantage                      # fixed when the phase starts: the visible part of the counter (its centre) changes while the base drives
    if target is None:
        c.counts['plan_fail'] += 1
        ce = w.counter
        c.change('look', f'no free vantage point (counter centre {np.round(ce.centre, 2).tolist()} n {np.round(ce.n, 2).tolist()} depth {ce.depth:.2f}, '
                         f'base {np.round(w.base, 2).tolist()}, radius {c.radius}): look from here')
        return
    dist, yaw_err = c.drive(action, q, w, target[0], target[1])
    if c.arrived_at(dist, yaw_err) or (dist < 0.15 and abs(yaw_err) < 0.25 and c.age > 120):
        c.after_look = 'choose'
        c.change('look', f'at vantage {c.shift_k}')
    elif c.age > P.NAV_TIMEOUT:
        c.after_look = 'choose'
        c.change('look', 'approach timed out')


def needed(c) -> tuple:
    """Roles the next stage needs."""
    st = c.stages
    if not st['onion']:
        return ('bowl', 'basin', 'onion')
    if not st['knife']:
        return ('parer', 'bowl', 'basin')
    return ('board', 'basin')


def missing(c, w: World) -> list:
    out = []
    for r in needed(c):
        if r == 'basin':
            if w.basin is None:
                out.append(r)
        elif r not in w.items or (r in ('board', 'parer') and w.items[r].fit <= 0.):
            out.append(r)
    return out


def step_look(c, action, q, w) -> None:
    c.to_rest(action, q)
    if c.age < P.LOOK_STEPS or (w.step - c.perc.last_anchor < P.LOOK_SETTLE_STEPS and c.age < P.LOOK_STEPS + P.LOOK_EXTRA_STEPS):
        return                       # the odometry is still being pulled onto the counter: the estimates of this view are not yet in its final frame
    if c.after_look == 'reach':
        role = c.role
        it = w.items.get(role)
        fresh = it is not None and it.seen_step >= c.phase_t0 - 1 and it.seen_step > c.perc.last_anchor and (role not in ('board', 'parer') or it.fit > 0.)
        old_ok = it is not None and w.step - it.seen_step <= P.STALE_OK_STEPS and (role not in ('board', 'parer') or it.fit > 0.)
        if not fresh and not old_ok:
            c.stale_looks += 1
            c.counts['stale'] = c.counts.get('stale', 0) + 1
            if c.stale_looks <= 1:
                c.change('look', f'{role} not seen at the stand: look again')
                c.after_look = 'reach'
                return
            c.stale_looks = 0
            c.counts['plan_fail'] += 1
            if c.stand is not None:
                c.bad_stands.append(np.asarray(c.stand[0], float).copy())
            c.change('plan', f'{role} not seen from the stand')
            return
        if not fresh:
            c.history.append((c.step, 'look', f'{role} hidden from the stand (rim): the estimate of the earlier view, {w.step - it.seen_step} controls old'))
        c.stale_looks = 0
        c.item = it
        c.perc.frozen.add(role)
        c.change('reach', f'{role} seen from the stand: {np.round(it.centre, 3).tolist()}')
        return
    gone = missing(c, w)
    if 'basin' in gone and c.close_view and not c.stages['onion']:
        c.counts['rejected_counters'] = c.counts.get('rejected_counters', 0) + 1
        c.history.append((c.step, 'look', f'no sink basin in the counter at {np.round(w.counter.centre, 2).tolist()} (seen from 0.8 m): another run of counters'))
        c.perc.reject_counter()
        c.shift_k, c.close_view = 0, False
        c.scan_yaw0, c.turned = float(w.base[2]), 0.
        c.scan_limit = P.SCAN_FULL_TURN_RAD
        c.change('scan', 'counter without a sink: scan for another one')
        return
    if ('basin' in gone or 'onion' in gone) and not c.close_view:
        c.close_view = True
        c.change('approach', f'missing {gone}: closer to the counter')
        return
    if gone and c.shift_k < P.MAX_SHIFTS:
        c.shift_k += 1
        c.counts['shifts'] += 1
        c.change('approach', f'missing {gone}: vantage {c.shift_k}' + (' (closer)' if c.close_view else ''))
        return
    if gone and not c.close_view:                    # shifts did not reveal it: one look from 0.8 m, the thin parer gives few cells from far
        c.close_view = True
        c.change('approach', f'missing {gone}: closer to the counter')
        return
    c.shift_k = 0
    c.close_view = False
    c.change('choose', f'view done, missing {gone}')


PHASES = {'settle': step_settle, 'scan': step_scan, 'survey': step_survey, 'approach': step_approach, 'look': step_look}
