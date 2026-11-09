"""Phases settle, scan and explore: bring the arms to the carry posture, turn in place to map the room and look for the cabinet top; without a
cabinet after a full turn, drive to the frontier of the seen floor that opens the most unseen ground and turn again.
Success: a cabinet estimate exists (plan_drawer). EXPLORE_TRIES frontiers without it fail the episode.
Also home of the shared recovery that changes the input: a base wedged on the way to a stand is backed straight off (arms at rest so the
head can look again), the failed stand is remembered, and the caller re-decides from the moved pose."""
from __future__ import annotations

import math

BACKOFF_M = 0.4          # how far the recovery backs the base straight away from the wedge or the target it could not reach
BACKOFF_MAX_CTRL = 240   # stop waiting for the back-off after this many controls (the caller re-decides anyway)
BACKOFF_NO_ROUTE = 30    # leave the recovery when even the back-off goal has no route
UNSTICK_MAX = 4          # at most this many recovery back-offs per episode; then the callers take their old fallbacks

import numpy as np
from scipy import ndimage

from core.skills_lib import nav_grid

from .. import params as P


def step_settle(c, action, q, w) -> None:
    c.to_rest(action, q)
    if c.age >= P.SETTLE_STEPS:
        c.scan_last_yaw = w.base[2]
        c.change('scan', 'settled')


def step_scan(c, action, q, w) -> None:
    c.to_rest(action, q)
    c.scan_total += abs(nav_grid.wrap(w.base[2] - c.scan_last_yaw))
    c.scan_last_yaw = w.base[2]
    cab = w.cabinet
    if c.scan_total >= P.SCAN_MIN_TURN_RAD or (cab is not None and cab.complete and c.counts['explore'] > 0):
        if cab is not None:
            c.change('plan_drawer', f'cabinet at {np.round(cab.centre, 2).tolist()} (complete {cab.complete}, face measured {cab.face_measured}) '
                                    f'after {c.scan_total:.1f} rad of turning')
        elif c.counts['explore'] >= P.EXPLORE_TRIES:
            c.fail(f'no cabinet after {c.counts["explore"]} frontiers')
        else:
            c.change('explore', f'no cabinet after {c.scan_total:.1f} rad of turning; to a frontier')
        return
    action[2] = P.SCAN_RATE / nav_grid.BASE_TURN
    c.reason = f'turning, {c.scan_total:.1f} rad, cabinet {cab is not None}'


def frontier_goal(c, w):
    """A free cell at the edge of the seen floor, EXPLORE_STEP_M or more away, with the most unseen cells around it; (xy, yaw) or None."""
    perc = c.perc
    free = perc.floor_seen & ~perc.blocked_cells()
    unknown = ~perc.floor_seen & ~perc.blocked_cells()
    edge = free & ndimage.binary_dilation(unknown, iterations=2)
    cells = np.argwhere(edge)
    if not len(cells):
        return None
    xy = (cells + 0.5) * P.MAP_RES_M - P.MAP_HALF_M
    d = np.linalg.norm(xy - w.base[:2], axis=1)
    sel = (d > P.EXPLORE_STEP_M) & (d < 4.0)
    if not sel.any():
        return None
    cells, xy, d = cells[sel][::4], xy[sel][::4], d[sel][::4]
    k = int(round(1.0 / P.MAP_RES_M))
    gain = np.array([unknown[max(i - k, 0):i + k, max(j - k, 0):j + k].sum() for i, j in cells]) * P.MAP_RES_M ** 2
    best = int(np.argmax(gain - 0.3 * d))
    goal = xy[best] - (xy[best] - w.base[:2]) / max(d[best], 1e-6) * 0.5        # stop short of the edge of the seen floor
    return goal, math.atan2(xy[best][1] - w.base[1], xy[best][0] - w.base[0])


def step_explore(c, action, q, w) -> None:
    if 'goal' not in c.mark:
        g = frontier_goal(c, w)
        if g is None:
            c.fail('no frontier left to explore')
            return
        c.mark['goal'] = g
        c.counts['explore'] += 1
    goal_xy, goal_yaw = c.mark['goal']
    dist, yaw_err = c.navigate(action, q, w, goal_xy, goal_yaw)
    if c.nav.arrived(dist, yaw_err) or (dist < 0.15 and c.age > 60) or c.nav.no_route > P.NO_ROUTE_MAX:
        c.scan_total, c.scan_last_yaw = 0., w.base[2]
        c.change('scan', f'at the frontier ({dist:.2f} m, no route {c.nav.no_route})')


def start_backoff(c, w, away, next_phase: str, why: str) -> None:
    """Record the recovery drive: BACKOFF_M straight away from `away` (odometry xy), arms at rest, then c.change(next_phase)."""
    m = c.mark.setdefault('backoff', {})
    if 'goal' in m:
        return
    base = np.asarray(w.base[:2], float)
    d = base - np.asarray(away, float)[:2]
    n = float(np.linalg.norm(d))
    d = d / n if n > 1e-6 else np.array([-math.cos(w.base[2]), -math.sin(w.base[2])])
    m['goal'] = base + d * BACKOFF_M
    m['yaw'] = float(w.base[2])
    m['next'] = next_phase
    m['why'] = why
    c.counts['unsticks'] = c.counts.get('unsticks', 0) + 1


def backoff_step(c, action, q, w) -> bool:
    """True = a recorded back-off is driving or just handed control to its next phase (the caller must return);
    False = no back-off recorded, the caller runs its own logic."""
    m = c.mark.get('backoff')
    if m is None:
        return False
    if 'goal' not in m:
        c.mark.pop('backoff', None)
        c.change(m.get('next', 'plan_jar'), m.get('why', 'back-off aborted'))
        return True
    c.to_rest(action, q)
    dist, yaw_err = c.navigate(action, q, w, m['goal'], m['yaw'])
    c.reason = f'backing off, {dist:.2f} m to go ({c.age} controls): {m["why"]}'
    if c.nav.arrived(dist, yaw_err) or c.age > BACKOFF_MAX_CTRL or c.nav.no_route > BACKOFF_NO_ROUTE:
        nxt, why = m.get('next', 'plan_jar'), m.get('why', 'backed off')
        c.mark.pop('backoff', None)
        c.change(nxt, why)
    return True


PHASES = {'settle': step_settle, 'scan': step_scan, 'explore': step_explore}
