"""Search phases of the fair planner of task 69: settle the arms (the rest trunk tilts the head camera down onto the floor near the robot, the
arms hang so that a turn in place sweeps nothing), turn in place until the vacuum is seen and the floor around has been looked at for dark
flakes, walk to a survey point and turn again when the vacuum is not found. Subgoal near_vacuum (base within 1 m of the vacuum) is reached
by these phases and the walk of the grasp phases."""
from __future__ import annotations

import math

import numpy as np

from .. import geometry as G
from .. import params as P


def _turn(c, w) -> None:
    if c.scan_from is not None:
        c.turned += abs(G.wrap(w.base[2] - c.scan_from))
    c.scan_from = float(w.base[2])


def settle(c, action, q, w) -> None:
    c.to_rest(action, q)
    if c.age >= P.SETTLE_STEPS:
        c.turned, c.scan_from, c.pause, c.turn_mark = 0., None, 0, 0.
        c.change('scan', 'settled; turning to find the vacuum and the dust')


def scan(c, action, q, w) -> None:
    c.to_rest(action, q)
    _turn(c, w)
    if c.pause > 0:                      # standing still after a step of the turn: the odometry catches up, the flakes are placed
        c.pause -= 1
    elif c.turned - c.turn_mark >= P.SCAN_STOP_RAD:
        c.pause, c.turn_mark = P.SCAN_PAUSE, c.turned
    else:
        action[2] = P.SCAN_RATE
    found = w.vacuum is not None
    c.reason = f'turning {c.turned:.2f} rad, vacuum {"seen" if found else "not seen"}, dust {len(w.dust)}'
    if found and c.turned >= P.SCAN_TURN_RAD:
        c.change('plan_grasp', f'vacuum found, {len(w.dust)} flakes seen after {c.turned:.2f} rad')
    elif c.turned >= P.SCAN_TURN_RAD + 0.5 and not found:
        if c.counts['surveys'] >= P.MAX_SURVEYS:
            c.change('failed', 'vacuum not found after every survey')
        else:
            c.change('survey', f'no vacuum after {c.turned:.2f} rad')


def survey(c, action, q, w) -> None:
    """Walk SURVEY_STEP_M towards the most open seen floor.

    Walk SURVEY_STEP_M towards the most open seen floor (a direction not tried before), then turn again."""
    c.to_rest(action, q)
    if c.age == 1:
        c.counts['surveys'] += 1
        c.build_grid(w, [])
        best, goal = -1e9, None
        for k in range(16):
            a = 2 * math.pi * k / 16
            xy = w.base[:2] + P.SURVEY_STEP_M * np.array([math.cos(a), math.sin(a)])
            if not c.free_pose(xy) or not math.isfinite(c.route_len(w, xy)):
                continue
            far = max(float(np.linalg.norm(xy - p)) for p in getattr(c, 'surveyed', [w.base[:2]]))
            score = c.inflated.clearance(xy) + 0.8 * far
            if score > best:
                best, goal = score, xy
        c.survey_goal = goal
        c.surveyed = getattr(c, 'surveyed', []) + [w.base[:2].copy()]
        if goal is None:
            c.change('failed', 'no survey point reachable')
            return
    goal = c.survey_goal
    yaw = math.atan2(goal[1] - w.base[1], goal[0] - w.base[0])
    dist, yaw_err = c.navigate(action, w, goal, yaw)
    c.reason = f'walking to the survey point, {dist:.2f} m'
    if w.vacuum is not None:
        c.change('plan_grasp', 'vacuum seen while walking')
    elif c.arrived(dist, yaw_err) or c.age > P.NAV_TIMEOUT // 2:
        c.turned, c.scan_from, c.pause, c.turn_mark = 0., None, 0, 0.
        c.change('scan', f'at the survey point ({dist:.2f} m)')


PHASES = dict(settle=settle, scan=scan, survey=survey)
