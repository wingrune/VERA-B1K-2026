"""Search phases of the fair planner of task 35: settle the arms (the rest trunk tilts the head camera down onto the floor near the robot), turn
in place until the camera is seen (and the routes have a map), walk to a survey point and turn again when it is not.
Subgoal near_camera (base within 1 m of the camera) is reached by these phases and the walk of the grasp phases."""
from __future__ import annotations

import math

import numpy as np

from core.skills_lib import nav_grid

from .. import params as P


def _turn(c, w) -> None:
    if c.scan_from is not None:
        c.turned += abs(nav_grid.wrap(w.base[2] - c.scan_from))
    c.scan_from = float(w.base[2])


def settle(c, action, q, w) -> None:
    c.to_rest(action, q)
    if c.age >= P.SETTLE_STEPS:
        c.turned, c.scan_from = 0., None
        c.change('scan', 'settled; turning to find the camera and the tripod')


def scan(c, action, q, w) -> None:
    c.to_rest(action, q)
    _turn(c, w)
    action[2] = P.SCAN_RATE
    found = w.camera is not None
    c.reason = f'turning {c.turned:.2f} rad, camera {"seen" if found else "not seen"}, tripod {"seen" if w.tripod is not None else "not seen"}'
    if found and c.turned >= P.SCAN_MIN_TURN_RAD and (w.tripod is not None or not P.SCAN_WANT_TRIPOD):
        c.change('plan_grasp', f'camera found after {c.turned:.2f} rad')
    elif c.turned >= P.SCAN_FULL_TURN_RAD:
        if found:
            c.change('plan_grasp', 'camera found')
        elif c.counts['surveys'] >= P.MAX_SURVEYS:
            c.change('failed', 'camera not found after every survey')
        else:
            c.change('survey', f'no camera after {c.turned:.2f} rad')


def survey(c, action, q, w) -> None:
    """Walk SURVEY_STEP_M towards the most open seen floor (a direction not tried before), then turn again."""
    if c.age == 1:
        c.counts['surveys'] += 1
        c.build_grid(w, [])
        best, goal = -1e9, None
        for k in range(16):
            a = 2 * math.pi * k / 16
            xy = w.base[:2] + P.SURVEY_STEP_M * np.array([math.cos(a), math.sin(a)])
            if not c.free_pose(xy) or not math.isfinite(c.route_len(w, xy)):
                continue
            far = max(xy_dist for xy_dist in [float(np.linalg.norm(xy - p)) for p in getattr(c, 'surveyed', [w.base[:2]])])
            score = c.inflated.clearance(xy) if hasattr(c.inflated, 'clearance') else 0.
            score += 0.8 * far
            if score > best:
                best, goal = score, xy
        c.survey_goal = goal
        c.surveyed = getattr(c, 'surveyed', []) + [w.base[:2].copy()]
        if goal is None:
            c.change('failed', 'no survey point reachable')
            return
    goal = c.survey_goal
    yaw = math.atan2(goal[1] - w.base[1], goal[0] - w.base[0])
    dist, yaw_err = c.navigate(action, q, w, goal, yaw)
    if w.camera is not None:
        c.change('plan_grasp', 'camera seen while walking')
    elif c.arrived(dist, abs(yaw_err) * 0) or c.age > P.NAV_TIMEOUT // 2:
        c.turned, c.scan_from = 0., None
        c.change('scan', f'at the survey point ({dist:.2f} m)')


PHASES = dict(settle=settle, scan=scan, survey=survey)
