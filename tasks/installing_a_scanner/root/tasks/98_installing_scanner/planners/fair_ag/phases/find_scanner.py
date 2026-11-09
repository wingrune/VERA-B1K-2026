"""Phases settle and scan: bring the arms to the carry posture and turn in place until perception has the desk, the laptop and the scanner with
its button end. Success: the world is ready (desk frame locked, scanner pose with the button end known, laptop box). A full turn without it
sends the robot a metre forward (the desk may be hidden behind a wall or out of range) and the turn is repeated."""
from __future__ import annotations

import numpy as np

from core.skills_lib import nav_grid

from .. import params as P


def ready(c, w) -> bool:
    return w.ready and c.perc.locked and w.scanner.front_known


def step_settle(c, action, q, w) -> None:
    c.to_rest(action, q)
    if c.age >= P.SETTLE_STEPS:
        c.scan_last_yaw = c.perc.odo.pose[2]
        c.scan_start = c.scan_total
        c.change('scan', 'settled')


def step_scan(c, action, q, w) -> None:
    c.to_rest(action, q)
    # A new spot (after settle or walk_on) needs a near full look around before planning: the obstacle grid around the
    # robot and the desk must be seen, else no base pose reaches the marker.
    session = c.scan_total - getattr(c, 'scan_start', 0.0)
    if ready(c, w) and c.age > 5 and (session >= P.SCAN_MIN_TURN_RAD or c.recoveries > 0):
        c.counts.update({f'wall_planes_{k}': v for k, v in c.perc.freeze_walls().items()})
        c.change('plan_grasp' if c.pressed else 'plan_button', f'world ready after {c.scan_total:.1f} rad of turning: scanner {np.round(w.scanner.centre, 2).tolist()}, '
                                f'desk locked, laptop and button end known')
        return
    yaw = c.perc.odo.pose[2]
    c.scan_total += abs(nav_grid.wrap(yaw - c.scan_last_yaw))
    c.scan_last_yaw = yaw
    if c.scan_total > 2 * np.pi * (1 + c.counts['turns']):
        c.counts['turns'] += 1
        c.change('walk_on', f'a full turn did not give the scene (desk area {c.perc.desk_area:.2f} m2, scanner {w.scanner is not None}, '
                            f'front {w.scanner is not None and w.scanner.front_known})')
        return
    action[2] = P.SCAN_RATE / nav_grid.BASE_TURN
    c.reason = f'turning, {c.scan_total:.1f} rad, desk area {c.perc.desk_area:.2f} m2'


def step_walk_on(c, action, q, w) -> None:
    """Walk a metre forward (robot frame) and turn again."""
    c.to_rest(action, q)
    action[0] = 0.5
    c.reason = f'walking forward, step {c.age}'
    if c.age >= 40:
        c.scan_last_yaw = c.perc.odo.pose[2]
        c.scan_start = c.scan_total
        c.change('scan', 'walked on, looking around again')


PHASES = {'settle': step_settle, 'scan': step_scan, 'walk_on': step_walk_on}
