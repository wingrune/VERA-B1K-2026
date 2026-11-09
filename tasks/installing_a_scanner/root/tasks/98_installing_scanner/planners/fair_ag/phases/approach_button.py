"""Phases plan_button and go_button: choose a base pose in front of the perceived button (rings around the marker, free cell and A* route on the
perceived obstacle grid, IK of the pre-press and press points), then drive there with the arms in the carry posture. Success: base within
NAV_POS_TOL and NAV_YAW_TOL of the pose. A base pose that is not reached, or a marker that moved more than 3 cm in the estimate while driving,
sends the phase back to planning (next candidate or re-plan)."""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from .. import geometry as G
from .. import params as P

NO_ROUTE_STEPS = 60
from core.skills_lib.nav_grid import wrap as G_wrap


def rot2(a: float) -> np.ndarray:
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def press_bearings(w) -> np.ndarray:
    """Bearings from the marker to the base: around the front normal first (every direction is ranked by path cost afterwards)."""
    f = G.ScannerFrame(w)
    centre = math.atan2(f.front[1], f.front[0])
    return np.array([centre + math.radians(a) for a in range(-180, 180, 20)])


def find_press_poses(c, w, q) -> List[Dict[str, Any]]:
    boxes = c.boxes_for_clearance(w)
    found: List[Dict[str, Any]] = []
    ranked = c.rank_candidates(w, w.marker[:2], bearings=press_bearings(w))
    for tilt in P.TILTS:
        plan = G.poke_plan(w, tilt)
        for cand in ranked[:P.CAND_LIMIT * 3]:
            for side in c.sides_for_point(w.marker, cand):
                base = np.array([cand['xy'][0], cand['xy'][1], cand['yaw']])
                if c.ik_feasible(side, base, q, [(plan['pre'], plan['rotation']), (plan['press'], plan['rotation'])], boxes):
                    found.append(dict(cand, side=side, tilt=tilt))
                    break
            if len(found) >= 3:
                break
        if found:
            break
    return found


def step_plan_button(c, action, q, w) -> None:
    c.to_rest(action, q)
    if not w.ready:
        c.change('scan', 'the world is not ready for planning')
        return
    c.build_grid(w)
    found = find_press_poses(c, w, q)
    if not found and not c.soft_clutter:
        c.build_grid(w, soft=True)
        found = find_press_poses(c, w, q)
    if not found:
        c.change('plan_push' if c.pick['placed'] else 'plan_grasp', 'no base pose reaches the marker; moving the scanner first')
        return
    found.sort(key=lambda cand: cand['cost'])
    for cand in found:      # the base pose is kept relative to the scanner, so a refined scanner estimate moves the goal with it
        cand['rel_xy'] = rot2(-w.scanner.yaw) @ (cand['xy'] - w.scanner.centre)
        cand['rel_yaw'] = G_wrap(cand['yaw'] - w.scanner.yaw)
    c.goal, c.cands = found[0], found[1:]
    c.side, c.tilt, c.servo = c.goal['side'], c.goal['tilt'], None
    c.change('go_button', f"base candidate {c.goal['xy'].round(2).tolist()} yaw {c.goal['yaw']:.2f} arm {c.side} tilt {c.tilt}"
                          f"{' (clutter treated as movable)' if c.soft_clutter else ''}")


def step_go_button(c, action, q, w) -> None:
    if w.scanner is not None:
        c.goal['xy'] = w.scanner.centre + rot2(w.scanner.yaw) @ c.goal['rel_xy']
        c.goal['yaw'] = G_wrap(w.scanner.yaw + c.goal['rel_yaw'])
    dist, yaw_err = c.navigate(action, q, w, c.goal['xy'], c.goal['yaw'])
    c.err = (dist, yaw_err)
    if c.arrived(dist, yaw_err):
        c.change('reach_button', 'at the base pose')
    elif c.counts['nav_stuck'] >= P.NAV_MAX_STUCK * (1 + c.counts['candidates_failed']) or c.age > P.NAV_TIMEOUT or c.no_route > NO_ROUTE_STEPS:
        c.no_route = 0
        c.next_candidate('go_button', 'base pose not reached')


PHASES = {'plan_button': step_plan_button, 'go_button': step_go_button}
