"""Base motion: route on the grid, stand candidates around a target, the drive and safe turns.
Mixin of logic.Planner of the task 55 fair planner; the methods were moved verbatim from logic.py."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.skills_lib import nav_grid

from .. import geometry as G
from ..world import Box, State
from ..params import (
    BEARING_STEP, CARRY_SLEW, CARRY_V, CARRY_W, CARRY_YAW_FIRST, NAV_GAIN, NAV_POS_TOL, NAV_REPLAN_EVERY, NAV_STUCK_STEPS, NAV_VFAR,
    NAV_VNEAR, NAV_WMAX, NAV_YAW_GAIN, NAV_YAW_TOL, RINGS, VARIANT)


class NavigateMixin:
    def free_pose(self, xy: np.ndarray) -> bool:
        return not self.inflated.blocked[self.inflated.cell(xy)]

    def route(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> Optional[List[np.ndarray]]:
        return self.inflated.path(start_xy, goal_xy)

    def rank_candidates(self, s: State, target_xy: np.ndarray, radii=RINGS, bearings=None, yaw_offsets=(0.,)) -> List[Dict[str, Any]]:
        bearings = bearings if bearings is not None else np.arange(0., 2 * math.pi, BEARING_STEP)
        ranked = []
        for xy, yaw in G.base_candidates(target_xy, radii, bearings, yaw_offsets):
            if not self.free_pose(xy):
                continue
            path = self.route(s.base[:2], xy)
            if path is None:
                continue
            ranked.append(dict(xy=xy, yaw=nav_grid.wrap(yaw), cost=nav_grid.path_length(path) + 0.15 * abs(nav_grid.wrap(yaw - s.base[2]))
                               + 1.5 * float(np.linalg.norm(xy - np.asarray(target_xy)[:2]))))
        ranked.sort(key=lambda c: c['cost'])
        return ranked

    def navigate(self, action: np.ndarray, q: np.ndarray, s: State, goal_xy: np.ndarray, goal_yaw: float) -> Tuple[float, float]:
        """One control towards the base pose, arms in the carry posture (a held book rides in the closed gripper, the arm keeps its joints and the
        base accelerates gently)."""
        self.to_rest(action, q, hold=self.carry_side())
        dist = float(np.linalg.norm(goal_xy - s.base[:2]))
        yaw_err = nav_grid.wrap(goal_yaw - s.base[2])
        if self.path is None or self.path_age >= NAV_REPLAN_EVERY:
            self.path = self.route(s.base[:2], goal_xy)
            self.path_age = 0
            self.counts['replans'] += 1
        self.path_age += 1
        if self.path is None:
            self.reason = 'no route to the base pose'
            return dist, yaw_err
        near = int(np.argmin([np.linalg.norm(p - s.base[:2]) for p in self.path]))
        waypoint = self.path[min(near + 1, len(self.path) - 1)]
        for p in self.path[near:]:
            waypoint = p
            if np.linalg.norm(p - s.base[:2]) > 0.4:
                break
        if dist < 0.4:
            waypoint = goal_xy
        carrying = self.holding is not None
        vmax = CARRY_V if carrying else (NAV_VNEAR if dist < 0.5 else NAV_VFAR)
        cmd = nav_grid.drive_command(s.base, waypoint, goal_yaw, NAV_GAIN, vmax, NAV_YAW_GAIN, CARRY_W if carrying else NAV_WMAX)
        if carrying and abs(yaw_err) > CARRY_YAW_FIRST:
            cmd[0], cmd[1] = 0., 0.                  # turn in place first; translating while yawing flings a friction pinch
            self.stuck_ref = (s.base[:2].copy(), self.age)
            if 'turn' in VARIANT or 'turnback' in VARIANT:
                self.safe_turn(cmd, s, yaw_err)
        elif carrying:
            self.turn_dir, self.turn_long = None, False
        if carrying:
            prev = self.cmd[:3] if self.cmd is not None else np.zeros(3)
            cmd = prev + np.clip(cmd - prev, -CARRY_SLEW, CARRY_SLEW)
        action[:3] = cmd
        self.reason = f'driving, {dist:.2f} m, yaw err {yaw_err:.2f}'
        if self.stuck_ref is None or np.linalg.norm(self.stuck_ref[0] - s.base[:2]) > 0.03:
            self.stuck_ref = (s.base[:2].copy(), self.age)
        elif self.age - self.stuck_ref[1] > NAV_STUCK_STEPS and dist > NAV_POS_TOL * 2:
            self.counts['nav_stuck'] += 1
            self.path, self.stuck_ref = None, None
        return dist, yaw_err

    def turn_hits(self, s: State, sweep: float) -> Optional[Box]:
        """What stands on the desk that the held book meets while the base turns in place by `sweep` rad (its footprint as a circle)."""
        b = s.books[self.holding].box
        rel = b.centre[:2] - s.base[:2]
        rad = 0.5 * float(np.linalg.norm(b.size[:2]))
        boxes = [Box(lo - rad, hi + rad) for lo, hi in self.desk_items(s)]
        for a in np.linspace(0., sweep, max(2, int(abs(sweep) / 0.05) + 1)):
            c, s_ = math.cos(a), math.sin(a)
            p = s.base[:2] + np.array([c * rel[0] - s_ * rel[1], s_ * rel[0] + c * rel[1]])
            for box in boxes:
                if np.all(p >= box.lo) and np.all(p <= box.hi):
                    return box
        return None

    def safe_turn(self, cmd: np.ndarray, s: State, yaw_err: float) -> None:
        """Turn in place the way whose arc keeps the held book off the lamp; when both ways meet it, back the base straight away first.
        Controls spent on a long way round or on backing away do not count against the carry timeout."""
        if self.turn_long:
            self.turn_extra += 1
        if self.turn_dir is None:
            short = yaw_err
            other = yaw_err - math.copysign(2 * math.pi, yaw_err)
            hit = self.turn_hits(s, short)
            if hit is None:
                self.turn_dir = math.copysign(1., short)
            elif self.turn_hits(s, other) is None and ('turnback' not in VARIANT or self.turn_extra > 300):
                self.turn_dir, self.turn_long = math.copysign(1., other), True
                self.note(f'carry turn: the short way sweeps the book through the desk item, turning {other:.2f} rad')
            else:
                away = s.base[:2] - np.clip(s.base[:2], hit.lo, hit.hi)
                away = away / max(float(np.linalg.norm(away)), 1e-6)
                c, s_ = math.cos(s.base[2]), math.sin(s.base[2])
                cmd[0], cmd[1] = CARRY_V * (c * away[0] + s_ * away[1]), CARRY_V * (-s_ * away[0] + c * away[1])
                cmd[2] = 0.
                self.turn_extra += 1
                self.reason = 'backing away so the held book can turn clear of the desk item'
                return
        cmd[2] = self.turn_dir * CARRY_W

    @staticmethod
    def arrived(dist: float, yaw_err: float) -> bool:
        return dist < NAV_POS_TOL and abs(yaw_err) < NAV_YAW_TOL
