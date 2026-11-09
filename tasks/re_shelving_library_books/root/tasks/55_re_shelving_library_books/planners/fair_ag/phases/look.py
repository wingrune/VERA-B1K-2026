"""Phases of the fair layer: settle, scan (a turn, then localise on the map), look, look_hand, goview.
Mixin of planner.Planner of the task 55 fair planner; the methods were moved verbatim from planner.py."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from core.skills_lib import nav_grid

from ..world import State

SCAN_RATE, SCAN_EVERY, SCAN_TURN = 0.5, 6, 2 * math.pi + 0.3
LOOK_MIN, LOOK_MAX, LOOK_EVERY = 18, 90, 6
DESK_FIX_RANGE_M = 2.6
MOVED_M, MOVED_YAW = 0.015, 0.06
VIEW_RADII, VIEW_BEARINGS = (1.2, 1.5), 16
MAX_VIEWS = 4


class LookMixin:
    # ---- phases of the fair layer -----------------------------------------------------------------------------------------------------------
    def phase_settle(self, action, q, s: State) -> None:
        self.to_rest(action, q)
        if self.age >= 12:
            self.finish_change('scan', 'settled')

    def phase_scan(self, action, q, s: State) -> None:
        """Turn in place once; then localise on the map."""
        self.to_rest(action, q)
        yaw = float(self.perc.odo.pose[2])
        if self.scan_prev is not None:
            self.scan_turn += abs(math.remainder(yaw - self.scan_prev, 2 * math.pi))
        self.scan_prev = yaw
        if self.age % SCAN_EVERY == 0:
            self.perc.collect_scan()
        if self.scan_turn < SCAN_TURN:
            action[2] = SCAN_RATE
            self.reason = f'scan turn {self.scan_turn:.2f} rad'
            return
        if not self.perc.localise():
            self.scan_turn, self.scan_prev = 0., None
            self.reason = 'localisation failed, turning again'
            return
        h = self.perc.hyps
        self.build_grid(self.belief())
        self.finish_change('look', f'localised: score {h[0]["score"]:.3f}' + (f' vs {h[1]["score"]:.3f}' if len(h) > 1 else ''))

    def phase_look(self, action, q, s: State) -> None:
        worst = self.to_rest(action, q, hold=self.carry_side())
        if self.holding is not None:
            self.freeze_hold_arm(action, q)
        if self.age < LOOK_MIN or (worst > 0.05 and self.age < LOOK_MAX):
            self.reason = f'look: arms to rest ({worst:.2f})'
            return
        if self.age % LOOK_EVERY:
            return
        p = self.perc
        p.track()
        if float(np.linalg.norm(p.base()[:2] - p.map.table.centre[:2])) < DESK_FIX_RANGE_M:
            p.desk_fix()
        seen = p.update_desk()
        self.looks = getattr(self, 'looks', 0) + 1
        if self.looks < 3 and self.age < LOOK_MAX:
            return
        self.looks = 0
        if not p.books and self.views < MAX_VIEWS:
            self.finish_change('goview', 'no book seen from here')
            return
        self.finish_change(self.after_look, f'looked ({self.look_reason}): books {sorted(p.books)} updated {seen}')

    def phase_look_hand(self, action, q, s: State) -> None:
        """At the stand of an action: re-measure, keep the plan when its book stayed put."""
        a = self.act_plan
        if self.holding is not None:
            self.freeze_hold_arm(action, q)
        if self.age == 1:
            b = s.books.get(a['book'])
            self.snapshot = dict(xy=None if b is None else b.box.centre[:2].copy(), yaw=None if b is None else b.yaw)
        if self.age < LOOK_MIN or self.age % LOOK_EVERY:
            return
        p = self.perc
        p.track()
        if float(np.linalg.norm(p.base()[:2] - p.map.table.centre[:2])) < DESK_FIX_RANGE_M:
            p.desk_fix()
        if a['kind'] != 'insert':
            p.update_desk()
        if self.age < LOOK_MIN + 2 * LOOK_EVERY:
            return
        s = self.belief()
        if a['kind'] == 'insert':
            self.refresh_insert(a, s, q)
            self.finish_change('hand', 'at the base pose (looked)')
            return
        b = s.books.get(a['book'])
        if b is None or self.snapshot.get('xy') is None:
            self.finish_change('hand', 'at the base pose')
            return
        moved = float(np.linalg.norm(b.box.centre[:2] - self.snapshot['xy']))
        turned = abs(math.remainder(b.yaw - self.snapshot['yaw'], math.pi))
        if moved > MOVED_M or turned > MOVED_YAW:
            self.attempts[self.key(a['book'], a['kind'])] = max(0, self.attempts.get(self.key(a['book'], a['kind']), 1) - 1)
            self.act_plan, self.servo = None, None
            self.finish_change('plan', f'{a["book"]} is {moved:.3f} m / {turned:.2f} rad off the plan: re-plan from here')
            return
        self.finish_change('hand', f'at the base pose (book within {moved:.3f} m)')

    def phase_goview(self, action, q, s: State) -> None:
        """Drive to a stop that sees the desk top (the nearest free one not used yet), then look."""
        if self.age == 1 or self.view_goal is None:
            self.views += 1
            self.view_goal = self.pick_view(s)
            if self.view_goal is None:
                self.finish_change('failed', 'no view stop of the desk')
                return
        dist, yaw_err = self.navigate(action, q, s, self.view_goal[:2], self.view_goal[2])
        if self.arrived(dist, yaw_err) or self.age > 1500:
            self.view_goal = None
            self.after_look = 'plan'
            self.finish_change('look', f'view stop {self.views} reached')

    def pick_view(self, s: State) -> Optional[np.ndarray]:
        if self.views <= 1 or not hasattr(self, 'used_views'):
            self.used_views = []
        used = self.used_views
        fresh = self.views >= 3          # repeat looks (planning already failed): never re-look from a pose already used
        best = self._best_view(s, VIEW_RADII + ((0.95,) if fresh else ()), used if fresh else [])
        if best is None and fresh:
            best = self._best_view(s, VIEW_RADII, [])
        if best is not None:
            used.append(np.asarray(best[:2], float).copy())
        return best

    def _best_view(self, s: State, radii, avoid) -> Optional[np.ndarray]:
        c = self.perc.map.table.centre[:2]
        here = np.asarray(s.base[:2], float)
        best = None
        for r in radii:
            for k in range(VIEW_BEARINGS):
                a = 2 * math.pi * k / VIEW_BEARINGS
                xy = c + r * np.array([math.cos(a), math.sin(a)])
                if avoid and (float(np.linalg.norm(xy - here)) < 0.5 or any(float(np.linalg.norm(xy - u)) < 0.5 for u in avoid)):
                    continue
                if not self.free_pose(xy):
                    continue
                path = self.route(s.base[:2], xy)
                if path is None:
                    continue
                cost = nav_grid.path_length(path) + 0.8 * (self.views - 1) * (k % 3)
                if best is None or cost < best[0]:
                    best = (cost, np.r_[xy, math.atan2(c[1] - xy[1], c[0] - xy[0])])
        return None if best is None else best[1]
