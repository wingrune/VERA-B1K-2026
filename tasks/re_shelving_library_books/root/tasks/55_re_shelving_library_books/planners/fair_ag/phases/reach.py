"""Arm reach checks: clearance of the arm and the sweep to the first waypoint, GPU-IK feasibility, stand choice.
Mixin of logic.Planner of the task 55 fair planner; the methods were moved verbatim from logic.py."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import ik
from ..world import Box, State
from ..params import VARIANT


class ReachMixin:
    def static_clearance_boxes(self, s: State) -> List[Box]:
        grow = np.array([0.02, 0.02, 0.0])
        boxes = [Box(s.table_box.lo - grow, s.table_box.hi + grow)]
        boxes += [Box(c.box.lo - grow, c.box.hi + grow) for c in s.cases.values()]
        return boxes

    def arm_clear(self, side: str, x: np.ndarray, q: np.ndarray, base: np.ndarray, base_z: float, boxes: List[Box]) -> bool:
        trial = q.copy()
        trial[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])] = x
        poses = K.kinematics().forward(trial)[0]
        for link in (f'{side}_arm_link4', f'{side}_arm_link5', f'{side}_arm_link6'):
            w = G.to_world(base, base_z, poses[link][:3, 3])
            if any(np.all(w >= b.lo) and np.all(w <= b.hi) for b in boxes):
                return False
        return True

    def desk_item_boxes(self, s: State) -> List[Box]:
        """3D boxes of what stands on the desk (the lamp, 0.47 m tall), from its footprint (grown 0.02 m) up to 0.45 m above the top."""
        top = float(s.table_box.hi[2])
        return [Box(np.r_[lo, top], np.r_[hi, top + 0.45]) for lo, hi in self.desk_items(s)]

    def sweep_clear(self, side: str, x: np.ndarray, q: np.ndarray, base: np.ndarray, base_z: float, boxes: List[Box]) -> bool:
        """The arm going from the carry posture towards the joint target `x` (joint-space line, as the servo moves) keeps its forearm, wrist
        and gripper out of `boxes` over the first three quarters of the way. A box that holds the gripper at the target is left out (a book
        right beside the lamp cannot be reached any other way)."""
        dofs = list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])
        start = np.asarray(K.rest(side), float)
        links = [f'{side}_arm_link{k}' for k in (5, 6, 7)] + [f'{side}_gripper_link']

        def points(t: float) -> List[np.ndarray]:
            trial = q.copy()
            trial[dofs] = start + t * (np.asarray(x, float) - start)
            poses = K.kinematics().forward(trial)[0]
            pts = [G.to_world(base, base_z, poses[n][:3, 3]) for n in links]
            return pts + [(a + b) / 2 for a, b in zip(pts[:-1], pts[1:])]

        inside = lambda p, b: bool(np.all(p >= b.lo) and np.all(p <= b.hi))  # noqa: E731
        end = points(1.)[len(links) - 1]
        boxes = [b for b in boxes if not inside(end, b)]
        for t in np.linspace(0., 0.75, 10):
            if any(inside(p, b) for p in points(t) for b in boxes):
                return False
        return True

    def ik_feasible(self, side: str, base: np.ndarray, base_z: float, q: np.ndarray, targets: List[Tuple[np.ndarray, np.ndarray]],
                    boxes: List[Box]) -> bool:
        """Every target reachable from `base` by solve_near seeded with the current trunk + arm joints (the frozen carrying arm), arm clear."""
        import time
        t0 = time.perf_counter()
        seed = q[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])]
        stand = np.asarray(base, float).reshape(1, 3)
        try:
            for point, rotation in targets:
                self.counts['ik_checks'] += 1
                self.ik_calls += 1
                out = ik.near(side, ik.pose_T(np.asarray(point, float), np.asarray(rotation, float)), stand, base_z, seed)
                if out is None:              # no worker: only reached with BIMBO_IK_REQUIRE_GPU=0, ik.STATS counts the CPU solve
                    ik.STATS['cpu_near'] += 1
                    servo = K.ArmServo(side)
                    servo.use_fingers_mid()
                    x = servo.solve(q, G.to_robot(base, base_z, point), G.rotation_to_robot(base, rotation))
                else:
                    x = out[1][0] if out[0][0] else None
                if x is None or not self.arm_clear(side, x, q, base, base_z, boxes):
                    return False
            return True
        finally:
            self.ik_ms += int(1000 * (time.perf_counter() - t0))

    def reach_pick(self, s: State, q: np.ndarray, ranked: List[Dict[str, Any]], targets: List[Tuple[np.ndarray, np.ndarray]],
                   boxes: List[Box]) -> Optional[Tuple[np.ndarray, str, Dict[str, Any]]]:
        """Cheapest free stand whose one arm reaches every waypoint: one batched GPU scan (ik.scan, reachable_stands) per arm and waypoint."""
        import time
        if not ranked:
            return None
        stands = np.array([[c['xy'][0], c['xy'][1], c['yaw']] for c in ranked], np.float64)
        t0 = time.perf_counter()
        per_arm = {}
        for side in ('left', 'right'):
            ok = np.ones(len(stands), bool)
            q_sol = q_first = None
            for wp, rot in targets:
                mask, q_sol = ik.scan(side, ik.pose_T(np.asarray(wp, float), np.asarray(rot, float)), stands, float(s.base_z))
                q_first = q_sol if q_first is None else q_first
                self.ik_calls += 1
                ok &= mask
            per_arm[side] = (ok, q_sol, q_first)
        tall = self.desk_item_boxes(s) if 'sweep' in VARIANT else []
        self.ik_ms += int(1000 * (time.perf_counter() - t0))
        self.ik_backend = 'gpu' if ik.gpu_on() else 'cpu'
        best = None
        for i, cand in enumerate(ranked):
            base = stands[i]
            for side in self.sides_for(targets[0][0], cand):
                ok, q_sol, q_first = per_arm[side]
                if not ok[i] or not self.arm_clear(side, q_sol[i], q, base, s.base_z, boxes):
                    continue
                if tall and not self.sweep_clear(side, q_first[i], q, base, s.base_z, tall):
                    self.note(f'stand {np.round(base[:2], 2).tolist()} arm {side}: the arm would sweep through what stands on the desk')
                    continue
                if best is None or cand['cost'] < best[2]['cost']:
                    best = (base, side, cand)
                break
        return best
