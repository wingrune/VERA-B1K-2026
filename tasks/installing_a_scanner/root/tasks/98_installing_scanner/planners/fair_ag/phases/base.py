"""Shared machinery of the phases of the fair planner: state of the plan, navigation over the perceived obstacle grid, arm posture helpers,
IK feasibility checks (GPU cuRobo: reachable_stands for a base pose, solve_near for the servo). Everything works on the World (estimates
in the desk frame); nothing reads the simulator."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import ik
from .. import params as P
from ..perception import Perception
from ..world import Box, World
from .pick_core import new_pick


class Core:
    """State and helpers shared by the phases. `step` of a phase module is called as step(core, action, q, world)."""

    def __init__(self) -> None:
        self.perc = Perception()
        self.reset_state()

    def reset_state(self) -> None:
        self.perc.reset()
        self.phase, self.reason, self.age, self.step = 'settle', 'start', 0, 0
        self.counts: Dict[str, int] = dict(replans=0, ik_checks=0, press_retries=0, push_rounds=0, nav_stuck=0, candidates_failed=0, ik_none=0,
                                           turns=0, regaze=0)
        self.cmd: Optional[np.ndarray] = None
        self.grid: Optional[nav_grid.Grid] = None
        self.inflated: Optional[nav_grid.Grid] = None
        self.soft_clutter = False
        self.side, self.tilt = 'left', P.TILTS[0]
        self.servo: Optional[K.ArmServo] = None
        self.cands: List[Dict[str, Any]] = []
        self.goal: Optional[Dict[str, Any]] = None
        self.path: Optional[List[np.ndarray]] = None
        self.path_age = 0
        self.stuck_ref: Optional[Tuple[np.ndarray, int]] = None
        self.press_s, self.press_hold, self.dither = P.PRE_OFFSET_M, 0, 0
        self.pressed = False                 # the press was carried out (dwell done); the button state is never observed
        self.push: Optional[Dict[str, Any]] = None
        self.push_p = 0.
        self.push_base0: Optional[np.ndarray] = None
        self.push_scanner0: Optional[np.ndarray] = None
        self.bad_push: set = set()
        self.bad_rounds: set = set()
        self.grip = {'left': K.GRIPPER_CLOSED, 'right': K.GRIPPER_CLOSED}
        self.push_yaw0 = 0.
        self.push_stall, self.push_moved = 0, -1.
        self.err = (float('nan'), float('nan'))
        self.history: List[Tuple[int, str, str]] = []
        self.tip: Optional[List[float]] = None
        self.gap_goal = P.GAP_GOAL_M
        self.scan_total = 0.
        self.scan_last_yaw: Optional[float] = None
        self.done_age = 0
        self.repressed = False
        self.press_world: Optional[World] = None
        self.grid_round = -1
        self.push_stats: Dict[str, int] = {}
        self.recoveries = 0
        self.face_filt = 0.
        self.tip_along: Optional[float] = None
        self.last_drive = np.zeros(3)
        self.unstick_dir = np.array([-1., 0.])
        self.resume_phase = 'plan_button'
        self.no_route = 0
        self.relax_level = 0                 # 0: full robot radius; 1, 2: smaller radii when no base pose is free or routable
        self.pick: Dict[str, Any] = new_pick()

    # ---- transitions ----------------------------------------------------------------------------------------------------------------
    def change(self, phase: str, reason: str) -> None:
        self.history.append((self.step, phase, reason))
        self.phase, self.reason, self.age = phase, reason, 0
        self.path, self.stuck_ref = None, None

    # ---- navigation -----------------------------------------------------------------------------------------------------------------
    def build_grid(self, w: World, soft: bool = False) -> None:
        pts = [w.base[:2], w.scanner.centre] + ([w.laptop_box.centre[:2]] if w.laptop_box is not None else [])
        lo, hi = np.min(pts, axis=0) - 2.0, np.max(pts, axis=0) + 2.0
        self.grid = self.make_grid(soft, lo, hi)
        self.inflated = self.grid.inflated((P.ROBOT_RADIUS, 0.5 * (P.ROBOT_RADIUS + P.ROBOT_TIGHT_RADIUS), P.ROBOT_TIGHT_RADIUS)[self.relax_level])
        # around the base itself only the true footprint counts: the robot is there, so a route out of a tight spot must exist
        tight = self.grid.inflated(P.ROBOT_TIGHT_RADIUS)
        ix, iy = np.meshgrid(np.arange(self.grid.shape[0]), np.arange(self.grid.shape[1]), indexing='ij')
        near = np.hypot(self.grid.lo[0] + (ix + .5) * self.grid.res - w.base[0], self.grid.lo[1] + (iy + .5) * self.grid.res - w.base[1]) < P.START_RELAX_M
        self.inflated.blocked = np.where(near, tight.blocked, self.inflated.blocked)
        self.soft_clutter = soft

    def make_grid(self, soft: bool, lo: np.ndarray, hi: np.ndarray) -> nav_grid.Grid:
        return self.perc.nav_grid(soft, lo, hi)

    def to_rest(self, action: np.ndarray, q: np.ndarray) -> float:
        """Walk the trunk and both arms to the carry posture; returns the largest remaining joint error."""
        worst = 0.
        for side in ('left', 'right'):
            target = K.rest(side)
            dofs = list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])
            cur = q[dofs]
            new = cur + np.clip(target - cur, -P.REST_STEP, P.REST_STEP)
            worst = max(worst, float(np.max(np.abs(target - cur))))
            action[K.TRUNK_ACTION] = new[:4]
            action[K.ARM_ACTION[side]] = new[4:]
        return worst

    def lane_length(self, xy: np.ndarray, e: np.ndarray, limit: float) -> float:
        d = 0.
        while d + 0.05 <= limit + 1e-9:
            if self.inflated.blocked[self.inflated.cell(xy + e[:2] * (d + 0.05))]:
                break
            d += 0.05
        return d

    def free_pose(self, xy: np.ndarray) -> bool:
        return not self.inflated.blocked[self.inflated.cell(xy)]

    def rank_candidates(self, w: World, target_xy: np.ndarray, radii=P.RINGS, bearings=None) -> List[Dict[str, Any]]:
        bearings = bearings if bearings is not None else np.arange(0., 2 * math.pi, math.radians(P.BEARING_STEP_DEG))
        while True:
            ranked = []
            for xy, yaw in G.base_candidates(target_xy, radii, bearings):
                if not self.free_pose(xy):
                    continue
                path = self.inflated.path(w.base[:2], xy)
                if path is None:
                    continue
                ranked.append(dict(xy=xy, yaw=nav_grid.wrap(yaw), cost=nav_grid.path_length(path) + 0.15 * abs(nav_grid.wrap(yaw - w.base[2]))))
            if ranked or self.relax_level >= 2 or self.grid is None:
                break
            # nothing free or routable: the desk is hemmed in by clutter; shrink the inflation (the arm reach is still checked by IK)
            self.relax_level += 1
            self.build_grid(w, self.soft_clutter)
            self.path = None
        ranked.sort(key=lambda c: c['cost'])
        return ranked

    def navigate(self, action: np.ndarray, q: np.ndarray, w: World, goal_xy: np.ndarray, goal_yaw: float, rest: bool = True,
                 vmax_far: float = P.NAV_VFAR, wmax: float = P.NAV_WMAX) -> Tuple[float, float]:
        """One control towards the base pose; arms in the carry posture when `rest`. Returns (distance m, yaw error rad)."""
        if rest:
            self.to_rest(action, q)
        dist = float(np.linalg.norm(goal_xy - w.base[:2]))
        yaw_err = nav_grid.wrap(goal_yaw - w.base[2])
        if self.path is None or self.path_age >= P.NAV_REPLAN_EVERY:
            if self.path is not None:
                self.build_grid(w, self.soft_clutter)         # the map grew since the last plan
            self.path = self.inflated.path(w.base[:2], goal_xy)
            self.path_age = 0
            self.counts['replans'] += 1
        self.path_age += 1
        if self.path is None:
            self.reason = 'no route to the base pose'
            self.no_route += 1
            return dist, yaw_err
        self.no_route = 0
        near = int(np.argmin([np.linalg.norm(p - w.base[:2]) for p in self.path]))
        waypoint = self.path[min(near + 1, len(self.path) - 1)]
        for p in self.path[near:]:
            waypoint = p
            if np.linalg.norm(p - w.base[:2]) > 0.4:
                break
        if dist < 0.4:
            waypoint = goal_xy
        vmax = min(P.NAV_VNEAR, vmax_far) if dist < 0.5 else vmax_far
        soft = getattr(self.grid, 'soft', None)
        if soft is not None:        # slow down where the route crosses a chair: at speed the base rides up onto it
            heading = (waypoint - w.base[:2]) / max(float(np.linalg.norm(waypoint - w.base[:2])), 1e-6)
            if any(soft[self.grid.cell(w.base[:2] + heading * d)] for d in (0.3, 0.6, 0.9)):
                vmax = min(vmax, P.NAV_VSOFT)
        action[:3] = nav_grid.drive_command(w.base, waypoint, goal_yaw, P.NAV_GAIN, vmax, P.NAV_YAW_GAIN, wmax)
        if np.linalg.norm(action[:2]) > 0.05:
            self.last_drive = action[:3].copy()
        self.reason = f'driving, {dist:.2f} m, yaw err {yaw_err:.2f}'
        if self.stuck_ref is None or np.linalg.norm(self.stuck_ref[0] - w.base[:2]) > 0.03:
            self.stuck_ref = (w.base[:2].copy(), self.age)
        elif self.age - self.stuck_ref[1] > P.NAV_STUCK_STEPS and dist > P.NAV_POS_TOL * 2:
            self.counts['nav_stuck'] += 1
            self.path, self.stuck_ref = None, None
        return dist, yaw_err

    @staticmethod
    def arrived(dist: float, yaw_err: float) -> bool:
        return dist < P.NAV_POS_TOL and abs(yaw_err) < P.NAV_YAW_TOL

    def next_candidate(self, back_to: str, why: str) -> None:
        self.counts['candidates_failed'] += 1
        if self.cands:
            self.goal = self.cands.pop(0)
            self.side, self.tilt = self.goal['side'], self.goal['tilt']
            self.servo = None
            self.change(back_to, why + '; next base candidate')
        else:
            self.change('plan_button' if back_to == 'go_button' else 'plan_push', why + '; no candidates left, re-planning')

    # ---- arm --------------------------------------------------------------------------------------------------------------------------
    def make_servo(self, mid: bool = False) -> K.ArmServo:
        return ik.make_servo(self.side, 'mid' if mid else 'tip')

    def open_grip(self, spread: float) -> None:
        self.grip[self.side] = float(np.clip(spread / 0.05 * 2. - 1., -1., 1.))

    def close_grips(self) -> None:
        self.grip = {'left': K.GRIPPER_CLOSED, 'right': K.GRIPPER_CLOSED}

    @staticmethod
    def boxes_for_clearance(w: World) -> List[Box]:
        grow = np.array([0.02, 0.02, 0.02])
        return [Box(b.lo - grow, b.hi + grow) for b in (w.table_box, w.laptop_box, w.scanner_box)]

    def arm_clear(self, side: str, x: np.ndarray, q: np.ndarray, base: np.ndarray, boxes: List[Box]) -> bool:
        trial = q.copy()
        trial[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])] = x
        poses = K.kinematics().forward(trial)[0]
        c, sn = math.cos(base[2]), math.sin(base[2])
        for link in (f'{side}_arm_link4', f'{side}_arm_link5', f'{side}_arm_link6', f'{side}_arm_link7'):
            p = poses[link][:3, 3]
            wp = np.array([base[0] + c * p[0] - sn * p[1], base[1] + sn * p[0] + c * p[1], p[2]])
            if any(np.all(wp >= b.lo) and np.all(wp <= b.hi) for b in boxes):
                return False
        return True

    def ik_feasible(self, side: str, base: np.ndarray, q: np.ndarray, targets: List[Tuple[np.ndarray, np.ndarray]], boxes: List[Box],
                    mid: bool = False) -> bool:
        servo = K.ArmServo(side)
        servo.use_fingers_mid() if mid else servo.use_tip()
        q0 = q.copy()
        q0[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])] = K.rest(side)
        tool, rel = ik.link_point_in_gripper(side, q0, servo.link, servo.point)
        goals = []
        for point, rotation in targets:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = ik.gripper_goal(rotation, rel), point
            goals.append(T)
        self.counts['ik_checks'] += len(goals)
        masks, qs = ik.scan(side, goals, base[None], 0., tool)
        return all(masks[k, 0] and self.arm_clear(side, qs[k, 0], q0, base, boxes) for k in range(len(goals)))

    @staticmethod
    def sides_for_point(point: np.ndarray, cand: Dict[str, Any]) -> List[str]:
        """Arm order: the arm on the side of the point as seen from the base pose first."""
        d = point[:2] - cand['xy']
        lateral = -math.sin(cand['yaw']) * d[0] + math.cos(cand['yaw']) * d[1]
        return ['left', 'right'] if lateral >= 0 else ['right', 'left']

    @staticmethod
    def tool_goal(point: np.ndarray, rotation: np.ndarray, w: World):
        return G.to_robot(w.base, 0., point), G.rotation_to_robot(w.base, rotation)
