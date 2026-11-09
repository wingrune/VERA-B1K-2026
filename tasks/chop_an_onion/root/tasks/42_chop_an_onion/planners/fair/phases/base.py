"""Shared machinery of the phases of the fair planner of task 42: plan state, transitions, arm helpers, stand choice. IK on the GPU: cuRobo
reachable_stands as a stand mask only, the servo solves solve_near from the current joints (or tracks resolved-rate for the short held moves).
Everything works on the World (estimates in the odometry frame); nothing reads the simulator."""
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
from ..world import ItemEst, World
from . import clearance
from .nav import NavMixin

DETECT_PHASES = ('scan', 'survey', 'approach', 'go', 'look', 'look_bowl', 'look_sink')
HOLD_PHASES = ('reach', 'refine', 'grasp', 'lift', 'carry', 'lower', 'release', 'chop', 'clear', 'wait_slicer', 'dice', 'sink_lower', 'chop_go', 'sink_go',
               'cut_up')


class Core(NavMixin):
    """State shared by the phases.
    State and helpers shared by the phases. A phase step is called as step(core, action, q, world)."""

    def __init__(self) -> None:
        self.perc = Perception()
        self.reset_state()

    def reset_state(self) -> None:
        self.perc.reset()
        self.phase, self.reason, self.age, self.step = 'settle', 'start', 0, 0
        self.phase_t0 = 0
        self.stale_looks = 0
        self.counts: Dict[str, int] = dict(replans=0, nav_stuck=0, plan_fail=0, grasp_fail=0, shifts=0, surveys=0, gpu_ik=0, drops=0,
                                           chop_rounds=0, dice_rounds=0, sink_tries=0)
        self.cmd: Optional[np.ndarray] = None
        self.inflated: Optional[nav_grid.Grid] = None
        self.raw_grid: Optional[nav_grid.Grid] = None
        self.radius = P.ROBOT_RADIUS
        self.servos = {s: ik.make_servo(s, 'mid') for s in ('left', 'right')}
        self.side, self.role = 'right', None
        self.mode, self.place_kind = 'top', None
        self.stand: Optional[Tuple[np.ndarray, float]] = None
        self.grip = {'left': K.GRIPPER_OPEN, 'right': K.GRIPPER_OPEN}
        self.hold_q: Optional[np.ndarray] = None
        self.tries: Dict[str, int] = {}
        self.path: Optional[List[np.ndarray]] = None
        self.path_age = 0
        self.stuck_ref: Optional[Tuple[np.ndarray, int]] = None
        self.routes: Dict[tuple, float] = {}
        self.err = (float('nan'), float('nan'))
        self.history: List[Tuple[int, str, str]] = []
        self.dwell = 0
        self.item: Optional[ItemEst] = None          # the frozen estimate of the object being handled
        self.hand_rot: Optional[np.ndarray] = None   # world rotation of the gripper when it closed
        self.lift_goal: Optional[np.ndarray] = None
        self.cut_target: Optional[np.ndarray] = None
        self.turned = 0.0
        self.scan_yaw0: Optional[float] = None
        self.scan_limit = P.SCAN_FULL_TURN_RAD
        self.shift_k = 0
        self.explored: List[np.ndarray] = []
        self.debug: Dict[str, Any] = {}
        self.reach_stats: Dict[str, Any] = {}
        self.last_cands: list = []
        self.close_view = False
        self.vantage_k: Optional[tuple] = None
        self.vantage_t0 = 0
        self.vantage: Optional[Tuple[np.ndarray, float]] = None
        self.after_look = 'choose'
        self.stages = dict(onion=False, parer_cut=False, knife=False, board=False)
        self.pick_stats: Dict[str, Any] = {}
        self.sink_rot: Optional[np.ndarray] = None
        self.sink_off: Optional[np.ndarray] = None
        self.release_xy: Optional[np.ndarray] = None
        self.bowl_before: Dict[str, Any] = {}
        self.board_v: Optional[np.ndarray] = None
        self.cut_rounds = 0
        self.cut_mode = 'chop'
        self.sink_sub = 'over'
        self.sink_goal = None
        self.age_down = 0
        self.bad_stands: List[np.ndarray] = []
        self.flip = False
        self.tilt = 0.0
        self.stand_rel: Optional[Tuple[float, float, float]] = None   # (offset along the counter from the target, distance from its front edge, yaw offset)
        self.stand_q: Optional[np.ndarray] = None
        self.pre_q: Optional[np.ndarray] = None
        self.reach_mode = 'solve'
        self.walk_age = 0
        self.stand_target: Optional[np.ndarray] = None
        self.jaw = 0.0
        self.held_offset = np.zeros(3)
        self.sink_try = 0
        self.stages_fail: List[str] = []
        self.stall_ref: Optional[Tuple[float, int]] = None

    # ---- transitions ----------------------------------------------------------------------------------------------------------------
    def change(self, phase: str, reason: str) -> None:
        self.history.append((self.step, phase, reason))
        self.phase, self.reason, self.age = phase, reason, 0
        self.phase_t0 = self.step
        if phase != 'approach':
            self.vantage = None
        self.path, self.stuck_ref, self.stall_ref = None, None, None
        self.perc.detect = phase in DETECT_PHASES
        self.perc.recent.clear()
        self.perc.map_hold = phase in HOLD_PHASES
        for servo in self.servos.values():
            servo.target = None
            servo.stepwise = False
            servo.sync = False

    # ---- frames and holds -----------------------------------------------------------------------------------------------------------
    def goal_robot(self, w: World, point: np.ndarray, rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return G.to_robot(w.base, 0., point), G.rotation_to_robot(w.base, rot)

    def mid_world(self, q: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray]:
        T = self.perc.mid_T(q, side)
        return T[:3, 3].copy(), T[:3, :3].copy()

    @staticmethod
    def finger_gap(q: np.ndarray, side: str) -> float:
        return float(sum(q[list(K.FINGER_DOFS[side])]))

    def holds(self, q: np.ndarray, side: str) -> bool:
        """Fingers commanded closed that stay apart: something is between them (the assisted grasp holds it). Proprio only."""
        return self.grip[side] == K.GRIPPER_CLOSED and self.finger_gap(q, side) > P.HELD_GAP_M

    # ---- arm --------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def rest_pose(side: str) -> np.ndarray:
        return K.rest(side).copy()

    def to_rest(self, action: np.ndarray, q: np.ndarray, sides=('left', 'right')) -> float:
        worst = 0.
        for side in sides:
            target = self.rest_pose(side)
            dofs = list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])
            cur = q[dofs]
            new = cur + np.clip(target - cur, -P.REST_STEP, P.REST_STEP)
            worst = max(worst, float(np.max(np.abs(target - cur))))
            action[K.TRUNK_ACTION] = new[:4]
            action[K.ARM_ACTION[side]] = new[4:]
        return worst

    def rest_one(self, action: np.ndarray, q: np.ndarray, side: str) -> None:
        target = self.rest_pose(side)
        dofs = list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side])
        cur = q[dofs]
        new = cur + np.clip(target - cur, -P.REST_STEP, P.REST_STEP)
        action[K.ARM_ACTION[side]] = new[4:]
        if side == 'left':
            action[K.TRUNK_ACTION] = new[:4]

    def hold_arm(self, action: np.ndarray, q: np.ndarray, side: str) -> None:
        src = self.hold_q if self.hold_q is not None else q
        action[K.TRUNK_ACTION] = src[list(K.TRUNK_DOFS)]
        action[K.ARM_ACTION[side]] = src[list(K.ARM_DOFS[side])]

    def servo(self, action: np.ndarray, q: np.ndarray, w: World, point: np.ndarray, rot: np.ndarray, plan: bool = True,
              stepwise: bool = False) -> Tuple[float, float]:
        """One control of the active arm towards (point, rot) of the finger midpoint (odometry frame); the other arm keeps its joints (or the
        joints of hold_q when it holds something)."""
        servo = self.servos[self.side]
        servo.use_fingers_mid() if servo.link != K.GRIPPER_LINK[self.side] or not np.allclose(servo.point, K.FINGER_MID_POINT) else None
        servo.stepwise = stepwise
        goal, rotation = self.goal_robot(w, point, rot)
        err = servo.step(action, q, goal, rotation, plan=plan and not stepwise)
        other = 'left' if self.side == 'right' else 'right'
        if self.hold_q is not None and self.holds(q, other):
            action[K.ARM_ACTION[other]] = self.hold_q[list(K.ARM_DOFS[other])]
        else:
            action[K.ARM_ACTION[other]] = q[list(K.ARM_DOFS[other])]
        self.err = err
        return err

    def hand_rotation(self, q: np.ndarray, w: World) -> np.ndarray:
        """World rotation of the gripper link of the active side now."""
        return G.rot_yaw(float(w.base[2])) @ self.servos[self.side].pose(q)[:3, :3]

    # ---- reach / stands -------------------------------------------------------------------------------------------------------------
    def reach_mask(self, side: str, goals: List[Tuple[np.ndarray, np.ndarray]], stands: np.ndarray, w: Optional[World] = None,
                   check_arm: bool = True, with_q: bool = False):
        """(N,) bool: the finger midpoint reaches every (point, rotation) goal from each stand (x, y, yaw), GPU cuRobo, and the arm at each goal
        does not touch the counter (clearance.hits on the joints of the solution)."""
        if len(stands) == 0:
            return (np.zeros(0, bool), None) if with_q else np.zeros(0, bool)
        Ts = []
        for p, r in goals:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = r, p
            Ts.append(T)
        masks, qs = ik.scan(side, Ts, stands, 0.0, K.FINGER_MID_POINT)
        self.counts['gpu_ik'] += 1
        ok = masks.all(axis=0)
        self.reach_stats = dict(ik=int(ok.sum()), n=int(len(ok)))
        if check_arm and w is not None and w.counter is not None:
            for i in np.where(ok)[0]:
                why: list = []
                if any(clearance.hits(w, side, qs[k, i], stands[i], why) for k in range(len(Ts))):
                    if 'why' not in self.reach_stats:
                        self.reach_stats['why'] = why[0]
                    ok[i] = False
                    self.counts['arm_clash'] = self.counts.get('arm_clash', 0) + 1
            self.reach_stats['clear'] = int(ok.sum())
        return (ok, qs) if with_q else ok

    def pick_stand(self, w: World, target: np.ndarray, goals_fn, side: str, yaw_near: Optional[float] = None, max_ring: Optional[float] = None
                   ) -> Optional[Tuple[np.ndarray, float]]:
        """A stand in front of the counter that reaches every goal of goals_fn() without the arm touching the counter (batched GPU scan), is free
        in the route grid and has a route. Candidates are laid out in the counter frame: at STAND_DISTS from its front edge, within STAND_SPAN_M
        along it of the target, facing the counter with small yaw offsets (the ring layout of the oracle serves only as the fallback without a
        counter). Nearest to the target along the counter first. Returns (xy, yaw) or None."""
        cands = []
        drop = dict(grid=0, front=0, bad=0, yaw=0)
        ce = w.counter
        if ce is not None:
            a_t, _ = ce.to_local(target[:2])
            raw = []
            for d in P.STAND_DISTS:
                for da in np.arange(-P.STAND_SPAN_M, P.STAND_SPAN_M + 1e-9, P.STAND_STEP_M):
                    for dyaw in P.STAND_YAWS:
                        xy = ce.from_local(a_t + da, 0.5 * ce.depth + d)
                        dist = float(np.linalg.norm(xy - target[:2]))
                        if dist > P.STAND_MAX_DIST_M:             # at the edge of the arm's workspace the solution exists but the arm cannot get there
                            continue
                        raw.append((dist + 0.5 * abs(dyaw), xy, nav_grid.wrap(ce.yaw_facing + dyaw)))
        else:
            raw = [(float(np.linalg.norm(xy - target[:2])), xy, yaw) for xy, yaw in
                   G.base_candidates(target[:2], P.RINGS, np.arange(P.N_BEARINGS) * 2 * math.pi / P.N_BEARINGS)]
        for score, xy, yaw in raw:
            if max_ring is not None and ce is None and float(np.linalg.norm(xy - target[:2])) > max_ring:
                continue
            if not self.free_pose(xy):
                drop['grid'] += 1
                continue
            if not self.in_front(w, xy):
                drop['front'] += 1
                continue
            if any(float(np.linalg.norm(xy - bad)) < P.BAD_STAND_M for bad in self.bad_stands):
                drop['bad'] += 1
                continue
            if yaw_near is not None and abs(nav_grid.wrap(float(yaw) - yaw_near)) > P.KEEP_YAW_RAD:
                drop['yaw'] += 1
                continue
            cands.append((score, xy, float(yaw)))
        cands.sort(key=lambda c: c[0])
        self.pick_stats = dict(cands=len(cands), drop=drop)
        if not cands:
            return None
        cands = cands[:P.STAND_MAX_SCAN]
        good, qs = self.reach_mask(side, goals_fn(), np.array([[xy[0], xy[1], yaw] for _, xy, yaw in cands]), w, with_q=True)
        routes = []
        for i, (score, xy, yaw) in enumerate(cands):
            if not good[i]:
                continue
            length = self.route_len(w, xy)
            if not math.isfinite(length):
                continue
            routes.append((score, length, xy, yaw, i))
            if len(routes) >= 6:
                break
        self.pick_stats = dict(cands=len(cands), reach=int(good.sum()), routed=len(routes), drop=drop, ik=getattr(self, 'reach_stats', None))
        self.last_cands = [(xy, bool(good[i])) for i, (_, xy, _) in enumerate(cands)]
        if not routes and good.any():
            dbg = []
            for i in np.where(good)[0][:5]:
                xy = cands[i][1]
                p = self.inflated.path(w.base[:2], xy)
                dbg.append(dict(xy=np.round(xy, 2).tolist(), blocked=bool(self.inflated.blocked[self.inflated.cell(xy)]), path=None if p is None else round(nav_grid.path_length(p), 2),
                                start_blocked=bool(self.inflated.blocked[self.inflated.cell(w.base[:2])]), clearance=round(self.raw_grid.clearance(w.base[:2]), 2)))
            self.pick_stats['route_debug'] = dbg
        if not routes:
            return None
        routes.sort(key=lambda r: (r[0], r[1]))
        self.stand_q = np.asarray(qs[:, routes[0][4]], float)
        if ce is not None:
            sxy, syaw = routes[0][2], routes[0][3]
            a_s, b_s = ce.to_local(sxy)
            a_t, _ = ce.to_local(target[:2])
            self.stand_rel = (float(a_s - a_t), float(b_s - 0.5 * ce.depth), float(nav_grid.wrap(syaw - ce.yaw_facing)))
            self.stand_target = np.asarray(target[:2], float).copy()         # the joint solutions of every goal from the chosen stand (trunk 4 + arm 7)
        return routes[0][2], routes[0][3]

    def in_front(self, w: World, xy: np.ndarray) -> bool:
        """A stand lies on the side of the counter where the robot saw it (not behind it)."""
        c = w.counter
        if c is None:
            return True
        a, b = c.to_local(xy)
        return b > 0.5 * c.depth + P.FRONT_MARGIN_M

    def grasp_failed(self, w: World, reason: str) -> None:
        from .pick import grasp_failed
        grasp_failed(self, w, reason)

    # ---- driving --------------------------------------------------------------------------------------------------------------------
    def drive(self, action: np.ndarray, q: np.ndarray, w: World, xy: np.ndarray, yaw: float, hold: bool = False, vmax: Optional[float] = None
              ) -> Tuple[float, float]:
        if hold and self.hold_q is not None:
            self.hold_arm(action, q, self.side)
            self.rest_one(action, q, 'left' if self.side == 'right' else 'right')
        else:
            self.to_rest(action, q)
        v = vmax if vmax is not None else (P.NAV_VNEAR if hold else P.NAV_VFAR)
        return self.navigate(action, q, w, xy, yaw, rest=False, vmax=v, wmax=P.NAV_WMAX)

    def arrived_at(self, dist: float, yaw_err: float) -> bool:
        return dist < P.NAV_POS_TOL and abs(yaw_err) < P.NAV_YAW_TOL
