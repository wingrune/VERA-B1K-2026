"""Shared machinery of the phases of the fair planner of task 69: plan state, transitions, joint walks of the arms, the travel posture (the
navigation part is phases/nav.py). IK on the GPU only: cuRobo reachable_stands as a stand mask, solve_near for every joint target (seeded with
the current command), the arm-only solve for the free hand. Everything works on the World (estimates in the odometry frame); nothing reads
the simulator."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.skills_lib import nav_grid
from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import ik
from .. import params as P
from ..perception import Perception
from ..world import World
from .nav import NavMixin


class Walker:
    """Joint walk of the trunk + one arm.

    Joint walk of the trunk + one arm (11) or of both arms (18) to a target, all joints arriving together, then `settle` controls of hold."""

    def __init__(self, start: np.ndarray, target: np.ndarray, step: float = P.JOINT_STEP, settle: int = P.WALK_SETTLE) -> None:
        self.cmd, self.target, self.step, self.settle = np.asarray(start, float).copy(), np.asarray(target, float), step, settle
        self.hold = 0

    def next(self) -> Tuple[np.ndarray, bool]:
        """The next joint command; done once the command has...

        The next joint command; done once the command has reached the target and then held `settle` more controls."""
        delta = self.target - self.cmd
        m = float(np.max(np.abs(delta)))
        if m > 1e-9:
            self.cmd = self.cmd + delta * min(1., self.step / m)
        if float(np.max(np.abs(self.target - self.cmd))) < 1e-6:
            self.hold += 1
        return self.cmd.copy(), self.hold > self.settle

    @property
    def gap(self) -> float:
        return float(np.max(np.abs(self.target - self.cmd)))


class Core(NavMixin):
    """State and helpers shared by the phases.

    State and helpers shared by the phases. A phase step is called as step(core, action, q, world)."""

    def __init__(self) -> None:
        self.perc = Perception()
        self.reset_state()

    def reset_state(self) -> None:
        self.perc.reset()
        self.phase, self.reason, self.age, self.step = 'settle', 'start', 0, 0
        self.counts: Dict[str, int] = dict(replans=0, nav_stuck=0, grasp_tries=0, grasp_fail=0, drops=0, scans=0, plans=0, ik_fail=0, targets=0,
                                           retargets=0, height_fixes=0, hard=0, surveys=0, press_tries=0, recoveries=0, stalls=0, regrasps=0, tilt_regrasps=0)
        self.cmd: Optional[np.ndarray] = None
        self.side = 'left'
        self.arm_cmd: Optional[np.ndarray] = None          # trunk 4 + working arm 7 joint command
        self.other_cmd: Optional[np.ndarray] = None        # the free arm 7 joint command
        self.grip = {'left': K.GRIPPER_CLOSED, 'right': K.GRIPPER_CLOSED}
        self.inflated: Optional[nav_grid.Grid] = None
        self.inflated_far: Optional[nav_grid.Grid] = None
        self.inflated_tight: Optional[nav_grid.Grid] = None
        self.raw_grid: Optional[nav_grid.Grid] = None
        self.path: Optional[List[np.ndarray]] = None
        self.path_age = 0
        self.stuck_ref: Optional[Tuple[np.ndarray, int, float]] = None
        self.routes: Dict[tuple, float] = {}
        self.pick_stats: Dict[str, Any] = {}
        self.grasp_cands: List[Dict[str, Any]] = []
        self.grasp: Optional[Dict[str, Any]] = None
        self.walker: Optional[Walker] = None
        self.line: Optional[List[np.ndarray]] = None
        self.line_rot: Optional[np.ndarray] = None
        self.line_i = 0
        self.line_seeds: Optional[list] = None
        self.held_age = 0
        self.start_q: Optional[np.ndarray] = None
        self.hold_arm: Optional[np.ndarray] = None
        self.turned = 0.
        self.pause, self.turn_mark = 0, 0.
        self.wait_fix, self.fix0 = 0, 0
        self.leg: Optional[str] = None
        self.relax = 0
        self.expl: Optional[Dict[str, Any]] = None
        self.explores = 0
        self.explore_resets = 0
        self.nopose = 0
        self.unjam = 0
        self.cand_relax = 0
        self.alarms = 0
        self.fail_count: Dict[tuple, int] = {}
        self.banned: List[Dict[str, Any]] = []
        self.wp_looked = False
        self.press_skips = 0
        self.searched = False
        self.lift_s = 0.
        self.cmd_last: Optional[np.ndarray] = None
        self.hold_high: Optional[np.ndarray] = None
        self.plan_vac: Optional[np.ndarray] = None
        self.wp: Optional[np.ndarray] = None
        self.scan_from: Optional[float] = None
        self.look_t0: Optional[int] = None
        self.err = (float('nan'), float('nan'))
        self.history: List[Tuple[int, str, str]] = []
        self.press_log: List[Dict[str, Any]] = []
        self.look_log: List[Dict[str, Any]] = []
        self.sweep_log: List[Dict[str, Any]] = []
        self.target: Optional[Dict[str, Any]] = None
        self.tries: Dict[int, int] = {}
        self.hard: set = set()
        self.height_tries: Dict[int, int] = {}
        self.bad_yaws: Dict[int, List[float]] = {}
        self.lift_round = 0
        self.toggle_step: Optional[int] = None
        self.grasp_step: Optional[int] = None

    # ---- transitions ----------------------------------------------------------------------------------------------------------------
    def change(self, phase: str, reason: str) -> None:
        self.history.append((self.step, phase, reason[:160]))
        self.phase, self.reason, self.age = phase, reason, 0
        self.path, self.stuck_ref = None, None
        self.perc.map_hold = phase in P.MAP_HOLD_PHASES      # the arms work at the vacuum: its points must not become obstacles

    def fixed_here(self, w: World) -> bool:
        """The marker blob has corrected the standing vacuum's...

        The marker blob has corrected the standing vacuum's pose since the arm started to the pre-grasp point."""
        return self.perc.counts.get('marker_fix_stand', 0) > getattr(self, 'fix0', 0)

    # ---- arms -------------------------------------------------------------------------------------------------------------------------
    @property
    def free_side(self) -> str:
        return 'right' if self.side == 'left' else 'left'

    def dofs(self, side: Optional[str] = None) -> List[int]:
        return list(K.TRUNK_DOFS) + list(K.ARM_DOFS[side or self.side])

    def take_arm(self, q: np.ndarray, side: str) -> None:
        """Start commanding the trunk + `side` arm from the measured joints.

        Start commanding the trunk + `side` arm from the measured joints; the other arm holds its current joints."""
        self.side = side
        self.arm_cmd = q[self.dofs(side)].copy()
        self.other_cmd = q[list(K.ARM_DOFS['right' if side == 'left' else 'left'])].copy()

    def release_arms(self) -> None:
        self.arm_cmd = self.other_cmd = None

    @staticmethod
    def rest_pose(side: str) -> np.ndarray:
        """Trunk 4 + arm 7 of the travel posture.

        Trunk 4 + arm 7 of the travel posture: the rest trunk (the head camera pitched down onto the floor) with the arms hanging."""
        target = K.rest(side).copy()
        target[4:] = 0.
        return target

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

    def hand(self, q: np.ndarray, side: Optional[str] = None) -> np.ndarray:
        """Finger-midpoint pose (4x4) of an arm in the odometry frame.

        Finger-midpoint pose (4x4) of an arm in the odometry frame, from the measured joints."""
        return self.perc.mid_T(q, side or self.side)

    @staticmethod
    def finger_gap(q: np.ndarray, side: str) -> float:
        return float(sum(q[list(K.FINGER_DOFS[side])]))

    def holds(self, q: np.ndarray, side: str) -> bool:
        """The closed fingers stay apart.

        The closed fingers stay apart: something is between them (the assisted grasp holds it). Proprio only."""
        return self.finger_gap(q, side) > P.HELD_GAP_M

    def solve_hand(self, w: World, T: np.ndarray, start: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Joints (trunk 4 + arm 7) of the working arm that put...

        Joints (trunk 4 + arm 7) of the working arm that put the finger midpoint on T (odometry frame), seeded with the command."""
        x = ik.hand_near(self.side, T, w.base, self.arm_cmd if start is None else start)
        if x is None:
            self.counts['ik_fail'] += 1
        return x

    # ---- the held vacuum's transit height --------------------------------------------------------------------------------------------------
    def transit_s(self, dist: float) -> float:
        """0 near the flake (level), 1 farther out than...

        0 near the flake (level), 1 farther out than TRANSIT_NEAR_M + TRANSIT_BLEND_M (raised): a held vacuum dragged on the floor tilts."""
        return 0. if self.hold_high is None else float(np.clip((dist - P.TRANSIT_NEAR_M) / P.TRANSIT_BLEND_M, 0., 1.))

    def transit(self, dist: float) -> np.ndarray:
        """The held arm's joints for the distance left to the flake.

        The held arm's joints for the distance left to the flake: level near it, raised farther out; the raise follows the distance at a
        limited rate (the arm cannot follow a jump of 2 cm, the pull alarm would read it as a drag)."""
        self.lift_s += float(np.clip(self.transit_s(dist) - self.lift_s, -P.TRANSIT_RATE, P.TRANSIT_RATE))
        return self.hold_arm if self.lift_s <= 1e-6 else self.hold_arm + self.lift_s * (self.hold_high - self.hold_arm)

    def hand_pulled(self, q: np.ndarray, w: World) -> float:
        """Distance (m) of the held hand from the pose of the last arm command.

        Distance (m) of the held hand from the pose of the last arm command (forward kinematics of the command against the measured joints):
        a vacuum dragged on the floor or caught on furniture pulls the arm off it."""
        cmd = self.cmd_last if self.cmd_last is not None else self.hold_arm
        want = ik.hand_fk(self.side, cmd[:4], cmd[4:], w.base)[:3, 3]
        return float(np.linalg.norm(want - self.hand(q)[:3, 3]))

    def raise_pose(self, w: World) -> None:
        """Joints of the held vacuum raised TRANSIT_LIFT_M above the levelled one.

        Joints of the held vacuum raised TRANSIT_LIFT_M above the levelled one (solve_near from the levelled joints)."""
        self.hold_high = None
        if P.TRANSIT_LIFT_M <= 0.:
            return
        mid = ik.hand_fk(self.side, self.hold_arm[:4], self.hold_arm[4:], w.base)
        mid[2, 3] += P.TRANSIT_LIFT_M
        x = ik.hand_near(self.side, mid, w.base, self.hold_arm)
        if x is not None and float(np.max(np.abs(x - self.hold_arm))) < 0.3:
            self.hold_high = x

    # ---- lines of finger-midpoint goals, followed by joint walks -------------------------------------------------------------------------
    def start_line(self, p0: np.ndarray, p1: np.ndarray, rot: np.ndarray, step: float = P.LINE_STEP_M, n: Optional[int] = None) -> None:
        n = n or max(1, int(np.ceil(float(np.linalg.norm(p1 - p0)) / step)))
        self.line_seeds = None
        self.line = [p0 + (p1 - p0) * k / n for k in range(1, n + 1)]
        self.line_rot, self.line_i, self.walker = rot, 0, None

    def follow_line(self, w: World) -> Optional[bool]:
        """One control along the line of finger-midpoint goals.

        One control along the line of finger-midpoint goals (solve_near per waypoint, joint walk). True when finished, False on an IK
        failure, None while moving."""
        if self.walker is None:
            seeds = self.line_seeds
            x = self.solve_hand(w, G.pose_T(self.line_rot, self.line[self.line_i]), start=None if not seeds else seeds[min(self.line_i, len(seeds) - 1)])
            if x is None:
                return False
            if float(np.max(np.abs(x - self.arm_cmd))) > P.JUMP_MAX:       # another IK branch: not a step along the line
                self.counts['ik_jump'] = self.counts.get('ik_jump', 0) + 1
                return False
            last = self.line_i == len(self.line) - 1
            self.walker = Walker(self.arm_cmd, x, P.APPROACH_STEP if last else 0.5 * P.JOINT_STEP, P.WALK_SETTLE if last else 0)
        self.arm_cmd, done = self.walker.next()
        if done:
            self.walker = None
            self.line_i += 1
            if self.line_i >= len(self.line):
                return True
        return None
