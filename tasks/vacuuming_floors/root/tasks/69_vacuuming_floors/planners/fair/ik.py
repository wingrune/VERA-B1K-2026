"""GPU IK of the task 69 fair planner (skills/gpu-ik through the worker planners/ik_worker.py); kinematics only, no simulator state. Frozen.

The planner's world is the odometry frame (the base at the first control, z from the robot frame origin), so a goal is a 4x4 pose of the
finger midpoint (gripper rotation) in that frame and `base` the estimated base pose (x, y, yaw).

  scan(side, goals, stands)                  which base poses (N, 3 = x, y, yaw) reach every goal (K, 4, 4): masks (K, N), joint solutions
                                             (K, N, 11 = trunk 4 + arm 7). Stand choice, branch-free (reachable_stands).
  hand_near(side, T, base, q_seed)           solve_near for the hand: trunk + arm joints (11) seeded with the current command (nearest branch)
  arm_near(side, T, base, trunk, q_seed)     the 7 arm joints only, the trunk held (the free hand's press must not move the shared trunk);
                                             cuRobo model of the arm from its mount link, re-checked in float64 by the CPU forward kinematics

One worker process in the cuRobo venv named by BIMBO_IK_PYTHON (the simulation venv has no cuRobo), started by start() at planner reset, so
its warm-up overlaps the episode load. The log gets one line `IK_BACKEND gpu` when the worker is ready. There is no CPU path: without the
worker or when it dies the call raises GpuIkRequired and the episode ends with cause infra.
"""
from __future__ import annotations

import json
import os
import select
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from core.skills_lib import r1pro_kinematics as K

ROOT = Path((os.environ.get('PYTHONPATH') or '').split(os.pathsep)[0] or Path(__file__).resolve().parents[4])
WORKER = ROOT / 'tasks' / '69_vacuuming_floors' / 'planners' / 'ik_worker.py'   # also for a candidate copy of this package elsewhere
READY_TIMEOUT_S, REQUEST_TIMEOUT_S = 900.0, 120.0
SCAN_TOL = (0.010, 0.10)    # metres, radians of the stand scans (a stand within a few centimetres of the scanned one still reaches)
TOL = (0.005, 0.05)         # the near solves (the joint targets the arms are driven to)
ARM_TOL = (0.010, 0.12)     # float64 re-check of the arm-only solve
LOOSE_TOL = (0.020, 0.35)   # the same for the loose solver (the pushes of the press: the closed fingers push a face, the orientation is not critical)
POINT = K.FINGER_MID_POINT
STATS: Dict[str, Any] = dict(backend='none', ready=False, warm_s=0.0, scans=0, scan_goals=0, scan_stands=0, scan_s=0.0, near=0, near_fail=0, near_s=0.0,
                             arm_near=0, arm_fail=0, arm_s=0.0)
_W: Dict[str, Any] = {}


class GpuIkRequired(RuntimeError):
    """No GPU IK (the worker is missing or died).

    No GPU IK (the worker is missing or died). The episode runner ends the episode with cause infra."""


class _Worker:
    def __init__(self, python: str) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix='ik69f_'))
        self.log = open(self.dir / 'worker.log', 'w')
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH')}
        self.proc = subprocess.Popen([python, '-u', str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                                     env=env, start_new_session=True)
        self.t0, self.ready = time.time(), False

    def _json(self, timeout: float) -> Dict[str, Any]:
        end = time.time() + timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                raise GpuIkRequired(f'IK worker exited with {self.proc.returncode}, log {self.dir / "worker.log"}')
            if select.select([self.proc.stdout], [], [], 1.0)[0]:
                line = self.proc.stdout.readline()
                if line == '':
                    raise GpuIkRequired('IK worker closed its output')
                try:
                    out = json.loads(line)
                except ValueError:
                    continue          # banner lines of warp and cuRobo
                if isinstance(out, dict):
                    return out
        raise GpuIkRequired('IK worker did not answer')

    def call(self, req: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ready:
            self._json(READY_TIMEOUT_S)
            self.ready = True
            STATS['warm_s'] = round(time.time() - self.t0, 1)
            STATS['backend'], STATS['ready'] = 'gpu', True
            print(f"IK_BACKEND gpu (cuRobo worker ready in {STATS['warm_s']} s)", flush=True)
        self.proc.stdin.write(json.dumps(req) + '\n')
        self.proc.stdin.flush()
        out = self._json(REQUEST_TIMEOUT_S)
        if 'error' in out:
            raise GpuIkRequired('IK worker: ' + str(out['error'])[-300:])
        return out


def start() -> None:
    """Start the GPU-IK worker once per process.

    Start the GPU-IK worker once per process (idempotent). No worker python: IK_BACKEND cpu is logged and GpuIkRequired raised."""
    if 'w' in _W:
        return
    py = os.environ.get('BIMBO_IK_PYTHON', '')
    if not py or not Path(py).exists():
        STATS['backend'] = 'cpu'
        print(f'IK_BACKEND cpu no cuRobo python (BIMBO_IK_PYTHON={py!r})', flush=True)
        raise GpuIkRequired(f'GPU IK required: BIMBO_IK_PYTHON={py!r} is not a cuRobo python')
    _W['w'] = _Worker(py)
    STATS['backend'] = 'gpu-worker'


def _worker() -> _Worker:
    start()
    return _W['w']


def scan(side, goals, stands, base_z=0.0) -> Tuple[np.ndarray, np.ndarray]:
    """masks (K, N) and q (K, N, 11) of every goal.

    masks (K, N) and q (K, N, 11) of every goal (finger-midpoint pose in the odometry frame) from every stand."""
    goals = np.asarray(goals, float).reshape(-1, 4, 4)
    stands = np.atleast_2d(np.asarray(stands, float)).reshape(-1, 3)
    k, n = len(goals), len(stands)
    STATS['scans'] += 1
    STATS['scan_goals'] += k
    STATS['scan_stands'] += n
    if n == 0 or k == 0:
        return np.zeros((k, n), bool), np.zeros((k, n, 11))
    t = time.time()
    out = _worker().call(dict(op='scan', side=side, T=goals.reshape(k, 16).tolist(), stands=stands.tolist(), base_z=float(base_z),
                              point=list(map(float, POINT)), tol=list(SCAN_TOL)))
    STATS['scan_s'] += time.time() - t
    return np.asarray(out['mask'], bool).reshape(k, n), np.asarray(out['q'], float).reshape(k, n, 11)


def to_robot(base: np.ndarray, T: np.ndarray) -> np.ndarray:
    c, s = np.cos(base[2]), np.sin(base[2])
    inv = np.eye(4)
    inv[:3, :3] = [[c, s, 0.], [-s, c, 0.], [0., 0., 1.]]
    inv[:3, 3] = -inv[:3, :3] @ np.array([base[0], base[1], 0.])
    return inv @ np.asarray(T, float)


def near(side, T_robot, q_seed, point=POINT) -> Optional[np.ndarray]:
    """solve_near seeded with q_seed.

    solve_near seeded with q_seed: trunk + arm joints (11) putting `point` of the gripper link on the robot-frame goal; None when no solve."""
    t = time.time()
    out = _worker().call(dict(op='near', side=side, T=np.asarray(T_robot, float).ravel().tolist(), q_seed=np.asarray(q_seed, float).tolist(),
                              point=list(map(float, point)), tol=list(TOL)))
    STATS['near'] += 1
    STATS['near_s'] += time.time() - t
    if not out['mask']:
        STATS['near_fail'] += 1
        return None
    return np.asarray(out['q'], float)


def hand_near(side, T_world, base, q_seed) -> Optional[np.ndarray]:
    """Joints (trunk 4 + arm 7) that put the finger midpoint on T_world.

    Joints (trunk 4 + arm 7) that put the finger midpoint on T_world (odometry frame), seeded with the current command."""
    return near(side, to_robot(base, T_world), q_seed)


def arm_near(side, T_world, base, trunk, q_seed, loose=False) -> Optional[np.ndarray]:
    """7 arm joints that put the finger midpoint on T_world...

    7 arm joints that put the finger midpoint on T_world with the trunk held at `trunk` (cuRobo on the arm's mount link, seeded with
    q_seed), re-checked in float64 with the CPU forward kinematics (ARM_TOL); None when no solve."""
    t = time.time()
    goal = np.asarray(T_world, float).copy()
    goal[:3, 3] = goal[:3, 3] - goal[:3, :3] @ np.asarray(POINT, float)        # finger midpoint -> gripper link origin
    goal_robot = to_robot(base, goal)
    qq = np.zeros(22)
    qq[list(K.TRUNK_DOFS)] = trunk
    mount = K.kinematics().forward(qq)[0][f'{side}_arm_base_link']
    out = _worker().call(dict(op='arm_near_loose' if loose else 'arm_near', side=side, T=(np.linalg.inv(mount) @ goal_robot).ravel().tolist(),
                              q_seed=np.asarray(q_seed, float).tolist()))
    STATS['arm_near'] += 1
    STATS['arm_s'] += time.time() - t
    if not out['mask']:
        STATS['arm_fail'] += 1
        return None
    q7 = np.asarray(out['q'], float)
    qq[list(K.ARM_DOFS[side])] = q7
    got = K.kinematics().forward(qq)[0][K.GRIPPER_LINK[side]]
    tol = LOOSE_TOL if loose else ARM_TOL
    if (np.linalg.norm(got[:3, 3] - goal_robot[:3, 3]) > tol[0]
            or Rot.from_matrix(got[:3, :3] @ goal_robot[:3, :3].T).magnitude() > tol[1]):
        STATS['arm_fail'] += 1
        return None
    return q7


def hand_fk(side, trunk, arm, base) -> np.ndarray:
    """Finger-midpoint pose (4x4.

    Finger-midpoint pose (4x4, gripper rotation) in the odometry frame at the joints (forward kinematics only)."""
    qq = np.zeros(22)
    qq[list(K.TRUNK_DOFS)] = trunk
    qq[list(K.ARM_DOFS[side])] = arm
    t = K.kinematics().forward(qq)[0][K.GRIPPER_LINK[side]]
    c, s = np.cos(base[2]), np.sin(base[2])
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ t[:3, :3]
    mid = t[:3, 3] + t[:3, :3] @ np.asarray(POINT, float)
    T[:3, 3] = [base[0] + c * mid[0] - s * mid[1], base[1] + s * mid[0] + c * mid[1], mid[2]]
    return T


def stats() -> Dict[str, Any]:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in STATS.items()}
