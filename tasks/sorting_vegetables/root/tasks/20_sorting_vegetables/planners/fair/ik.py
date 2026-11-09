"""GPU IK of the task 20 fair planner (skills/gpu-ik through the worker planners/ik_worker.py); kinematics only, no simulator state.

  scan(side, goals, stands, base_z, point)   which base poses (N, 3 = x, y, yaw) reach every world gripper-link goal (K, 4, 4): masks (K, N),
                                             joint solutions (K, N, 11 = trunk 4 + arm 7). Stand choice, branch-free (reachable_stands).
  IKServo                                    ArmServo whose joint-target solve is solve_near (seeded with the current joints, nearest branch).

The worker runs in the cuRobo venv named by BIMBO_IK_PYTHON (the simulator venv has no cuRobo); it is started by start() at planner reset,
so its warm-up (about 35 s) overlaps the episode load. The log gets one line `IK_BACKEND gpu` when the worker is ready, or `IK_BACKEND cpu
<reason>`. With BIMBO_IK_REQUIRE_GPU=1 (the default) a missing or failed worker raises; only with 0 do both fall back to the CPU solvers
(skills/gpu-ik backend 'cpu' for scans, limited to the CPU_STANDS cheapest stands; the r1pro_kinematics LM for the servo); STATS says which.
"""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from core.skills_lib import r1pro_kinematics as K

ROOT = Path((os.environ.get('PYTHONPATH') or '').split(os.pathsep)[0] or Path(__file__).resolve().parents[4])
WORKER = ROOT / 'tasks' / '20_sorting_vegetables' / 'planners' / 'ik_worker.py'   # also for a candidate copy of this package elsewhere
READY_TIMEOUT_S, REQUEST_TIMEOUT_S = 900.0, 120.0
TOL = (0.010, 0.10)
CPU_STANDS = 12
REQUIRE_GPU = os.environ.get('BIMBO_IK_REQUIRE_GPU', '1') == '1'
STATS: Dict[str, Any] = dict(backend='cpu', warm_s=0.0, scans=0, scan_goals=0, scan_stands=0, scan_s=0.0, near=0, near_fail=0, near_s=0.0,
                             cpu_near=0, fallback='')
import types
_W: Dict[str, Any] = sys.modules.setdefault('bimbo_20_fair_ik_state', types.ModuleType('bimbo_20_fair_ik_state')).__dict__.setdefault('W', {})   # kept when the package is reloaded in one process


class _Worker:
    def __init__(self, python: str) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix='ik20f_'))
        self.log = open(self.dir / 'worker.log', 'w')
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        self.proc = subprocess.Popen([python, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, env=env)
        self.t0, self.ready = time.time(), False

    def _json(self, timeout: float) -> Dict[str, Any]:
        end = time.time() + timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                raise RuntimeError(f'IK worker exited with {self.proc.returncode}, log {self.dir / "worker.log"}')
            if select.select([self.proc.stdout], [], [], 1.0)[0]:
                line = self.proc.stdout.readline()
                if line == '':
                    raise RuntimeError('IK worker closed its output')
                try:
                    out = json.loads(line)
                except ValueError:
                    continue          # banner lines of warp and cuRobo
                if isinstance(out, dict):
                    return out
        raise TimeoutError('IK worker did not answer')

    def call(self, req: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ready:
            self._json(READY_TIMEOUT_S)
            self.ready = True
            STATS['warm_s'] = round(time.time() - self.t0, 1)
            print(f"IK_BACKEND gpu (cuRobo worker ready in {STATS['warm_s']} s)", flush=True)
        self.proc.stdin.write(json.dumps(req) + '\n')
        self.proc.stdin.flush()
        out = self._json(REQUEST_TIMEOUT_S)
        if 'error' in out:
            raise RuntimeError(out['error'])
        return out


def start() -> None:
    """Start the worker once per process (idempotent). Leaves STATS['backend'] = 'cpu' with the reason when there is none."""
    if 'w' in _W:
        return
    _W['w'] = None
    py = os.environ.get('BIMBO_IK_PYTHON', '')
    if not py or not Path(py).exists():
        _cpu(f'no cuRobo python (BIMBO_IK_PYTHON={py!r})')
        return
    try:
        _W['w'] = _Worker(py)
        STATS['backend'] = 'gpu-worker'
    except Exception as exc:  # noqa: BLE001
        _cpu(f'worker start failed: {exc!r}')


class GpuIkRequired(RuntimeError):
    """No GPU IK while BIMBO_IK_REQUIRE_GPU=1: the episode runner ends the episode with cause infra."""


def _cpu(why: str) -> None:
    STATS['backend'], STATS['fallback'] = 'cpu', why[:300]
    print(f'IK_BACKEND cpu {why[:300]}', flush=True)
    if REQUIRE_GPU:
        raise GpuIkRequired(f'GPU IK required (BIMBO_IK_REQUIRE_GPU=1): {why[:300]}')


def _require() -> None:
    """Every IK call after a lost worker raises again (not only the first): a runner that survives the first exception must not go on with CPU IK."""
    if REQUIRE_GPU and _W.get('w') is None:
        raise GpuIkRequired(f"GPU IK required (BIMBO_IK_REQUIRE_GPU=1): {STATS['fallback'] or 'no worker'}")


def _drop(why: str) -> None:
    w = _W.get('w')
    if w is not None and w.proc.poll() is None:
        w.proc.kill()
    _W['w'] = None
    _cpu(why)


def gpu_on() -> bool:
    start()
    return _W.get('w') is not None


def _gpu_ik_module():
    path = str(ROOT / 'skills' / 'gpu-ik')
    if path not in sys.path:
        sys.path.insert(0, path)
    import gpu_ik
    return gpu_ik


def scan(side: str, goals: Sequence[np.ndarray], stands: np.ndarray, base_z: float, point: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """masks (K, N) and q (K, N, 11) of every goal from every stand. The CPU fallback solves only the first CPU_STANDS stands (callers sort
    the stands cheapest first); the others stay unreachable."""
    start()
    goals = np.asarray(goals, float).reshape(-1, 4, 4)
    stands = np.atleast_2d(np.asarray(stands, float)).reshape(-1, 3)
    k, n = len(goals), len(stands)
    STATS['scans'] += 1
    STATS['scan_goals'] += k
    STATS['scan_stands'] += n
    t = time.time()
    if n == 0:
        return np.zeros((k, 0), bool), np.zeros((k, 0, 11))
    w = _W.get('w')
    if w is not None:
        try:
            out = w.call(dict(op='scan', side=side, T=goals.reshape(k, 16).tolist(), stands=stands.tolist(), base_z=float(base_z),
                              point=list(map(float, point))))
            STATS['scan_s'] += time.time() - t
            return np.asarray(out['mask'], bool).reshape(k, n), np.asarray(out['q'], float).reshape(k, n, 11)
        except Exception as exc:  # noqa: BLE001
            _drop(f'scan failed, CPU from here: {exc!r}')
    _require()
    g = _gpu_ik_module()
    masks, qs = np.zeros((k, n), bool), np.zeros((k, n, 11))
    m = min(n, CPU_STANDS)
    for i, T in enumerate(goals):
        mk, qk = g.reachable_stands(T, stands[:m], side, tol=TOL, point=point, base_z=base_z, backend='cpu')
        masks[i, :m], qs[i, :m] = mk, qk
    STATS['scan_s'] += time.time() - t
    return masks, qs


def near(side: str, T_robot: np.ndarray, q_seed: np.ndarray, point: Sequence[float]) -> Optional[np.ndarray]:
    """solve_near on the worker: trunk + arm joints (11) reaching the robot-frame gripper-link goal, nearest to q_seed; None when no solve."""
    w = _W.get('w')
    if w is None:
        _require()
        return None
    t = time.time()
    try:
        out = w.call(dict(op='near', side=side, T=np.asarray(T_robot, float).ravel().tolist(), q_seed=np.asarray(q_seed, float).tolist(),
                          point=list(map(float, point))))
    except Exception as exc:  # noqa: BLE001
        _drop(f'near failed, CPU from here: {exc!r}')
        return None
    STATS['near'] += 1
    STATS['near_s'] += time.time() - t
    if not out['mask']:
        STATS['near_fail'] += 1
        return None
    return np.asarray(out['q'], float)


def link_point_in_gripper(side: str, q: np.ndarray, link: str, point: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """(point in gripper-link coordinates, rotation of `link` in the gripper-link frame) at the joints q (the finger links slide)."""
    poses = K.kinematics().forward(q)[0]
    rel = np.linalg.inv(poses[K.GRIPPER_LINK[side]]) @ poses[link]
    return rel[:3, :3] @ np.asarray(point, float) + rel[:3, 3], rel[:3, :3]


def gripper_goal(rotation_link: np.ndarray, rel_rot: np.ndarray) -> np.ndarray:
    """Gripper-link rotation that puts the servoed link at `rotation_link` (the link is rigid on the gripper apart from its slide)."""
    return np.asarray(rotation_link, float) @ rel_rot.T


class IKServo(K.ArmServo):
    """ArmServo whose joint-target solve is solve_near on the GPU worker (same signature, None on failure); the CPU LM when there is no worker.
    sync=True walks to a joint target with all joints arriving together (the plain servo clips each joint, so the small joints finish
    first and the hand leaves the straight joint path). The walk moves the command, not the measured joints: a command one step ahead of
    the measured joints caps the PD error at max_step, and a loaded trunk joint (torso leaning over the floor) then sags and never arrives."""

    sync = False
    cmd: Optional[np.ndarray] = None    # the walked command; None starts the next walk from the measured joints

    def step(self, action, q, goal, rotation, plan: bool = False):
        if not self.sync or self.target is None:
            return super().step(action, q, goal, rotation, plan)
        qa = np.asarray(q, float)[self.dofs]
        if self.cmd is None:
            self.cmd = qa.copy()
        delta = self.target - self.cmd
        scale = min(1., self.max_step / max(float(np.max(np.abs(delta))), 1e-9))
        lo, hi = self.limits
        self.cmd = command = np.clip(self.cmd + delta * scale, lo + 1e-5, hi - 1e-5)
        if np.max(np.abs(self.target - qa)) < .035:
            self.target, self.cmd = None, None
        action[K.TRUNK_ACTION] = command[:4]
        action[K.ARM_ACTION[self.side]] = command[4:]
        return self.errors(q, goal, rotation)

    def solve(self, q, goal, rotation, pos_tol: float = .015, ori_tol: float = .12, ori_weight: float = 1.0):
        if _W.get('w') is None:
            _require()
            STATS['cpu_near'] += 1
            return super().solve(q, goal, rotation, pos_tol, ori_tol, ori_weight)
        point, rel = link_point_in_gripper(self.side, q, self.link, self.point)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = gripper_goal(rotation, rel), goal
        out = near(self.side, T, np.asarray(q, float)[list(self.dofs)], point)
        if out is None and _W.get('w') is None:     # the worker died during the call
            return super().solve(q, goal, rotation, pos_tol, ori_tol, ori_weight)
        return out


def make_servo(side: str, mode: str = 'tip', sync: bool = False) -> IKServo:
    servo = IKServo(side)
    servo.sync = sync
    {'tip': servo.use_tip, 'mid': servo.use_fingers_mid, 'palm': servo.use_palm}[mode]()
    return servo


def stats() -> Dict[str, Any]:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in STATS.items()}
