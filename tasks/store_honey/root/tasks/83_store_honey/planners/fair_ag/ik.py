"""GPU IK of the task 83 fair planner (skills/gpu-ik through the worker planners/ik_worker.py); the same client as oracle_ag/ik.py.

  scan(side, T, stands, base_z)          which base poses (N, 3 = x, y, yaw) reach their world finger-midpoint goal T (N, 4, 4): mask (N,),
                                         joints (N, 11 = trunk 4 + arm 7). Stand choice, branch-free (reachable_stands).
  near(side, T, stands, base_z, q_seed)  solve_near on the same batch, every problem seeded with q_seed (N, 11): nearest branch.
  IKServo                                ArmServo whose joint-target solve is solve_near seeded with the current joints. A failed solve keeps
                                         the previous joint command (the next control solves again); the resolved-rate fallback of ArmServo
                                         runs only with BIMBO_IK_REQUIRE_GPU=0. plan=False steps stay resolved-rate tracking (Jacobian, no solve).

The worker runs in the cuRobo venv named by BIMBO_IK_PYTHON (the simulator venv has no cuRobo); it is started by start() at planner reset,
so its warm-up (about 35 s) overlaps the episode load. The log gets one line `IK_BACKEND gpu` when the worker is ready, or `IK_BACKEND cpu
<reason>`. With BIMBO_IK_REQUIRE_GPU=1 (the default) a missing or failed worker raises GpuIkRequired; only with 0 do the scans fall back to
the CPU solver of skills/gpu-ik (backend 'cpu', the CPU_STANDS cheapest stands) and the servo to the r1pro_kinematics LM; STATS says which.
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
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.skills_lib import r1pro_kinematics as K

# the repo: the first PYTHONPATH entry (run_job.sh and the evolution worker job set it), so a candidate copy of this package
# outside the repo (evo worker: <queue>/work/<id>/planner) still finds the worker and skills/gpu-ik; as task 98
ROOT = Path((os.environ.get('PYTHONPATH') or '').split(os.pathsep)[0] or Path(__file__).resolve().parents[4])
WORKER = ROOT / 'tasks' / '83_store_honey' / 'planners' / 'ik_worker.py'
READY_TIMEOUT_S, REQUEST_TIMEOUT_S = 900.0, 180.0
TOL = (0.010, 0.12)
CPU_STANDS = 24
LOCK_TOL = 0.02                  # rad: a solution counts as trunk-locked when every trunk joint stays this close to the locked value
POINT = np.asarray(K.FINGER_MID_POINT, float)
REQUIRE_GPU = os.environ.get('BIMBO_IK_REQUIRE_GPU', '1') == '1'
STATS: Dict[str, Any] = dict(backend='cpu', warm_s=0.0, scans=0, scan_stands=0, scan_s=0.0, near=0, near_fail=0, near_s=0.0, lock_reject=0,
                             hold_on_fail=0, cpu_near=0, rr_fallback=0, fallback='')
_W: Dict[str, Any] = {}


class _Worker:
    def __init__(self, python: str) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix='ik83_'))
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


class GpuIkRequired(RuntimeError):
    """No GPU IK while BIMBO_IK_REQUIRE_GPU=1: the episode runner ends the episode with cause infra."""


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


def _cpu(why: str) -> None:
    STATS['backend'], STATS['fallback'] = 'cpu', why[:300]
    print(f'IK_BACKEND cpu {why[:300]}', flush=True)
    if REQUIRE_GPU:
        raise GpuIkRequired(f'GPU IK required (BIMBO_IK_REQUIRE_GPU=1): {why[:300]}')


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


def _batch(op: str, side: str, T: np.ndarray, stands: np.ndarray, base_z: float, q_seed: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(stands)
    req = dict(op=op, side=side, T=T.reshape(n, 16).tolist(), stands=stands.tolist(), base_z=float(base_z), point=POINT.tolist())
    if q_seed is not None:
        req['q_seed'] = np.broadcast_to(np.asarray(q_seed, float), (n, 11)).tolist()
    out = _W['w'].call(req)
    return np.asarray(out['mask'], bool).reshape(n), np.asarray(out['q'], float).reshape(n, 11)


def scan(side: str, T: np.ndarray, stands: np.ndarray, base_z: float) -> Tuple[np.ndarray, np.ndarray]:
    """mask (N,) and q (N, 11) of the world goal T[i] (finger-midpoint frame) from stands[i]. The CPU fallback (BIMBO_IK_REQUIRE_GPU=0 only)
    solves the first CPU_STANDS stands (callers sort the stands cheapest first); the others stay unreachable."""
    start()
    stands = np.atleast_2d(np.asarray(stands, float)).reshape(-1, 3)
    n = len(stands)
    T = np.broadcast_to(np.asarray(T, float), (n, 4, 4))
    STATS['scans'] += 1
    STATS['scan_stands'] += n
    if n == 0:
        return np.zeros(0, bool), np.zeros((0, 11))
    t = time.time()
    if _W.get('w') is not None:
        try:
            out = _batch('scan', side, T, stands, base_z, None)
            STATS['scan_s'] += time.time() - t
            return out
        except Exception as exc:  # noqa: BLE001
            _drop(f'scan failed: {exc!r}')
    g = _gpu_ik_module()
    mask, q = np.zeros(n, bool), np.zeros((n, 11))
    for i in range(min(n, CPU_STANDS)):
        mk, qk = g.reachable_stands(T[i], stands[i:i + 1], side, tol=TOL, point=POINT, base_z=base_z, backend='cpu')
        mask[i], q[i] = mk[0], qk[0]
    STATS['scan_s'] += time.time() - t
    return mask, q


def near(side: str, T: np.ndarray, stands: np.ndarray, base_z: float, q_seed: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """solve_near on the worker: mask (N,) and q (N, 11) nearest to q_seed; None when there is no worker (CPU mode)."""
    if _W.get('w') is None:
        return None
    stands = np.atleast_2d(np.asarray(stands, float)).reshape(-1, 3)
    T = np.broadcast_to(np.asarray(T, float), (len(stands), 4, 4))
    t = time.time()
    try:
        mask, q = _batch('near', side, T, stands, base_z, q_seed)
    except Exception as exc:  # noqa: BLE001
        _drop(f'near failed: {exc!r}')
        return None
    STATS['near'] += 1
    STATS['near_s'] += time.time() - t
    STATS['near_fail'] += int((~mask).sum())
    return mask, q


def pose_T(point: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rotation, point
    return T


class IKServo(K.ArmServo):
    """Servo of the finger midpoint whose joint target comes from solve_near on the GPU worker (seeded with the current trunk + arm joints).
    With the trunk locked (narrowed trunk limits) a solution that moves the trunk more than LOCK_TOL is rejected."""

    locked = False

    def solve(self, q, goal, rotation, pos_tol: float = .015, ori_tol: float = .12, ori_weight: float = 1.0):
        if _W.get('w') is None:
            STATS['cpu_near'] += 1
            return super().solve(q, goal, rotation, pos_tol, ori_tol, ori_weight)
        seed = np.asarray(q, float)[list(self.dofs)]
        out = near(self.side, pose_T(goal, rotation), np.zeros((1, 3)), 0.0, seed)
        if out is None:                              # the worker died during the call (only reached with BIMBO_IK_REQUIRE_GPU=0)
            return super().solve(q, goal, rotation, pos_tol, ori_tol, ori_weight)
        mask, x = out
        if not mask[0]:
            return None
        x = x[0]
        if self.locked and np.max(np.abs(x[:4] - seed[:4])) > LOCK_TOL:
            STATS['lock_reject'] += 1
            return None
        lo, hi = self.limits
        return np.clip(x, lo + 1e-5, hi - 1e-5)

    def step(self, action, q, goal, rotation, plan: bool = False):
        """ArmServo.step; with BIMBO_IK_REQUIRE_GPU=1 a plan step whose solve fails leaves the previous joint command in `action`
        (no resolved-rate fallback) and solves again on the next control."""
        if plan and self.target is None:
            self.target = self.solve(q, goal, rotation)
            self.plan_failed = self.target is None
            if self.target is None:
                if REQUIRE_GPU:
                    STATS['hold_on_fail'] += 1
                    return self.errors(q, goal, rotation)
                STATS['rr_fallback'] += 1
        return super().step(action, q, goal, rotation, False)


def make_servo(side: str, q: Optional[np.ndarray] = None, lock_trunk: bool = False, max_step: float = 0.016) -> IKServo:
    """Servo of the finger midpoint of `side`. lock_trunk keeps the trunk at its value in q (the other arm holds something that the trunk carries)."""
    servo = IKServo(side, max_step=max_step)
    servo.use_fingers_mid()
    if lock_trunk and q is not None:
        servo.limits = servo.limits.copy()
        t = np.asarray(q, float)[list(K.TRUNK_DOFS)]
        servo.limits[0, :4], servo.limits[1, :4] = t - 1e-4, t + 1e-4
        servo.locked = True
    return servo


def stats() -> Dict[str, Any]:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in STATS.items()}
