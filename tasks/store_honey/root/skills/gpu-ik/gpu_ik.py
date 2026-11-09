"""Batched stand-reachability for R1Pro (trunk 4 + one arm 7 DoF): which base poses (x, y, yaw) can reach a world-frame gripper pose.

    from gpu_ik import reachable_stands          # skills/gpu-ik on sys.path
    mask, q = reachable_stands(T_world(4x4), stands(N,3), 'left', tol=(0.005, 0.05))

mask (N,) bool; q (N, 11) = trunk 0-3 + arm joints (same order as ArmServo.dofs), valid where mask. The stand is the base_link pose on the floor
(x, y, yaw, plus optional base_z). The target is the pose of the gripper link, or of a frame at `point` (gripper-link coordinates) with the
gripper orientation. The GPU path (cuRobo) is created once per process (warm-up is tens of seconds); without CUDA/cuRobo it falls back to the
CPU solver of core.skills_lib.r1pro_kinematics with the same signature. Joint limits and kinematics come from that module (JOINTS table).
A process without cuRobo (the simulation venv) gets the GPU through a helper process when BIMBO_CUROBO_PY names a cuRobo venv python:
backend 'gpu' / 'auto' / 'server' then run ik_server.py under it (started on first use or by warmup, one per process, same results).
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from core.skills_lib import r1pro_kinematics as K  # noqa: E402

log = logging.getLogger('gpu_ik')
DEFAULT_TOL = (0.005, 0.05)  # metres, radians (rotation vector norm)
NUM_SEEDS = 32
MAX_BATCH = 1024
_GPU = {}


def stand_to_base_T(stands: np.ndarray, base_z: float = 0.0) -> np.ndarray:
    """(N,3) x, y, yaw -> (N,4,4) base_link pose in the world."""
    s = np.atleast_2d(np.asarray(stands, float))
    T = np.tile(np.eye(4), (len(s), 1, 1))
    c, sn = np.cos(s[:, 2]), np.sin(s[:, 2])
    T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1] = c, -sn, sn, c
    T[:, 0, 3], T[:, 1, 3], T[:, 2, 3] = s[:, 0], s[:, 1], base_z
    return T


def targets_in_base(target_world: np.ndarray, stands: np.ndarray, point: Optional[Sequence[float]] = None, base_z: float = 0.0) -> np.ndarray:
    """Gripper-link goal poses in base_link, one per stand. target_world is (4,4) or (N,4,4). `point` shifts the goal frame to the gripper link."""
    Tb = stand_to_base_T(stands, base_z)
    Tw = np.broadcast_to(np.asarray(target_world, float), (len(Tb), 4, 4)).copy()
    if point is not None:  # goal frame sits at `point` in gripper coordinates: T_link = T_goal * trans(-point)
        Tw[:, :3, 3] -= Tw[:, :3, :3] @ np.asarray(point, float)
    Tinv = np.linalg.inv(Tb)
    return Tinv @ Tw


def pose_error(T_a: np.ndarray, T_b: np.ndarray) -> Tuple[float, float]:
    return float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])), float(np.linalg.norm(R.from_matrix(T_b[:3, :3] @ T_a[:3, :3].T).as_rotvec()))


def fk_gripper(arm: str, q11: np.ndarray) -> np.ndarray:
    q = np.zeros(22)
    q[list(K.TRUNK_DOFS) + list(K.ARM_DOFS[arm])] = q11
    return K.kinematics().forward(q)[0][K.GRIPPER_LINK[arm]]


# ------------------------------------------------------------------ CPU
def reachable_stands_cpu(target_world, stands, arm: str, tol=DEFAULT_TOL, point=None, base_z: float = 0.0, n_starts: int = 3,
                         seed: int = 0, q_seed=None) -> Tuple[np.ndarray, np.ndarray]:
    """Same signature as the GPU entry; LM solve_lsq from rest, the mirrored rest and random in-limit starts (first success wins)."""
    stands = np.atleast_2d(np.asarray(stands, float))
    Tg = targets_in_base(target_world, stands, point, base_z)
    lim = K.arm_limits(arm)
    lo, hi = lim
    rst = K.rest(arm)
    rng = np.random.default_rng(seed)
    starts = [rst] + [rng.uniform(lo, hi) for _ in range(max(n_starts - 1, 0))]
    seeds = None if q_seed is None else np.broadcast_to(np.asarray(q_seed, float), (len(stands), 11))
    mask, Q = np.zeros(len(stands), bool), np.zeros((len(stands), 11))
    for i, T in enumerate(Tg):
        p, Rm = T[:3, 3], T[:3, :3]

        def fun(x):
            Tq = fk_gripper(arm, x)
            return np.r_[Tq[:3, 3] - p, R.from_matrix(Tq[:3, :3] @ Rm.T).as_rotvec() * 0.3]
        for s0 in ([seeds[i]] if seeds is not None else starts[:n_starts]):
            x = K.solve_lsq(fun, np.clip(s0, lo + 1e-3, hi - 1e-3), lo, hi)
            e = pose_error(fk_gripper(arm, x), T)
            if e[0] < tol[0] and e[1] < tol[1]:
                mask[i], Q[i] = True, x
                break
            if not Q[i].any():
                Q[i] = x
    return mask, Q


# ------------------------------------------------------------------ GPU
def _write_model(arm: str, out: Path) -> Path:
    by_child = {j[2]: j for j in K.JOINTS}
    chain, link = [], K.GRIPPER_LINK[arm]
    while link in by_child:
        chain.append(by_child[link])
        link = by_child[link][1]
    chain.reverse()
    lim = K.arm_limits(arm)
    dof_cols = list(K.TRUNK_DOFS) + list(K.ARM_DOFS[arm])
    names, x = [], ['<robot name="r1pro_%s">' % arm]
    x += [f'<link name="{l}"/>' for l in sorted({'base_link'} | {j[2] for j in chain})]
    j_ = lambda v: ' '.join(map(repr, map(float, v)))  # noqa: E731
    for dof, parent, child, kind, xyz, rpy, axis in chain:
        if dof < 0:
            x.append(f'<joint name="{child}_fixed" type="fixed"><parent link="{parent}"/><child link="{child}"/><origin xyz="{j_(xyz)}" rpy="{j_(rpy)}"/></joint>')
            continue
        k = dof_cols.index(dof)
        names.append(f'j{dof}')
        x.append(f'<joint name="j{dof}" type="revolute"><parent link="{parent}"/><child link="{child}"/><origin xyz="{j_(xyz)}" rpy="{j_(rpy)}"/>'
                 f'<axis xyz="{j_(axis)}"/><limit lower="{lim[0, k]}" upper="{lim[1, k]}" effort="100" velocity="3"/></joint>')
    x.append('</robot>')
    assert len(names) == 11
    (out / f'r1pro_{arm}.urdf').write_text('\n'.join(x))
    (out / f'r1pro_{arm}.yml').write_text(f"""robot_cfg:
  kinematics:
    format_version: 2.0
    asset_root_path: {out}
    urdf_path: {out}/r1pro_{arm}.urdf
    base_link: base_link
    tool_frames: [{K.GRIPPER_LINK[arm]}]
    collision_link_names: []
    cspace:
      joint_names: {names}
      default_joint_position: {list(map(float, K.rest(arm)))}
      null_space_weight: {[1.0] * 11}
      cspace_distance_weight: {[1.0] * 11}
      max_acceleration: 15.0
      max_jerk: 500.0
""")
    return out / f'r1pro_{arm}.yml'


def gpu_available() -> bool:
    try:
        import torch
        import curobo  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _patch_graph_executor() -> bool:
    """Plug the GPU memory leak of cuRobo 0.8.0.post1.dev43 (78fd485): the LM seed solver feeds its CUDA-graph executor a fresh
    `joint_position.detach().clone().requires_grad_(True)` on every call, and GraphExecutor.__call__ copy_()s it into its persistent
    graph input with autograd on. The graph input is a non-leaf tensor, so each copy_ chains a new CopyBackwards node that keeps the
    leaf alive through its AccumulateGrad edge: 1024 x 32 x 11 float32 tensor, measured 1.43 MiB of GPU memory per solve, never freed (18 GiB after
    ~13000 calls). A replay needs no autograd, so the replay path now runs under torch.no_grad(); results are bit-identical.
    BIMBO_GPU_IK_NO_GRAPH_PATCH=1 skips the patch (A/B measurements only). Returns True when active."""
    if os.environ.get('BIMBO_GPU_IK_NO_GRAPH_PATCH') == '1':
        return False
    import torch
    from curobo._src.util import cuda_graph_util as cgu
    cls = cgu.GraphExecutor
    if getattr(cls.__call__, '_bimbo_nograd', False):
        return True
    orig = cls.__call__

    def call(self, *inputs, **kw):
        if getattr(self, '_graph', None) is not None:    # replay: no autograd needed (first call records the graph with grad on)
            with torch.no_grad():
                return orig(self, *inputs, **kw)
        return orig(self, *inputs, **kw)
    call._bimbo_nograd = True
    cls.__call__ = call
    return True


def _solver(arm: str, tol):
    key = (arm, tuple(tol))
    if key not in _GPU:
        _patch_graph_executor()
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        out = Path(tempfile.mkdtemp(prefix='gpu_ik_'))
        cfg = InverseKinematicsCfg.create(robot=str(_write_model(arm, out)), num_seeds=NUM_SEEDS, max_batch_size=MAX_BATCH,
                                          self_collision_check=False, load_collision_spheres=False,
                                          position_tolerance=float(tol[0]), orientation_tolerance=float(tol[1]))
        ik = InverseKinematics(cfg)
        _GPU[key] = ik
    return _GPU[key]


def warmup(arms=('left', 'right'), tol=DEFAULT_TOL, wait: bool = True) -> None:
    """Build both solvers and run one full-size batch (JIT + CUDA graph capture). Call once per process.
    Through the server (no cuRobo here, BIMBO_CUROBO_PY set) wait=False returns at once and the warm-up overlaps the caller's work."""
    if not gpu_available() and server_configured():
        _server().call('warmup', dict(arms=list(arms), tol=list(map(float, tol))), wait=wait)
        return
    for arm in arms:
        reachable_stands_gpu(np.eye(4), np.zeros((MAX_BATCH, 3)), arm, tol)


# ------------------------------------------------------------------ server (cuRobo in another python)
SERVER_PY_ENV = 'BIMBO_CUROBO_PY'
SERVER_TIMEOUT_S = float(os.environ.get('BIMBO_GPU_IK_TIMEOUT_S', 600))   # per reply; a cold warp kernel cache makes the warm-up long
_MARK = 'IKR '
_SRV = {}


def server_configured() -> bool:
    return bool(os.environ.get(SERVER_PY_ENV)) and _SRV.get('dead') is None


class _Server:
    """ik_server.py under the cuRobo python, JSON lines over a pipe. Any failure marks it dead for the process (no restarts)."""

    def __init__(self, py: str) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', SERVER_PY_ENV)}
        env['PYTHONNOUSERSITE'] = '1'
        self.proc = subprocess.Popen([py, '-u', str(Path(__file__).with_name('ik_server.py'))], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, env=env, start_new_session=True)
        self.lines: 'queue.Queue[str]' = queue.Queue()
        self.id, self.pending = 0, {}
        self.stats = dict(calls=0, server_ms=0.0, wall_ms=0.0, warmup_s=None)
        threading.Thread(target=self._pump, args=(self.proc.stdout,), daemon=True).start()
        atexit.register(self.close)

    def _pump(self, out) -> None:
        with out:
            for line in out:
                if line.startswith(_MARK):
                    self.lines.put(line[len(_MARK):])
        self.lines.put('')

    def _die(self, why: str):
        _SRV['dead'] = why
        self.close()
        raise RuntimeError('gpu_ik server: ' + why)

    def call(self, op: str, args: dict, wait: bool = True) -> Optional[dict]:
        t0 = time.perf_counter()
        self.id += 1
        rid = self.id
        try:
            self.proc.stdin.write(json.dumps(dict(id=rid, op=op, args=args)) + '\n')
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            self._die(f'write {exc!r}')
        if not wait:
            self.pending[rid] = op
            return None
        while True:
            try:
                line = self.lines.get(timeout=SERVER_TIMEOUT_S)
            except queue.Empty:
                self._die(f'no reply in {SERVER_TIMEOUT_S:.0f}s')
            if not line:
                self._die(f'server exited rc={self.proc.poll()}')
            msg = json.loads(line)
            if 'error' in msg:
                self._die('error ' + str(msg['error'])[-400:])
            done = self.pending.pop(msg['id'], None) or (op if msg['id'] == rid else None)
            if done == 'warmup':
                self.stats['warmup_s'] = round(msg['ms'] / 1e3, 1)
            if msg['id'] == rid:
                if op != 'warmup':
                    self.stats['calls'] += 1
                    self.stats['server_ms'] = round(self.stats['server_ms'] + msg['ms'], 2)
                    self.stats['wall_ms'] = round(self.stats['wall_ms'] + 1e3 * (time.perf_counter() - t0), 2)
                return msg

    def close(self) -> None:
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.stdin.write('{"op": "quit"}\n')
            p.stdin.flush()
            p.stdin.close()
            p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            p.kill()
            p.wait()


def _server() -> _Server:
    if _SRV.get('dead') is not None:
        raise RuntimeError('gpu_ik server: ' + _SRV['dead'])
    if _SRV.get('srv') is None:
        py = os.environ.get(SERVER_PY_ENV, '')
        try:
            _SRV['srv'] = _Server(py)
        except OSError as exc:
            _SRV['dead'] = f'cannot start {py}: {exc!r}'
            raise RuntimeError('gpu_ik server: ' + _SRV['dead'])
        log.info('gpu_ik: cuRobo server started under %s', py)
    return _SRV['srv']


def close_server() -> None:
    """Stop the helper and forget its state (also a failure), so the next call starts a fresh one. Runs at exit anyway."""
    s = _SRV.pop('srv', None)
    _SRV.clear()
    if s is not None:
        s.close()


def server_stats() -> dict:
    """Counters of the server backend: calls, server_ms (solver time), wall_ms (with the pipe), warmup_s, dead (failure reason or None)."""
    s = _SRV.get('srv')
    return dict(s.stats if s is not None else {}, configured=bool(os.environ.get(SERVER_PY_ENV)), dead=_SRV.get('dead'))


def _route(backend: str) -> str:
    """'cpu' | 'gpu' (in this process) | 'server'; 'cpu' from 'auto' only with a logged warning."""
    if backend == 'cpu':
        return 'cpu'
    if backend not in ('gpu', 'auto', 'server'):
        raise ValueError(backend)
    if backend != 'server' and gpu_available():
        return 'gpu'
    if server_configured():
        return 'server'
    if backend == 'auto':
        why = f'server failed ({_SRV["dead"]})' if _SRV.get('dead') else 'CUDA/cuRobo unavailable'
        log.warning('gpu_ik: %s, FALLING BACK TO CPU solver (about 1e3-1e4 times slower)', why)
        return 'cpu'
    raise RuntimeError(f"gpu_ik: backend='{backend}' needs CUDA and cuRobo here or {SERVER_PY_ENV} "
                       f"(use backend='cpu' or 'auto' for an explicit fallback){'; server: ' + _SRV['dead'] if _SRV.get('dead') else ''}")


def _via_server(op: str, backend: str, **args) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Solve on the server; None after a failure under backend 'auto' (warning logged, the caller goes to the CPU)."""
    js = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in args.items()}
    try:
        msg = _server().call(op, js)
    except RuntimeError as exc:
        if backend != 'auto':
            raise
        log.warning('gpu_ik: %s, FALLING BACK TO CPU solver (about 1e3-1e4 times slower)', exc)
        return None
    return np.asarray(msg['mask'], bool), np.asarray(msg['q'], float).reshape(-1, 11)


def _args(stands, arm, tol, point, base_z, verify) -> dict:
    return dict(stands=np.atleast_2d(np.asarray(stands, float)), arm=arm, tol=[float(v) for v in tol],
                point=None if point is None else np.asarray(point, float), base_z=float(base_z), verify=bool(verify))


def reachable_stands_gpu(target_world, stands, arm: str, tol=DEFAULT_TOL, point=None, base_z: float = 0.0,
                         q_seed=None) -> Tuple[np.ndarray, np.ndarray]:
    """q_seed (11,) or (N,11): seed every problem with it and return, among the successful seeds, the solution nearest to it (limit-normalised L2)."""
    import torch
    from curobo.types import GoalToolPose, Pose
    stands = np.atleast_2d(np.asarray(stands, float))
    Tg = targets_in_base(target_world, stands, point, base_z)
    ik = _solver(arm, tol)
    frame = ik.tool_frames[0]
    n = len(Tg)
    mask, Q = np.zeros(n, bool), np.zeros((n, 11))
    for s in range(0, n, MAX_BATCH):
        T = Tg[s:s + MAX_BATCH]
        m = len(T)
        if m < MAX_BATCH:  # fixed batch shape (CUDA graph): pad with the last target
            T = np.concatenate([T, np.repeat(T[-1:], MAX_BATCH - m, 0)])
        xyzw = R.from_matrix(T[:, :3, :3]).as_quat()
        pos = torch.tensor(T[:, :3, 3], dtype=torch.float32, device='cuda')
        quat = torch.tensor(np.concatenate([xyzw[:, 3:], xyzw[:, :3]], 1), dtype=torch.float32, device='cuda')
        goal = GoalToolPose.from_poses({frame: Pose(position=pos, quaternion=quat)}, num_goalset=1)
        if q_seed is None:
            r = ik.solve_pose(goal)
            torch.cuda.synchronize()
            mask[s:s + m] = r.success.reshape(-1).cpu().numpy()[:m]
            Q[s:s + m] = r.solution.reshape(-1, 11).cpu().numpy()[:m]
            continue
        sd = np.broadcast_to(np.asarray(q_seed, float), (n, 11))[s:s + m]
        if m < MAX_BATCH:
            sd = np.concatenate([sd, np.repeat(sd[-1:], MAX_BATCH - m, 0)])
        seed_t = torch.tensor(sd, dtype=torch.float32, device='cuda').view(MAX_BATCH, 1, 11)
        r = ik.solve_pose(goal, seed_config=seed_t, return_seeds=NUM_SEEDS)
        torch.cuda.synchronize()
        ok = r.success.reshape(MAX_BATCH, -1).cpu().numpy()[:m]
        sol = r.solution.reshape(MAX_BATCH, ok.shape[1], 11).cpu().numpy()[:m].astype(float)
        lo, hi = K.arm_limits(arm)
        d = np.linalg.norm((sol - sd[:m, None, :]) / (hi - lo), axis=2)
        d[~ok] = np.inf
        j = d.argmin(1)
        mask[s:s + m] = np.isfinite(d[np.arange(m), j])
        Q[s:s + m] = sol[np.arange(m), j]
    return mask, Q


def _verify64(target_world, stands, arm, mask, q, tol, point, base_z):
    """float64 re-check of the float32 GPU solutions (pose tolerance and joint limits) with the CPU FK of r1pro_kinematics; drops any claim that does not hold."""
    Tg = targets_in_base(target_world, stands, point, base_z)
    lo, hi = K.arm_limits(arm)
    for i in np.where(mask)[0]:
        e = pose_error(fk_gripper(arm, q[i]), Tg[i])
        if not (e[0] < tol[0] and e[1] < tol[1]) or np.any(q[i] < lo - 1e-6) or np.any(q[i] > hi + 1e-6):
            mask[i] = False
    return mask


def reachable_stands(target_pose_world, stands, arm: str, tol=DEFAULT_TOL, point=None, base_z: float = 0.0, backend: str = 'gpu',
                     verify: bool = True):
    """-> (mask (N,) bool, q (N,11)). backend 'gpu' (default; raises if CUDA/cuRobo is missing here and BIMBO_CUROBO_PY is unset), 'cpu',
    'server' (only the helper process), or 'auto' = GPU, and if it is not available an explicit CPU fallback with a logged warning (never
    silent). verify=True re-checks GPU hits in float64 on the CPU FK."""
    if arm not in ('left', 'right'):
        raise ValueError(arm)
    route = _route(backend)
    if route == 'server':
        out = _via_server('reachable_stands', backend, target_pose_world=np.asarray(target_pose_world, float),
                          **_args(stands, arm, tol, point, base_z, verify))
        if out is not None:
            return out
        route = 'cpu'
    if route == 'cpu':
        return reachable_stands_cpu(target_pose_world, stands, arm, tol, point, base_z)
    mask, q = reachable_stands_gpu(target_pose_world, stands, arm, tol, point, base_z)
    if verify:
        mask = _verify64(target_pose_world, np.atleast_2d(stands), arm, mask, q, tol, point, base_z)
    return mask, q


def best_stand(target_pose_world, stands, arm: str = 'both', tol=DEFAULT_TOL, point=None, base_z: float = 0.0, backend: str = 'gpu',
               q_ref: Optional[np.ndarray] = None):
    """Pick one stand of `stands` for the target. Cost = joint-space distance of the solution to rest (or to q_ref, 11 values of that arm)
    plus nothing else; collisions and path length are the planner's job. arm 'left' | 'right' | 'both'.
    -> dict(index, arm, stand, q, n_reachable) or None when no stand/arm reaches."""
    arms = ('left', 'right') if arm == 'both' else (arm,)
    best, n_ok = None, 0
    for a in arms:
        mask, q = reachable_stands(target_pose_world, stands, a, tol, point, base_z, backend)
        n_ok += int(mask.sum())
        ref = K.rest(a) if q_ref is None else np.asarray(q_ref, float)
        cost = np.linalg.norm(q - ref, axis=1)
        cost[~mask] = np.inf
        i = int(np.argmin(cost))
        if np.isfinite(cost[i]) and (best is None or cost[i] < best[0]):
            best = (cost[i], dict(index=i, arm=a, stand=np.atleast_2d(stands)[i].copy(), q=q[i].copy()))
    if best is None:
        return None
    best[1]['n_reachable'] = n_ok
    return best[1]


def solve_near(target_world, stands, arm: str, q_seed, tol=DEFAULT_TOL, point=None, base_z: float = 0.0, backend: str = 'gpu',
               verify: bool = True):
    """Branch-continuous IK for servoing: like reachable_stands, but every problem is seeded with q_seed (11,) or (N,11) (current trunk + arm
    joints, ArmServo.dofs order) and, among the successful solutions, the one nearest q_seed (L2 normalised by the joint ranges) is returned.
    backend 'gpu' (raises without CUDA here and without BIMBO_CUROBO_PY), 'server', 'cpu' (LM started at q_seed, as ArmServo.solve does)
    or 'auto' (explicit logged CPU fallback). Use reachable_stands to choose a stand, solve_near to track a moving goal from the current joints."""
    if arm not in ('left', 'right'):
        raise ValueError(arm)
    route = _route(backend)
    if route == 'server':
        out = _via_server('solve_near', backend, target_world=np.asarray(target_world, float), q_seed=np.asarray(q_seed, float),
                          **_args(stands, arm, tol, point, base_z, verify))
        if out is not None:
            return out
        route = 'cpu'
    if route == 'cpu':
        return reachable_stands_cpu(target_world, stands, arm, tol, point, base_z, q_seed=q_seed)
    mask, q = reachable_stands_gpu(target_world, stands, arm, tol, point, base_z, q_seed=q_seed)
    if verify:
        mask = _verify64(target_world, np.atleast_2d(stands), arm, mask, q, tol, point, base_z)
    return mask, q
