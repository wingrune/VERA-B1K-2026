"""Native episode runner: a planner against the pinned OmniGibson evaluator (BatchedEvaluator, num_envs=1, CPU PhysX), with video.

Task independent. The harness steps the native env, records everything, and feeds the planner the native observation:
obs (torch tensors, batch dim 1, exactly the keys of core.env.native_iface) -> planner.act -> (1, 23) action -> env.step.
Privileged state exists only on the harness side (TaskAdapter.read_state); it is handed to the planner only when the adapter
says so (the oracle). Success is the native BDDL predicate (task.success), Q is the native metric (TaskMetric q_score).

One episode per process (`main`): Kit does not survive a second load of a scene reliably and its teardown can hang, so `main`
leaves through os._exit after the files are written. Written under --out:
  result.json         success, q_score, steps, seconds, stop reason, native max steps, guards, planner log, subgoals, files
  trace.jsonl         core.harness.trace events (phases, subgoals, exceptions) and a state sample every STATE_EVERY steps
  planner_trace.jsonl planner.log() every LOG_EVERY steps and at each phase change
  planner_trace_tail.jsonl  planner.log() of each of the last TAIL_LEN steps
  progress.json       latest heartbeat while the episode runs
  rollout.mp4         head camera large, both wrist cameras stacked, H.264, every VIDEO_EVERY-th control (fragmented mp4)
  frames/             PNG of the same composite at the start, at phase changes, every FRAME_EVERY steps and at the end
Nothing here imports the simulator at module import time; NativeEnv imports it when built.
"""
from __future__ import annotations

import argparse
import collections
import errno
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from core.env.native_iface import spec
from core.harness.trace import EpisodeFacts, Trace, classify_failure, compress

STATE_EVERY = 30
LOG_EVERY = 10
TAIL_LEN = 300
HEARTBEAT_EVERY = 200
VIDEO_EVERY = 3          # control 30 Hz: 10 video frames per simulated second
VIDEO_FPS = 10           # so the mp4 plays in real (simulated) time
VIDEO_CRF = 27
FRAME_EVERY = 300
MAX_FRAMES = 60
MAX_EXCEPTIONS = 20
FAILED_HOLD_STEPS = 60
HORIZON_DEFAULT = 6000
HEAD_PX, WRIST_PX = 448, 224
EVAL_SEED_FALLBACK = 0


class StopRequested(BaseException):
    """SIGTERM (the timeout of the launcher): unwinds the episode so that result.json, trace and the video index are still written."""


def write_retry(path: Path, text: str, tries: int = 40, wait: float = 30.0) -> None:
    """Write a file, waiting out a full disk: the render host's disk is shared and runs out of space for minutes at a time (ENOSPC kills the
    episode otherwise). Gives up after tries * wait seconds."""
    for k in range(tries):
        try:
            Path(path).write_text(text)
            return
        except OSError as exc:
            if exc.errno != errno.ENOSPC or k == tries - 1:
                raise
            time.sleep(wait)


def sha12(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def file_hashes(folder: Path) -> Dict[str, str]:
    return {str(p.relative_to(folder)): sha12(p) for p in sorted(Path(folder).rglob('*.py'))}


def to_numpy(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def jsonable(x: Any) -> Any:
    """Plain JSON values from planner logs and numpy results (long arrays are summarised, not dumped)."""
    if isinstance(x, Mapping):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if hasattr(x, 'detach'):
        x = to_numpy(x)
    if isinstance(x, np.ndarray):
        return np.round(x.astype(float), 4).tolist() if x.size <= 32 else f'<array {tuple(x.shape)}>'
    if isinstance(x, (np.floating, float)):
        return round(float(x), 4)
    if isinstance(x, (np.bool_, np.integer)):
        return x.item()
    if isinstance(x, (bool, int, str)) or x is None:
        return x
    return str(x)[:200]


# ---------------------------------------------------------------------------------------------------------------- video
def _image(obs: Mapping[str, Any], key: str) -> np.ndarray:
    a = to_numpy(obs[key])
    return a.reshape(-1, *a.shape[-3:])[0][..., :3].astype(np.uint8)


def resize(img: np.ndarray, side: int) -> np.ndarray:
    """Square resize (area interpolation with OpenCV, strided/nearest without)."""
    try:
        import cv2
        return cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
    except ImportError:
        ys = np.arange(side) * img.shape[0] // side
        xs = np.arange(side) * img.shape[1] // side
        return img[ys][:, xs]


def compose(obs: Mapping[str, Any]) -> np.ndarray:
    """Head camera (HEAD_PX square) beside the left and right wrist cameras stacked (WRIST_PX each): HEAD_PX x (HEAD_PX + WRIST_PX) x 3."""
    cams = {c.name: c for c in spec.CAMERAS}
    head = resize(_image(obs, cams['head'].rgb_key), HEAD_PX)
    wrists = [resize(_image(obs, cams[n].rgb_key), WRIST_PX) for n in ('left_wrist', 'right_wrist')]
    return np.concatenate([head, np.concatenate(wrists, axis=0)], axis=1)


class Video:
    """H.264 mp4 through an ffmpeg pipe, fragmented (empty_moov) so the file plays while it is written or after the process was killed.
    Never raises into the episode: on any failure it turns itself off and keeps the reason in `error`."""

    def __init__(self, path: Path, fps: int = VIDEO_FPS, crf: int = VIDEO_CRF) -> None:
        self.path, self.fps, self.crf = Path(path), fps, crf
        self.proc: Optional[subprocess.Popen] = None
        self.frames, self.error = 0, None
        self.dead = shutil.which('ffmpeg') is None
        if self.dead:
            self.error = 'ffmpeg not found'

    def add(self, frame: np.ndarray) -> None:
        if self.dead:
            return
        try:
            h, w = frame.shape[:2]
            frame = np.ascontiguousarray(frame[:h // 2 * 2, :w // 2 * 2, :3], dtype=np.uint8)  # yuv420p needs even sides
            if self.proc is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                cmd = ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{frame.shape[1]}x{frame.shape[0]}',
                       '-r', str(self.fps), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(self.crf), '-pix_fmt', 'yuv420p',
                       '-movflags', '+frag_keyframe+empty_moov+default_base_moof', str(self.path)]
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.proc.stdin.write(frame.tobytes())
            self.frames += 1
        except Exception as exc:  # noqa: BLE001 - the video must never end the episode
            self.dead, self.error = True, repr(exc)[:200]

    def close(self) -> Optional[Path]:
        proc, self.proc = self.proc, None
        if proc is None:
            return None
        try:
            proc.stdin.close()
            proc.wait(timeout=180)
        except Exception as exc:  # noqa: BLE001
            self.error = repr(exc)[:200]
            return None
        return self.path if proc.returncode == 0 else None


def save_png(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        cv2.imwrite(str(path), np.ascontiguousarray(frame[..., ::-1]))
    except ImportError:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.imsave(str(path), frame)


# ---------------------------------------------------------------------------------------------------------------- adapter
class TaskAdapter:
    """What a task contributes: planner loading, harness-side privileged state, subgoal predicates, failure facts.
    The defaults give an episode with no privileged state, no subgoals and a plain timeout cause."""

    task = ''                       # BEHAVIOR task name (the evaluator's task.name)
    subgoals: Tuple[str, ...] = ()
    privileged_planners: Tuple[str, ...] = ()    # planner kinds whose act() takes the state as a second argument

    def load_planner(self, kind: str) -> Any:
        """Module or object with entrypoint() -> planner (reset(seed), act(obs[, state]), log())."""
        raise NotImplementedError

    def planner_dir(self, kind: str) -> Optional[Path]:
        return None

    def bind(self, env: Any, instance: int) -> None:
        """After the instance is loaded: set up the readers on the live env."""

    def read_state(self, step: int) -> Any:
        """Harness-side privileged state of this step (or None)."""
        return None

    def evaluate(self, state: Any) -> Dict[str, bool]:
        """Subgoal predicates by state, keys = self.subgoals."""
        return {}

    def overview_pose(self) -> Optional[Tuple[Sequence[float], Sequence[float]]]:
        """(camera position, look-at target) of the fixed third-person review camera, world frame; None: no overview video."""
        return None

    def sample(self, state: Any, log: Mapping[str, Any]) -> Dict[str, Any]:
        """Extra fields of the periodic state event of the trace."""
        return {}

    def perception(self, state: Any, log: Mapping[str, Any]) -> Dict[str, float]:
        """Errors (m) of the estimates the planner reports in its log against the truth, by object. Feedback for the trace only."""
        return {}

    def dump(self, path: Path, obs: Mapping[str, Any], state: Any, step: int) -> None:
        """Save the observation and the truth of this step (offline development of perception). Harness only."""

    def facts(self, env: Any, exceptions: int, trace: Trace, success: bool) -> Optional[EpisodeFacts]:
        """Measurements for classify_failure (None: the cause is the stop reason)."""
        return None

    def extra_result(self) -> Dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------------------------------------------- episode
def run_episode(env: Any, planner: Any, adapter: TaskAdapter, instance: int, out: Path, *, planner_kind: str = 'planner',
                max_steps: int = HORIZON_DEFAULT, video: bool = True, video_every: int = VIDEO_EVERY, video_fps: int = VIDEO_FPS,
                dump_steps: Sequence[int] = (), dump_every: int = 0, max_dumps: int = 60, overview: bool = True, log: Callable[..., None] = print) -> Dict[str, Any]:
    """One episode on a built env: load, step the planner to success, termination, truncation or max_steps; write the files.

    env protocol: load(instance) -> obs; step(action) -> (obs, terminated, truncated); success; goal_options(); outcome() ->
    {q_score, ...}; native_max_steps; no_op() -> action; guard (CPU physics guard line). Never raises: failures end in result.json."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gets_state = planner_kind in adapter.privileged_planners
    result: Dict[str, Any] = dict(task=adapter.task, planner=planner_kind, instance=instance, max_steps=max_steps, success=False, q_score=0.0, steps=0,
                                  pin=os.environ.get('PIN'), cpu_physics_guard=getattr(env, 'guard', None),
                                  native_max_steps=getattr(env, 'native_max_steps', None), stop_reason='init')
    trace = Trace(adapter.task, instance, max_steps, list(adapter.subgoals))
    vid = Video(out / 'rollout.mp4', video_fps) if video else None
    vid2 = Video(out / 'overview.mp4', video_fps) if video and overview else None
    overview_on = False
    tail: collections.deque = collections.deque(maxlen=TAIL_LEN)
    exceptions, frames_saved, step, last_phase, dumps = 0, 0, 0, None, 0
    plog = open(out / 'planner_trace.jsonl', 'w', encoding='utf-8')
    log_errors = 0

    def plog_write(row: Mapping[str, Any]) -> None:
        nonlocal log_errors
        try:
            plog.write(json.dumps(row) + '\n')
        except OSError:      # a full disk loses a log row, not the episode
            log_errors += 1
    state = None
    spent = collections.defaultdict(float)     # seconds in the env step, the planner, the truth reads and the video

    def frame_of(obs: Mapping[str, Any]) -> Optional[np.ndarray]:
        try:
            return compose(obs)
        except Exception as exc:  # noqa: BLE001
            result.setdefault('frame_error', repr(exc)[:200])
            return None

    def snap(obs: Mapping[str, Any], tag: str) -> None:
        nonlocal frames_saved
        if frames_saved >= MAX_FRAMES:
            return
        fr = frame_of(obs)
        if fr is not None:
            save_png(fr, out / 'frames' / f'{step:05d}_{tag}.png')
            frames_saved += 1

    try:
        obs = env.load(instance)
        result['t_load_s'] = round(time.time() - t0, 1)
        result['obs_contract'] = check_obs_contract(obs)
        planner.reset(instance)
        adapter.bind(env, instance)
        pose = adapter.overview_pose() if vid2 is not None and hasattr(env, 'add_overview_camera') else None
        if pose is not None:
            try:
                env.add_overview_camera(*pose)
                overview_on = True
            except Exception as exc:  # noqa: BLE001 - the review camera is optional
                result['overview_error'] = repr(exc)[:300]
        initial_options = env.goal_options()
        result['initial_options'] = jsonable(initial_options)
        t_loop = time.time()
        result['stop_reason'] = 'max_steps'
        snap(obs, 'start')
        while step < max_steps:
            t_a = time.time()
            state = adapter.read_state(step)
            if state is not None:
                trace.subgoal_state(step, adapter.evaluate(state))
            spent['truth_s'] += time.time() - t_a
            if state is not None and dumps < max_dumps and (step in dump_steps or (dump_every and step % dump_every == 0)):
                adapter.dump(out / 'obs' / f'obs_{step:05d}.npz', obs, state, step)
                dumps += 1
            t_a = time.time()
            try:
                action = planner.act(obs, state) if gets_state else planner.act(obs)
            except Exception:  # noqa: BLE001 - a planner exception is data, not an episode crash
                exceptions += 1
                trace.exception(step, traceback.format_exc()[-300:])
                log('PLANNER_EXCEPTION step', step, traceback.format_exc())
                if exceptions > MAX_EXCEPTIONS:
                    result['stop_reason'] = 'planner_exceptions'
                    break
                action = env.no_op()
            spent['planner_s'] += time.time() - t_a
            info = jsonable(planner.log())
            tail.append({**info, 'step': step})
            phase = info.get('phase')
            if phase != last_phase:
                trace.phase(step, str(phase), str(info.get('reason', '')))
                plog_write({**info, 'step': step})
                snap(obs, f'phase_{phase}')
                log(f"EPISODE_PHASE step={step} phase={phase} reason={info.get('reason')}")
                last_phase = phase
            elif step % LOG_EVERY == 0:
                plog_write({**info, 'step': step})
            if step % STATE_EVERY == 0:
                trace.events.append(dict(t=step, ev='state', phase=phase, reason=info.get('reason'), err=info.get('err'), **adapter.sample(state, info)))
                if state is not None:
                    for obj, err in adapter.perception(state, info).items():
                        trace.perception_error(step, obj, err)
            if step % FRAME_EVERY == 0 and step:
                snap(obs, 't')
            if step % HEARTBEAT_EVERY == 0:
                rate = step / max(1e-9, time.time() - t_loop)
                try:
                    (out / 'progress.json').write_text(json.dumps(dict(step=step, phase=phase, steps_per_s=round(rate, 2), seconds=round(time.time() - t0, 1))))
                except OSError:
                    log_errors += 1
                log(f'EPISODE_PROGRESS step={step} phase={phase} steps_per_s={rate:.2f}')
            if vid is not None and step % video_every == 0:
                t_a = time.time()
                fr = frame_of(obs)
                if fr is not None:
                    vid.add(fr)
                if overview_on:
                    try:
                        vid2.add(env.overview_frame())
                    except Exception as exc:  # noqa: BLE001
                        overview_on = False
                        result['overview_error'] = repr(exc)[:300]
                spent['video_s'] += time.time() - t_a
            if phase == 'failed' and info.get('age', 0) > FAILED_HOLD_STEPS:
                result['stop_reason'] = 'planner_failed'
                break
            t_a = time.time()
            obs, terminated, truncated = env.step(action)
            spent['env_step_s'] += time.time() - t_a
            step += 1
            if env.success:
                result['stop_reason'] = 'success'
                break
            if terminated or truncated:
                result['stop_reason'] = 'terminated' if terminated else 'truncated'
                break
        state = adapter.read_state(step)
        if state is not None:
            trace.subgoal_state(step, adapter.evaluate(state))
        snap(obs, 'end')
        if vid is not None:
            fr = frame_of(obs)
            if fr is not None:
                vid.add(fr)
        success = bool(env.success)
        outcome = env.outcome()
        facts = adapter.facts(env, exceptions, trace, success)
        cause = None if success else (classify_failure(facts) if facts is not None else result['stop_reason'])
        result.update(success=success, q_score=float(outcome.get('q_score', 0.0)), steps=step, native=jsonable(outcome), cause=cause,
                      progress_env=trace.progress_env(), subgoals={g: trace._reached.get(g) for g in trace.subgoals},
                      subgoals_held=trace.held_goals(), exceptions=exceptions, goal_options=jsonable(env.goal_options()),
                      planner_log=jsonable(planner.log()), compressed=compress(trace, cause),
                      loop_seconds=round(time.time() - t_loop, 1), time_split_s={k: round(v, 1) for k, v in spent.items()}, steps_per_s=round(step / max(1e-9, time.time() - t_loop), 2),
                      **adapter.extra_result())
    except BaseException as exc:  # noqa: BLE001 - the episode record must survive any failure
        result.update(success=False, cause='stopped' if isinstance(exc, StopRequested) else 'infra',
                      stop_reason='sigterm' if isinstance(exc, StopRequested) else 'crash', steps=step, error=''.join(traceback.format_exception_only(type(exc), exc))[-500:])
        log('EPISODE_ERROR', traceback.format_exc())
        trace.exception(step, traceback.format_exc()[-300:])
    try:
        plog.close()
    except OSError:
        log_errors += 1
    write_retry(out / 'planner_trace_tail.jsonl', ''.join(json.dumps(r) + '\n' for r in tail))
    if vid is not None:
        path = vid.close()
        result['video'] = dict(path=path.name if path else None, frames=vid.frames, every=video_every, fps=video_fps, error=vid.error,
                               bytes=path.stat().st_size if path else 0)
    if vid2 is not None and vid2.frames:
        path2 = vid2.close()
        result['overview_video'] = dict(path=path2.name if path2 else None, frames=vid2.frames, error=vid2.error, bytes=path2.stat().st_size if path2 else 0)
    result['frames'] = frames_saved
    result['seconds'] = round(time.time() - t0, 1)
    result['log_write_errors'] = log_errors
    write_retry(out / 'trace.jsonl', trace.to_jsonl() + '\n')
    write_retry(out / 'result.json', json.dumps(result, indent=1, default=str))
    log('EPISODE_RESULT ' + json.dumps({k: result.get(k) for k in ('instance', 'success', 'q_score', 'steps', 'stop_reason', 'cause', 'seconds')}))
    return result


# ---------------------------------------------------------------------------------------------------------------- native env
def physics_guard_line() -> Tuple[bool, str]:
    """True only if GPU dynamics are off in the macros and in the live PhysX context, broadphase is MBP and the sim device is cpu."""
    import omnigibson as og
    from omnigibson.macros import gm
    ctx = getattr(og.sim, '_physics_context', None)
    gpu_dyn = ctx.is_gpu_dynamics_enabled() if ctx is not None else None
    broad = ctx.get_broadphase_type() if ctx is not None else None
    try:
        device = str(og.sim.device)
    except (AttributeError, RuntimeError) as exc:
        device = f'unreadable:{type(exc).__name__}'
    ok = gm.USE_GPU_DYNAMICS is False and gpu_dyn is False and broad == 'MBP' and device.startswith('cpu')
    return ok, f'use_gpu_dynamics={gm.USE_GPU_DYNAMICS} physx_gpu_dynamics={gpu_dyn} broadphase={broad} sim_device={device}'


class NativeEnv:
    """The pinned BatchedEvaluator with one logical env and a placeholder policy: the planner acts through step().

    Same construction as the evaluator entry point (RGBDFullResWrapper: rgb + depth_linear, head 720, wrists 480; robot config eval/r1pro.yaml),
    so the observation and the physics are the native ones. Raises unless physics is CPU PhysX."""

    def __init__(self, task: str, mode: str = 'public_test', seed: Optional[int] = None) -> None:
        import omnigibson as og
        import torch
        from omegaconf import OmegaConf
        from omnigibson.macros import gm
        gm.HEADLESS, gm.RENDER_VIEWER_CAMERA = True, False
        from omnigibson.eval.utils import eval_utils
        seed = eval_utils.DEFAULT_EVAL_SEED if seed is None else seed
        eval_utils.seed_everything(seed)
        from omnigibson.eval.evaluator import BatchedEvaluator
        cfg = OmegaConf.create({
            'env_wrapper': {'_target_': 'omnigibson.eval.wrappers.RGBDFullResWrapper'}, 'policy_name': 'local',
            'model': {'_target_': 'omnigibson.eval.policies.LocalPolicy', 'action_dim': None}, 'headless': True, 'partial_scene_load': True,
            'max_steps': None, 'write_video': False, 'mode': mode, 'seed': seed, 'num_envs': 1, 'task': {'name': task}, 'robot': None})
        self.og, self.gm, self.torch, self.task, self.mode = og, gm, torch, task, mode
        self.ev = BatchedEvaluator(cfg)
        ok, line = physics_guard_line()
        self.guard = 'ok ' + line if ok else 'FAIL ' + line
        print('CPU_PHYSICS_GUARD', self.guard, flush=True)
        if not ok:
            raise RuntimeError('CPU_PHYSICS_GUARD FAIL ' + line)
        self.state = self.ev.instance_eval_states[0]
        self.acc = self.state.env_accessor
        self.robot = self.acc.robot
        self.native_max_steps = int(self.ev.human_stats['length'] * eval_utils.EVAL_TIMEOUT_MULTIPLIER)
        self.steps = 0
        self.instance: Optional[int] = None

    def _obs(self) -> Dict[str, Any]:
        """Evaluator observation of the instance, batch dim 1, the keys of the contract (a missing or extra key is a contract check failure)."""
        raw = self.state.obs
        return {k: (v if hasattr(v, 'detach') else self.torch.as_tensor(v)).detach().cpu().unsqueeze(0) for k, v in raw.items()}

    def load(self, instance: int) -> Dict[str, Any]:
        """Evaluator.load_batch({0: instance}): TRO restore, 25 keep_still steps, update_initial_file, env.reset."""
        self.ev.cfg.mode = self.mode
        self.ev.load_batch({0: int(instance)})
        self.instance, self.steps = int(instance), 0
        return self._obs()

    def step(self, action: Any) -> Tuple[Dict[str, Any], bool, bool]:
        act = np.asarray(to_numpy(action), dtype=np.float32).reshape(1, spec.ACTION_DIM)
        if not np.isfinite(act).all():
            raise ValueError('non-finite action')
        terminated, truncated, _ = self.ev._apply_actions(self.torch.from_numpy(act), [0])
        self.steps += 1
        return self._obs(), bool(terminated[0]), bool(truncated[0])

    @property
    def success(self) -> bool:
        return bool(self.acc.success)

    def goal_options(self) -> List[List[bool]]:
        return [[bool(a) for a in option] for option in self.acc.get_goal_option_satisfaction()]

    def outcome(self) -> Dict[str, Any]:
        """Native metrics of the episode (AgentMetric, TaskMetric), q_score = the final value of the TaskMetric."""
        merged: Dict[str, Any] = {}
        for metric in self.state.metrics:
            merged.update(metric.aggregate())
        merged['q_score'] = float((merged.get('q_score') or {}).get('final') or 0.0)
        return merged

    def add_overview_camera(self, position: Sequence[float], target: Sequence[float], size: Tuple[int, int] = (480, 640), focal_length: float = 14.0) -> None:
        """A fixed third-person RGB camera at `position` looking at `target` (world frame, z up), for the review video. Not part of the observation."""
        from omnigibson.sensors import VisionSensor
        cam = VisionSensor(relative_prim_path='/overview_cam', name='overview_cam', modalities=['rgb'], image_height=size[0], image_width=size[1],
                           focal_length=focal_length)
        cam.load(self.acc.scene)
        cam.initialize()
        self.overview = cam
        self.move_overview_camera(position, target)

    def move_overview_camera(self, position: Sequence[float], target: Sequence[float]) -> None:
        """Place the overview camera (USD camera: looks along its -Z, up is +Y)."""
        from scipy.spatial.transform import Rotation
        position, target = np.asarray(position, float), np.asarray(target, float)
        z = position - target
        z /= np.linalg.norm(z)
        x = np.cross([0., 0., 1.], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        quat = Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat()
        self.overview.set_position_orientation(position=self.torch.as_tensor(position, dtype=self.torch.float32),
                                               orientation=self.torch.as_tensor(quat, dtype=self.torch.float32))

    def overview_frame(self) -> Optional[np.ndarray]:
        obs, _ = self.overview.get_obs()
        return to_numpy(obs['rgb'])[..., :3].astype(np.uint8)

    def no_op(self) -> np.ndarray:
        from omnigibson.controllers import ControllerView
        parts = [to_numpy(ControllerView.compute_no_op_action(g, i)).reshape(-1) for g, i in self.robot.controllers.values()]
        return np.concatenate(parts).astype(np.float32)[None]

    def commit(self) -> str:
        import inspect
        import omnigibson
        root = Path(inspect.getfile(omnigibson)).parent
        return subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()

    def close(self) -> None:
        """Left to process exit: shutting Kit down here can hang or end the process before the files are written."""


def check_obs_contract(obs: Mapping[str, Any]) -> str:
    """'ok' or the first difference to the native observation contract (core.env.native_iface.obs.check_obs); the task_id key is optional."""
    from core.env.native_iface.obs import check_obs
    try:
        check_obs({k: v for k, v in obs.items()})
        return 'ok'
    except ValueError as exc:
        return str(exc)[:300]


# ---------------------------------------------------------------------------------------------------------------- cli
def main(adapter: TaskAdapter, argv: Optional[Sequence[str]] = None, env_factory: Optional[Callable[..., Any]] = None, hard_exit: bool = True) -> int:
    ap = argparse.ArgumentParser(description='One native episode of a planner, with video.')
    ap.add_argument('--planner', required=True)
    ap.add_argument('--instance', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-steps', type=int, default=HORIZON_DEFAULT)
    ap.add_argument('--mode', default='public_test', choices=['train', 'public_test', 'hidden_test'])
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--no-overview', action='store_true', help='skip the third-person review camera and its video')
    ap.add_argument('--video-every', type=int, default=VIDEO_EVERY)
    ap.add_argument('--video-fps', type=int, default=VIDEO_FPS)
    ap.add_argument('--dump-steps', default='', help='comma separated steps at which obs + truth are saved to <out>/obs/')
    ap.add_argument('--dump-every', type=int, default=0)
    ap.add_argument('--max-dumps', type=int, default=60)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rc, env = 0, None

    def on_term(signum: int, frame: Any) -> None:
        raise StopRequested(f'signal {signum}')
    signal.signal(signal.SIGTERM, on_term)
    try:
        env = (env_factory or NativeEnv)(adapter.task, args.mode)
        planner_mod = adapter.load_planner(args.planner)
        planner = planner_mod.entrypoint()
        result = run_episode(env, planner, adapter, args.instance, out, planner_kind=args.planner, max_steps=args.max_steps, video=not args.no_video,
                             video_every=args.video_every, video_fps=args.video_fps,
                             dump_steps={int(v) for v in args.dump_steps.split(',') if v.strip()}, dump_every=args.dump_every, max_dumps=args.max_dumps, overview=not args.no_overview)
        folder = adapter.planner_dir(args.planner)
        if folder is not None:
            result['planner_files'] = file_hashes(folder)
        if hasattr(env, 'commit'):
            result['behavior_commit'] = env.commit()
        write_retry(out / 'result.json', json.dumps(result, indent=1, default=str))
        rc = 3 if result.get('cause') == 'infra' else (124 if result.get('cause') == 'stopped' else 0)
    except BaseException as exc:  # noqa: BLE001 - env build or planner load failed: still leave a record
        (out / 'result.json').write_text(json.dumps(dict(task=adapter.task, planner=args.planner, instance=args.instance, success=False, cause='infra',
                                                         error=''.join(traceback.format_exception_only(type(exc), exc))[-500:]), indent=1))
        print('EPISODE_ERROR', traceback.format_exc(), flush=True)
        rc = 3
    sys.stdout.flush()
    sys.stderr.flush()
    if hard_exit:
        os._exit(rc)
    return rc
