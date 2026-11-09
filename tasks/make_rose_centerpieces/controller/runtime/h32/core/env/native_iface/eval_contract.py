"""Simulator-free pieces of the native BEHAVIOR-1K evaluator (OmniGibson bd049de), copied for the harness.

Sources, semantics kept:
  ``eval/evaluator.py``      resolve_instance_ids, evaluate_instances_batched, EVAL_MODES, the result row layout
  ``eval/utils/eval_utils``  DEFAULT_EVAL_SEED, EVAL_TIMEOUT_MULTIPLIER, the test instance split
  ``metrics/task_metric.py`` compute_q_score and the time block
  ``metrics/agent_metric.py`` per-step displacement sums and their normalisation
Per-task human statistics (length and distances) are inputs from task.yaml, not constants of this module.
"""
from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

DEFAULT_EVAL_SEED = 0
EVAL_TIMEOUT_MULTIPLIER = 1.5
NUM_TEST_INSTANCES = 40
NUM_PUBLIC_TEST_INSTANCES = 20
NUM_HIDDEN_TEST_INSTANCES = NUM_TEST_INSTANCES - NUM_PUBLIC_TEST_INSTANCES
TEST_INSTANCE_IDS = list(range(301, 341))
EVAL_MODES = ('train', 'public_test', 'hidden_test')
RENDERING_DT = 1 / 30  # og.sim.get_rendering_dt() at the evaluator's rendering_frequency 30

RESULT_KEYS = ('task', 'instance_id', 'rollout_id', 'steps', 'success', 'agent_distance', 'normalized_agent_distance',
               'q_score', 'time')


def horizon(human_length: float, cfg_max_steps: Optional[int]) -> int:
    """Native termination_config max_steps: the cfg value, else int(1.5 x human length). The episode is truncated at
    the step that makes the step count exceed it, so it records max_steps + 1 steps (3225 for human length 2149.64).
    human_length is the per-task row of the native human statistics (task.yaml)."""
    if cfg_max_steps is not None:
        return int(cfg_max_steps)
    return int(human_length * EVAL_TIMEOUT_MULTIPLIER)


def resolve_instance_ids(task_name: str, instance_indices: List[int], mode: str = 'public_test') -> List[int]:
    if mode not in EVAL_MODES:
        raise ValueError(f'Mode must be one of {EVAL_MODES}, got {mode}')
    if mode == 'train':
        return [int(instance_id) for instance_id in instance_indices]
    test_instances = (TEST_INSTANCE_IDS[:NUM_PUBLIC_TEST_INSTANCES] if mode == 'public_test'
                      else TEST_INSTANCE_IDS[NUM_PUBLIC_TEST_INSTANCES:])
    num_split_instances = NUM_PUBLIC_TEST_INSTANCES if mode == 'public_test' else NUM_HIDDEN_TEST_INSTANCES
    if not set(instance_indices).issubset(set(range(num_split_instances))):
        raise ValueError(f'Instance indices must be in range({num_split_instances}) for mode {mode}')
    return [int(test_instances[i]) for i in instance_indices]


def evaluate_instances_batched(instances: Sequence, num_envs: int, load_fn: Callable[[Dict[int, object]], None],
                               step_fn: Callable[[List[int]], Any], record_fn: Callable[..., object],
                               max_steps: Optional[int] = None) -> Dict[object, object]:
    """Native batch driver, unchanged: one instance per logical env, an env leaves the active set when finished."""
    if num_envs < 1:
        raise ValueError(f'num_envs must be >= 1, got {num_envs}')
    instances = list(instances)
    if len(instances) != num_envs:
        raise ValueError(f'Evaluation requires exactly one instance per logical environment: '
                         f'got {len(instances)} instances and num_envs={num_envs}.')
    results: Dict[object, object] = {}
    env_idx_to_instance = {env_idx: instance for env_idx, instance in enumerate(instances)}
    load_fn(dict(env_idx_to_instance))
    active = {env_idx: True for env_idx in env_idx_to_instance}
    step = 0
    while any(active.values()):
        active_env_indices = sorted(env_idx for env_idx, is_active in active.items() if is_active)
        terminated, truncated = step_fn(active_env_indices)
        step += 1
        hit_cap = max_steps is not None and step >= max_steps
        for env_idx in active_env_indices:
            term = bool(terminated[env_idx])
            trunc = bool(truncated[env_idx]) or hit_cap
            if term or trunc:
                results[env_idx_to_instance[env_idx]] = record_fn(env_idx=env_idx, instance=env_idx_to_instance[env_idx],
                                                                  step=step, terminated=term, truncated=trunc)
                active[env_idx] = False
    return results


def compute_q_score(success: bool, now_satisfied_options: Sequence[Sequence[bool]],
                    initial_satisfied_options: Sequence[Sequence[bool]]) -> float:
    """Native partial success: 1.0 on success, else the best fraction of newly satisfied goal predicates."""
    if success:
        return 1.0
    if not now_satisfied_options:
        return 0.0
    option_scores = []
    for now_opt, init_opt in zip(now_satisfied_options, initial_satisfied_options):
        if len(now_opt) == 0:
            option_scores.append(0.0)
            continue
        newly_satisfied = sum(int((not init) and now) for now, init in zip(now_opt, init_opt))
        option_scores.append(newly_satisfied / len(now_opt))
    return max(option_scores) if option_scores else 0.0


class AgentDistance:
    """Native AgentMetric: the first step caches the pose (delta 0), each step appends |p_t - p_{t-1}|."""

    PARTS = {'base': 'distance_traveled', 'left': 'left_eef_displacement', 'right': 'right_eef_displacement'}

    def __init__(self, human_stats: Dict[str, float]):
        self.human = {k: human_stats[v] for k, v in self.PARTS.items()}
        self.reset()

    def reset(self) -> None:
        self.cache: Optional[Dict[str, Sequence[float]]] = None
        self.delta: Dict[str, List[float]] = {k: [] for k in self.PARTS}

    def step(self, positions: Dict[str, Sequence[float]]) -> None:
        if self.cache is None:
            self.cache = positions
        for k in self.PARTS:
            self.delta[k].append(math.dist(positions[k], self.cache[k]))
        self.cache = positions

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        total = {k: sum(v) for k, v in self.delta.items()}
        return {'agent_distance': total,
                'normalized_agent_distance': {k: (self.human[k] / v if v != 0 else float('inf')) for k, v in total.items()}}


def task_block(success: bool, now_options, initial_options, timesteps: int, length: float) -> Dict[str, Any]:
    """Native TaskMetric aggregate."""
    return {'q_score': {'final': compute_q_score(success, now_options, initial_options)},
            'time': {'simulator_steps': timesteps, 'simulator_time': timesteps * RENDERING_DT,
                     'normalized_time': length / timesteps if timesteps > 0 else float('inf')}}


def result_row(task: str, instance: int, rollout_id: int, step: int, success: bool, agent: Dict[str, Any],
               task_metric: Dict[str, Any]) -> Dict[str, Any]:
    """Native record_fn row: header, success, then AgentMetric and TaskMetric aggregates in that order."""
    result = {'task': task, 'instance_id': int(instance), 'rollout_id': rollout_id, 'steps': step}
    result['success'] = success
    result.update(agent)
    result.update(task_metric)
    return result


def compare_actions(a, b, n: int) -> Dict[str, Any]:
    """First n rows of two (T, 23) action logs: bit equality, max |diff| and the first differing step (1-based)."""
    import numpy as np
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = min(n, len(a), len(b))
    diff = np.abs(a[:m] - b[:m])
    rows = np.flatnonzero(diff.max(-1) > 0) if m else np.array([], int)
    return dict(n_compared=int(m), n_requested=int(n), identical=bool(m == n and not len(rows)),
                max_abs_diff=float(diff.max()) if m else None,
                first_diff_step=int(rows[0]) + 1 if len(rows) else None, n_diff_steps=int(len(rows)))


NESTED_KEYS = {'agent_distance': ('base', 'left', 'right'), 'normalized_agent_distance': ('base', 'left', 'right'),
               'q_score': ('final',), 'time': ('simulator_steps', 'simulator_time', 'normalized_time')}


def check_episode(folder, task: str, max_steps: int, exact: bool = False,
                  extra_files: Sequence[str] = ()) -> Dict[str, Any]:
    """Acceptance on one evaluation output folder: files, one native row with every contract key,
    steps <= max_steps (== with exact), simulator_steps == steps, simulator_time == steps / 30.
    extra_files: task entry-point specific files that must exist beside config.json, policy and summary.json."""
    from pathlib import Path
    folder, problems = Path(folder), []
    for name in ('config.json', 'policy', 'summary.json') + tuple(extra_files):
        if not (folder / name).exists():
            problems.append('missing ' + name)
    if problems:
        return dict(ok=False, problems=problems)
    rows = json.loads((folder / 'summary.json').read_text())['results']
    if len(rows) != 1:
        return dict(ok=False, problems=[f'{len(rows)} result rows'])
    row = rows[0]
    if tuple(row) != RESULT_KEYS:
        problems.append(f'row keys {list(row)}')
    for key, sub in NESTED_KEYS.items():
        if tuple(row.get(key, {})) != sub:
            problems.append(f'{key} keys {list(row.get(key, {}))}')
    steps = row.get('steps')
    if not isinstance(steps, int) or not 0 < steps <= max_steps or (exact and steps != max_steps):
        problems.append(f'steps {steps} against {"==" if exact else "<="} {max_steps}')
    t = row.get('time', {})
    if t.get('simulator_steps') != steps or not math.isclose(t.get('simulator_time', -1.), (steps or 0) / 30, rel_tol=1e-9):
        problems.append(f'time block {t} for {steps} steps')
    if not isinstance(row.get('success'), bool) or row.get('task') != task:
        problems.append('success or task field')
    per_episode = folder / 'json' / f'{task}_{row.get("instance_id")}_0.json'
    if not per_episode.exists() or json.loads(per_episode.read_text()) != row:
        problems.append('per-episode json differs from the summary row')
    if not list((folder / 'videos').glob('*.mp4')):
        problems.append('no video')
    return dict(ok=not problems, problems=problems, steps=steps, success=row.get('success'),
                q=row.get('q_score', {}).get('final'), simulator_time=t.get('simulator_time'))
