"""Run the 2025 BEHAVIOR Challenge winner (Larchenko, Pi0.5-based, checkpoints 1-4 of IliaLarchenko/behavior_submission)
under the 2026 evaluator.

What differs between the 2025 setup the checkpoints were trained for and the 2026 evaluator, and what this adapter does:

- Task id. The 2026 evaluator puts its own index (B100_task_misc.csv, 100 tasks) into obs["task_id"]; the checkpoints embed
  the 2025 index (50 tasks). The ids are remapped by task name (task_ids_2025.json vs server/task_ids_2026.json; for the 50
  shared tasks both tables currently agree, the remap keeps that an explicit check). Tasks new in 2026 have no embedding and
  are refused.
- Stage counts. The upstream wrapper votes stages against the 2026 stage table; the 2025 checkpoints use TASK_NUM_STAGES_2025
  (it differs, e.g. can_meat 14 vs 13), so the wrapper's table is swapped for the 2025 one.
- Model config. pi_behavior_2025_submission: the pi_behavior_b1k_fast recipe with the 50-task embedding tables.
- Actions. convert_2025_actions=True (23-d base, trunk, left arm, left gripper, right arm, right gripper; checked identical
  on demo frames of both years, see upstream/src/b1k/shared/convert_2025_action.py).
- Proprio and images. The 2026 compact 61-d proprio is mapped to the 23-d model state by upstream b1k_proprio; RGB of the
  RGBDFullResWrapper cameras (720 / 480 px) is resized with padding to 224 as in 2025; depth is not used.
- Batch. The upstream wrapper keeps one episode state: only --num-envs 1.

Known gap: in the 2025 data the base velocity in the state was in the world frame, 2026 proprio gives it in the robot frame.
The world yaw is privileged in 2026, so the state keeps the robot-frame velocity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASKS_2025 = json.loads((HERE / 'task_ids_2025.json').read_text())
TASKS_2026 = json.loads((HERE.parents[1] / 'server' / 'task_ids_2026.json').read_text())
ID_2026_TO_2025 = {TASKS_2026[name]: idx for name, idx in TASKS_2025.items()}
NAME_2026 = {idx: name for name, idx in TASKS_2026.items()}
CONFIG_2025 = 'pi_behavior_2025_submission'


def task_2025(task_id_2026: int) -> int:
    if task_id_2026 not in ID_2026_TO_2025:
        raise ValueError(f'task {NAME_2026.get(task_id_2026, task_id_2026)} (2026 id {task_id_2026}) has no 2025 embedding')
    return ID_2026_TO_2025[task_id_2026]


class Adapter2026:
    """reset() / act(obs) -> (1, 23) float32 around the upstream B1KPolicyWrapper."""

    def __init__(self, wrapper) -> None:
        self.wrapper = wrapper

    def reset(self) -> None:
        self.wrapper.reset()

    def act(self, obs: dict) -> np.ndarray:
        batch = np.shape(obs['robot_r1::proprio'])[0]
        if batch != 1:
            raise ValueError(f'the VLA backend serves one environment, got a batch of {batch}: run the evaluator with --num-envs 1')
        obs = {k: v for k, v in obs.items() if not k.endswith('::depth_linear')}
        obs['task_id'] = np.array([task_2025(int(np.asarray(obs['task_id']).reshape(-1)[0]))], dtype=np.int64)
        action = self.wrapper.act(obs)
        return np.asarray(action.detach().cpu().numpy() if hasattr(action, 'detach') else action, np.float32).reshape(1, -1)


def build(checkpoint: str, num_steps: int = 20) -> Adapter2026:
    """Load one checkpoint as the 2025 submission served it (wrapper defaults of scripts/serve_b1k.py)."""
    from b1k.models import pi_behavior_config
    from b1k.policies import policy_config
    from b1k.shared import eval_b1k_wrapper
    from b1k.training import config as train_config

    eval_b1k_wrapper.TASK_NUM_STAGES = pi_behavior_config.TASK_NUM_STAGES_2025
    policy = policy_config.create_trained_policy(train_config.get_config(CONFIG_2025), checkpoint, sample_kwargs={'num_steps': num_steps})
    wrapper = eval_b1k_wrapper.B1KPolicyWrapper(
        policy, config=eval_b1k_wrapper.B1KWrapperConfig(num_steps=num_steps, convert_2025_actions=True))
    return Adapter2026(wrapper)
