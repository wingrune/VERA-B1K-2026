"""Policy server of one classical controller of the b1k_rl_guys repository (tasks/<task>/controller/): RGB-D + proprio in,
23-d R1Pro action out.

usage: python solutions/b1k_classical/backend.py --task clean_a_trumpet --port 8101

The controller runs as under the native evaluator in b1k_rl_guys (its run_native.py): `observation_policy.Policy()` from the
controller folder, `reset()` at an episode's start, `forward(obs)` per step on the evaluator's observation dict as torch
tensors with the env batch dimension (only --num-envs 1), returning a (1, 23) action. It reads RGB, depth, proprioception and
the robot-relative camera poses only, plus the static priors in its own folder (robot model, a map of the task's scene, object
models). Its event trace goes to $B1K_CLASSICAL_OUTPUT (default: a temporary folder).

Controllers with a different Policy API select an adapter through solution.json entry.adapter.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from server.protocol import serve  # noqa: E402


class ClassicalPolicy:
    def __init__(self, task: str) -> None:
        folder = REPO / 'tasks' / task
        spec = json.loads((folder / 'solution.json').read_text())
        code = folder / spec['entry']['code']
        out = Path(os.environ.get('B1K_CLASSICAL_OUTPUT') or tempfile.mkdtemp(prefix=f'b1k_classical_{task}_'))
        os.environ['POLICY_OUTPUT'] = str(out / 'policy'); os.environ['TASK_NAME'] = task
        os.environ.update(spec['entry'].get('env', {}))
        sys.path.insert(0, str(code))
        import torch
        self.torch = torch
        from observation_policy import Policy           # the controller's own entry point (one controller per process)
        self.policy = Policy()
        if hasattr(self.policy, 'set_action_dim'):
            self.policy.set_action_dim(23)
        self.ready = True                               # (Policy() starts with a fresh episode)
        logging.info('classical controller %s (%s) loaded from %s, trace in %s', task, spec.get('candidate'), code, out)

    def reset(self) -> None:
        if not self.ready:
            self.policy.reset()
        self.ready = False

    def act(self, obs: dict) -> np.ndarray:
        batch = np.shape(obs['robot_r1::proprio'])[0]
        if batch != 1:
            raise ValueError(f'classical controllers serve one environment, got a batch of {batch}: run the evaluator with --num-envs 1')
        self.ready = False
        tensors = {k: self.torch.as_tensor(np.array(v)) for k, v in obs.items() if not isinstance(v, (str, bytes))}
        action = self.policy.forward(tensors)
        return np.asarray(action.detach().cpu().numpy() if hasattr(action, 'detach') else action, dtype=np.float32).reshape(1, -1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--port', type=int, required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s classical[%(process)d] %(message)s')
    spec = json.loads((REPO / 'tasks' / a.task / 'solution.json').read_text())
    adapter = spec['entry'].get('adapter', 'observation_policy')
    if adapter == 'rose_m15':
        from solutions.b1k_classical.rose_m15 import RoseM15Policy
        policy = RoseM15Policy(a.task)
    elif adapter == 'observation_policy':
        policy = ClassicalPolicy(a.task)
    else:
        raise ValueError(f'Unknown classical Policy adapter: {adapter!r}')
    serve(policy, a.port)


if __name__ == '__main__':
    main()
