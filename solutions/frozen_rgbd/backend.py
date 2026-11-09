"""Transport for five unchanged classical RGB-D policies in their evaluated CPU runtime.

The router gives each task/episode a fresh process. Only the policy's KEYS enter
action(); the Cans adapter removes the single environment axis, as evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from server.protocol import serve


def verify_core(task: str) -> tuple[Path, dict]:
    folder = REPO / 'tasks' / task
    spec = json.loads((folder / 'solution.json').read_text())
    if spec.get('kind') != 'frozen_rgbd':
        raise ValueError(f'{task} is not a frozen_rgbd policy')
    root = folder / spec['entry']['code']
    manifest = json.loads((folder / 'fingerprint.json').read_text())
    actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    # Keep the exact compact-JSON fingerprint convention of the evaluated packages.
    digest = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if actual != manifest['files'] or digest != manifest['sha256'] or digest != spec['source']['fingerprint']:
        raise RuntimeError(f'Frozen {task} controller fingerprint mismatch')
    return root, spec


class FrozenRGBDPolicy:
    def __init__(self, task: str):
        root, self.spec = verify_core(task)
        self.task = task
        sys.dont_write_bytecode = True
        loaded = sys.modules.get('observation_policy')
        if loaded is not None and Path(loaded.__file__).resolve().parent != root.resolve():
            raise RuntimeError('Different frozen policy loaded; use a fresh backend process')
        if not os.environ.get('POLICY_OUTPUT'):
            os.environ['POLICY_OUTPUT'] = tempfile.mkdtemp(prefix=f'frozen_rgbd_{task}_')
        sys.path.insert(0, str(root))
        from observation_policy import Policy, KEYS
        self.keys = tuple(KEYS)
        self.policy = Policy()

    def reset(self):
        self.policy.reset()

    def act(self, obs: dict) -> np.ndarray:
        if np.shape(obs.get('robot_r1::proprio')) != (1, 61):
            raise ValueError('Frozen RGB-D requires --num-envs 1 and R1Pro proprio shape (1, 61)')
        if np.size(obs.get('task_id')) != 1 or int(np.asarray(obs['task_id']).item()) != self.spec['task_id_2026']:
            raise ValueError(f"Requires task_id={self.spec['task_id_2026']}")
        data = {k: obs[k] for k in self.keys}
        if self.spec['entry'].get('unbatch_observations', False):
            data = {k: np.asarray(v) for k, v in data.items()}
            if any(v.ndim == 0 or v.shape[0] != 1 for v in data.values()):
                raise ValueError('Expected one environment for every observation')
            data = {k: v[0] for k, v in data.items()}
        action = np.asarray(self.policy.action(data), dtype=np.float32)
        if action.shape != (23,) or not np.isfinite(action).all():
            raise ValueError('Invalid frozen controller action')
        return action.reshape(1, 23)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--port', type=int, required=True)
    a = p.parse_args()
    policy = FrozenRGBDPolicy(a.task)
    serve(policy, a.port, metadata={'policy': a.task, 'candidate': policy.spec['candidate'], 'action_dim': 23})


if __name__ == '__main__':
    main()
