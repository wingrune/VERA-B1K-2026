"""Adapt the frozen task-56 M15 Policy API for the common classical backend.

Runtime code and static priors are byte-identical to the evaluated M15 controller. Its onboard-observation
whitelist and action checks remain active.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[2]


class RoseM15Policy:
    def __init__(self, task: str) -> None:
        if task != 'make_rose_centerpieces':
            raise ValueError(f'M15 does not serve {task!r}')
        folder = REPO / 'tasks' / task
        spec = json.loads((folder / 'solution.json').read_text())
        package = folder / spec['entry']['code']
        hashes = {
            str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in package.rglob('*')
            if path.is_file() and '__pycache__' not in path.relative_to(package).parts and path.name != '.DS_Store'
        }
        fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        if fingerprint != spec['source']['fingerprint'] or len(hashes) != spec['source']['files']:
            raise ValueError('Frozen M15 package differs from its imported source fingerprint')
        out = Path(os.environ.get('B1K_CLASSICAL_OUTPUT') or os.environ.get('ROSE_M15_OUTPUT')
                   or tempfile.mkdtemp(prefix='rose_m15_'))
        sys.path.insert(0, str(package))
        from policy import Policy
        self.policy = Policy(output_dir=out, variant='h32', backend='cuda')
        self.policy.set_action_dim(23)
        atexit.register(self.policy.close)
        logging.info('Loaded frozen M15/h32: %s files; trace: %s', len(hashes), out)

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs: dict) -> np.ndarray:
        # Pass the observation unchanged. The frozen adapter validates its keys,
        # shapes and dtypes, excludes task_id, and accepts only one environment.
        action = self.policy.forward(obs)
        return action.detach().cpu().numpy()
