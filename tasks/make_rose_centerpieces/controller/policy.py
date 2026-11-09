"""BEHAVIOR Policy API for exact preserved rose controller/helper sources."""
import importlib
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from configure import ROOT, activate_imports, profile_scope

class Policy:
    def __init__(self, output_dir=None, variant='h30', backend='cuda'):
        output = output_dir or os.environ.get('POLICY_OUTPUT')
        if not output:
            raise ValueError('Provide output_dir or POLICY_OUTPUT')
        self.output = Path(output).resolve()
        if self.output == ROOT or self.output.is_relative_to(ROOT):
            raise ValueError('Use an output directory outside the immutable package')
        self.variant, self.backend = variant, backend
        activate_imports(variant)
        bridge = importlib.import_module('tasks.56_make_rose_centerpieces.evo.bridge')
        with profile_scope(self.output, variant, backend):
            self.planner = bridge.Planner(ROOT / 'controller', self.output, ROOT / 'candidates' / variant)

    def set_action_dim(self, n):
        if n != 23:
            raise ValueError('R1 Pro requires 23D native actions')

    def reset(self):
        with profile_scope(self.output, self.variant, self.backend):
            self.planner.reset()

    def forward(self, observation):
        import torch
        with profile_scope(self.output, self.variant, self.backend):
            return torch.from_numpy(self.planner.act(observation))

    def close(self):
        self.planner.close()
