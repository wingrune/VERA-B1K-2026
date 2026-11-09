"""Reset-only diagnostic. No simulator state is available to this policy."""
import os
from pathlib import Path
import numpy as np
import torch

class ResetProbe:
    def __init__(self):
        self.output = Path(os.environ['POLICY_OUTPUT'])
        self.output.mkdir(parents=True, exist_ok=True)

    def set_action_dim(self, n):
        assert n == 23, n

    def reset(self):
        self.step = 0

    def forward(self, obs):
        allowed = {k: v[0].detach().cpu().numpy() for k, v in obs.items()
                   if k in ('robot_r1::proprio', 'robot_r1::cam_rel_poses')
                   or k.endswith(('::rgb', '::depth_linear'))}
        if self.step == 0:
            np.savez_compressed(self.output / 'reset_observation.npz', **allowed)
        s = allowed['robot_r1::proprio']
        assert s.shape == (61,) and np.isfinite(s).all(), s.shape
        action = np.r_[np.zeros(3), s[53:57], s[3:10], 1., s[28:35], 1.]
        assert action.shape == (23,) and np.isfinite(action).all()
        self.step += 1
        return torch.as_tensor(action[None], dtype=torch.float32)
