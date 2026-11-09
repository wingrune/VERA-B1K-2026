"""Development probe: hold the arms, turn the base in place, save the allowed observations every 15 steps."""
import json
import os
from pathlib import Path

import numpy as np
import torch


def allowed(obs):                                                                             # the closed input set: proprio, camera poses, RGB-D
    return {k: v.detach().cpu().numpy() for k, v in obs.items()
            if k == "robot_r1::proprio" or k == "robot_r1::cam_rel_poses" or (k.startswith("robot_r1::robot_r1:") and k.endswith(("::rgb", "::depth_linear")))}


class ScanProbe:
    turn = 0.4                                                                                # rad/s, input [-1, 1] maps to [-1, 1] rad/s

    def __init__(self):
        self.output = Path(os.environ["POLICY_OUTPUT"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode = -1

    def set_action_dim(self, n):
        assert n == 23

    def reset(self):
        self.episode += 1
        self.step = 0
        self.hold = None

    def forward(self, obs):
        data = allowed(obs)
        s = data["robot_r1::proprio"].reshape(-1)                                          # [61]
        if self.step == 0:
            (self.output / "obs_spec.json").write_text(json.dumps({k: [list(v.shape), str(v.dtype)] for k, v in data.items()}, indent=1))
        if self.hold is None:
            self.hold = np.r_[np.zeros(3), s[53:57], s[3:10], 1.0, s[28:35], 1.0]           # [23] trunk, left arm, left grip, right arm, right grip
        if self.step % 15 == 0:
            np.savez_compressed(self.output / f"step_{self.step:05d}.npz", **data)
        action = self.hold.copy()
        action[2] = self.turn
        self.step += 1
        return torch.as_tensor(action[None, :], dtype=torch.float32)
