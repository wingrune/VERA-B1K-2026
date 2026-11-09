"""Policy entry point for the native evaluator: putting away the Halloween decorations, observation-based.

Runtime inputs: head RGB-D (depth for everything; colour for one thing, a pumpkin told from a candle), proprioception,
robot-relative camera calibration. Static priors: the robot model, a map of the house's lower floor (the same scene in
every instance) and the cabinet's and the coffee table's geometry. No simulator state, segmentation,
contacts or evaluator results are read.
"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from scene_map import SceneMap
from manip import Context
from observer import Observer
from car_observed import Runtime
from decor_observed import DecorEstimator, observed_pipeline

KEYS = ['robot_r1::proprio', 'robot_r1::cam_rel_poses'] + [f'robot_r1::robot_r1:{link}:Camera:0::{kind}' for link in ('zed_link', 'left_realsense_link', 'right_realsense_link') for kind in ('rgb', 'depth_linear')]


class Policy:
    def __init__(self):
        self.root = Path(__file__).parent
        self.output = Path(os.environ.get('POLICY_OUTPUT', '.')); self.output.mkdir(parents=True, exist_ok=True)
        self.scene = SceneMap(self.root / 'priors' / 'house_map.npz')
        self.reset()

    def set_action_dim(self, n):
        assert n == 23

    def reset(self):
        self.step = 0
        self.observer = Observer(self.root, self.scene); self.estimator = DecorEstimator(self.observer, self.scene, self.root)
        self.context = Context(self.root, self.scene, self.estimator, log=self._log)
        self.runtime = Runtime(self.context, self.observer, self.estimator, observed_pipeline)

    def _log(self, *args):
        if args and args[0] == 'EVENT':
            with (self.output / 'trace.jsonl').open('a') as f:
                f.write(json.dumps(args[1], default=str) + '\n')
        print(*args, flush=True)

    def forward(self, obs):
        self.step += 1
        data = {k: obs[k][0].detach().cpu().numpy() for k in KEYS}
        action = self.runtime.step(data)
        if self.runtime.finished is not None and self.step % 300 == 0:
            self._log('EVENT', dict(event='idle', step=self.step, script=self.runtime.finished))
        if self.step % 10 == 0 and self.observer.loc.pose is not None:
            # What the controller believes, for drawing the episode afterwards (dev/render_videos.py): its pose in
            # the map, the things it knows and what its hands hold.
            est = self.estimator
            row = dict(step=self.step, pose=[round(float(v), 3) for v in est.base()], holding={arm: str(v) for arm, v in getattr(self.context, 'holding', {}).items()},
                       drawers_out=list(getattr(est, 'drawers_out', [])), things=[[item['kind']] + [round(float(v), 3) for v in item['xy']] for item in est.known()])
            with (self.output / 'poses.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
        assert np.isfinite(action).all()
        return torch.as_tensor(action, dtype=torch.float32)[None]
