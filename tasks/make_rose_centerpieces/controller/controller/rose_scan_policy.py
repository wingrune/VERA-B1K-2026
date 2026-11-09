"""Allowed-observation perception/navigation diagnostic, without manipulation."""
import json
import numpy as np
from scan_policy import ScanPolicy
from rose_perception import RosePerception
from perception import make_cameras

class RoseScanPolicy(ScanPolicy):
    def __init__(self):
        super().__init__()
        self.perception=RosePerception(__import__('pathlib').Path(__file__).parent/'priors')

    def forward(self,obs):
        action=super().forward(obs)
        if self.step%30==1:
            data={k:v[0].detach().cpu().numpy() for k,v in obs.items() if k in ('robot_r1::proprio','robot_r1::cam_rel_poses') or k.endswith(('::rgb','::depth_linear'))}
            cameras=make_cameras(data,size=None);self.perception.level=self.ground.transform
            detections={}
            for cat in ('rose','vase','coffee_table'):
                result=self.perception.fit(cat,cameras)
                if result is not None:
                    detections[cat]=dict(body_pose=result[0].tolist(),**result[1])
            with (self.output/'perception.jsonl').open('a') as f:f.write(json.dumps(dict(step=self.step,detections=detections,rose_candidates=[dict(body_pose=p.tolist(),**s) for p,s in self.perception.candidates]))+'\n')
        return action
