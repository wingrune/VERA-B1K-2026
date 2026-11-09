"""RGB-D-only room exploration diagnostic; no task-state or asset queries."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from perception import make_cameras
from navigation import DepthMap,rot2
from ground_frame import GroundFrameEstimator,level_cameras
from visual_odometry import VisualOdometry
from active_search import ActiveSearch
from controller import ArmServo
from robot_collision import RobotCollision
from scene_collision import ObservedSceneCollision

class ScanPolicy:
    def __init__(self):
        self.output=Path(os.environ['POLICY_OUTPUT']);self.output.mkdir(parents=True,exist_ok=True)
        self.servo=ArmServo(Path(__file__).parent)
        self.robot=RobotCollision(Path(__file__).parent/'robot_collision_spheres.json')
        self.reset()

    def set_action_dim(self,n):
        assert n==23

    def reset(self):
        self.step=0;self.age=0;self.odom=np.zeros(3);self.ground=GroundFrameEstimator();self.vo=VisualOdometry()
        self.map=DepthMap();self.map.n=384;self.map.log_odds=np.zeros((384,384,self.map.nz),np.int8)
        self.searcher=ActiveSearch(global_frontiers=True);self.moving=False;self.holds=0;self.phase='scan'

    def forward(self,obs):
        data={k:v[0].detach().cpu().numpy() for k,v in obs.items() if k in ('robot_r1::proprio','robot_r1::cam_rel_poses') or k.endswith(('::rgb','::depth_linear'))}
        s=data['robot_r1::proprio'];q=self.servo.q(s);self.step+=1;self.age+=1
        self.odom[:2]+=rot2(self.odom[2])@s[:2]/30;self.odom[2]+=s[2]/30
        cameras=make_cameras(data,size=320)
        level=self.ground.update(cameras['head']) if self.step%10==1 else self.ground.transform
        cams=level_cameras(cameras,level)
        if self.step%5==1:self.odom=self.vo.update(cams['head'],self.step,self.odom)
        if self.step%10==1:
            capsules=[(level[:3,:3]@a+level[:3,3],level[:3,:3]@b+level[:3,3],r) for a,b,r in self.servo.capsules(q)]
            self.map.update(cams,self.odom,capsules)
        action=np.r_[np.zeros(3),s[53:57],s[3:10],1.,s[28:35],1.]
        if self.moving:
            if self.searcher.step(self.map,self.odom,action):self.moving=False;self.age=0;self.phase='scan'
        elif self.age<540 if self.step<=540 else self.age<120:
            action[2]=.34
        elif self.searcher.plan(self.map,self.odom,'scan'):
            self.moving=True;self.age=0;self.phase='explore'
        else:
            action[2]=.25
        # Gate the whole robot's next swept base transform using current depth.
        scene=ObservedSceneCollision(cameras,self.robot,self.servo.kin.forward(q)[0],None,[-.02]*3,[.02]*3)
        poses=self.servo.kin.forward(q)[0]
        from scipy.spatial.transform import Rotation as R
        transform=np.eye(4);transform[:2,3]=action[:2]/30;transform[:3,:3]=R.from_euler('z',action[2]/30).as_matrix()
        if not scene.valid({k:transform@v for k,v in poses.items()},None):action[:3]=0;self.holds+=1
        if self.step==1 or self.step%90==0:np.savez_compressed(self.output/f'observation_{self.step:05d}.npz',**data)
        if self.step%30==0:
            row=dict(step=self.step,phase=self.phase,age=self.age,odom=self.odom.tolist(),search=self.searcher.debug,
                live_clearance_holds=self.holds,ground=self.ground.stats,visual_odometry=self.vo.stats,
                map=self.map.last_update,action_base=action[:3].tolist())
            with (self.output/'trace.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return torch.as_tensor(action[None],dtype=torch.float32)
