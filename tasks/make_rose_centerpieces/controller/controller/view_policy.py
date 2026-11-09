"""Diagnostic: free-wrist floor observation and RGB-D search, no task attempts."""
import numpy as np
import torch
from compost_policy import Policy
from perception import make_cameras
from wrist_view import plan_floor_view
from scene_collision import ObservedSceneCollision
class ViewPolicy(Policy):
    def reset(self):
        super().reset();self.view_goal=None;self.view_attempted=False
    def plan_navigation(self,*args,**kwargs):return False
    def plan_inspection(self):return False
    def forward(self,obs):
        result=super().forward(obs);action=result[0].detach().cpu().numpy().copy()
        data={k:v[0].detach().cpu().numpy() for k,v in obs.items() if k in ('robot_r1::proprio','robot_r1::cam_rel_poses') or k.endswith(('::rgb','::depth_linear'))}
        s=data['robot_r1::proprio'];q=self.servo.q(s);cams=make_cameras(data,size=480)
        for c in cams.values():c['ground_from_body']=self.level.copy()
        if not self.view_attempted:
            self.view_attempted=True;self.view_diagnostics=[];self.view_goal=plan_floor_view(self.root,q,cams,self.view_diagnostics)
        if self.view_goal is not None:
            target=s[28:35]+np.clip(self.view_goal-s[28:35],-.014,.014);trial=q.copy();ids=[self.servo.names.index(f'right_arm_joint{i}') for i in range(1,8)];trial[ids]=target
            scene=ObservedSceneCollision(cams,self.robot,self.servo.kin.forward(q)[0],None,[0,0,0],[0,0,0])
            if scene.valid(self.servo.kin.forward(trial)[0],None):action[15:22]=target
            if np.max(abs(self.view_goal-s[28:35]))>.04:action[:3]=0
        self.action=action
        return torch.as_tensor(action[None],dtype=torch.float32)
