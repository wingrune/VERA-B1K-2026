"""Composting waste: whitelist RGB-D/proprio/robot-relative camera extrinsics only."""
import os,json,time
from pathlib import Path
import numpy as np
import torch
from scene_map import SceneMap
from compost_skills import Context
from compost_perception import Observer,Estimator
from compost_pipeline import task

KEYS=['robot_r1::proprio','robot_r1::cam_rel_poses']+[f'robot_r1::robot_r1:{link}:Camera:0::{kind}'
      for link in ('zed_link','left_realsense_link','right_realsense_link') for kind in ('rgb','depth_linear')]

class Policy:
    def __init__(self):
        self.root=Path(__file__).parent
        self.output=Path(os.environ.get('POLICY_OUTPUT','/tmp/compost_policy'));self.output.mkdir(parents=True,exist_ok=True)
        self.scene=SceneMap(self.root/'priors/scene_map.npz');self.reset()
    def set_action_dim(self,n):
        if n!=23:raise ValueError('Expected native 23-D action space')
    def reset(self):
        self.ob=Observer(self.root,self.scene);self.est=Estimator(self.ob,self.scene)
        self.ctx=Context(self.root,self.scene,self.est,log=self.log)
        self.script=task(self.ctx,self.est);self.last=np.zeros(23);self.steps=0;self.finished=None;self.unknown_floor=[]
    def log(self,*args):
        if args and args[0]=='EVENT':
            with (self.output/'trace.jsonl').open('a') as f:f.write(json.dumps(args[1],default=str)+'\n')
        print(*args,flush=True)
    def checkpoint(self,data):
        """Failure forensics from permitted inputs and policy beliefs only."""
        ctx=self.ctx
        state=dict(arm=ctx.arm,held=ctx.held,grip=ctx.grip,trunk_command=ctx.trunk_command,
                   arm_command=ctx.arm_command,estimate=None if self.ob.loc.pose is None else list(self.est.base()),steps=self.steps,
                   result=self.finished,
                   observed_guard=np.asarray(ctx.guard.boxes if hasattr(ctx.guard,'boxes') else ctx.guard),
                   guard_frame=getattr(ctx.guard,'frame',None),unknown_floor=self.unknown_floor,
                   tracks={} if self.ob.loc.pose is None else {k:self.est.object(k,max_age=20000) for k in self.est.tracks})
        target=self.output/'finished_checkpoint'
        np.savez_compressed(str(target)+'_rgbd.npz',**data)
        Path(str(target)+'_context.json').write_text(json.dumps(state,default=lambda x:x.tolist(),indent=2))
    def action(self,obs):
        data={}
        for k in KEYS:
            value=obs[k]
            if hasattr(value,'detach'):value=value.detach().cpu().numpy()
            value=np.asarray(value)
            expected=1 if k.endswith(('proprio','cam_rel_poses')) else 2 if k.endswith('depth_linear') else 3
            if value.ndim==expected+1:value=value[0]
            if value.ndim!=expected:raise ValueError(f'Bad observation shape for {k}: {value.shape}')
            data[k]=value
        payload=None
        if self.ctx.held is not None:
            arm,category,relation=self.ctx.held;bounds=self.ctx.models[category]['bounds']
            axes=[np.linspace(a,b,3 if i<2 else 7) for i,(a,b) in enumerate(zip(*bounds))]
            points=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
            in_link=points@relation[:3,:3].T+relation[:3,3]+self.ctx.iks[arm].tool
            payload=(self.ctx.iks[arm].link,in_link,.07)
        self.steps+=1;self.ob.frame(data,payload);self.ctx.update(data['robot_r1::proprio'])
        self.est.predict(self.last,data['robot_r1::proprio'])
        # A stationary base must not walk through the map while the trunk/arms change its view.
        moving=np.linalg.norm(data['robot_r1::proprio'][:2])>.025 or abs(float(self.last[2]))>.06
        if self.ob.loc.pose is not None and moving and self.steps%15==0:self.ob.loc.correct(self.ob.scan())
        if self.ob.loc.pose is not None and self.steps%30==0 and self.est.object('ashcan',max_age=20000) is None:
            from floor_obstacles import observe
            boxes=observe(self.ob,self.ctx.base_matrix())
            for box in boxes:
                middle=(box[:3]+box[3:])/2
                for n,old in enumerate(self.unknown_floor):
                    if np.linalg.norm(middle[:2]-(old[:2]+old[3:5])/2)<.25:
                        self.unknown_floor[n]=np.r_[np.minimum(old[:3],box[:3]),np.maximum(old[3:],box[3:])];break
                else:self.unknown_floor.append(box)
            if self.unknown_floor:self.ctx.guard=np.array(self.unknown_floor[-8:])
        if self.finished is None:
            started=time.monotonic()
            try:action=next(self.script)
            except StopIteration as done:
                self.finished=str(done.value);self.ctx.event('finished',result=self.finished);action=self.ctx.action()
                try:self.checkpoint(data)
                except OSError as error:self.ctx.event('checkpoint_error',reason=str(error))
            elapsed=time.monotonic()-started
            if elapsed>2.:self.ctx.event('planning_latency',seconds=elapsed)
        else:action=self.ctx.action()
        action=np.asarray(action,dtype=np.float32)
        if action.shape!=(23,) or not np.isfinite(action).all():raise ValueError('Invalid action')
        self.last=action.copy();return action
    def forward(self,obs):return torch.as_tensor(self.action(obs),dtype=torch.float32).reshape(1,23)
    def act(self,obs):return self.forward(obs)
