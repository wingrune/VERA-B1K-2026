"""Finite held-rose view recovery from current RGB-D and public kinematics."""
import time
import numpy as np
from scene_collision import ObservedSceneCollision

OFFSETS=((0.,.15,0.),(-.10,.20,0.),(0.,.20,0.),(-.10,.15,0.))

def arm_ray_exposure(p,q,vase,camera):
    points=p.perception.models['vase']['xyz'][::5]@vase[:3,:3].T+vase[:3,3]
    origin=camera['t'];rays=points-origin;length=np.linalg.norm(rays,axis=1)
    direction=rays/np.maximum(length[:,None],1e-8)
    ids=[i for i,n in enumerate(p.robot.names) if n.startswith(('left_arm','left_gripper','left_realsense'))]
    delta=p.robot.centers(p.servo.kin.forward(q)[0])[ids]-origin
    along=direction@delta.T
    distance=np.linalg.norm(delta[None]-along[:,:,None]*direction[:,None],axis=2)
    blocked=((distance<p.robot.radii[ids][None])&(along>0)&(along<length[:,None])).any(axis=1)
    return float((~blocked).mean())

def plan_view(p,seconds=5.):
    q=p.current_q;hand=p.servo.pose(q);vase=np.linalg.inv(p.base())@p.poses['vase']
    camera=p.current_cameras['head'];current=arm_ray_exposure(p,q,vase,camera)
    scene=ObservedSceneCollision(p.current_cameras,p.robot,p.servo.kin.forward(q)[0],hand@p.held,*p.bounds('rose'))
    deadline=time.monotonic()+seconds;choices=[];trials=[]
    for offset in OFFSETS:
        if time.monotonic()>deadline:break
        endpoint=p.servo.solve(q,hand[:3,3]+offset,hand[:3,:3],locked={i:q[p.servo.indices[i]] for i in range(4)},validator=lambda trial:scene.valid(p.servo.kin.forward(trial)[0],p.servo.pose(trial)@p.held))
        row={'offset':offset,'endpoint_ik':endpoint is not None};trials.append(row)
        if endpoint is None:continue
        target=q.copy();target[p.servo.indices]=endpoint
        row['checked101_sweep']=all(scene.valid(p.servo.kin.forward(q+a*(target-q))[0],p.servo.pose(q+a*(target-q))@p.held) for a in np.linspace(0,1,101))
        if not row['checked101_sweep']:continue
        row['arm_ray_clear_fraction']=arm_ray_exposure(p,target,vase,camera)
        gain=row['arm_ray_clear_fraction']-current
        if gain<.15:continue
        cost=np.linalg.norm(endpoint-q[p.servo.indices]);choices.append((gain-.05*cost,endpoint,row))
    debug={'reason':'checked_view_found' if choices else 'no_checked_view','current_arm_ray_clear_fraction':current,'trials':trials,'future_depth_not_generated':True,'freshness_not_changed':True}
    if not choices:return None,debug
    _,endpoint,row=max(choices,key=lambda x:x[0]);debug['selected']=row
    return endpoint,debug

class HeldVaseView:
    def __init__(self,p):self.p=p;self.active=False;self.attempted=set();self.debug={}
    def maybe_start(self,s):
        p=self.p;key=len(p.rose_releases)
        if self.active or key in self.attempted or p.child!='rose' or p.parent!='vase' or p.held is None:return False
        if p.dual.active or p.dual.right_held is not None or getattr(p,'paired',None) is not None and p.paired.active:return False
        if p.phase!='target_navigate' or p.age<100 or p.step-p.seen.get('vase',-1000)<=90:return False
        if p.pose_route is None or p.pose_index<len(p.pose_route) or np.linalg.norm(s[:3])>=.015:return False
        if p.step-p.seen.get('rose',-1000)>45:return False
        self.attempted.add(key);self.target,self.debug=plan_view(p)
        if self.target is None:return False
        self.anchor_trunk=p.current_q[p.servo.indices[:4]].copy();self.command=np.r_[p.action[3:7],p.action[7:14]].copy();self.started=p.step;self.settled=None;self.active=True;p.change('held_vase_view');return True
    def fail(self,reason):self.debug=dict(self.debug,failure=reason);self.p.change('done_hold');self.failed=True
    def tick(self,action,s):
        if not self.active:return action
        p=self.p;action=np.asarray(action).copy();action[:3]=0.;action[14]=-1.;action[22]=1.
        if getattr(self,'failed',False):
            action[3:7]=s[53:57];action[7:14]=s[3:10];action[15:22]=s[28:35];return action
        measured=p.current_q[p.servo.indices]
        previous=np.clip(self.command,measured-.12,measured+.12);advance=self.target-previous
        self.command=previous+advance*min(1.,.024/max(np.max(abs(advance)),1e-9));self.command[:4]=self.anchor_trunk
        action[3:7]=self.command[:4];action[7:14]=self.command[4:];action[15:22]=s[28:35]
        if p.step-p.seen.get('rose',-1000)>60:self.fail('held_rose_not_fresh_during_view');return self.tick(action,s)
        if np.max(abs(self.target-measured))<.035:
            if self.settled is None:self.settled=p.step
            if p.step-p.seen.get('vase',-1000)<=45:
                self.debug=dict(self.debug,allowed_vase_fit_step=p.seen['vase']);self.active=False;p.change('target_navigate')
            elif p.step-self.settled>90:self.fail('view_pose_did_not_reacquire_vase')
        elif p.step-self.started>240:self.fail('view_motor_convergence_timeout')
        return action
