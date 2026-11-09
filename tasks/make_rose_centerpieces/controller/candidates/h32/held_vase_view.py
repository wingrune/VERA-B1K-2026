import numpy as np
from scipy.optimize import least_squares
from .skills import ObservedSceneCollision,transform

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

def plan_left_view(p,seconds=5.):
    q=p.current_q;hand=p.servo.pose(q);vase=np.linalg.inv(p.base())@p.poses['vase']
    camera=p.current_cameras['head'];current=arm_ray_exposure(p,q,vase,camera)
    scene=ObservedSceneCollision(p.current_cameras,p.robot,p.servo.kin.forward(q)[0],hand@p.held,*p.bounds('rose'))
    choices=[];trials=[]
    for offset in OFFSETS:
        if False:break
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

def parent_visibility(p,q,vase,camera):
    table=p.parent=='coffee_table'
    model=p.perception.models[p.parent];xyz=model['xyz'][::32]
    if table:
        lo,hi=model['bounds'];top=float(np.quantile(model['xyz'][:,2],.99))
        xyz=np.array([[x,y,top] for x in np.linspace(lo[0],hi[0],25) for y in np.linspace(lo[1],hi[1],51)])
    points=xyz@vase[:3,:3].T+vase[:3,3];origin=camera['t']
    local=(points-origin)@camera['R'];image=local@camera['K'].T
    uv=image[:,:2]/np.maximum(local[:,2,None],1e-8)
    height,width=camera['rgb'].shape[:2]
    border=0. if table else .05
    inside=(local[:,2]>.06)&(uv[:,0]>border*width)&(uv[:,0]<(1-border)*width)&(uv[:,1]>border*height)&(uv[:,1]<(1-border)*height)
    rays=points-origin;length=np.linalg.norm(rays,axis=1)
    direction=rays/np.maximum(length[:,None],1e-8)
    delta=p.robot.centers(p.servo.kin.forward(q)[0])-origin
    along=direction@delta.T
    distance=np.linalg.norm(delta[None]-along[:,:,None]*direction[:,None],axis=2)
    blocked=((distance<p.robot.radii[None]+.008)&(along>0)&(along<length[:,None])).any(axis=1)
    payload=p.servo.pose(q)@p.held;lo,hi=p.bounds(p.child)
    start=(origin-payload[:3,3])@payload[:3,:3]
    end=(points-payload[:3,3])@payload[:3,:3];segments=end-start
    enter=np.zeros(len(points))-1e9;leave=np.zeros(len(points))+1e9
    outside=np.zeros(len(points),dtype=bool)
    for axis in range(3):
        denominator=segments[:,axis].copy();parallel=abs(denominator)<1e-12
        denominator[parallel]=1.
        a=(lo[axis]-.008-start[axis])/denominator
        b=(hi[axis]+.008-start[axis])/denominator
        near=np.minimum(a,b);far=np.maximum(a,b)
        near[parallel]=-1e9;far[parallel]=1e9
        outside|=parallel&((start[axis]<lo[axis]-.008)|(start[axis]>hi[axis]+.008))
        enter=np.maximum(enter,near);leave=np.minimum(leave,far)
    payload_blocked=(~outside)&(enter<leave)&(leave>0)&(enter<1)
    visible=inside&~blocked&~payload_blocked
    edge=np.minimum(np.minimum(uv[:,0]/width,1-uv[:,0]/width),np.minimum(uv[:,1]/height,1-uv[:,1]/height))
    fraction=float(visible.mean())
    if table:fraction=float(np.min(np.ptp(xyz[visible,:2],axis=0)/(model['bounds'][1]-model['bounds'][0])[:2])) if visible.any() else 0.
    return fraction,float(np.quantile(edge[visible],.05)) if visible.any() else -1.

def plan_view(p,seconds=5.,excluded=()):
    table=p.child=='vase' and p.parent=='coffee_table'
    if 'right_wrist' not in p.current_cameras:return (None,dict(reason='no_right_table_camera')) if table else plan_left_view(p,seconds)
    q=p.current_q;hand=p.servo.pose(q);poses=p.servo.kin.forward(q)[0]
    vase=np.linalg.inv(p.base())@p.poses[p.parent];camera=p.current_cameras['right_wrist']
    scene=ObservedSceneCollision(p.current_cameras,p.robot,poses,hand@p.held,*p.bounds(p.child))
    baseline=p.robot.margins(poses);ids=[11,13,15,17,19,21,23]
    mount=np.linalg.inv(poses['right_gripper_link'])@transform(camera['R'],camera['t'])
    choices=[];trials=[]
    bank=[(a,-.6,-.15,-.7,b,c,-1.4) for a in (.3,.5,.7) for b in (2.15,2.28) for c in (.85,.95)]
    if table:
        bank.extend((a,-.6,-.15,-.7,b,c,d) for a in (.3,.7) for b in (1.2,1.7,2.28) for c in (.2,.6,1.) for d in (-1.4,-.8,0.))
        model=p.perception.models['coffee_table'];lo,hi=model['bounds']
        point=vase[:3,:3]@np.r_[(lo[:2]+hi[:2])/2,float(np.quantile(model['xyz'][:,2],.99))]+vase[:3,3]
        lower=np.array([-4.4505,-3.1415,-2.3561,-2.0943,-2.3561,-1.0471,-1.5707])
        upper=np.array([1.3089,.1744,2.3561,.3490,2.3561,1.0471,1.5707])
        def residual(joints):
            trial=q.copy();trial[ids]=joints
            cam=p.servo.kin.forward(trial)[0]['right_gripper_link']@mount
            direction=point-cam[:3,3];direction/=max(float(np.linalg.norm(direction)),1e-9)
            return np.r_[2.*(cam[:3,2]-direction),.25*(cam[:3,3]-camera['t']),.015*(joints-q[ids])]
        seeds=[q[ids],np.array([.7,-.6,-.15,-.7,2.28,.6,-1.4]),np.array([.3,-.6,-.15,-.7,2.28,.6,-1.4]),np.array([-.7,-1.2,0.,-.7,1.4,.6,0.])]
        for seed in seeds:
            fit=least_squares(residual,np.clip(seed,lower,upper),bounds=(lower,upper),max_nfev=80)
            bank.append(fit['x'])
    for joints in bank:
        joints=np.array(joints)
        if table and any(np.linalg.norm(joints-np.asarray(old))<.10 for old in excluded):continue
        trial=q.copy();trial[ids]=joints
        cam=p.servo.kin.forward(trial)[0]['right_gripper_link']@mount
        fraction,edge=parent_visibility(p,trial,vase,dict(camera,R=cam[:3,:3],t=cam[:3,3]))
        row=dict(right_target=joints.tolist(),visible_fraction=fraction,edge_margin=edge)
        trials.append(row)
        if fraction>(.90 if table else .60):choices.append(((fraction,edge,-float(np.linalg.norm(joints-q[ids]))),trial,row))
    for score,target,row in sorted(choices,key=lambda item:item[0],reverse=True):
        issued=q.copy();issued[p.servo.indices]=np.r_[p.action[3:7],p.action[7:14]];issued[ids]=p.action[15:22]
        issued_target=issued.copy();issued_target[ids]=target[ids]
        path=[]
        for start,end in ((q,target),(issued,issued_target)):
            path.extend((p.servo.kin.forward(start+a*(end-start))[0],p.servo.pose(start+a*(end-start))) for a in np.linspace(0.,1.,101))
        row['measured_and_issued_path_samples']=[101,101]
        row['checked101_sweep']=all(p.robot.valid(tf,baseline) and scene.valid(tf,pose@p.held) for tf,pose in path)
        if not row['checked101_sweep']:continue
        robust=True
        for shift in ((.005,0.,0.),(-.005,0.,0.),(0.,.005,0.),(0.,-.005,0.),(0.,0.,.005),(0.,0.,-.005)):
            held=p.held.copy();held[:3,3]+=shift
            if not all(scene.valid(tf,pose@held) for tf,pose in path):robust=False;break
        row['checked5mm_envelopes']=robust
        if robust:
            debug=dict(reason='checked_parent_right_view',mode='right_wrist_parent',parent_role=p.parent,child_role=p.child,selected=row,trials=trials,
                       right_target=target[ids].tolist(),right_command=p.action[15:22].tolist(),
                       future_depth_not_generated=True,freshness_not_changed=True)
            return q[p.servo.indices].copy(),debug
    if table:return None,dict(reason='no_checked_table_view',right_view_trials=trials,future_depth_not_generated=True,freshness_not_changed=True)
    endpoint,debug=plan_left_view(p,seconds);debug=dict(debug,right_view_trials=trials)
    return endpoint,debug

def table_color(p,pose):
    model=p.perception.models['coffee_table'];lo,hi=model['bounds'];body=np.linalg.inv(p.base())@pose;colors=[]
    for role in p.current_cameras:
        cam=p.current_cameras[role]
        depth=cam['depth'];width=depth.shape[1];index=np.arange(depth.shape[0]*width)[::3];y=index//width;x=index%width;z=depth[y,x]
        valid=np.isfinite(z)&(z>.06)&(z<5);x=x[valid];y=y[valid];z=z[valid];K=cam['K']
        points=np.c_[(x+.5-K[0,2])/K[0,0]*z,(y+.5-K[1,2])/K[1,1]*z,z]@cam['R'].T+cam['t']
        local=(points-body[:3,3])@body[:3,:3]
        top=(abs(local[:,2]-hi[2])<.015)&(local[:,0]>lo[0]-.015)&(local[:,0]<hi[0]+.015)&(local[:,1]>lo[1]-.015)&(local[:,1]<hi[1]+.015)
        colors.extend(cam['rgb'][y[top],x[top],:3].tolist())
    reference=np.median(model['rgb'][model['xyz'][:,2]>np.quantile(model['xyz'][:,2],.95)],axis=0)
    reference=reference/max(float(reference.sum()),1.)
    rgb=np.asarray(colors,dtype=float) if colors else np.zeros((0,3));chroma=rgb/np.maximum(rgb.sum(axis=1)[:,None],1.)
    required=.25*(reference[:2]-reference[2]);warm=(chroma[:,:2]-chroma[:,2,None]>=required).all(axis=1)
    fraction=float(warm.mean()) if len(warm) else 0.
    return len(rgb)>=80 and fraction>=.60,dict(points=len(rgb),warm_fraction=fraction,required=required.tolist(),median_rgb=np.median(rgb,axis=0).tolist() if len(rgb) else [])

class HeldVaseView:
    def __init__(self,p):self.p=p;self.active=False;self.attempted=set();self.debug={};self.anchor=None
    def maybe_start(self,s):
        p=self.p;table=p.child=='vase' and p.parent=='coffee_table'
        self.anchor=None
        if table:
            self.anchor=dict(pose=p.poses['coffee_table'].copy() if 'coffee_table' in p.poses else None,seen=p.seen.get('coffee_table',-1000),phase=p.phase,command=None if p.action is None else p.action.copy())
        key=('table',len(p.rose_releases)) if table else len(p.rose_releases)
        if self.active and self.debug.get('mode')=='right_wrist_parent':
            self.debug['right_command']=p.action[15:22].tolist()
        retry=table and self.debug.get('rejected_view_retry',False)
        if self.active or key in self.attempted and not retry or not (table or p.child=='rose' and p.parent=='vase') or p.held is None or p.parent not in p.poses:return False
        if p.dual.active or p.dual.right_held is not None or getattr(p,'paired',None) is not None and p.paired.active:return False
        if p.phase!='target_navigate' or (not table and p.age<100) or p.step-p.seen.get(p.parent,-1000)<=90:return False
        if p.pose_route is None or p.pose_index<len(p.pose_route) or np.linalg.norm(s[:3])>=.015:return False
        if p.step-p.seen.get(p.child,-1000)>45:return False
        prior=dict(self.debug) if retry else {}
        excluded=prior.get('rejected_views',[])
        history=self.debug.get('table_height_rejections',[])
        self.attempted.add(key);self.target,self.debug=plan_view(p,excluded=excluded)
        if table:self.debug['table_height_rejections']=history
        if table:self.debug.update(table_reference=p.poses[p.parent].tolist(),table_reference_seen=p.seen[p.parent],rejected_views=excluded,table_fit_rejections=prior.get('table_fit_rejections',[]))
        if self.target is None:return False
        self.anchor_trunk=p.current_q[p.servo.indices[:4]].copy();self.command=np.r_[p.action[3:7],p.action[7:14]].copy();self.started=p.step;self.settled=None;self.active=True;p.change('held_vase_view');return True
    def fail(self,reason):self.debug=dict(self.debug,failure=reason);self.p.change('done_hold');self.failed=True
    def tick(self,action,s):
        p=self.p
        if self.anchor is not None and self.anchor['pose'] is None and p.seen.get('coffee_table',-1000)==p.step:
            accepted,row=table_color(p,p.poses['coffee_table']);self.debug['cold_table_color']=dict(row,step=p.step,accepted=accepted)
            if not accepted:
                p.poses.pop('coffee_table',None);p.seen.pop('coffee_table',None);p.change(self.anchor['phase'])
                action=action.copy();action[:3]=0.;action[3:7]=s[53:57];action[7:14]=s[3:10];action[15:22]=s[28:35]
        if not self.active and self.anchor is not None and self.anchor['pose'] is not None and p.seen.get('coffee_table',-1000)>self.anchor['seen']:
            observed=p.poses['coffee_table'];reference=self.anchor['pose']
            height=float(observed[2,3]-reference[2,3])
            if abs(height)>.03:
                self.debug['table_height_rejections']=self.debug.get('table_height_rejections',[])+[dict(step=p.seen['coffee_table'],height_m=height,restored_seen=self.anchor['seen'],before_phase=self.anchor['phase'],proposed_phase=p.phase)]
                p.poses['coffee_table']=reference.copy();p.seen['coffee_table']=self.anchor['seen']
                if p.held is not None and self.anchor['phase'] in ('carry','search_target','target_navigate'):
                    p.change(self.anchor['phase'])
                    action=self.anchor['command'].copy();action[:3]=0.;action[14]=-1.
        if not self.active:return action
        p=self.p;action=np.asarray(action).copy();action[:3]=0.;action[14]=-1.;action[22]=1.
        if getattr(self,'failed',False):
            action[3:7]=s[53:57];action[7:14]=s[3:10];action[15:22]=s[28:35];return action
        if self.debug.get('mode')=='right_wrist_parent':
            ids=[11,13,15,17,19,21,23];measured=p.current_q[ids]
            previous=np.asarray(self.debug['right_command'])
            target=np.asarray(self.debug['right_target']);advance=target-previous
            proposed=previous+advance*min(1.,.024/max(float(np.max(abs(advance))),1e-9))
            action[3:7]=self.command[:4];action[7:14]=self.command[4:];action[15:22]=proposed
            self.debug['right_command']=proposed.tolist()
            error=float(np.max(abs(target-measured)))
        else:
            measured=p.current_q[p.servo.indices]
            previous=np.clip(self.command,measured-.12,measured+.12);advance=self.target-previous
            self.command=previous+advance*min(1.,.024/max(np.max(abs(advance)),1e-9));self.command[:4]=self.anchor_trunk
            action[3:7]=self.command[:4];action[7:14]=self.command[4:];action[15:22]=s[28:35]
            error=float(np.max(abs(self.target-measured)))
        if p.step-p.seen.get(p.child,-1000)>60:self.fail('held_child_not_fresh_during_view');return self.tick(action,s)
        if p.child=='vase' and p.parent=='coffee_table':
            reference=np.asarray(self.debug['table_reference']);seen=self.debug['table_reference_seen']
            if p.seen.get(p.parent,-1000)>seen:
                observed=p.poses[p.parent];delta=observed[:3,3]-reference[:3,3]
                a=observed[:2,0];b=reference[:2,0]
                alignment=float(np.clip(abs(a@b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-9),0.,1.))
                if abs(delta[2])>.03 or alignment<np.cos(.20):
                    self.debug['table_fit_rejections'].append(dict(step=p.seen[p.parent],translation_m=float(np.linalg.norm(delta)),height_m=float(delta[2]),axis_alignment=alignment))
                    p.poses[p.parent]=reference.copy();p.seen[p.parent]=seen
        if error<.035:
            if self.settled is None:self.settled=p.step
            if p.step-p.seen.get(p.parent,-1000)<=45 and (p.child!='vase' or p.seen[p.parent]>=self.settled):
                self.debug=dict(self.debug,allowed_vase_fit_step=p.seen[p.parent],allowed_parent_fit_role=p.parent);self.active=False;p.change('target_navigate')
            elif p.child=='vase' and self.debug.get('table_fit_rejections') and p.step-self.settled>=18:
                self.debug['rejected_views'].append(self.debug['right_target'])
                self.debug['rejected_view_retry']=len(self.debug['rejected_views'])<3
                if self.debug['rejected_view_retry']:self.active=False;p.change('target_navigate')
                else:self.fail('table_views_only_inconsistent_fits')
            elif p.step-self.settled>90:self.fail('view_pose_did_not_reacquire_parent')
        elif p.step-self.started>240:self.fail('view_motor_convergence_timeout')
        return action
