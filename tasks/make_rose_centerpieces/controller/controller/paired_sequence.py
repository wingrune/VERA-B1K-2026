from grasp_regions import supported_aperture_minimum
"""Observed paired planning before closure; independently confirmed dual pickup."""
import json,time,os
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from paired_grasp_planner import plan_pair,TargetProtectedScene
from dual_scene import DualSceneCollision,obb_separated

class PairedPickup:
    def __init__(self,p):
        self.p=p;self.active=False;self.attempted=set();self.failed_pregrasp=set();self.attempt_count={};self.last_scan=-1000;self.debug={};self.left_possible=False;self.right_possible=False
    def change(self,stage):
        self.stage=stage;self.started=self.p.step;self.node=0;self.history=[];self.relatives=[];self.fits=[];self.confirmations=0;self.gaps=[]
        self.p.change('paired_'+stage)
    def maybe_start(self,s):
        p=self.p;key=len(p.rose_releases)
        transition=(p.phase=='descend' and p.age<=1) or (p.phase=='close' and p.age<=1)
        retry=key in self.failed_pregrasp and transition and self.attempt_count.get(key,0)<2
        if self.active or key>=2 or (key in self.attempted and not retry) or p.child!='rose' or p.held is not None or p.dual.right_held is not None:return False
        if p.phase not in ('pregrasp','descend','close') or (p.phase=='close' and p.age>1):return False
        if np.linalg.norm(s[:3])>=.015 or s[24:26].sum()<.04:return False
        # First descent changes the camera/arm occlusion pattern: two targets
        # can be freshly supported here and disappear before first closure.
        # Inspect each transition even when the periodic pregrasp scan is recent.
        transition=p.phase=='close' or (p.phase=='descend' and p.age<=1)
        if p.step-self.last_scan<15 and not transition:return False
        if p.step-p.seen.get('vase',-1000)>90:return False
        self.last_scan=p.step;tracks=p.dual.perception;candidates=tracks.acquire(p.current_cameras,p.base(),p.level,p.step)
        if len(candidates)<2:self.debug={'reason':'fewer_than_two_supported_targets'};return False
        self.attempted.add(key);self.attempt_count[key]=self.attempt_count.get(key,0)+1
        if os.environ.get('POLICY_SAVE_OBSERVATIONS','1')=='1':
            allowed=dict(q=p.current_q,level=p.level,base=p.base())
            for role,camera in p.current_cameras.items():
                for name,value in camera.items():
                    if isinstance(value,np.ndarray):allowed[role+'::'+name]=value
            np.savez_compressed(p.output/f'paired_attempt_{p.step:05d}.npz',**allowed)
            (p.output/f'paired_candidates_{p.step:05d}.json').write_text(json.dumps(candidates,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v))
        plan,debug=plan_pair(p.robot,p.servo,p.dual.servo,p.current_cameras,p.current_q,p.perception.models['rose'],p.level,candidates,np.linalg.inv(p.base())@p.poses['vase'],p.step,seconds=35.)
        self.debug=debug
        with (p.output/'paired_plan.jsonl').open('a') as out:out.write(json.dumps(dict(step=p.step,plan=plan,debug=debug),default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v)+'\n')
        if plan is None:
            if p.phase=='pregrasp':self.failed_pregrasp.add(key)
            return False
        self.plan=plan;self.anchor=p.base().copy();self.initial_right=p.current_q[p.dual.servo.indices[4:]].copy();self.world={h:self.anchor@plan['initial_'+h] for h in ('left','right')};self.initial_height={h:float(self.center(self.world[h])[2]) for h in ('left','right')}
        for hand in ('left','right'):
            ident=plan['pair'][hand+'_id']
            if not tracks.assign(hand,ident):
                tracks.clear_hand('left');tracks.clear_hand('right');self.debug={'reason':'independent_assignment_failed'};return False
        self.clear_shared_motion()
        self.active=True;self.left_possible=False;self.right_possible=False;self.hold_command=np.asarray(p.action).copy();self.change('stage');return True
    def clear_shared_motion(self):
        # A paired path supersedes the serial pregrasp/descent joint target.
        # Never resume that old target when handing back to shared carry.
        p=self.p;p.servo.target=None;p.servo.synchronized=False
        p.dual.servo.target=None;p.dual.servo.synchronized=False
        p.grasp_path=[];p.descend_path=[];p.task_joint_path=[]
    def center(self,world):
        lo,hi=self.p.perception.models['rose']['bounds'];return world[:3,:3]@((lo+hi)/2)+world[:3,3]
    def hand_pose(self,hand,q=None):
        q=self.p.current_q if q is None else q;servo=self.p.servo if hand=='left' else self.p.dual.servo
        return self.p.base()@servo.pose(q)
    def track(self):
        p=self.p
        if p.step%15:return
        for hand in ('left','right'):
            local=p.held if hand=='left' else p.dual.right_held
            predicted=np.linalg.inv(p.base())@self.world[hand] if local is None else np.linalg.inv(p.base())@self.hand_pose(hand)@local
            fit=p.dual.perception.fit_held(hand,p.current_cameras,predicted,p.base(),p.step,carried_blossom=local is not None)
            if fit is not None:
                moved=np.linalg.norm(fit[0][:3,3]-predicted[:3,3]);angle=R.from_matrix(fit[0][:3,:3].T@predicted[:3,:3]).magnitude()
                possible=self.left_possible if hand=='left' else self.right_possible
                if local is None and not possible and (moved>.020 or angle>.10):self.fail('unheld_target_moved_requires_new_complete_plan');return
                confirmed=hand=='left' and p.left_pickup_confirmed
                if local is not None and confirmed and (moved>.025 or angle>.18):self.fail('carried_target_changed_relative_to_confirmed_hand');return
                if local is not None and not confirmed:
                    # Close-time attachment/fit corrections seed later observed
                    # fits; independent rise/stability/aperture checks still
                    # decide holding. No Native flag or assumed freshness.
                    updated=np.linalg.inv(self.hand_pose(hand))@(p.base()@fit[0])
                    if hand=='left':p.held=updated
                    else:p.dual.right_held=updated
                self.world[hand]=p.base()@fit[0]
                if hand=='left':
                    p.poses['rose']=self.world[hand].copy();p.seen['rose']=p.step;p.stats['rose']=dict(fit[1]);p.perception.active=self.world[hand][:3,3].copy()
                else:p.dual.right_pose_world=self.world[hand].copy()
    def scene(self):
        p=self.p;bounds=p.perception.models['rose']['bounds'];payloads=[]
        if p.held is not None:payloads.append(dict(arm='left',local=p.held,bounds=bounds))
        if p.dual.right_held is not None:payloads.append(dict(arm='right',local=p.dual.right_held,bounds=bounds))
        protected=[(h,np.linalg.inv(p.base())@self.world[h],bounds) for h,local in [('left',p.held),('right',p.dual.right_held)] if local is None]
        targets=[t[1] for t in protected]
        scene=TargetProtectedScene(p.current_cameras,p.robot,p.servo.kin.forward(p.current_q)[0],payloads,target=None if not targets else dict(pose=targets[0],bounds=bounds),protected_targets=protected)
        for target in targets[1:]:
            local=(scene.points-target[:3,3])@target[:3,:3];keep=~np.all((local>bounds[0]-.055)&(local<bounds[1]+.055),axis=1);scene.points=scene.points[keep]
        scene.tree=cKDTree(scene.points);scene.scene_baseline=scene.sphere_margins(p.servo.kin.forward(p.current_q)[0]);return scene
    def fail(self,reason):
        self.debug={'reason':reason,'stage':self.stage,'step':self.p.step}
        if self.left_possible or self.right_possible:
            self.stage='failed_hold';self.p.change('done_hold')
        else:
            self.active=False;self.p.dual.perception.clear_hand('left');self.p.dual.perception.clear_hand('right');self.p.nav=None;self.p.pose_route=None;self.p.perception.active=None;self.p.change('search')
    def confirmed(self,hand,s):
        p=self.p;pose=self.hand_pose(hand);self.history.append(pose);self.history=self.history[-12:]
        ident=self.plan['pair'][hand+'_id'];t=p.dual.perception.tracks[ident];gap=s[24:26].sum() if hand=='left' else s[49:51].sum()
        stationary=len(self.history)==12 and np.max(np.ptp([x[:3,3] for x in self.history],axis=0))<.012 and max(R.from_matrix(self.history[0][:3,:3].T@x[:3,:3]).magnitude() for x in self.history)<.12 and np.linalg.norm(s[:3])<.015
        fresh=p.step-t['seen_step']<25;risen=self.center(self.world[hand])[2]-self.initial_height[hand]>.080
        if not(stationary and fresh and risen and gap>supported_aperture_minimum(self.plan[hand+'_width'])):self.confirmations=0;self.relatives=[];self.fits=[];return False
        relative=np.linalg.inv(pose)@self.world[hand];self.relatives.append(relative);self.relatives=self.relatives[-12:]
        consistent=len(self.relatives)==12 and np.max(np.ptp([x[:3,3] for x in self.relatives],axis=0))<.025 and max(R.from_matrix(self.relatives[0][:3,:3].T@x[:3,:3]).magnitude() for x in self.relatives)<.18
        if t['seen_step'] not in self.fits:self.fits.append(t['seen_step'])
        self.confirmations=self.confirmations+1 if consistent else 0;return self.confirmations>=18 and len(self.fits)>=3
    def tick(self,action,s):
        if not self.active:return action
        p=self.p;action=np.asarray(action).copy();action[:3]=0.;self.track()
        if not self.active:
            action[3:7]=s[53:57];action[7:14]=s[3:10];action[15:22]=s[28:35];action[14]=1.;action[22]=1.;return action
        action[14]=-1. if self.left_possible else 1.;action[22]=-1. if self.right_possible else 1.
        if self.stage=='failed_hold':
            action[3:7]=s[53:57];action[7:14]=s[3:10];action[15:22]=s[28:35];return action
        age=p.step-self.started
        if age>({'stage':600,'left_descend':450,'left_close':180,'left_lift':700,'left_verify':300,'right_descend':450,'right_close':180,'right_lift':500,'right_verify':300}.get(self.stage,300)):
            self.fail('finite_paired_stage_timeout');return self.tick(action,s)
        if p.held is not None and self.stage in ('right_descend','right_close','right_lift','right_verify'):
            ident=self.plan['pair']['left_id']
            if p.step-p.dual.perception.tracks[ident]['seen_step']>45:
                self.fail('confirmed_left_payload_not_fresh_during_right_motion');return self.tick(action,s)
        if self.stage in ('stage','left_descend','right_descend'):
            hands=['left','right'] if self.stage=='stage' else ['left'] if self.stage=='left_descend' else ['right']
            if any(p.step-p.dual.perception.tracks[self.plan['pair'][h+'_id']]['seen_step']>45 for h in hands):self.fail('unheld_supported_target_stale');return self.tick(action,s)
        if self.stage.endswith('_close'):
            hand=self.stage.split('_')[0]
            if hand=='left':self.left_possible=True;action[14]=-1.
            else:self.right_possible=True;action[22]=-1.
            action[3:14]=self.hold_command[3:14];action[15:22]=self.hold_command[15:22]
            gap=float(s[24:26].sum() if hand=='left' else s[49:51].sum());self.gaps.append(gap);self.gaps=self.gaps[-8:]
            ready=age>45 and len(self.gaps)==8 and np.ptp(self.gaps)<.001 and gap<max(.015,self.plan[hand+'_width']+.01)
            track=p.dual.perception.tracks[self.plan['pair'][hand+'_id']]
            if ready and p.step-track['seen_step']<25:
                local=np.linalg.inv(self.hand_pose(hand))@self.world[hand]
                if hand=='left':
                    p.held=local;p.pregrasp_rose_world=self.world[hand].copy();p.grasp=dict(region=self.plan['left_region'],nominal_width=self.plan['left_width'],initial_height=self.initial_height['left'],world_pose=self.anchor@self.plan['left_grasp'])
                else:p.dual.right_held=local;p.dual.grasp_region=self.plan['right_region']
                self.change(hand+'_lift')
            return action
        if self.stage.endswith('_verify'):
            hand=self.stage.split('_')[0];action[3:14]=self.hold_command[3:14];action[15:22]=self.hold_command[15:22]
            if self.confirmed(hand,s):
                local=np.linalg.inv(self.hand_pose(hand))@self.world[hand]
                if hand=='left':p.held=local;p.left_pickup_confirmed=True;self.change('right_descend')
                else:
                    p.dual.right_held=local;p.dual.right_hold_command=action[15:22].copy();p.dual.track_id=self.plan['pair']['right_id'];p.dual.initial_right_joints=self.initial_right.copy();p.dual.grasp_width=self.plan['right_width'];p.dual.possible_held=True;p.dual.stage='ready_pair';p.dual.active=False;p.dual.attempted.add(len(p.rose_releases));self.clear_shared_motion();self.active=False;p.change('carry');self.debug={'reason':'independently_confirmed_pair','step':p.step}
            return action
        nodes=self.plan['paths'][self.stage];node=nodes[self.node];indices=p.servo.indices+p.dual.servo.indices[4:]
        measured=p.current_q[indices];previous=np.r_[p.action[3:7],p.action[7:14],p.action[15:22]];previous=np.clip(previous,measured-.12,measured+.12);delta=node[indices]-previous;command=previous+delta*min(1.,.024/max(float(np.max(np.abs(delta))),1e-9));command=np.clip(command,measured-.12,measured+.12)
        action[3:7]=command[:4];action[7:14]=command[4:11];action[15:22]=command[11:]
        scene=self.scene();trial=p.current_q.copy();trial[indices]=command
        def valid(q):
            if not scene.valid(p.servo.kin.forward(q)[0]):return False
            if p.held is not None and p.dual.right_held is None:
                return obb_separated(p.servo.pose(q)@p.held,p.perception.models['rose']['bounds'],np.linalg.inv(p.base())@self.world['right'],p.perception.models['rose']['bounds'])
            return True
        if not all(valid(p.current_q+(trial-p.current_q)*f) for f in np.linspace(0.,1.,11)):
            action[3:7]=s[53:57];action[7:14]=s[3:10];action[15:22]=s[28:35];self.debug={'reason':'paired_live_command_blocked','stage':self.stage};return action
        self.hold_command=action.copy()
        errors=[]
        for servo in (p.servo,p.dual.servo):
            a=servo.pose(p.current_q);b=servo.pose(node);errors.append(np.linalg.norm(a[:3,3]-b[:3,3])<.015 and R.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude()<.12)
        if all(errors) and np.max(np.abs(p.current_q[p.servo.indices[:4]]-node[p.servo.indices[:4]]))<.035:
            self.node+=1
            if self.node==len(nodes):self.change({'stage':'left_descend','left_descend':'left_close','left_lift':'left_verify','right_descend':'right_close','right_lift':'right_verify'}[self.stage])
        return action
