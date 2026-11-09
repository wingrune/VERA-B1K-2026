"""Conditional serial dual-arm rose sequence; RGB-D/proprio/static geometry only.

The outer policy owns acquisition/map updates and the final BOTH-held live gate.
Custom dual_* phases intentionally have no shared left-arm action branch.
"""
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from dual_arm_right import RightArmServo
from dual_rose_perception import DualRosePerception
from dual_scene import DualSceneCollision,obb_separated
from pair_geometry import pair_condition
from grasp_regions import grasp_regions,supported_aperture_minimum

class DualRoseSequence:
    def __init__(self, policy):
        self.p=policy;self.servo=RightArmServo(policy.root)
        self.perception=DualRosePerception(policy.root/'priors')
        self.active=False;self.right_held=None;self.right_pose_world=None
        self.right_hold_command=None;self.stage=None;self.stage_start=0
        self.attempted=set();self.debug={'reason':'not_attempted'}
        self.possible_held=False;self.release_world=None
        self.hand_history=[];self.relative_history=[];self.observation_steps=[]
        self.confirmations=0;self.close_history=[];self.clear_confirmations=0

    def _stage(self,name):
        self.stage=name;self.stage_start=self.p.step
        self.p.change('dual_'+name)
        self.debug={'reason':name,'step':int(self.p.step)}

    def _world_hand(self):return self.p.base()@self.servo.pose(self.p.current_q)

    def _center(self,world):
        lo,hi=self.p.perception.models['rose']['bounds']
        return world[:3,:3]@((lo+hi)/2)+world[:3,3]

    def _scene(self):
        p=self.p;payloads=[]
        bounds=p.perception.models['rose']['bounds']
        if p.held is not None:payloads.append({'arm':'left','local':p.held,'bounds':bounds})
        if self.right_held is not None:payloads.append({'arm':'right','local':self.right_held,'bounds':bounds})
        target=None
        if self.right_held is None and self.right_pose_world is not None:
            targetworld=self.release_world if self.stage in ('open','withdraw','park') and self.release_world is not None else self.right_pose_world
            target={'pose':np.linalg.inv(p.base())@targetworld,'bounds':bounds}
        return DualSceneCollision(p.current_cameras,p.robot,
                                  p.servo.kin.forward(p.current_q)[0],payloads,target=target)

    def _validator(self,scene):
        def valid(q):
            if not scene.valid(self.servo.kin.forward(q)[0]):return False
            # The grasp target is excluded from scene points for right-finger
            # contact, but may never be struck by the already-held left rose.
            if self.p.held is not None and self.right_held is None and self.right_pose_world is not None:
                bounds=self.p.perception.models['rose']['bounds']
                left=self.p.servo.pose(q)@self.p.held
                target=np.linalg.inv(self.p.base())@self.right_pose_world
                if not obb_separated(left,bounds,target,bounds,margin=.005):return False
            return True
        return valid

    def _admit(self,worldgoal):
        body=np.linalg.inv(self.p.base())@worldgoal
        scene=self._scene()
        if not self.servo.set_goal(self.p.current_q,body[:3,3],body[:3,:3],self._validator(scene)):
            self.debug={'reason':self.servo.last_status,'stage':self.stage}
            return False
        self.goal=worldgoal.copy();return True

    def _fail(self,reason):
        self.debug={'reason':reason,'stage':self.stage,'step':int(self.p.step)}
        self.servo.target=None
        if self.possible_held or self.right_held is not None or self.stage in ('place','open','withdraw','park'):
            self.active=True;self.stage='failed_hold';self.p.change('done_hold')
        else:
            self.active=False;self.perception.clear_hand('right');self.stage='aborted'
            self.p.change('carry')

    def maybe_start(self):
        p=self.p;key=len(p.rose_releases)
        if (self.active or self.right_held is not None or key in self.attempted or
            not (p.phase in ('carry','search_target','target_navigate') or (p.phase=='execute' and 0<=getattr(p,'age',1000)<=5)) or p.child!='rose' or p.held is None or key>=2 or
            not (getattr(p,'left_pickup_confirmed',False) or getattr(p,'pickup_ready',False)) or
            p.step-p.seen.get('rose',-1000)>25 or p.step-p.seen.get('vase',-1000)>90):return False
        proprio=getattr(p,'last_proprio',None)
        if proprio is None or np.linalg.norm(proprio[:3])>=.015:return False
        self.attempted.add(key)
        leftworld=p.poses['rose'].copy()
        ident=self.perception.register_observed('left',leftworld,p.base(),p.seen['rose'],p.stats.get('rose',{}))
        if ident is None:self.debug={'reason':'left_registration_failed'};return False
        leftfk=p.base()@p.servo.pose(p.current_q)@p.held
        self.perception.update_held_fk('left',leftfk,p.step)
        candidate=self.perception.tracks[ident].copy()
        candidate.update(track_id=ident,body_pose=np.linalg.inv(p.base())@leftfk,
                         state='held',pickup_confirmed=True,
                         source_world_pose=getattr(p,'pregrasp_rose_world',leftworld).copy())
        available=self.perception.acquire(p.current_cameras,p.base(),p.level,p.step)
        rgb=p.perception.models['rose']['rgb'].astype(float);xyz=p.perception.models['rose']['xyz']
        red=(rgb[:,0]>70)&(rgb[:,0]>rgb[:,1]*1.5)&(rgb[:,0]>rgb[:,2]*1.5)
        if red.sum()>20:
            anchors=[world[:3,:3]@np.median(xyz[red],axis=0)+world[:3,3] for world in (leftworld,leftfk)]
            available=[t for t in available if all(np.linalg.norm(t.get('observed_flower_world',t['world_pose'][:3,3])-anchor)>=.035 for anchor in anchors)]
        pair=pair_condition([candidate]+available,np.linalg.inv(p.base())@p.poses['vase'],p.level,p.step,base=p.base())
        self.debug=pair;self.pair_debug=pair;self.plan_diagnostics=[]
        eligible=[x for x in pair.get('pairs',[]) if x['left_id']==ident]
        if not pair.get('eligible') or not eligible:
            self.perception.clear_hand('left');return False
        # Keep exactly the previous checked left motor command while right moves.
        deadline=time.monotonic()+3.
        previous=np.asarray(p.action)
        self.fixed_left=previous[3:15].copy();self.fixed_left[:4]=p.current_q[self.servo.indices[:4]]
        self.fixed_left[-1]=-1.
        self.initial_right_joints=p.current_q[self.servo.indices[4:]].copy()
        bounds=p.perception.models['rose']['bounds'];lo,hi=bounds
        for chosen in eligible:
            if time.monotonic()>deadline:break
            track=self.perception.tracks[chosen['right_id']]
            obj=np.linalg.inv(p.base())@track['world_pose']
            self.right_pose_world=track['world_pose'].copy()
            model=p.perception.models['rose']
            for center,axis,width,region in grasp_regions('rose',model):
                if time.monotonic()>deadline:break
                close=obj[:3,axis]
                if abs((p.level[:3,:3]@close)[2])>.30:continue
                for sign in (1.,-1.):
                    if time.monotonic()>deadline:break
                    closing=close*sign;up=p.level[:3,:3].T@np.array([0.,0.,1.])
                    approach=up-closing*np.dot(closing,up)
                    if np.linalg.norm(approach)<.35:continue
                    approach/=np.linalg.norm(approach)
                    rotation=np.column_stack([np.cross(closing,approach),closing,approach])
                    point=obj[:3,:3]@center+obj[:3,3]+up*.007
                    grasp=np.eye(4);grasp[:3,:3]=rotation;grasp[:3,3]=point
                    world=p.base()@grasp;pre=world.copy();pre[:3,3]+=p.base()[:3,:3]@approach*.14
                    if not self._admit(pre):
                        self.plan_diagnostics.append({'track_id':chosen['right_id'],'region':region,'sign':sign,'reason':self.debug.get('reason')})
                        continue
                    # Endpoint and swept descend check are required before commitment.
                    qpre=p.current_q.copy();qpre[self.servo.indices]=self.servo.target
                    scene=self._scene();valid=self._validator(scene)
                    down=self.servo.solve(qpre,point,rotation,validator=valid)
                    if down is None:
                        self.plan_diagnostics.append({'track_id':chosen['right_id'],'region':region,'sign':sign,'reason':'descend_ik_or_endpoint_blocked'})
                        continue
                    if not all(valid(qpre+(self._solution_q(qpre,down)-qpre)*f) for f in np.linspace(0.,1.,101)):continue
                    if not self.perception.assign('right',chosen['right_id']):continue
                    self.track_id=chosen['right_id'];self.grasp_world=world;self.grasp_object_local=np.linalg.inv(self.right_pose_world)@world;self.grasp_width=width;self.grasp_region=region
                    self.initial_height=float(self._center(self.right_pose_world)[2]);self.active=True
                    self.possible_held=False;self.right_held=None;self.close_history=[]
                    self.hand_history=[];self.relative_history=[];self.observation_steps=[];self.confirmations=0
                    self._stage('pregrasp');return True
        self.servo.target=None;self.right_pose_world=None;self.perception.clear_hand('left')
        self.debug={'reason':'no_checked_right_grasp','attempts':self.plan_diagnostics,'pair':self.pair_debug};return False

    def _solution_q(self,q,solution):
        trial=q.copy();trial[self.servo.indices]=solution;return trial

    def _track(self,s):
        # Camera fitting is periodic; seen_step changes only on accepted RGB-D.
        if self.p.step%15:return
        prediction=(self.servo.pose(self.p.current_q)@self.right_held if self.right_held is not None
                    else np.linalg.inv(self.p.base())@self.right_pose_world)
        fit=self.perception.fit_held('right',self.p.current_cameras,prediction,self.p.base(),self.p.step,
                carried_blossom=bool(self.right_held is not None and self.grasp_region.startswith('blossom') and s[49:51].sum()>supported_aperture_minimum(self.grasp_width)))
        if fit is not None:
            self.right_pose_world=self.p.base()@fit[0]
            if self.stage in ('pregrasp','descend'):
                updated=self.right_pose_world@self.grasp_object_local
                moved=np.linalg.norm(updated[:3,3]-self.grasp_world[:3,3])
                angle=R.from_matrix(updated[:3,:3].T@self.grasp_world[:3,:3]).magnitude()
                if moved>.010 or angle>.08:
                    self.grasp_world=updated
                    goal=updated.copy()
                    if self.stage=='pregrasp':goal[:3,3]+=updated[:3,2]*.14
                    if not self._admit(goal):self._fail('fresh_right_target_replan_blocked')

    def _pickup_confirmation(self,s):
        hand=self._world_hand();self.hand_history.append(hand.copy());self.hand_history=self.hand_history[-12:]
        stationary=(len(self.hand_history)==12 and np.max(np.ptp([h[:3,3] for h in self.hand_history],axis=0))<.012
            and max(R.from_matrix(self.hand_history[0][:3,:3].T@h[:3,:3]).magnitude() for h in self.hand_history)<.12
            and np.linalg.norm(s[:3])<.015)
        track=self.perception.tracks[self.track_id];fresh=self.p.step-track['seen_step']<25
        risen=(fresh and self.right_pose_world is not None and self._center(self.right_pose_world)[2]-self.initial_height>.080
               and s[49:51].sum()>supported_aperture_minimum(self.grasp_width))
        if not (stationary and risen):
            self.confirmations=0;self.relative_history=[];self.observation_steps=[];return False
        relative=np.linalg.inv(hand)@self.right_pose_world
        self.relative_history.append(relative);self.relative_history=self.relative_history[-12:]
        consistent=(len(self.relative_history)==12 and np.max(np.ptp([h[:3,3] for h in self.relative_history],axis=0))<.025
             and max(R.from_matrix(self.relative_history[0][:3,:3].T@h[:3,:3]).magnitude() for h in self.relative_history)<.18)
        observed=track['seen_step']
        if observed not in self.observation_steps:self.observation_steps.append(observed)
        self.confirmations=self.confirmations+1 if consistent else 0
        return self.confirmations>=18 and len(self.observation_steps)>=3

    def _holding_action(self,action,s):
        action=action.copy();q=self.p.current_q
        previous=self.right_hold_command if self.right_hold_command is not None else s[28:35]
        action[15:22]=np.clip(previous,s[28:35]-.12,s[28:35]+.12);action[22]=-1.
        scene=self._scene();trial=q.copy();trial[self.servo.indices[:4]]=action[3:7]
        left=[self.servo.names.index('left_arm_joint'+str(i)) for i in range(1,8)]
        trial[left]=action[7:14];trial[self.servo.indices[4:]]=action[15:22]
        if not all(self._validator(scene)(q+(trial-q)*f) for f in np.linspace(0.,1.,11)):
            action[:3]=0.;action[3:7]=s[53:57];action[7:14]=s[3:10];action[15:22]=s[28:35]
            self.debug={'reason':'both_held_command_blocked'}
        return action

    def begin_right_place(self):
        p=self.p
        if self.right_held is None or p.held is not None or self.active:return False
        if p.step-p.seen.get('vase',-1000)>90:
            self._fail('right_place_parent_stale');return False
        model=p.perception.models['rose'];xyz=model['xyz'];center=xyz.mean(0)
        _,_,v=np.linalg.svd(xyz-center,full_matrices=False);axis=v[0]
        rgb=model['rgb'].astype(float);green=(rgb[:,1]>rgb[:,0]*1.15)&(rgb[:,1]>rgb[:,2]*1.15)
        if green.sum()>20 and (xyz[~green].mean(0)-xyz[green].mean(0))@axis<0:axis=-axis
        rotation=R.align_vectors([[0.,0.,1.]],[axis])[0].as_matrix();rotated=xyz@rotation.T
        vase=p.poses['vase'];vxyz=p.perception.models['vase']['xyz']@vase[:3,:3].T+vase[:3,3]
        rim=float(vxyz[:,2].max());bottom=float(vxyz[:,2].min())
        angle=2*np.pi*len(p.rose_releases)/3
        xy=np.median(vxyz[:,:2],axis=0)+.045*np.array([np.cos(angle),np.sin(angle)])
        insertion=max(0.,min((rim-bottom)*.65,np.ptp(rotated[:,2])*.70)-.024)
        objectgoal=np.eye(4);objectgoal[:3,:3]=rotation
        objectgoal[:3,3]=np.r_[xy,rim-insertion-rotated[:,2].min()];objectgoal[:2,3]-=np.median(rotated[:,:2],axis=0)
        above=objectgoal.copy();above[2,3]+=insertion+.10
        middle=objectgoal.copy();middle[2,3]+=.04
        self.place_goals=[g@np.linalg.inv(self.right_held) for g in (above,middle,objectgoal)]
        self.place_index=0;self.fixed_left=np.asarray(p.action)[3:15].copy();self.fixed_left[:4]=p.current_q[self.servo.indices[:4]]
        self.active=True;self._stage('place')
        if not self._admit(self.place_goals[0]):self._fail('right_place_path_blocked');return False
        return True

    def tick(self,action,s):
        action=np.asarray(action).copy()
        if self.p.held is not None and self.perception.hands['left'] is not None:
            self.perception.update_held_fk('left',self.p.base()@self.p.servo.pose(self.p.current_q)@self.p.held,self.p.step)
        if not self.active:
            if self.right_held is not None:
                self._track(s)
                self.perception.update_held_fk('right',self._world_hand()@self.right_held,self.p.step)
                return self._holding_action(action,s)
            return action
        action[:3]=0.;action[3:15]=self.fixed_left
        action[3:7]=np.clip(action[3:7],s[53:57]-.12,s[53:57]+.12)
        action[7:14]=np.clip(action[7:14],s[3:10]-.12,s[3:10]+.12)
        action[22]=-1. if self.possible_held or self.right_held is not None else 1.
        if self.stage=='failed_hold':
            action[3:7]=s[53:57];action[7:14]=s[3:10];action[15:22]=s[28:35]
            return action
        age=self.p.step-self.stage_start
        if age>({'pregrasp':600,'descend':350,'close':160,'lift':500,'place':700,'open':90,'withdraw':450,'park':450}.get(self.stage,450)):
            self._fail('dual_stage_timeout');return action
        if self.stage not in ('open','withdraw','park'):self._track(s)
        if self.stage=='failed_hold' or not self.active:
            action[15:22]=s[28:35];action[22]=-1. if self.possible_held else 1.
            return action
        if self.stage in ('pregrasp','descend'):
            if self.p.step-self.perception.tracks[self.track_id]['seen_step']>60:
                self._fail('right_target_stale');return action
        if self.stage=='close':
            action[15:22]=np.clip(self.right_hold_command,s[28:35]-.12,s[28:35]+.12);action[22]=-1.
            self.possible_held=True;gap=float(s[49:51].sum());self.close_history.append(gap);self.close_history=self.close_history[-8:]
            stable=len(self.close_history)==8 and np.ptp(self.close_history)<.001 and gap<max(.015,self.grasp_width+.01)
            if age>45 and stable and self.p.step-self.perception.tracks[self.track_id]['seen_step']<25:
                self.right_held=np.linalg.inv(self._world_hand())@self.right_pose_world
                self.lift_start=self._world_hand();goal=self.lift_start.copy();goal[2,3]+=.18
                self._stage('lift')
                if not self._admit(goal):self._fail('right_lift_path_blocked')
            return action
        if self.stage=='open':
            action[15:22]=np.clip(self.right_hold_command,s[28:35]-.12,s[28:35]+.12);action[22]=1.
            if age>=30 and s[49:51].sum()>max(.04,self.grasp_width+.01):
                self.right_held=None;self.possible_held=False
                self.withdraw_start=self._world_hand();goal=self.withdraw_start.copy();goal[2,3]+=.16
                self.clear_confirmations=0;self._stage('withdraw')
                if not self._admit(goal):self._fail('right_empty_withdrawal_blocked')
            return action
        scene=self._scene();body=np.linalg.inv(self.p.base())@self.goal
        error=self.servo.step(action,self.p.current_q,body[:3,3],body[:3,:3],self._validator(scene),grip=action[22])
        if self.servo.last_status!='moving':
            self.debug={'reason':self.servo.last_status,'stage':self.stage};return action
        self.right_hold_command=action[15:22].copy()
        if self.stage=='pregrasp' and error[0]<.045 and error[1]<.35:
            self._stage('descend')
            if not self._admit(self.grasp_world):self._fail('right_descend_path_blocked')
        elif self.stage=='descend' and error[0]<.025 and error[1]<.25:
            self._stage('close')
        elif self.stage=='lift' and error[0]<.025 and error[1]<.25 and self._pickup_confirmation(s):
            self.right_held=np.linalg.inv(self._world_hand())@self.right_pose_world
            self.active=False;self.stage='ready_pair';self.p.change('carry');self.debug={'reason':'ready_pair','confirmations':self.confirmations,'observations':len(self.observation_steps)}
        elif self.stage=='place' and error[0]<.026 and error[1]<.12:
            self.place_index+=1
            if self.place_index<len(self.place_goals):
                if not self._admit(self.place_goals[self.place_index]):self._fail('right_next_insertion_path_blocked')
            else:
                if self.p.step-self.p.seen.get('vase',-1000)>90:
                    self._fail('right_release_parent_stale');return action
                self.release_world=self._world_hand()@self.right_held
                self.perception.mark_released('right',self.release_world)
                self._stage('open');action[22]=1.
        elif self.stage=='withdraw':
            current=self._world_hand();up=current[2,3]-self.withdraw_start[2,3]
            lateral=np.linalg.norm(current[:2,3]-self.withdraw_start[:2,3]);angle=R.from_matrix(current[:3,:3].T@self.withdraw_start[:3,:3]).magnitude()
            self.clear_confirmations=self.clear_confirmations+1 if up>=.12 and lateral<.025 and angle<.15 else 0
            if self.clear_confirmations>=12:
                self._stage('park');self.park_confirmations=0
                scene=self._scene()
                if not self.servo.set_joint_goal(self.p.current_q,self.initial_right_joints,self._validator(scene)):
                    self._fail('right_empty_parking_blocked');return action
                qgoal=self.p.current_q.copy();qgoal[self.servo.indices[4:]]=self.initial_right_joints
                self.goal=self.p.base()@self.servo.pose(qgoal)
        elif self.stage=='park':
            close=np.max(np.abs(self.p.current_q[self.servo.indices[4:]]-self.initial_right_joints))<.035
            self.park_confirmations=self.park_confirmations+1 if close and error[0]<.026 and error[1]<.12 else 0
            if self.park_confirmations>=12:
                self.active=False;self.stage='released';self.right_pose_world=None
                self.p._record_rose_release(self.release_world.copy());self.p._resume_after_rose_release()
        return action
