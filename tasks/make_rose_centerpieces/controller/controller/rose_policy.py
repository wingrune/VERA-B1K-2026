"""RGB-D rose insertion and vase transfer. No simulator-state policy inputs."""
from pathlib import Path
import numpy as np
import copy,time,json
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import observation_policy as shared
from rose_perception import RosePerception
from scene_collision import ObservedSceneCollision
from pose_navigation import PosePlanner,compress_route
from navigation import rot2,wrap
from retreat_recovery import start_vertical_retreat,step_vertical_retreat
from release_view import start_release_view,step_release_view
from camera_centering import start_camera_centering,step_camera_centering
from dual_sequence import DualRoseSequence
from held_vase_view import HeldVaseView
from paired_sequence import PairedPickup
from grasp_regions import supported_aperture_minimum

shared.TASKS['make_rose_centerpieces']=('rose','vase','attach')

class Policy(shared.Policy):
    def __init__(self):
        # The shared initializer expects these object priors and public kinematics.
        super().__init__()
        self.servo.point=np.array([0.,0.,-.072])  # Public finger pad centers: jointZ-.03689 plus padZ-.035.
        self.perception=RosePerception(self.root/'priors')
        self.perception.world_transform=self.base;self.perception.protected_parent=lambda:self.poses.get("vase")
        self.categories=['rose','vase','coffee_table']
        self.map.res=.04;self.map.n=384;self.map.nz=48;self.map.z_min=.025
        self.map.log_odds=np.zeros((384,384,48),np.int8)
        self.dual=DualRoseSequence(self)
        self.paired=PairedPickup(self);self.vase_view=HeldVaseView(self)

    def reset(self):
        super().reset()
        self.phase='search';self.child='rose';self.parent='vase';self.rose_releases=[];self.failed_releases=[];self.rose_assembly=[];self.vase_payload_bounds=None;self.motion_history=[];self.close_history=[];self.pickup_confirmations=0;self.pickup_ready=False;self.hand_history=[];self.relative_history=[];self.pickup_observation_steps=[];self.lift_hold_command=None
        self.map.res=.04;self.map.n=384;self.map.nz=48;self.map.z_min=.025
        self.map.log_odds=np.zeros((384,384,48),np.int8)
        self.pose_route=None;self.pose_index=0;self.next_plan=0;self.route_progress=None
        self.task_index=0;self.clearance_holds=0;self.pose_debug={}
        self.retreat_state=None;self.retreat_debug={}
        self.release_view_state=None;self.release_view_debug={}
        self.camera_center_state=None;self.camera_center_debug={};self.camera_center_deferred_used=False
        self.commanded_release_world=None;self.left_pickup_confirmed=False
        self.stage='roses';self.action_attempts=0
        self.pickup_reacquire_command=None;self.pickup_reacquire_started=None;self.pickup_reacquire_failed=False;self.unconfirmed_refit_debug={}
        if hasattr(self,'perception') and isinstance(self.perception,RosePerception):
            self.perception.active=None;self.perception.excluded=[]
            self.dual=DualRoseSequence(self)
            self.paired=PairedPickup(self);self.vase_view=HeldVaseView(self)
        self.robot.extra_payloads=[]

    def bounds(self,category):
        if category=='vase' and self.vase_payload_bounds is not None and self.stage=='vase_transfer':return self.vase_payload_bounds
        return super().bounds(category)

    def change(self,phase):
        if phase=='lift' and self.child=='rose' and self.phase=='close' and hasattr(self,'pregrasp_rose_world'):
            self.held=np.linalg.inv(self.grasp['world_pose'])@self.pregrasp_rose_world
        if phase=='release' and self.child=='rose' and self.held is not None:
            self.commanded_release_world=self.base()@self.servo.pose(self.current_q)@self.held
        if phase=='verify_release' and self.phase=='release_retract' and self.child=='rose' and self.parent=='vase':
            self._record_rose_release(self.commanded_release_world)
            self._resume_after_rose_release()
            return
        if phase=='carry' and self.phase=='execute' and self.child=='rose':
            # Timeout while inserted: withdraw vertically before any carry motor.
            self.retreat_state=start_vertical_retreat(self)
            self.retreat_debug={'reason':'execute_timeout_vertical_withdrawal'}
            phase='vertical_retreat'
        if phase=='carry' and self.phase=='lift' and not self.pickup_ready:return
        if phase=='carry' and self.phase=='lift':self.left_pickup_confirmed=True
        if phase=='observe_result':
            if self.child=='rose' and self.parent=='vase':
                self.release_view_state=start_release_view(self)
                self.release_view_debug={'reason':'open_hand_vertical_view_clearance'}
                phase='release_retract'
            else:phase='verify_release'
        self._reset_route_progress(phase);super().change(phase)

    def _record_rose_release(self,world):
        # Record the issued release, without observing or asserting placement.
        if world is None or np.asarray(world).shape!=(4,4) or not np.isfinite(world).all():
            raise RuntimeError('Missing allowed FK release pose')
        world=np.asarray(world).copy();point=world[:3,3].copy()
        self.rose_releases.append(point.tolist());self.perception.excluded.append(point)
        self.rose_assembly.append(np.linalg.inv(self.poses['vase'])@world)
        self.commanded_release_world=None;self.left_pickup_confirmed=False
        if hasattr(self,'dual'):
            tracks=self.dual.perception
            if tracks.hands['left'] is not None:tracks.mark_released('left',world)
            elif not any(np.array_equal(old,world) for old in tracks.released):tracks.released.append(world.copy())

    def _resume_after_rose_release(self):
        self.perception.active=None;self.poses.pop('rose',None);self.seen.pop('rose',None)
        self.nav=None;self.pose_route=None;self.held=None
        if hasattr(self,'dual') and self.dual.right_held is not None:
            if self.dual.begin_right_place():return
            self.change('done_hold');return
        if len(self.rose_releases)>=3:
            self.child='vase';self.parent='coffee_table';self.stage='vase_transfer'
            model=self.perception.models['rose']['xyz']
            parts=[self.perception.models['vase']['xyz']]
            parts.extend(model@t[:3,:3].T+t[:3,3] for t in self.rose_assembly)
            assembly=np.concatenate(parts);self.vase_payload_bounds=np.array([assembly.min(0),assembly.max(0)])
        self.change('search')

    def plan_navigation(self,target,inspect=False):
        if self.step<self.next_plan:return False
        self.next_plan=self.step+45
        if self.child in self.poses and self.held is None and not inspect:
            body=np.linalg.inv(self.base())@self.poses[self.child]
            if np.linalg.norm(body[:2,3])<.85 and self.step-self.seen.get(self.child,-1000)<45:
                # Validate IK/path at the actual observed stance before changing base.
                if self.choose_grasp(self.current_q,self.base(),self.current_cameras):
                    self.nav=self.odom[:2].copy();self.heading=self.odom[2]
                    self.pose_route=[self.odom.copy()];self.pose_index=0
                    self.pose_debug={'reason':'current_stance_grasp_feasible'}
                    return True
        planner=PosePlanner(self.map,self.robot,self.servo,self.current_q,self.level,self.held,None if self.held is None else self.bounds(self.child))
        proposals=[]
        for radius in (.55,.65,.75,.90,1.05):
            for theta in np.linspace(-np.pi,np.pi,16,endpoint=False):
                yaw=wrap(theta+np.pi);point=target[:2]+radius*np.array([np.cos(theta),np.sin(theta)])+rot2(yaw)@np.array([0.,-.12])
                if inspect and np.linalg.norm(point-self.odom[:2])<.25:continue
                proposals.append((np.linalg.norm(point-self.odom[:2])+.12*abs(wrap(yaw-self.odom[2]))+abs(radius-.75),point,yaw))
        original_scene=None
        if self.held is None and not inspect and self.child in self.poses:
            obj=np.linalg.inv(self.base())@self.poses[self.child]
            original_scene=ObservedSceneCollision(self.current_cameras,self.robot,self.servo.kin.forward(self.current_q)[0],obj,*self.bounds(self.child))
        deadline=time.monotonic()+12.;rejected=0
        for _,point,yaw in sorted(proposals,key=lambda x:x[0])[:12]:
            if time.monotonic()>deadline:break
            route=planner.plan(self.odom,np.r_[point,yaw],seconds=.4)
            if route is None:continue
            if original_scene is not None:
                future=shared.transform(R.from_euler('z',yaw).as_matrix(),np.r_[point,0.])@self.level
                if not self.future_grasp_feasible(future,original_scene):rejected+=1;continue
            self.nav=point;self.heading=yaw;self.pose_route=compress_route(route);self.pose_index=0
            self.pose_debug=dict(planner.stats,lattice_pose_count=len(route),compressed_pose_count=len(self.pose_route),route=[p.tolist() for p in self.pose_route],selected_goal=np.r_[point,yaw].tolist())
            return True
        self.pose_debug=dict(planner.stats,grasp_rejections=rejected)
        return False

    def future_grasp_feasible(self,future,original_scene):
        # Transform the already observed/static cloud. Do not re-unproject pixels
        # using a hypothetical robot/camera pose.
        scene=copy.copy(original_scene);change=np.linalg.inv(future)@self.base()
        scene.points=scene.points@change[:3,:3].T+change[:3,3];scene.tree=cKDTree(scene.points)
        scene.scene_baseline=scene.sphere_margins(self.servo.kin.forward(self.current_q)[0])
        names=('grasp','grasp_path','descend_path','grasp_region_debug')
        saved={n:getattr(self,n) for n in names if hasattr(self,n)};target=self.servo.target;synchronized=self.servo.synchronized
        try:return self.choose_grasp(self.current_q,future,scene_override=scene)
        finally:
            self.servo.target=target;self.servo.synchronized=synchronized
            for n in names:
                if n in saved:setattr(self,n,saved[n])
                elif hasattr(self,n):delattr(self,n)

    def choose_grasp(self,q,base,cams=None,scene_override=None):
        found=super().choose_grasp(q,base,cams,scene_override)
        if found:
            self.grasp['world_pose']=base@shared.transform(self.grasp['rotation'],self.grasp['point'])
            if self.child=='rose':self.pregrasp_rose_world=self.poses['rose'].copy()
        return found

    def drive(self,action):
        if self.pose_route is None:return super().drive(action)
        while self.pose_index<len(self.pose_route):
            p=self.pose_route[self.pose_index]
            if np.linalg.norm(p[:2]-self.odom[:2])<.045 and abs(wrap(p[2]-self.odom[2]))<.10:self.pose_index+=1
            else:break
        if self.pose_index==len(self.pose_route):return True
        p=self.pose_route[self.pose_index]
        delta=rot2(self.odom[2]).T@(p[:2]-self.odom[:2]);distance=np.linalg.norm(delta)
        action[:2]=delta*max(.10,min(.16,.8*distance))/distance if distance>=.045 else np.zeros(2)
        action[2]=np.clip(wrap(p[2]-self.odom[2])*.8,-.25,.25)
        if abs(wrap(p[2]-self.odom[2]))>.25:action[:2]=0.
        return False

    def make_task_path(self,base,q,cams):
        if self.held is None or self.parent not in self.poses or self.step-self.seen.get(self.parent,-1000)>90:return False
        target=self.poses[self.parent];lo,hi=self.bounds(self.parent)
        model=self.perception.models[self.child];xyz=model['xyz']
        if self.child=='rose':
            # Object-local principal axis, signed toward the flower by stem color.
            center=xyz.mean(0);_,_,v=np.linalg.svd(xyz-center,full_matrices=False);axis=v[0]
            rgb=model['rgb'].astype(float);green=(rgb[:,1]>rgb[:,0]*1.15)&(rgb[:,1]>rgb[:,2]*1.15)
            if green.sum()>20 and (xyz[~green].mean(0)-xyz[green].mean(0))@axis<0:axis=-axis
            rotation=R.align_vectors([[0.,0.,1.]],[axis])[0].as_matrix()
            rotated=xyz@rotation.T
            # Use observed vase geometry to compute rim and available insertion depth.
            vase_xyz=self.perception.models['vase']['xyz']@target[:3,:3].T+target[:3,3]
            rim=float(vase_xyz[:,2].max());bottom=float(vase_xyz[:,2].min())
            vase_center=np.median(vase_xyz[:,:2],axis=0)
            angle=2*np.pi*len(self.rose_releases)/3
            vase_center+=.045*np.array([np.cos(angle),np.sin(angle)])
            height=np.ptp(rotated[:,2])
            nominal_insertion=min((rim-bottom)*.65,height*.70)
            insertion=max(0.,nominal_insertion-.024)
            position=np.r_[vase_center,rim-insertion-rotated[:,2].min()]
            position[:2]-=np.median(rotated[:,:2],axis=0)
            goal=shared.transform(rotation,position)
            above=goal.copy();above[2,3]+=insertion+.10
            middle=goal.copy();middle[2,3]+=.04
            self.waypoints=[(above,10),(middle,10),(goal,35)]
        else:
            # Place the vase and its contents on the visually localized table top.
            table_xyz=self.perception.models['coffee_table']['xyz']@target[:3,:3].T+target[:3,3]
            top=np.quantile(table_xyz[:,2],.99)
            center=np.median(table_xyz[table_xyz[:,2]>top-.03,:2],axis=0)
            current=base@self.servo.pose(q)@self.held
            rotation=current[:3,:3]
            bottom=(xyz@rotation.T)[:,2].min()
            goal=shared.transform(rotation,np.r_[center,float(top+.008-bottom)])
            above=goal.copy();above[2,3]+=.14
            self.waypoints=[(above,15),(goal,45)]
        self.release_pose=self.waypoints[-1][0].copy();self.waypoint=0;self.stable=0
        return True

    def release_observed(self):
        if self.child not in self.poses or self.parent not in self.poses:return False
        if self.step-self.seen.get(self.child,-1000)>45:return False
        child=self.poses[self.child];parent=self.poses[self.parent]
        points=self.perception.models[self.child]['xyz']@child[:3,:3].T+child[:3,3]
        if self.child=='rose':return False  # Placement verification disabled by user.
        table=self.perception.models['coffee_table']['xyz']@parent[:3,:3].T+parent[:3,3]
        error=abs(float(points[:,2].min()-np.quantile(table[:,2],.99)))
        center=points.mean(0);inside=np.all(center[:2]>table[:,:2].min(0)+.05) and np.all(center[:2]<table[:,:2].max(0)-.05)
        self.release_debug={'support_height_error_m':error,'within_table':bool(inside)}
        return error<.035 and inside

    def forward(self,obs):
        action=self._forward_control(obs)
        if self.phase=='lift':
            hand=self.servo.pose(self.current_q)
            goal=(np.linalg.inv(self.base())@self.lift_world_goal)[:3]
            rotation=(np.linalg.inv(self.base())@self.grasp['world_pose'])[:3,:3]
            error=float(np.linalg.norm(hand[:3,3]-goal))
            angle=float(R.from_matrix(hand[:3,:3].T@rotation).magnitude())
            if self.lift_hold_command is None and error<.012 and angle<.12:
                self.lift_hold_command=action[0,3:14].detach().cpu().numpy().copy()
            if self.lift_hold_command is not None and error<.04:
                action[0,3:14]=__import__('torch').as_tensor(self.lift_hold_command,dtype=action.dtype,device=action.device)
                s=obs['robot_r1::proprio'][0].detach().cpu().numpy()
                checked=self.check_live_action(action[0].detach().cpu().numpy().copy(),s)
                action[0]=__import__('torch').as_tensor(checked,dtype=action.dtype,device=action.device)
                self.action=checked.copy()
        else:self.lift_hold_command=None
        if self.phase in ('descend','close','lift','carry','verify_release','vertical_retreat','release_retract','release_center'):
            s=obs['robot_r1::proprio'][0].detach().cpu().numpy()
            row=dict(step=self.step,phase=self.phase,age=self.age,
                     gripper_left_qpos=s[24:26].tolist(),
                     closure_ready=bool(self.close_ready),
                     pickup_ready=bool(self.pickup_ready),
                     pickup_confirmations=self.pickup_confirmations,
                     stationary_taskspace=bool(self.stationary_taskspace),
                     pickup_observation_steps=self.pickup_observation_steps,
                     hand_pose=self.servo.pose(self.current_q).tolist(),
                     observed_arm_joint_velocities=s[10:17].tolist(),
                     observed_base_velocity=s[:3].tolist(),
                     held_transform=None if self.held is None else self.held.tolist(),
                     registration=getattr(self.perception,'stats',{}))
            with (self.output/'manipulation_allowed.jsonl').open('a') as f:
                f.write(json.dumps(row,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
        # Archive the actual final allowed 23D tensor after every modification.
        issued=action[0].detach().cpu().numpy().copy()
        with (self.output/'issued_actions_allowed.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=int(self.step),phase=self.phase,
                                   action23=issued.tolist(),finite=bool(np.isfinite(issued).all()),
                                   clearance_holds=int(self.clearance_holds)))+'\n')
        return action

    def _forward_control(self,obs):
        s=obs['robot_r1::proprio'][0].detach().cpu().numpy()
        self.current_q=self.servo.q(s)
        self.last_proprio=s.copy()
        self.perception.identity_world=None
        if self.child=='rose' and self.held is None:
            if self.phase=='navigate' and not self._route_progress_ok() and self.age>180 and self.step-self.seen.get('rose',-1000)>180:
                self.nav=None;self.pose_route=None;self.change('search')
            if self.phase in ('search','inspect_parent') and self.step-self.seen.get('rose',-1000)>180:
                self.perception.active=None
            if self.phase in ('pregrasp','descend','close') and hasattr(self,'pregrasp_rose_world'):
                self.perception.identity_world=self.pregrasp_rose_world.copy()
        self.perception.vase_predicted=None if 'vase' not in self.poses else np.linalg.inv(self.base())@self.poses['vase']
        self.perception.vase_robot=self.robot
        self.perception.vase_robot_poses=self.servo.kin.forward(self.current_q)[0]
        joints=self.current_q[self.servo.indices].copy();self.motion_history.append(joints);self.motion_history=self.motion_history[-12:]
        hand=self.servo.pose(self.current_q)
        self.hand_history.append(hand.copy());self.hand_history=self.hand_history[-12:]
        stationary=len(self.hand_history)==12 and np.max(np.ptp(np.array([h[:3,3] for h in self.hand_history]),axis=0))<.012 and max(R.from_matrix(self.hand_history[0][:3,:3].T@h[:3,:3]).magnitude() for h in self.hand_history)<.12 and np.linalg.norm(s[:3])<.015
        self.stationary_taskspace=stationary
        if self.phase=='close':
            gap=float(s[24:26].sum());self.close_history.append(gap);self.close_history=self.close_history[-8:]
            width=self.grasp.get('nominal_width',.03)
            self.close_ready=len(self.close_history)==8 and np.ptp(self.close_history)<.001 and gap<max(.015,width+.01)
        else:self.close_ready=False;self.close_history=[]
        self.pickup_ready=False
        if self.phase in ('lift','pickup_reacquire') and stationary:
            fresh=self.step-self.seen.get(self.child,-1000)<25
            pose=self.poses.get(self.child)
            rose_risen=fresh and pose is not None and self.center(self.child)[2]-self.grasp['initial_height']>.080 and float(s[24:26].sum())>supported_aperture_minimum(self.grasp.get('nominal_width',.03))
            if rose_risen:
                relative=np.linalg.inv(self.base()@hand)@pose
                self.relative_history.append(relative.copy());self.relative_history=self.relative_history[-12:]
                consistent=len(self.relative_history)==12 and np.max(np.ptp(np.array([h[:3,3] for h in self.relative_history]),axis=0))<.025 and max(R.from_matrix(self.relative_history[0][:3,:3].T@h[:3,:3]).magnitude() for h in self.relative_history)<.18
                seen_step=self.seen[self.child]
                if seen_step not in self.pickup_observation_steps:self.pickup_observation_steps.append(seen_step)
                self.pickup_confirmations=self.pickup_confirmations+1 if consistent else 0
            else:self.pickup_confirmations=0;self.relative_history=[];self.pickup_observation_steps=[]
            self.pickup_ready=self.pickup_confirmations>=18 and len(self.pickup_observation_steps)>=3
        else:self.pickup_confirmations=0;self.relative_history=[];self.pickup_observation_steps=[]
        self.perception.carried_blossom=bool((self.held is not None or self.phase=='close') and self.grasp is not None and self.grasp.get('region','').startswith('blossom') and float(s[24:26].sum())>supported_aperture_minimum(self.grasp.get('nominal_width',.03)))
        self.perception.predicted=None
        self.perception.unheld_predicted=None
        self.perception.unheld_navigation=self.phase=='navigate'
        self.perception.released_vase_body=None
        self.perception.released=False
        self.perception.suppress_rose_fit=self.phase in ('release','release_retract') or ((self.phase.startswith('dual_') or self.phase.startswith('paired_')) and self.held is None)
        if (self.perception.released and self.parent=='vase' and 'vase' in self.poses
                and self.step-self.seen.get('vase',-1000)<=90):
            # Only the last accepted RGB-D parent pose, in the current body.
            self.perception.released_vase_body=np.linalg.inv(self.base())@self.poses['vase']
        if self.child=='rose':
            if self.held is not None:self.perception.predicted=self.servo.pose(self.current_q)@self.held
            elif self.phase=='close' and self.grasp is not None and hasattr(self,'pregrasp_rose_world'):
                local=np.linalg.inv(self.grasp['world_pose'])@self.pregrasp_rose_world
                self.perception.predicted=self.servo.pose(self.current_q)@local
            elif self.perception.released and self.child in self.poses:self.perception.predicted=np.linalg.inv(self.base())@self.poses[self.child]
            elif (self.phase in ('navigate','park_arm','grasp_plan','pregrasp','descend')
                    and self.child in self.poses and self.perception.active is not None
                    and self.step-self.seen.get(self.child,-1000)<=(900 if self.phase=='navigate' else 180)):
                # This prior seeds current-image matching, never a freshness
                # override. The shared seen[] update still requires a valid fit.
                self.perception.unheld_predicted=np.linalg.inv(self.base())@self.poses[self.child]
        if self.phase=='verify_release' and self.child!='rose':
            if self.age>60 and self.release_observed():self.change('done_hold')
            elif self.age>180:
                self.failed_releases.append(self.child);self.action_attempts+=1
                if self.action_attempts>=3:self.change('done_hold')
                else:self.change('search')
        if (self.phase=='lift' and self.child=='rose' and self.held is not None
                and not self.pickup_ready and self.age>=170):
            self.pickup_reacquire_command=self.action.copy()
            self.pickup_reacquire_started=self.step
            self.pickup_reacquire_failed=False
            self.change('pickup_reacquire')
        self.vase_view.maybe_start(s)
        # Bound observation reacquisition after arrival; the shared driver's
        # drive(True)/path(False) branch otherwise stalls indefinitely.
        if self.phase=='target_navigate' and self.age>160 and self.step-self.seen.get(self.parent,-1000)>90:
            self.nav=None;self.pose_route=None;self.change('search_target')
        action=super().forward(obs).detach().cpu().numpy()[0]
        # Refinement of a possible-held prior is not pickup confirmation. Only
        # a newly accepted independent RGB-D fit supplies this local transform.
        if (self.phase in ('lift','pickup_reacquire') and self.child=='rose'
                and self.held is not None and not self.left_pickup_confirmed
                and self.seen.get('rose',-1000)==self.step and float(s[24:26].sum())>supported_aperture_minimum(self.grasp.get('nominal_width',.03))):
            observed=self.poses['rose'];world_hand=self.base()@self.servo.pose(self.current_q)
            risen=self.center('rose')[2]-self.grasp['initial_height']>.055
            if risen and np.linalg.norm(self.center('rose')-world_hand[:3,3])<.20:
                previous=self.held.copy();self.held=np.linalg.inv(world_hand)@observed
                self.unconfirmed_refit_debug=dict(step=self.step,source='newly_accepted_rgbd_fit',position_delta_m=float(np.linalg.norm(self.held[:3,3]-previous[:3,3])),rotation_delta_rad=float(R.from_matrix(previous[:3,:3].T@self.held[:3,:3]).magnitude()),confirmation_not_changed=True)
        if self.pickup_reacquire_command is not None:
            if self.phase=='pickup_reacquire' and self.pickup_ready:
                self.held=np.linalg.inv(self.base()@self.servo.pose(self.current_q))@self.poses['rose']
                self.left_pickup_confirmed=True;self.pickup_reacquire_command=None
                self.change('carry')
            else:
                action[:3]=0.;action[3:14]=self.pickup_reacquire_command[3:14]
                action[15:22]=self.pickup_reacquire_command[15:22];action[14]=-1.;action[22]=1.
                if self.phase=='pickup_reacquire' and self.step-self.pickup_reacquire_started>90:
                    self.pickup_reacquire_failed=True;self.change('done_hold')
        # M10: acquire one rose at a time; keep the right wrist camera free.
        # Optional paired and dual acquisition starts are disabled; safety ticks remain.
        if self.paired.active:action=self.paired.tick(action,s)
        else:action=self.dual.tick(action,s)
        action=self.vase_view.tick(action,s)
        self.robot.extra_payloads=[] if self.dual.right_held is None else [dict(arm='right',local=self.dual.right_held,bounds=self.perception.models['rose']['bounds'])]
        if self.phase=='release_retract':
            action,complete,self.release_view_debug=step_release_view(self,self.release_view_state,action,s)
            if complete:self.change('verify_release')
        if self.phase=='release_center':
            action,complete,self.camera_center_debug=step_camera_centering(self,self.camera_center_state,action,s)
            if complete:self.change('verify_release')
        if self.phase=='vertical_retreat':
            action,complete,self.retreat_debug=step_vertical_retreat(self,self.retreat_state,action,s)
            if complete:
                self.nav=None;self.pose_route=None
                self.change('carry')
        action=self.check_live_action(action,s)
        if self.vase_view.active:self.vase_view.command=np.r_[action[3:7],action[7:14]].copy()
        if self.pickup_reacquire_command is not None:self.pickup_reacquire_command=action.copy()
        self.action=action.copy()
        if self.step%30==0:
            import json
            with (self.output/'rose_trace.jsonl').open('a') as f:f.write(json.dumps(dict(step=self.step,stage=self.stage,phase=self.phase,rose_releases=self.rose_releases,failed_releases=self.failed_releases,clearance_holds=self.clearance_holds,retreat_debug=self.retreat_debug,release_view_debug=self.release_view_debug,camera_center_debug=self.camera_center_debug,paired_debug=self.paired.debug,paired_active=self.paired.active,dual_debug=self.dual.debug,dual_active=self.dual.active,right_held=self.dual.right_held is not None,release_count_semantics="issued_release_without_rgbd_placement_verification",pose_plan=self.pose_debug,held_vase_view=self.vase_view.debug,unconfirmed_held_refit=self.unconfirmed_refit_debug,pickup_reacquire_failed=self.pickup_reacquire_failed,release_debug=getattr(self,'release_debug',{})))+'\n')
        import torch
        return torch.as_tensor(action[None],dtype=torch.float32)

    def check_live_action(self,action,s):
        cameras=self.current_cameras;q=self.current_q;poses=self.servo.kin.forward(q)[0]
        payload=None if self.held is None else self.servo.pose(q)@self.held
        obj=payload if payload is not None else (np.linalg.inv(self.base())@self.poses[self.child] if self.child in self.poses else None)
        scene=ObservedSceneCollision(cameras,self.robot,poses,obj,*self.bounds(self.child))
        if self.paired.active:scene=self.paired.scene()
        elif self.dual.active:scene=self.dual._scene()
        trial=q.copy();trial[self.servo.indices]=np.r_[action[3:7],action[7:14]]
        right_ids=[self.servo.names.index(f'right_arm_joint{i}') for i in range(1,8)]
        trial[right_ids]=action[15:22]
        valid=True
        for alpha in (.33,.67,1.):
            tq=q+alpha*(trial-q);tf=self.servo.kin.forward(tq)[0]
            step=shared.transform(R.from_euler('z',float(action[2])*alpha/30).as_matrix(),np.r_[action[:2]*alpha/30,0.])
            future={k:step@t for k,t in tf.items()}
            future_payload=None if payload is None else step@self.servo.pose(tq)@self.held
            if not scene.valid(future,None if self.dual.active or self.paired.active else future_payload):valid=False;break
        if not valid:
            action[:3]=0.;action[3:7]=s[53:57];action[7:14]=s[3:10];action[15:22]=s[28:35];self.clearance_holds+=1
        return action

    def _reset_route_progress(self,phase):
        # Route replacement must not restart this absolute navigation budget.
        if phase!='navigate' or getattr(self,'phase',None)!='navigate':
            self.route_progress=None

    def _route_progress_ok(self):
        # Navigation-only prior: never writes object poses, seen, or grasp state.
        if self.phase!='navigate' or self.child!='rose' or self.held is not None:
            self.route_progress=None
            return False
        route=self.pose_route
        if route is None or not len(route) or 'rose' not in self.poses:
            return False
        points=np.asarray(route,dtype=float)
        target=np.asarray(self.poses['rose'],dtype=float)[:3,3]
        current=np.asarray(self.odom,dtype=float)
        if (points.ndim!=2 or points.shape[1]!=3 or current.shape!=(3,)
                or not np.isfinite(points).all() or not np.isfinite(current).all()
                or not np.isfinite(target).all() or not np.isfinite(self.last_proprio).all()):
            return False
        state=getattr(self,'route_progress',None)
        if state is None:
            state=dict(start=self.step-self.age,route=None,best=None,progress=self.step,
                       target=target.copy())
            self.route_progress=state
        if (self.step-state['start']>900 or self.step-self.seen.get('rose',-1000)>600
                or np.linalg.norm(target-state['target'])>.05):
            return False
        index=int(self.pose_index)
        if not 0<=index<=len(points):return False
        remaining=points[index:]
        chain=np.vstack([current,remaining])
        cost=sum(float(np.linalg.norm(b[:2]-a[:2]))+.25*abs(wrap(float(b[2]-a[2])))
                 for a,b in zip(chain[:-1],chain[1:]))
        # Preserve absolute start and stall timer even if a caller replaces route.
        signature=tuple(map(tuple,points))
        if state['route']!=signature:
            state['route']=signature;state['best']=cost
        elif cost<=state['best']-.015:
            state['best']=cost;state['progress']=self.step
        return self.step-state['progress']<=90
