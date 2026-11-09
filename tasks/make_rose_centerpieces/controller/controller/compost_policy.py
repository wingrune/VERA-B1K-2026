"""Two-fruit RGB-D search, grasp, carry and release planner.

Only onboard images, proprioception, relative calibration and static canonical
priors enter control. Evaluator diagnostics are in a different module/process
interface and never read here.
"""
from pathlib import Path
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import observation_policy as shared
from food_perception import FoodPerception
from perception import make_cameras
from scene_collision import ObservedSceneCollision
from active_search import ActiveSearch

shared.TASKS['composting_waste'] = ('half_banana', 'trash_can', 'attach')

class Policy(shared.Policy):
    def __init__(self):
        super().__init__()
        self.categories=['half_banana','half_pomegranate','trash_can']
        self.perception=FoodPerception(self.root/'priors',self.categories)
        self.perception.world_transform=self.base
        self.configure_map()

    def configure_map(self):
        self.map.res=.04;self.map.n=512;self.map.nz=48;self.map.z_min=.025
        self.map.log_odds=np.zeros((512,512,48),np.int8)
        mask=np.array([name=='base_link' or name.startswith('wheel_motor') for name in self.robot.names])
        centers=self.robot.centers(self.servo.kin.forward(np.zeros(len(self.servo.names)))[0])
        self.map.radius=float(np.max(np.linalg.norm(centers[mask,:2],axis=1)+self.robot.radii[mask]))+.015

    def reset(self):
        super().reset()
        if hasattr(self,'robot'):self.configure_map()
        self.phase='search'
        self.delivery_mode='single_fruit_first'
        from bin_search import BinSearch
        self.bin_search=BinSearch()
        self.child='half_banana'
        self.parent='trash_can'
        self.completed_releases=[]
        from pose_search import PoseSearch
        self.searcher=PoseSearch(self)
        self.pose_route=None;self.pose_debug={};self.current_q=None;self.next_pose_plan=0
        self.live_holds=0;self.inspection_goals=[];self.confirm_wait=0;self.bin_view_done=False;self.bin_view_target=None
        self.approach_proposal_cursor=0
        self.bin_confirmation_wait=0
        self.bin_focus_next_step=0
        self.live_hold_streak=0
        self.next_collision_replan=0

    def plan_bin_camera(self,q,views):
        from wrist_view import plan_floor_view
        looks=None;self.view_diagnostics=[]
        if self.parent in self.poses:
            parent=self.poses[self.parent];lo,hi=self.bounds(self.parent);ground_from_world=self.level@np.linalg.inv(self.base())
            looks=[(ground_from_world@np.r_[parent[:3,3]+parent[:3,:3]@np.array([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]+fraction*(hi[2]-lo[2])]),1.])[:3].tolist() for fraction in (.45,.3,.6)]
        payload=None if self.held is None else self.servo.pose(q)@self.held
        return plan_floor_view(self.root,q,views,self.view_diagnostics,look_points=looks,
                               payload=payload,payload_bounds=None if payload is None else self.bounds(self.child))

    def plan_navigation(self,target,inspect=False,radii=None):
        from pose_navigation import PosePlanner
        if self.current_q is None or self.step<self.next_pose_plan:return False
        if self.held is not None and (not self.bin_search.confirmed or self.step-self.seen.get(self.parent,-1000)>90):
            self.pose_debug=dict(reason='explore_with_payload_until_fresh_confirmed_bin')
            return False
        self.next_pose_plan=self.step+45
        planner=PosePlanner(self.map,self.robot,self.servo,self.current_q,self.level,self.held,
                            None if self.held is None else self.bounds(self.child))
        proposals=[]
        for radius in ((.48,.64,.80,.95,1.05,1.20) if radii is None else radii):
            for theta in np.linspace(-np.pi,np.pi,24,endpoint=False):
                yaw=theta+np.pi;goal=target[:2]+radius*np.array([np.cos(theta),np.sin(theta)])
                goal+=shared.rot2(yaw)@np.array([0.,-.12])
                if inspect and (np.linalg.norm(goal-self.odom[:2])<.25 or any(np.linalg.norm(goal-old)<.3 for old in self.inspection_goals)):continue
                score=np.linalg.norm(goal-self.odom[:2])+.12*abs(shared.wrap(yaw-self.odom[2]))+1.8*abs(radius-.64)
                proposals.append((score,goal,yaw))
        deadline=time.monotonic()+8.
        reach_rejections=0
        approach_scene=None
        if not inspect and self.held is None and getattr(self,'current_cameras',None):
            obj=np.linalg.inv(self.base())@self.poses[self.child]
            approach_scene=ObservedSceneCollision(self.current_cameras,self.robot,self.servo.kin.forward(self.current_q)[0],obj,*self.bounds(self.child))
        ordered=sorted(proposals,key=lambda x:x[0])
        start=0 if inspect or self.held is not None else self.approach_proposal_cursor%max(1,len(ordered))
        ordered=ordered[start:]+ordered[:start]
        examined=0
        for _,goal,yaw in ordered[:30]:
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            examined+=1
            route=planner.plan(self.odom,np.r_[goal,yaw],seconds=min(.7,remaining))
            if route is None:continue
            # A free base stance can still leave the fruit outside the arm's
            # feasible grasp orientations. This static check is necessary only;
            # actual RGB-D clearance and joint paths are checked after arrival.
            if not inspect and self.held is None and not self.approach_grasp_reachable(goal,yaw,approach_scene):
                reach_rejections+=1
                continue
            # Compress only collinear translation segments with identical heading.
            compressed=[route[0]]
            for point in route[1:]:
                if len(compressed)>=2:
                    a,b=compressed[-2:];u=b[:2]-a[:2];v=point[:2]-b[:2]
                    same=abs(shared.wrap(a[2]-b[2]))<1e-6 and abs(shared.wrap(b[2]-point[2]))<1e-6
                    if same and np.linalg.norm(u)>1e-5 and np.linalg.norm(v)>1e-5 and abs(u[0]*v[1]-u[1]*v[0])<1e-6 and u@v>0:compressed[-1]=point;continue
                compressed.append(point)
            self.pose_route=compressed;self.pose_index=0;self.pose_planner=planner
            self.nav=goal;self.heading=yaw;self.path=[p[:2] for p in route];self.pose_debug=dict(planner.stats,grasp_reach_rejections=reach_rejections)
            if not inspect and self.held is None:self.approach_proposal_cursor=0
            return True
        if not inspect and self.held is None:self.approach_proposal_cursor=(start+examined)%max(1,len(ordered))
        self.pose_debug=dict(planner.stats,grasp_reach_rejections=reach_rejections)
        return False

    def approach_grasp_reachable(self,goal,yaw,scene=None):
        """Check public arm geometry without changing active manipulation state."""
        if self.child not in self.poses:return False
        base=shared.transform(R.from_euler('z',yaw).as_matrix(),np.r_[goal,0.])@self.level
        names=('grasp','grasp_path','descend_path','grasp_region_debug')
        saved={name:getattr(self,name) for name in names if hasattr(self,name)}
        target=self.servo.target;synchronized=self.servo.synchronized
        if scene is not None:
            import copy
            from scipy.spatial import cKDTree
            change=np.linalg.inv(base)@self.base()
            scene=copy.copy(scene)
            scene.points=scene.points@change[:3,:3].T+change[:3,3]
            scene.tree=cKDTree(scene.points)
            scene.scene_baseline=scene.sphere_margins(self.servo.kin.forward(self.current_q)[0])
        try:
            return shared.Policy.choose_grasp(self,self.current_q,base,None,scene_override=scene)
        finally:
            for name in names:
                if name in saved:setattr(self,name,saved[name])
                elif hasattr(self,name):delattr(self,name)
            self.servo.target=target;self.servo.synchronized=synchronized

    def drive(self,action):
        if self.pose_route is None:return super().drive(action)
        while self.pose_index<len(self.pose_route):
            point=self.pose_route[self.pose_index]
            if np.linalg.norm(point[:2]-self.odom[:2])<.045 and abs(shared.wrap(point[2]-self.odom[2]))<.10:self.pose_index+=1
            else:break
        if self.pose_index>=len(self.pose_route):return True
        point=self.pose_route[self.pose_index]
        action[:2]=np.clip(shared.rot2(self.odom[2]).T@(point[:2]-self.odom[:2])*.9,-.20,.20)
        action[2]=np.clip(shared.wrap(point[2]-self.odom[2])*.9,-.30,.30)
        if abs(shared.wrap(point[2]-self.odom[2]))>.25:action[:2]=0.
        return False

    def plan_inspection(self):
        # Give a newly selected observation stance its own bounded search.
        self.next_pose_plan=0
        found=self.plan_navigation(self.center(self.child),inspect=True,radii=(1.05,1.25,1.5,1.8))
        if found:self.inspection_goals.append(self.nav.copy())
        return found

    def choose_grasp(self,q,base,cams=None):
        found=super().choose_grasp(q,base,cams)
        if found:
            goal=np.eye(4);goal[:3,:3]=self.grasp['rotation'];goal[:3,3]=self.grasp['point']
            self.grasp['world_pose']=base@goal
            self.carry_height=float(np.clip(self.grasp['initial_height']+.15,.95,1.3))
        return found

    def replan_stalled(self,q,point,rotation,cams,base):
        if self.live_hold_streak>=12:
            if self.step<self.next_collision_replan:return
            self.next_collision_replan=self.step+90
            obj=self.servo.pose(q)@self.held if self.held is not None else np.linalg.inv(base)@self.poses[self.child]
            scene=ObservedSceneCollision(cams,self.robot,self.servo.kin.forward(q)[0],obj,*self.bounds(self.child))
            def valid(trial):
                payload=None if self.held is None else self.servo.pose(trial)@self.held
                return scene.valid(self.servo.kin.forward(trial)[0],payload)
            target=self.servo.solve(q,point,rotation,validator=valid)
            self.stall_debug=dict(step=self.step,reason='observed_collision_replan',locked={},alternative_found=target is not None)
            if target is None:return
            from reach_planner import joint_path
            path=joint_path(self.servo,self.robot,q,target,cams,obj,self.bounds(self.child),seconds=5.,payload_local=self.held)
            self.stall_debug['collision_free_path']=path is not None
            if path is None:return
            self.servo.target=path.pop(0);self.servo.synchronized=True
            if self.phase=='pregrasp':self.grasp_path=path;self.descend_path=[]
            elif self.phase=='descend':self.descend_path=path
            elif self.phase=='execute':self.task_joint_path=path
            return
        return super().replan_stalled(q,point,rotation,cams,base)

    def make_task_path(self, base, q, cams):
        if self.held is None or self.parent not in self.poses or self.step-self.seen[self.parent]>90:
            return False
        parent=self.poses[self.parent]
        lo,hi=self.bounds(self.parent)
        rim=parent[:3,3]+parent[:3,:3]@np.array([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,hi[2]])
        current=base@self.servo.pose(q)@self.held
        bottom=float(np.min(self.perception.models[self.child]['xyz']@current[:3,:3].T,axis=0)[2])
        above=current.copy();above[:3,3]=rim;above[2,3]+=.16-bottom
        release=above.copy();release[2,3]-=.10
        self.waypoints=[(above,10),(release,20)]
        self.waypoint=0;self.stable=0
        return True

    def forward(self, obs):
        current_q=self.servo.q(obs['robot_r1::proprio'][0].detach().cpu().numpy())
        self.current_q=current_q
        current_fk=self.servo.kin.forward(current_q)[0]
        self.perception.robot_mask=lambda points:self.robot.point_self_mask(points,current_fk,padding=.003)
        self.perception.carried_prediction=lambda cat: self.servo.pose(current_q)@self.held if self.held is not None and cat==self.child else None
        self.perception.bin_prediction=lambda:np.linalg.inv(self.base())@self.poses[self.parent] if self.bin_search.confirmed and self.parent in self.poses else None
        previous=self.phase
        result=super().forward(obs)
        if previous=='carry' and self.phase=='search_target' and (not self.bin_view_done or self.step-self.seen.get(self.parent,-1000)>54):
            self.bin_view_done=True;self.change('bin_view')
        if self.phase=='search_target' and not self.search_moving and self.held is not None and self.bin_search.confirmed and self.step-self.seen.get(self.parent,-1000)>90 and self.step>=self.bin_focus_next_step:
            self.nav=None;self.pose_route=None;self.search_moving=False
            self.bin_focus_next_step=self.step+240
            self.change('bin_view')
        if self.phase=='target_navigate' and self.pose_route is not None and self.pose_index>=len(self.pose_route) and self.step-self.seen.get(self.parent,-1000)>90:
            self.bin_view_done=True;self.change('bin_view')
        # A completed release is an attempted placement, never native success.
        # Native goals/SR/Q remain exclusively evaluator outputs.
        if previous=='release' and self.phase=='observe_result':
            self.completed_releases.append(self.child)
            if len(self.completed_releases)==1:
                self.nav=None;self.held=None;self.search_moving=False;self.change('observe_delivery')
        # Require visual pickup evidence; reject the inherited gripper-only
        # fallback when a lift transitions to carry.
        if previous=='lift' and self.phase=='carry':
            fresh=self.child in self.poses and self.step-self.seen.get(self.child,-1000)<45
            risen=fresh and self.center(self.child)[2]-self.grasp['initial_height']>.055
            s=obs['robot_r1::proprio'][0].detach().cpu().numpy();q=self.servo.q(s)
            hand=self.base()@self.servo.pose(q)
            close=fresh and np.linalg.norm(self.center(self.child)-hand[:3,3])<.10
            if not (risen and close):
                self.held=None;self.retries+=1;self.change('search')
        # Stop commanded base motion when the observed floor indicates substantial tilt.
        self.observed_tilt=float(np.arccos(np.clip(self.level[2,2],-1,1)))
        # A physically useful lift can stop short of an exact IK waypoint.
        # Confirm it from independent, robot-masked RGB-D evidence before timeout.
        if self.phase=='lift' and self.age>60 and self.held is not None:
            fresh=self.step-self.seen.get(self.child,-1000)<45
            hand=self.base()@self.servo.pose(current_q)
            risen=fresh and self.center(self.child)[2]-self.grasp['initial_height']>.08
            near=fresh and np.linalg.norm(self.center(self.child)-hand[:3,3])<.10
            if risen and near:
                self.held=np.linalg.inv(hand)@self.poses[self.child];self.change('carry')
        # Current-depth clearance gate for manipulation, base and payload.
        data={k:v[0].detach().cpu().numpy() for k,v in obs.items() if k in ('robot_r1::proprio','robot_r1::cam_rel_poses') or k.endswith(('::rgb','::depth_linear'))}
        s=data['robot_r1::proprio'];q=self.servo.q(s);action=result[0].detach().cpu().numpy().copy()
        if self.step%18==1 and self.phase not in ('find_bin_view','find_bin','find_bin_focus'):
            views=make_cameras(data,size=480)
            food=[self.center(cat) for cat in ('half_banana','half_pomegranate')
                  if cat in self.poses and self.step-self.seen.get(cat,-1000)<60
                  and not (self.held is not None and cat==self.child)]
            self.bin_search.observe_counter(views['head'],self.level,self.base(),self.odom,food)
            self.bin_search.observe_fridge(views['head'],self.level,self.base(),self.step)
            if self.parent in self.poses:
                self.bin_search.observe_bin(self.step,self.seen[self.parent],self.poses[self.parent],self.center(self.parent))
            if self.searcher.anchor_changed():
                # A landmark update may restart exploration, but must not
                # abort a fruit approach, grasp or placement joint path.
                if self.phase=='search_target':self.search_moving=False
        if self.phase=='observe_delivery':
            action[:3]=0.;action[14]=action[22]=1.
        if self.phase in ('find_bin_view','find_bin','find_bin_focus'):
            # These phases have no fruit/manipulation branch in the inherited
            # state machine. Both grippers stay open until bin confirmation.
            action[14]=action[22]=1.
            if self.step%18==1:
                cameras=make_cameras(data,size=480)
                food=[self.center(cat) for cat in ('half_banana','half_pomegranate')
                      if cat in self.poses and self.step-self.seen.get(cat,-1000)<60]
                self.bin_search.observe_counter(cameras['head'],self.level,self.base(),self.odom,food)
                self.bin_search.observe_fridge(cameras['head'],self.level,self.base(),self.step)
                if self.searcher.anchor_changed():
                    self.search_moving=False;self.nav=None;self.pose_route=None;self.age=120 if self.bin_search.counter_from_food else 0;action[:3]=0
            confirmed=False
            if self.parent in self.poses:
                confirmed=self.bin_search.observe_bin(self.step,self.seen[self.parent],
                    self.poses[self.parent],self.center(self.parent))
            if confirmed and self.searcher.rear_area(self.odom):self.searcher.counter_backside_reached=True
            pending=self.perception.provisional.get(self.parent)
            waiting_bin=pending is not None and (pending[2]<3 or not confirmed) and self.bin_confirmation_wait<90
            self.bin_confirmation_wait=self.bin_confirmation_wait+1 if waiting_bin else 0 if pending is None or confirmed and pending[2]>=3 else self.bin_confirmation_wait
            rear_reached=self.searcher.counter_backside_reached
            fresh_bin=self.step-self.seen.get(self.parent,-1000)<54
            if confirmed and rear_reached and fresh_bin:
                action[:3]=0;self.nav=None;self.pose_route=None;self.search_moving=False
                self.searcher.visited=[];self.exploration=0
                self.bin_view_done=True;self.change('search')
            elif self.phase=='find_bin_focus':
                action[:3]=0
                if self.age<=1:
                    views=make_cameras(data,size=480)
                    for camera in views.values():camera['ground_from_body']=self.level.copy()
                    self.bin_view_target=self.plan_bin_camera(q,views)
                if self.bin_view_target is None or self.age>240:
                    self.bin_focus_next_step=self.step+120;self.change('find_bin');self.age=120
                else:action[15:22]=s[28:35]+np.clip(self.bin_view_target-s[28:35],-.014,.014)
            elif confirmed and rear_reached and not fresh_bin and self.step>=self.bin_focus_next_step:
                action[:3]=0;self.search_moving=False;self.nav=None;self.pose_route=None;self.change('find_bin_focus')
            elif self.phase=='find_bin_view':
                action[:3]=0
                if self.age<=1:
                    from wrist_view import plan_floor_view
                    self.view_diagnostics=[]
                    views=make_cameras(data,size=480)
                    for camera in views.values():camera['ground_from_body']=self.level.copy()
                    self.bin_view_target=plan_floor_view(self.root,q,views,self.view_diagnostics)
                if self.bin_view_target is None:self.change('find_bin')
                else:
                    action[15:22]=s[28:35]+np.clip(self.bin_view_target-s[28:35],-.014,.014)
                    if np.max(abs(self.bin_view_target-s[28:35]))<.035 or self.age>240:self.change('find_bin')
            else:
                # Start explicit counter navigation once the initial views
                # provide an anchor; retain a brief scan at each new stance.
                if waiting_bin:
                    action[:3]=0
                elif 0<self.bin_search.fridge_samples<3 and self.step-self.bin_search.fridge_seen<54:
                    action[:3]=0
                elif self.bin_search.counter_from_food and not self.search_moving and self.age>=120:
                    if self.searcher.plan(self.map,self.odom,'find_bin'):self.search_moving=True
                    else:action[2]=.25
                elif self.search_moving:
                    if self.searcher.step(self.map,self.odom,action):
                        local_arrival=self.searcher.debug.get('kind')=='counter_clearance' and self.searcher.debug.get('viewpoint_reached') and not self.searcher.counter_backside_reached
                        self.search_moving=False;self.age=120 if local_arrival else 0;self.nav=None;self.exploration+=1
                else:action[2]=.34
        if self.observed_tilt>.17:action[:3]=0
        if self.phase=='search' and self.held is None:
            fresh=[cat for cat in ('half_banana','half_pomegranate') if cat not in self.completed_releases and self.step-self.seen.get(cat,-1000)<60]
            if self.child not in fresh and fresh:
                self.child=min(fresh,key=lambda cat:np.linalg.norm(self.center(cat)[:2]-self.odom[:2]));self.grasp_trials={};self.nav=None
            pending=self.perception.provisional.get(self.child)
            if pending is not None and pending[2]<3 and self.confirm_wait<90:
                action[:3]=0;self.confirm_wait+=1
            else:self.confirm_wait=0
        if self.phase=='bin_view':
            if self.age<=1:
                views=make_cameras(data,size=480)
                for camera in views.values():camera['ground_from_body']=self.level.copy()
                self.bin_view_target=self.plan_bin_camera(q,views)
            if self.bin_view_target is None:
                self.change('search_target')
            else:
                action[15:22]=s[28:35]+np.clip(self.bin_view_target-s[28:35],-.014,.014)
                reached=np.max(abs(self.bin_view_target-s[28:35]))<.035
                fresh=self.step-self.seen.get(self.parent,-1000)<54
                if self.age>240 or reached and (not self.bin_search.confirmed or fresh):
                    self.bin_focus_next_step=max(self.bin_focus_next_step,self.step+120)
                    self.change('search_target')
        if previous=='inspect_child':
            # A new view can expose a feasible manipulation approach before
            # the entire inspection route finishes. Do not spend its whole
            # timeout driving away from an already reachable fruit stance.
            fresh=self.child in self.poses and self.step-self.seen.get(self.child,-1000)<45
            if fresh and self.step%60==0 and self.plan_navigation(self.center(self.child)):
                action[:3]=0;self.grasp=None;self.change('navigate')
            elif self.drive(action) or self.age>360:
                self.nav=None;self.search_moving=False;self.change('search')
        elif self.phase=='search' and self.age>0 and self.child in self.poses and self.step-self.seen.get(self.child,-1000)<360 and self.age%60==0:
            if self.plan_inspection():
                action[:3]=0;self.search_moving=False;self.change('inspect_child')
        obj=None if self.held is None else self.servo.pose(q)@self.held
        if obj is None and self.child in self.poses:
            obj=np.linalg.inv(self.base())@self.poses[self.child]
        live_cameras=make_cameras(data,size=480)
        for camera in live_cameras.values():camera['ground_from_body']=self.level.copy()
        scene=ObservedSceneCollision(live_cameras,self.robot,self.servo.kin.forward(q)[0],obj,*self.bounds(self.child))
        trial=q.copy();trial[self.servo.indices]=np.r_[action[3:7],action[7:14]]
        right_ids=[self.servo.names.index(f'right_arm_joint{i}') for i in range(1,8)]
        trial[right_ids]=action[15:22]
        valid=True;self.gate_failure=None
        for alpha in (.5,1.):
            poses=self.servo.kin.forward(q+alpha*(trial-q))[0]
            delta=np.eye(4);delta[:3,:3]=R.from_euler('z',action[2]/30*alpha).as_matrix();delta[:2,3]=action[:2]/30*alpha
            poses={key:delta@pose for key,pose in poses.items()}
            held=None if self.held is None else delta@self.servo.pose(q+alpha*(trial-q))@self.held
            if not scene.valid(poses,held):
                valid=False
                if self.step%30==0:self.gate_failure=scene.diagnose(poses,held)
                break
        if not valid:
            action[:3]=0;action[3:7]=s[53:57];action[7:14]=s[3:10];action[15:22]=s[28:35];self.live_holds+=1;self.live_hold_streak+=1
        else:self.live_hold_streak=0
        if self.step%180==0:
            np.savez_compressed(self.output/f'map_{self.step:05d}.npz',log_odds=self.map.log_odds,odom=self.odom,level=self.level,q=q,res=self.map.res,z_min=self.map.z_min,radius=self.map.radius)
        if self.step%30==0:
            import json
            with (self.output/'clearance_trace.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=self.step,phase=self.phase,bin_confirmed=self.bin_search.confirmed,rear_route_completed=self.searcher.counter_backside_reached,bin_confirmation_samples=self.bin_search.samples,counter_hint=None if self.bin_search.counter is None else self.bin_search.counter.tolist(),counter_from_food=self.bin_search.counter_from_food,counter_origin=None if self.bin_search.counter_origin is None else self.bin_search.counter_origin.tolist(),counter_revision=self.bin_search.counter_revision,fridge_route_hint=None if self.bin_search.fridge is None else self.bin_search.fridge.tolist(),fridge_hint_samples=self.bin_search.fridge_samples,live_holds=self.live_holds,gate_failure=self.gate_failure,observed_tilt=self.observed_tilt,pose_planning=self.pose_debug,provisional={cat:proposal[2] for cat,proposal in self.perception.provisional.items()},bin_diagnostics=getattr(self.perception,'bin_diagnostics',[]),view_diagnostics=getattr(self,'view_diagnostics',[]),bin_view_target=None if self.bin_view_target is None else self.bin_view_target.tolist(),current_grasp_pose=None if self.grasp is None or 'world_pose' not in self.grasp else (np.linalg.inv(self.base())@self.grasp['world_pose']).tolist()))+'\n')
        self.action=action
        if self.step%30==0:
            import json
            route=getattr(self.searcher,'route',None);index=getattr(self.searcher,'index',0)
            with (self.output/'navigation_control.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=self.step,phase=self.phase,odom=self.odom.tolist(),
                    commanded_base=action[:3].tolist(),measured_base_velocity=s[:3].tolist(),
                    search_moving=self.search_moving,route_index=index,
                    route_point=None if route is None or index>=len(route) else route[index].tolist(),
                    routing=self.searcher.debug))+'\n')
        import torch
        return torch.as_tensor(action[None],dtype=torch.float32)
