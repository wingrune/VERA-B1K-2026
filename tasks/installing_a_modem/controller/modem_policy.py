"""RGB-D development controller for modem pickup, cabinet placement, and physical power-button pressing."""
import os,json
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial.transform import Rotation as R
from perception import make_cameras
from task_perception import TelevisionDetector
from known_object import ModemPose
from controller import ArmServo
from navigation import DepthMap,rot2,wrap
from visual_odometry import VisualOdometry
from ground_frame import GroundFrameEstimator,level_cameras
from perception import pointcloud,red_mask
class ModemTracker:
    def __init__(self,assets):self.model=ModemPose(Path(assets)/'model.npz')
    def fit(self,cameras):
        small={}
        for key,c in cameras.items():
            x=c.copy();scale=360/c['rgb'].shape[1];x['rgb']=cv2.resize(c['rgb'],(360,360));x['depth']=cv2.resize(c['depth'],(360,360),interpolation=cv2.INTER_NEAREST);x['K']=c['K'].copy();x['K'][:2]*=scale;small[key]=x
        p=self.model.fit(small)
        if p is not None:p['rotation']=p['rotation']@np.array([[0,-1,0],[1,0,0],[0,0,1]]);p['visible_side']=1
        return p

class InstallationPolicy:
    def __init__(self):
        self.assets=Path(__file__).parent;self.output=Path(os.environ['POLICY_OUTPUT']);self.output.mkdir(parents=True,exist_ok=True);self.servo=ArmServo(self.assets);self.tracker=ModemTracker(self.assets/'assets');self.tv_detector=TelevisionDetector();self.episode=-1
    def set_action_dim(self,n):assert n==23
    def reset(self):
        self.episode+=1;self.step=0;self.age=0;self.phase='search';self.odom=np.zeros(3);self.vo=VisualOdometry();self.ground=GroundFrameEstimator();self.level=np.eye(4);self.map=DepthMap();self.object=None;self.last_seen=-1000;self.action=None;self.path=None;self.nav=None;self.grasp=None;self.servo.target=None;self.last_plan=-1000;self.grasp_attempt=0;self.trace=[];self.pregrasp_ready=False;self.tv=None;self.place_goal=None;self.mount=None;self.grip_object=None;self.mount_plan=None;self.frozen_goal=None;self.released_object=None;self.place_debug={};self.press_ready=False;self.press_count=0;self.press_orientation_ready=False;self.button_world=None;self.support_height=None;self.button_grip=None;self.yaw_votes=0;self.yaw_resolved=False;self.marker_debug={}
    def change(self,phase):
        self.phase=phase;self.age=0;self.servo.target=None;self.last_plan=-1000;print('INSTALL_PHASE',self.step,phase,flush=True)
    def base_transform(self):
        T=np.eye(4);T[:3,:3]=R.from_euler('z',self.odom[2]).as_matrix();T[:2,3]=self.odom[:2];return T@self.level
    def red_points(self,cameras):
        candidates=[]
        for camera in cameras.values():
            mask=red_mask(camera['rgb']);n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            for label in range(1,n):
                if stats[label,cv2.CC_STAT_AREA]<3:continue
                points,_,_=pointcloud(camera,labels==label)
                if len(points)<3 or np.linalg.norm(np.ptp(points,axis=0))>.065:continue
                candidates.append(np.median(points,axis=0))
        return candidates
    def resolve_held_yaw(self,cameras,q):
        """Break the blue/white housing symmetry with the observed red power marker."""
        grip=self.servo.pose(q);held=grip@self.grip_object
        offset=np.array([-.001497,-.066087,-.032603]);flipped=offset*np.array([-1,-1,1])
        front=held[:3,3]+held[:3,:3]@offset;back=held[:3,3]+held[:3,:3]@flipped
        nearby=[p for p in self.red_points(cameras) if np.linalg.norm(p-held[:3,3])<.17]
        if not nearby:return
        p=min(nearby,key=lambda p:min(np.linalg.norm(p-front),np.linalg.norm(p-back)))
        df=float(np.linalg.norm(p-front));db=float(np.linalg.norm(p-back))
        if not self.yaw_resolved:
            vote=1 if db<.07 and df-db>.045 else (-1 if df<.07 and db-df>.045 else 0)
            self.yaw_votes=self.yaw_votes+vote if vote*self.yaw_votes>=0 else vote
            if abs(self.yaw_votes)>=3:
                if self.yaw_votes>0:self.grip_object[:3,:3]=self.grip_object[:3,:3]@np.diag([-1.,-1.,1.])
                self.yaw_resolved=True;self.servo.target=None
        if min(df,db)<.07:
            relative=(np.linalg.inv(grip)@np.r_[p,1.])[:3]
            self.button_grip=relative if self.button_grip is None else .8*self.button_grip+.2*relative
        self.marker_debug={'front_distance':df,'back_distance':db,'votes':self.yaw_votes,'resolved':self.yaw_resolved}
    def plan_navigation(self):
        center=self.object[:2,3];normal=self.visible_side*self.object[:2,0];normal/=max(np.linalg.norm(normal),1e-6);choices=[]
        for distance in [.65,.73,.81,.90]:
            for angle in np.linspace(-1.2,1.2,11):
                direction=rot2(angle)@normal;yaw=np.arctan2(-direction[1],-direction[0]);left=np.array([-np.sin(yaw),np.cos(yaw)]);goal=center+distance*direction-.12*left;path=self.map.path(self.odom[:2],goal)
                if path:choices.append((sum(np.linalg.norm(b-a) for a,b in zip(path,path[1:]))+.3*abs(angle)+distance,goal,yaw,path))
        if choices:
            _,self.nav,self.heading,self.path=min(choices,key=lambda x:x[0]);return True
        return False
    def forward(self,obs):
        data={k:v[0].detach().cpu().numpy() for k,v in obs.items() if k in ['robot_r1::proprio','robot_r1::cam_rel_poses'] or (k.startswith('robot_r1::robot_r1:') and k.endswith(('::rgb','::depth_linear')))}
        s=data['robot_r1::proprio'];q=self.servo.q(s);self.step+=1;self.age+=1
        self.odom[:2]+=rot2(self.odom[2])@s[:2]/30;self.odom[2]+=s[2]/30
        cams=make_cameras(data,size=None)
        if self.step%5==1:self.level=self.ground.update(make_cameras(data,size=480)['head'])
        if self.step%5==1 and self.phase in ['search','navigate','find_tv','place_navigate']:self.odom=self.vo.update(level_cameras(make_cameras(data,size=480),self.level)['head'],self.step,self.odom)
        base=self.base_transform();inverse=np.linalg.inv(base)
        if self.grip_object is not None and self.phase in ['lift','carry','find_tv','place_navigate','place'] and self.step%6==0:self.resolve_held_yaw(cams,q)
        if self.step%24==1:
            tv=self.tv_detector.fit(cams['head'])
            if tv is not None:
                T=np.eye(4);T[:3,3]=tv['position'];T[:3,:3]=tv['rotation'];self.tv=base@T
                cloud,_,_=pointcloud(cams['head'],stride=3);candidates=[]
                for sign in [-1,1]:
                    for gap in [.05,.08,.11]:
                        for front in [.08,.14]:
                            p=tv['position']+tv['rotation'][:,1]*sign*(tv['width']/2+gap)+tv['rotation'][:,0]*front
                            nearby=cloud[(np.linalg.norm(cloud[:,:2]-p[:2],axis=1)<.07)&(cloud[:,2]<tv['position'][2]-.16)&(cloud[:,2]>tv['position'][2]-.6)]
                            if len(nearby)<15:continue
                            bins=np.round(nearby[:,2]/.01);values,counts=np.unique(bins,return_counts=True);height=values[np.argmax(counts)]*.01
                            if np.sum(np.abs(nearby[:,2]-height)<.018)<12 or np.any(nearby[:,2]>height+.05):continue
                            p[2]=height+.085;candidates.append((gap+front*.2,p.copy()))
                if candidates:
                    point=min(candidates,key=lambda x:x[0])[1];self.place_goal=base[:3,:3]@point+base[:3,3]
        if self.action is None:self.action=np.r_[np.zeros(3),s[53:57],s[3:10],1.,s[28:35],1.]
        action=self.action.copy();action[:3]=0;action[14]=-1 if self.phase in ['close','lift','carry','find_tv','place_navigate','place','press_power'] else 1
        if self.step%15==1:
            capsules=[(self.level[:3,:3]@a+self.level[:3,3],self.level[:3,:3]@b+self.level[:3,3],radius) for a,b,radius in self.servo.capsules(q)];self.map.radius=.46 if self.grip_object is not None else .276;self.map.update(level_cameras(make_cameras(data,size=256),self.level),self.odom,capsules)
        fit=None
        if self.step%12==1 and self.phase in ['search','navigate','raise_hand','settle_view','pregrasp','descend','close','lift']:
            fit=self.tracker.fit({'head':cams['head']} if self.phase in ['search','navigate'] else cams)
            if fit is not None:
                T=np.eye(4);T[:3,3]=fit['position'];T[:3,:3]=fit['rotation'];self.object=base@T;self.visible_side=fit['visible_side'];self.last_seen=self.step
        if self.step==1 or self.step%180==0:
            np.savez_compressed(self.output/f'observation_{self.step:05d}.npz',**data)
        error=None
        if self.phase=='search':
            if self.object is not None and self.step-self.last_seen<20:
                if self.plan_navigation():self.change('navigate')
            else:
                action[2]=.32
                if self.age>600:
                    # A finite local exploration move, accepted only through the depth occupancy map.
                    options=[]
                    for theta in np.linspace(-np.pi,np.pi,12,endpoint=False):
                        goal=self.odom[:2]+.65*np.array([np.cos(theta),np.sin(theta)]);path=self.map.path(self.odom[:2],goal)
                        if path:options.append((abs(wrap(theta-self.odom[2])),goal))
                    if options:
                        goal=min(options,key=lambda x:x[0])[1];delta=rot2(self.odom[2]).T@(goal-self.odom[:2]);action[:2]=np.clip(delta*.4,-.18,.18);action[2]=0
                    if self.age>780:self.age=0
        elif self.phase=='navigate':
            if self.age%90==1:self.plan_navigation()
            delta=self.nav-self.odom[:2];distance=np.linalg.norm(delta)
            if self.path:
                near=np.argmin([np.linalg.norm(x-self.odom[:2]) for x in self.path]);waypoint=self.path[min(near+3,len(self.path)-1)];v=rot2(self.odom[2]).T@(waypoint-self.odom[:2]);action[:2]=np.clip(v*.9,-.22,.22)
            action[2]=np.clip(wrap(self.heading-self.odom[2])*.8,-.35,.35)
            if distance<.045 and abs(wrap(self.heading-self.odom[2]))<.14:
                self.change('raise_hand');self.grasp=None
        elif self.phase=='raise_hand':
            error=self.servo.step(action,q,np.array([.38,.38,.90]),np.eye(3),plan=self.age==1)
            if error[0]<.035 and error[1]<.2:self.change('settle_view')
        elif self.phase=='settle_view':
            if self.age>36 and self.step-self.last_seen<15:self.change('pregrasp')
        elif self.phase in ['pregrasp','descend','close','lift']:
            if self.grasp is None and self.object is not None:
                obj=inverse@self.object;normal=obj[:3,0]*self.visible_side;up=obj[:3,2];closing=normal;forward=up/np.linalg.norm(up);rotation=np.column_stack([np.cross(closing,forward),closing,forward]);rotation=R.from_matrix(rotation).as_matrix()
                # Grasp off-center along the top edge, leaving the central nail attachment clear.
                edge=-.020*np.sign(obj[:2,1]@obj[:2,3])
                grip=obj[:3,3]+obj[:3,:3]@np.array([0.,edge,.025]);choices=[]
                for sign in [1,-1]:
                    candidate=rotation@np.diag([sign,sign,1]);solution=self.servo.solve(q,grip+up*.09,candidate)
                    if solution is not None:choices.append((np.linalg.norm(solution-q[self.servo.indices]),candidate,solution))
                if choices:
                    _,rotation,solution=min(choices,key=lambda x:x[0])
                self.grasp={'point':grip,'rotation':rotation,'initial_object_z':obj[2,3]}
            if self.grasp is not None:
                goal=self.grasp['point'].copy();rotation=self.grasp['rotation']
                if self.phase=='pregrasp':
                    goal+=rotation[:,2]*.15
                    if not self.pregrasp_ready:
                        raised=self.servo.pose(q)[:3,3].copy()
                        if self.age==1:self.orientation_point=raised
                        goal=self.orientation_point
                    else:goal[2]=max(goal[2],self.orientation_point[2])
                if self.phase=='lift':
                    goal=self.lift_goal.copy();rotation=self.lift_rotation
                if self.phase!='close':
                    make_plan=False
                    error=self.servo.step(action,q,goal,rotation,plan=make_plan)
                    if make_plan:self.last_plan=self.age
                    if self.phase=='pregrasp' and error[0]<.045 and error[1]<.40:
                        if not self.pregrasp_ready:self.pregrasp_ready=True
                        else:self.change('descend')
                    elif self.phase=='descend' and (error[0]<.018 or (self.age>100 and error[0]<.030)) and error[1]<.22:self.change('close')
                    elif self.phase=='lift' and (error[0]<.06 or self.age>210):
                        measured=None if self.object is None or self.step-self.last_seen>36 else (inverse@self.object)[2,3]
                        motion=np.linalg.norm(self.servo.pose(q)[:3,3]-self.close_pose[:3,3]);finger_contact=float(s[24:26].sum())>.004
                        if (measured is not None and measured-self.grasp['initial_object_z']>.06) or (finger_contact and motion>.06 and self.age>90):
                            self.change('carry')
                        elif self.age>240:self.change('pickup_unconfirmed')
                elif self.age>45:
                    self.close_pose=self.servo.pose(q).copy();self.lift_rotation=self.close_pose[:3,:3].copy();direction=-self.close_pose[:3,3].copy();direction[2]=0;direction/=max(np.linalg.norm(direction),1e-6);self.lift_goal=self.close_pose[:3,3]+direction*.15+np.array([0.,0.,.10]);self.grip_object=np.linalg.inv(base@self.close_pose)@self.object;self.change('lift')
        elif self.phase=='carry':
            held=self.servo.pose(q)@self.grip_object;held[:3,3]=np.array([.40,0.,.80]);goal_grip=held@np.linalg.inv(self.grip_object);error=self.servo.step(action,q,goal_grip[:3,3],goal_grip[:3,:3],plan=False)
            if error[0]<.06 or self.age>300:self.change('find_tv')
        elif self.phase=='find_tv':
            if self.tv is None or self.place_goal is None:action[2]=.30
            else:
                normal=self.tv[:2,0];options=[]
                for radius in [.60,.68,.76,.84]:
                    goal=self.place_goal[:2]+normal*radius;path=self.map.path(self.odom[:2],goal)
                    if path:options.append((sum(np.linalg.norm(b-a) for a,b in zip(path,path[1:])),goal,path))
                if options:
                    _,self.nav,self.path=min(options,key=lambda x:x[0]);self.heading=np.arctan2(-normal[1],-normal[0]);self.change('place_navigate')
                else:action[2]=.2
        elif self.phase=='place_navigate':
            near=np.argmin([np.linalg.norm(x-self.odom[:2]) for x in self.path]);waypoint=self.path[min(near+3,len(self.path)-1)];v=rot2(self.odom[2]).T@(waypoint-self.odom[:2]);action[:2]=np.clip(v*.8,-.18,.18);action[2]=np.clip(wrap(self.heading-self.odom[2])*.7,-.3,.3)
            if np.linalg.norm(self.nav-self.odom[:2])<.06 and abs(wrap(self.heading-self.odom[2]))<.15:self.change('place')
        elif self.phase=='place':
            if self.frozen_goal is None:self.frozen_goal=self.place_goal.copy();self.frozen_tv=self.tv.copy()
            desired=np.eye(4);desired[:3,:3]=np.column_stack([self.frozen_tv[:3,1],-self.frozen_tv[:3,0],np.array([0.,0.,1.])]);desired[:3,3]=self.frozen_goal;desired[2,3]+=max(.006,.20-self.age*.001)
            goal_grip=inverse@desired@np.linalg.inv(self.grip_object);self.release_goal=goal_grip.copy();error=self.servo.step(action,q,goal_grip[:3,3],goal_grip[:3,:3],plan=False)
            held=base@self.servo.pose(q)@self.grip_object;local=inverse@held;bottom=held[2,3]-.06434;near_support=[]
            for camera in cams.values():
                cloud,_,_=pointcloud(camera,stride=4);cloud=cloud@base[:3,:3].T+base[:3,3];distance=np.linalg.norm(cloud[:,:2]-held[:2,3],axis=1);near=cloud[(distance>.065)&(distance<.19)&(cloud[:,2]>self.frozen_goal[2]-.18)&(cloud[:,2]<self.frozen_goal[2]-.025)]
                if len(near)>15:near_support.extend(near[:,2].tolist())
            supported=False;self.place_debug={'support_points':len(near_support),'object_bottom':float(bottom),'held_center':held[:3,3].tolist()}
            if len(near_support)>25:
                values,counts=np.unique(np.round(np.array(near_support)/.01),return_counts=True);height=values[np.argmax(counts)]*.01;supported=(-.012<bottom-height<.055 and held[2,2]>.88 and np.linalg.norm(held[:2,3]-self.frozen_tv[:2,3])<.80)
            self.place_debug['supported']=bool(supported)
            if supported:self.support_height=float(height)
            if self.age>180 and ((error[0]<.025 and self.age>200) or supported):
                self.released_object=held.copy();
                if self.support_height is not None:self.released_object[2,3]=self.support_height+.06434
                if self.button_grip is not None:
                    observed=(base@self.servo.pose(q)@np.r_[self.button_grip,1.])[:3]
                    observed[2]+=self.released_object[2,3]-held[2,3];self.button_world=observed
                self.change('release')
        elif self.phase=='release':
            if self.age>45:self.change('retract')
        elif self.phase=='retract':
            goal=self.release_goal[:3,3]+np.array([0.,0.,.18]);error=self.servo.step(action,q,goal,self.release_goal[:3,:3],plan=False)
            if error[0]<.04:
                tip=self.servo.spec['finger_tip_offsets']['left'];self.servo.link=tip['link_name'];self.servo.point=np.asarray(tip['position_link']);self.change('press_power')
        elif self.phase=='press_power':
            action[14]=-1
            # FK at release is anchored by the observed support plane. Free upright-model
            # fitting can jump vertically or flip yaw when the hand occludes the blue base.
            target=None if self.released_object is None else self.released_object.copy()
            if target is not None:
                if self.object is not None and self.step-self.last_seen<36:
                    delta=self.object[:2,3]-target[:2,3]
                    if np.linalg.norm(delta)<.06:target[:2,3]+=.35*delta
                prior=target[:3,3]+target[:3,:3]@np.array([-.001497,-.066087,-.032603])
                button=(inverse@np.r_[prior,1])[:3];candidates=[]
                for camera in cams.values():
                    mask=red_mask(camera['rgb']);n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
                    for label in range(1,n):
                        if stats[label,cv2.CC_STAT_AREA]<2:continue
                        points,_,_=pointcloud(camera,labels==label)
                        if len(points)<2:continue
                        center=np.median(points,axis=0);world=base[:3,:3]@center+base[:3,3]
                        if np.linalg.norm(center-button)<.055 and np.linalg.norm(np.ptp(points,axis=0))<.04 and (self.support_height is None or world[2]>self.support_height+.008):candidates.append((np.linalg.norm(center-button),world))
                if candidates:self.button_world=min(candidates,key=lambda x:x[0])[1]
                if self.button_world is not None and np.linalg.norm(self.button_world-prior)<.08:button=(inverse@np.r_[self.button_world,1])[:3]
                normal=inverse[:3,:3]@(-target[:3,1]);normal[2]=0;normal/=max(np.linalg.norm(normal),1e-6);up=np.array([0.,0.,1.]);side=np.cross(up,normal)
                # Tilt the palm above the cabinet: a horizontal wrist puts its
                # collision geometry below the observed support plane.
                direction=normal+.65*up;direction/=np.linalg.norm(direction);vertical=np.cross(direction,side);rotation=np.column_stack([side,vertical,direction])
                green_seen=False
                for camera in cams.values():
                    rgb=camera['rgb'].astype(float);mask=(rgb[...,1]>90)&(rgb[...,1]>1.4*rgb[...,0])&(rgb[...,1]>1.3*rgb[...,2]);points,_,_=pointcloud(camera,mask)
                    if np.count_nonzero(np.linalg.norm(points-button,axis=1)<.04)>=3:green_seen=True
                if green_seen:
                    self.power_retreat=button+direction*.12;self.power_rotation=rotation.copy();self.change('verify_power')
                if not self.press_orientation_ready:
                    goal=button+direction*.18;error=self.servo.step(action,q,goal,rotation,plan=self.age==1)
                    if error[0]<.035 and error[1]<.20:self.press_orientation_ready=True;self.servo.target=None
                else:
                    offset=max(-.008,.075-self.press_count*.0006);goal=button+direction*offset
                    if self.press_count>165:
                        grid=[(0.,0.),(.009,0.),(-.009,0.),(0.,.009),(0.,-.009),(.009,.009),(-.009,.009)]
                        du,dv=grid[((self.press_count-165)//60)%len(grid)];goal+=side*du+vertical*dv
                    error=self.servo.step(action,q,goal,rotation,plan=False)
                    if error[0]<.012 and error[1]<.15:self.press_ready=True
                    if self.press_ready:self.press_count+=1
                self.button_debug={'support_height':self.support_height,'goal':goal.tolist(),'point':button.tolist(),'rgb_candidates':len(candidates),'error':error[0],'orientation_error':error[1],'ready':self.press_orientation_ready,'contact_age':self.press_count}
        elif self.phase=='verify_power':
            error=self.servo.step(action,q,self.power_retreat,self.power_rotation,plan=False)
        if self.step%15==0:
            row={'step':self.step,'phase':self.phase,'age':self.age,'odom':self.odom.tolist(),'last_seen':self.last_seen,'object':None if self.object is None else self.object.tolist(),'gripper':self.servo.pose(q).tolist(),'error':error,'fit':None if fit is None else {k:v for k,v in fit.items() if k in ['matches','depth_error','pixel_error','view']},'grip_joints':s[24:26].tolist(),'ground_frame':self.ground.stats,'button':getattr(self,'button_debug',{}),'placement':self.place_debug,'television_seen':self.tv is not None,'place_goal':None if self.place_goal is None else self.place_goal.tolist(),'grasp_transform':None if self.grip_object is None else self.grip_object.tolist(),'mount':None if self.mount is None else self.mount['point'].tolist()}
            row['marker_orientation']=self.marker_debug
            with (self.output/'trace.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        self.action=action.copy();assert np.isfinite(action).all();return torch.as_tensor(action[None],dtype=torch.float32)
