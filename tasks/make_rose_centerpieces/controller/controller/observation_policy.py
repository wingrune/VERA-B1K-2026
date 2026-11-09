"""Task-specific RGB-D manipulation controllers using allowed observations only."""
import os,json
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial.transform import Rotation as R
from perception import make_cameras,pointcloud
from object_perception import ObjectPerception
from controller import ArmServo
from navigation import DepthMap,rot2,wrap
from ground_frame import GroundFrameEstimator,level_cameras
from visual_odometry import VisualOdometry
from robot_collision import RobotCollision
from arm_parking import plan_right_parking
from active_search import ActiveSearch
from reach_planner import joint_path
from wrist_view import plan_floor_view
from floor_coverage import FloorCoverage
from floor_dirt import floor_marks
from grasp_regions import grasp_regions
from held_tool_tracking import refine_held
from base_recovery import BaseRecovery
TASKS={
 'scrubbing_bathroom_floor':('scrub_brush',None,'floor'),
 'clean_a_keyboard':('pipe_cleaner','keyboard','wipe'),
 'attach_a_camera_to_a_tripod':('digital_camera','camera_tripod','attach'),
 'make_cabinet_doors':('cabinet_door','cabinet_base','attach'),
 'installing_a_fax_machine':('facsimile',None,'place_toggle')}
def transform(rot=None,p=None):
 t=np.eye(4)
 if rot is not None:t[:3,:3]=rot
 if p is not None:t[:3,3]=p
 return t
def link_pose(meta,kind):
 item=next(v for v in meta['links'].values() if kind in v['meta_link_id'])
 return transform(R.from_quat(item['quaternion_xyzw']).as_matrix(),np.array(item['position']))
class Policy:
 def __init__(self):
  self.root=Path(__file__).parent;self.output=Path(os.environ['POLICY_OUTPUT']);self.output.mkdir(parents=True,exist_ok=True)
  self.task=os.environ['TASK_NAME'];self.child,self.parent,self.mode=TASKS[self.task];self.categories=[x for x in [self.child,self.parent] if x]
  self.perception=ObjectPerception(self.root/'priors',self.categories);self.servo=ArmServo(self.root);self.robot=RobotCollision(self.root/'robot_collision_spheres.json')
  self.reset()
 def set_action_dim(self,n):assert n==23
 def reset(self):
  self.task_failed_stances=[];self.grasp_trials={};self.step=0;self.age=0;self.phase='view_floor';self.odom=np.zeros(3);self.ground=GroundFrameEstimator();self.level=np.eye(4);self.vo=VisualOdometry();self.map=DepthMap();self.map.res=.08;self.map.n=512;self.map.log_odds=np.zeros((512,512,self.map.nz),np.int8);self.recovery=BaseRecovery()
  self.poses={};self.seen={};self.stats={};self.action=None;self.grasp=None;self.held=None;self.nav=None;self.path=None;self.retries=0;self.waypoints=[];self.waypoint=0;self.servo.target=None;self.place=None;self.floor_patches=[];self.exploration=0;self.blocked=0;self.support_candidates=[];self.parked=False;self.park_target=None;self.initial_gaze=None;self.searcher=ActiveSearch(global_frontiers=self.mode=='place_toggle');self.search_moving=False;self.view_target=None;self.inspection_positions=[];self.coverage=FloorCoverage()
 def change(self,phase):
  self.phase=phase;self.age=0;self.servo.target=None;print('PHASE',self.step,phase,flush=True)
 def base(self):
  t=transform(R.from_euler('z',self.odom[2]).as_matrix(),np.r_[self.odom[:2],0.]);return t@self.level
 def bounds(self,cat):return self.perception.models[cat]['bounds']
 def center(self,cat):
  center=self.bounds(cat).mean(0)
  if cat=='pipe_cleaner':
   model=self.perception.models[cat];rgb=model['rgb'].astype(float);mask=rgb.max(1)-rgb.min(1)>90
   if mask.sum()>20:center=np.median(model['xyz'][mask],axis=0)
  return self.poses[cat][:3,3]+self.poses[cat][:3,:3]@center
 def plan_navigation(self,target,inspect=False):
  choices=[];proposals=[];occupied=self.map.occupancy()
  radii=[.64,.80,.95,1.10,1.25] if self.held is not None and self.mode in ['wipe','place_toggle'] else [.48,.56,.64,.74,.84,.95,1.05]
  for radius in radii:
   for theta in np.linspace(-np.pi,np.pi,24,endpoint=False):
    pos=target[:2]+radius*np.array([np.cos(theta),np.sin(theta)]);yaw=wrap(theta+np.pi)
    pos+=rot2(yaw)@np.array([0,-.12])
    if self.held is not None and any(np.linalg.norm(pos-old)<.25 for old in self.task_failed_stances):continue
    if inspect and (np.linalg.norm(pos-self.odom[:2])<.5 or any(np.linalg.norm(pos-v)<.35 for v in self.inspection_positions)):continue
    cell=self.map.ij(pos)
    if not np.all((cell>=0)&(cell<self.map.n)) or occupied[tuple(cell)]:continue
    score=np.linalg.norm(pos-self.odom[:2])+.12*abs(wrap(yaw-self.odom[2]))+(.8 if self.held is not None else 1.8)*abs(radius-.60)
    proposals.append((score,pos,yaw,radius))
  for _,pos,yaw,radius in sorted(proposals,key=lambda x:x[0])[:16]:
    path=self.map.path(self.odom[:2],pos,occupancy=occupied)
    if path:
     length=sum(np.linalg.norm(b-a) for a,b in zip(path,path[1:]));choices.append((length+.12*abs(wrap(yaw-self.odom[2]))+(.8 if self.held is not None else 1.8)*abs(radius-.60),pos,yaw,path))
  if not choices:return False
  _,self.nav,self.heading,self.path=min(choices,key=lambda x:x[0]);return True
 def drive(self,action):
  if self.path:
   near=np.argmin([np.linalg.norm(x-self.odom[:2]) for x in self.path]);point=self.path[min(near+3,len(self.path)-1)]
   action[:2]=np.clip(rot2(self.odom[2]).T@(point-self.odom[:2])*.8,-.18,.18)
  action[2]=np.clip(wrap(self.heading-self.odom[2])*.8,-.30,.30)
  return np.linalg.norm(self.nav-self.odom[:2])<.05 and abs(wrap(self.heading-self.odom[2]))<.13
 def choose_grasp(self,q,base,cams=None,scene_override=None):
  obj=np.linalg.inv(base)@self.poses[self.child];lo,hi=self.bounds(self.child);extent=hi-lo;center=(lo+hi)/2;choices=[]
  from scene_collision import ObservedSceneCollision
  scene=scene_override if scene_override is not None else (None if cams is None else ObservedSceneCollision(cams,self.robot,self.servo.kin.forward(q)[0],obj,lo,hi))
  grasp_center=center.copy();grasp_extent=extent.copy()
  if self.child=='pipe_cleaner':
   model=self.perception.models[self.child];rgb=model['rgb'].astype(float);saturation=rgb.max(1)-rgb.min(1);colored=model['xyz'][saturation>90]
   if len(colored)>20:
    grasp_center=np.median(colored,axis=0)
    section=model['xyz'][abs(model['xyz'][:,0]-grasp_center[0])<.015]
    if len(section)>15:grasp_extent[1:]=np.ptp(section,axis=0)[1:]
  # Pinch a thin dimension. Wide planar objects are approached from their top
  # edge; narrow tools are pinched from above while preserving their working end.
  regions=grasp_regions(self.child,self.perception.models[self.child])
  for axis in ([1] if self.child in ['pipe_cleaner','scrub_brush'] else np.argsort(grasp_extent)):
   if grasp_extent[axis]<=.105:regions.append((grasp_center.copy(),int(axis),float(grasp_extent[axis]),'bounding_box'))
  self.grasp_region_debug=[]
  for region_center,axis,width,region_name in regions:
   close=obj[:3,axis]
   if self.child=='rose' and abs((self.level[:3,:3]@close)[2])>.30:continue
   for sign in [1.,-1.]:
    closing=close*sign
    up=np.array([0.,0.,1.]);approach=up-closing*np.dot(closing,up)
    if np.linalg.norm(approach)<.35:
     approach=-obj[:3,3];approach[2]=0;approach-=closing*np.dot(closing,approach)
    approach/=max(np.linalg.norm(approach),1e-8)
    rotation=np.column_stack([np.cross(closing,approach),closing,approach])
    local=region_center.copy()
    # For a tall door, avoid grasping its bottom edge or hinge.
    upaxis=int(np.argmax(np.abs(obj[2,:3])))
    if extent[upaxis]>.3 and self.child=='cabinet_door' and region_name=='bounding_box':local[upaxis]=hi[upaxis]-.020 if obj[2,upaxis]>0 else lo[upaxis]+.020
    point=obj[:3,:3]@local+obj[:3,3]
    if self.child=='rose':point+=self.level[:3,:3].T@np.array([0.,0.,.007])
    pre=point+approach*.14
    solution=self.servo.solve(q,pre,rotation,validator=None if scene is None else lambda trial:scene.valid(self.servo.kin.forward(trial)[0],None))
    if solution is None:continue
    trial=q.copy();trial[self.servo.indices]=solution
    if not self.robot.valid(self.servo.kin.forward(trial)[0],self.robot.margins(self.servo.kin.forward(q)[0])):continue
    self.grasp_region_debug.append(dict(name=region_name,point=point.tolist(),width=width))
    region_penalty=1.5 if region_name=='bounding_box' and len(regions)>1 else 0.
    # M15: prefer the wider blossom body, keeping base as a feasible fallback.
    if self.child=='rose' and region_name.startswith('blossom_base_'):region_penalty+=.5
    if self.child=='cabinet_door' and abs(obj[2,0])<.45:region_penalty=0. if region_name=='bounding_box' else 3.
    choices.append((np.linalg.norm(solution-q[self.servo.indices])+.7*width+region_penalty+2.5*self.grasp_trials.get(region_name,0),point,rotation,solution,region_name))
  if not choices:return False
  selected=None
  for _,point,rot,sol,region_name in sorted(choices,key=lambda x:x[0]):
   path=[sol] if cams is None and scene_override is None else joint_path(self.servo,self.robot,q,sol,cams,obj,self.bounds(self.child),seconds=3.,scene_override=scene)
   if path is not None:
    preq=q.copy();preq[self.servo.indices]=sol
    baseline=self.robot.margins(self.servo.kin.forward(preq)[0])
    end=self.servo.solve(preq,point,rot,validator=lambda trial:self.robot.valid(self.servo.kin.forward(trial)[0],baseline) and (scene is None or scene.valid(self.servo.kin.forward(trial)[0],None)))
    if end is None:continue
    descend=[end] if cams is None and scene_override is None else joint_path(self.servo,self.robot,preq,end,cams,obj,self.bounds(self.child),seconds=3.,scene_override=scene)
    if descend is not None:selected=(point,rot,sol,path,descend,region_name);break
  if selected is None:return False
  point,rot,sol,path,descend,region_name=selected
  self.grasp_path=list(path);self.descend_path=list(descend)
  self.grasp=dict(region=region_name,nominal_width=float(next((w for c,a,w,n in regions if n==region_name),.03)),point=point,rotation=rot,initial_height=self.center(self.child)[2],solution=sol.copy());self.servo.target=sol;self.servo.synchronized=True;return True
 def replan_stalled(self,q,point,rotation,cams,base):
  if self.age==1:self.motion_anchor=q[self.servo.indices].copy();self.stall_debug={}
  if self.age%90 or self.servo.target is None:return
  current=q[self.servo.indices];previous=getattr(self,'motion_anchor',current);self.motion_anchor=current.copy()
  stuck=np.where((abs(current-previous)<.006)&(abs(self.servo.target-current)>.075))[0]
  if not len(stuck):return
  baseline=self.robot.margins(self.servo.kin.forward(q)[0]);locked={int(i):float(current[i]) for i in stuck}
  from scene_collision import ObservedSceneCollision
  obj=self.servo.pose(q)@self.held if self.held is not None else np.linalg.inv(base)@self.poses[self.child]
  scene=ObservedSceneCollision(cams,self.robot,self.servo.kin.forward(q)[0],obj,*self.bounds(self.child))
  solution=self.servo.solve(q,point,rotation,validator=lambda trial:scene.valid(self.servo.kin.forward(trial)[0],None),locked=locked)
  self.stall_debug=dict(step=self.step,locked=locked,alternative_found=solution is not None)
  if solution is None:return
  obj=self.servo.pose(q)@self.held if self.held is not None else np.linalg.inv(base)@self.poses[self.child]
  path=joint_path(self.servo,self.robot,q,solution,cams,obj,self.bounds(self.child),seconds=3.,locked=tuple(locked))
  if path is None:self.stall_debug['collision_free_path']=False;return
  self.stall_debug['collision_free_path']=True
  self.servo.target=path.pop(0);self.servo.synchronized=True
  if self.phase=='pregrasp':self.grasp_path=path;self.descend_path=[]
  elif self.phase=='execute':self.task_joint_path=path
 def explore(self,action):
  if self.search_moving:
   if self.searcher.step(self.map,self.odom,action):
    local=self.searcher.debug.get('kind')=='counter_clearance' and self.searcher.debug.get('viewpoint_reached')
    self.search_moving=False;self.age=120 if local and self.held is not None else 0;self.nav=None;self.exploration+=1
   return
  if self.age<(540 if self.exploration==0 and self.held is None else 120):action[2]=.34;return
  if self.searcher.plan(self.map,self.odom,self.phase):self.search_moving=True
  else:action[2]=.25
  if self.age>720:self.age=0
 def make_task_path(self,base,q,cams):
  if self.held is None:return False
  obj=base@self.servo.pose(q)@self.held
  if self.mode=='attach':
   if self.parent not in self.poses:return False
   target=self.poses[self.parent]
   cm=self.perception.models[self.child]['metadata'];pm=self.perception.models[self.parent]['metadata']
   goal=target@link_pose(pm,'F')@np.linalg.inv(link_pose(cm,'M'))
   direction=goal[:3,3]-target[:3,3];direction[2]=max(direction[2],.1);direction/=max(np.linalg.norm(direction),1e-8)
   # Contact approach followed by a small bounded geometric search.
   for distance in [.12,.06,.025,0.,-.006]:
    t=goal.copy();t[:3,3]+=direction*distance;self.waypoints.append((t,35 if distance<=0 else 5))
   for dx,dy in [(0,.012),(0,-.012),(.012,0),(-.012,0),(0,0)]:
    t=goal.copy();t[:3,3]+=np.array([dx,dy,0]);self.waypoints.append((t,20))
  elif self.mode=='wipe':
   if self.parent not in self.poses:return False
   target=self.poses[self.parent];lo,hi=self.bounds(self.parent);clo,chi=self.bounds(self.child)
   # Brush bristles touch the keyboard plane; overlapping lanes cover the face.
   rotation=R.from_euler('z',np.arctan2(obj[1,0],obj[0,0])).as_matrix();long=int(np.argmax(hi-lo));cross=1-long
   model=self.perception.models[self.child];rgb=model['rgb'].astype(float);white=model['xyz'][(rgb.min(1)>155)&(rgb.max(1)-rgb.min(1)<60)];working=np.median(white,axis=0) if len(white)>20 else np.zeros(3)
   for lane in np.linspace(lo[cross]+.01,hi[cross]-.01,4):
    for endpoint in ([lo[long]+.025,hi[long]-.025] if len(self.waypoints)%4==0 else [hi[long]-.025,lo[long]+.025]):
     local=(lo+hi)/2;local[cross]=lane;local[long]=endpoint;local[2]=hi[2]-.004-clo[2]
     position=target[:3,:3]@local+target[:3,3]-rotation@np.r_[working[:2],0.]
     self.waypoints.append((transform(rotation,position),20))
   first=self.waypoints[0][0].copy();first[2,3]+=.12;self.waypoints.insert(0,(first,10))
  elif self.mode=='floor':
   center=self.coverage.target
   if center is None:return False
   yaw=R.from_euler('z',np.arctan2(obj[1,0],obj[0,0])).as_matrix()
   rot=yaw@R.from_euler('y',-.65).as_matrix()
   bottom=float(np.min(self.perception.models[self.child]['xyz']@rot[2]))
   if self.coverage.target_is_mark:
    rotated=self.perception.models[self.child]['xyz']@rot.T
    tip=np.median(rotated[rotated[:,2]<bottom+.006],axis=0)
    offsets=[(-.045,-.045),(.045,-.045),(.045,.045),(-.045,.045),(0.,0.)]
    for x,y in offsets:
     pos=center+yaw@np.array([x,y,0])-np.r_[tip[:2],0.]
     pos[2]=-.002-bottom;self.waypoints.append((transform(rot,pos),8))
    first=self.waypoints[0][0].copy();first[2,3]+=.09;self.waypoints.insert(0,(first,8))
   else:
    for i,x in enumerate(np.linspace(-.18,.18,4)):
     for y in ([-.20,.20] if i%2==0 else [.20,-.20]):
      pos=center+yaw@np.array([x,y,0]);pos[2]=-.001-bottom;self.waypoints.append((transform(rot,pos),6))
    first=self.waypoints[0][0].copy();first[2,3]+=.12;self.waypoints.insert(0,(first,10))
  else:
   if self.place is None:return False
   rot=R.from_euler('z',self.odom[2]).as_matrix();goal=transform(rot,self.place.copy());goal[2,3]-=self.bounds(self.child)[0,2];self.release_pose=goal.copy()
   above=goal.copy();above[2,3]+=.13;self.waypoints=[(above,15),(goal,60)]
  self.waypoint=0;self.stable=0;return True
 def find_support(self,cams,base):
  from carrel_support import carrel_support
  found=carrel_support(cams,self.level)
  if found is None:return None
  point,self.support_debug=found
  return np.r_[rot2(self.odom[2])@point[:2]+self.odom[:2],point[2]]
 def forward(self,obs):
  data={k:v[0].detach().cpu().numpy().copy() for k,v in obs.items() if k in ['robot_r1::proprio','robot_r1::cam_rel_poses'] or (k.startswith('robot_r1::robot_r1:') and k.endswith(('::rgb','::depth_linear')))}
  s=data['robot_r1::proprio'];q=self.servo.q(s);self.step+=1;self.age+=1
  if self.initial_gaze is None:self.initial_gaze=float(s[53]+s[54]-s[55])
  self.odom[:2]+=rot2(self.odom[2])@s[:2]/30;self.odom[2]+=s[2]/30;cams=make_cameras(data,size=None)
  self.current_cameras=cams
  if self.step%10==1:self.level=self.ground.update(make_cameras(data,size=480)['head'])
  if self.step%5==1:
   self.odom=self.vo.update(level_cameras(make_cameras(data,size=480),self.level)['head'],self.step,self.odom)
  for camera in cams.values():camera['ground_from_body']=self.level.copy()
  base=self.base();inverse=np.linalg.inv(base);self.perception.level=self.level;self.perception.held=self.child if self.held is not None else None;self.perception.robot_capsules=self.servo.capsules(q)
  if self.mode=='floor' and self.step%30==1:
   self.coverage.update(cams,self.level,self.odom)
   if self.held is not None and self.phase not in ['execute','descend','close']:
    small=make_cameras(data,size=640);marks=floor_marks(small,self.level,self.servo.capsules(q))
    self.coverage.observe_marks(marks,self.odom,self.step)
    self.floor_mark_debug=marks
  if self.step%15==1:
   small=level_cameras(make_cameras(data,size=320),self.level);caps=[(self.level[:3,:3]@a+self.level[:3,3],self.level[:3,:3]@b+self.level[:3,3],r) for a,b,r in self.servo.capsules(q)]
   centers=self.robot.centers(self.servo.kin.forward(q)[0])
   caps.extend((self.level[:3,:3]@center+self.level[:3,3],self.level[:3,:3]@center+self.level[:3,3],radius+.015) for center,radius in zip(centers,self.robot.radii))
   carried=None if self.held is None else (self.level@self.servo.pose(q)@self.held,*self.bounds(self.child))
   fk=self.servo.kin.forward(q)[0];levinv=np.linalg.inv(self.level)
   extra_boxes=[]
   for extra in getattr(self.robot,'extra_payloads',[]):
    hand=fk[extra['arm']+'_gripper_link'].copy();hand[:3,3]+=hand[:3,:3]@np.array([0.,0.,-.072])
    extra_boxes.append((self.level@hand@extra['local'],*extra['bounds']))
   def mask(points):
    result=self.robot.point_self_mask(points@levinv[:3,:3].T+levinv[:3,3],fk) if self.robot.hulls else self.map.self_mask(points,caps)
    for pose,lo,hi in extra_boxes:
     local=(points-pose[:3,3])@pose[:3,:3]
     result|=((local>=lo-.055)&(local<=hi+.055)).all(axis=1)
    return result
   self.map.update(small,self.odom,caps,exclude_box=carried,geometry_mask=mask)
  if self.step%18==1:
   selected=cams if self.task=='make_rose_centerpieces' else ({k:cams[k] for k in ['head','right_wrist']} if self.phase in ['search','navigate','search_target','target_navigate','inspect_parent'] else cams)
   feats=self.perception.features(selected)
   for cat in self.categories:
    fit=self.perception.fit(cat,selected,feats)
    if fit:
     pose,stats=fit;world=base@pose
     if cat not in self.poses or self.step-self.seen[cat]>100 or np.linalg.norm(world[:3,3]-self.poses[cat][:3,3])<.25:
      self.poses[cat]=world;self.seen[cat]=self.step;self.stats[cat]=stats
   if self.mode=='place_toggle' and self.held is not None:
    support=self.find_support(cams,base)
    if support is not None:self.place=support
  if self.action is None:self.action=np.r_[np.zeros(3),s[53:57],s[3:10],1.,s[28:35],1.]
  action=self.action.copy();action[:3]=0.;action[14]=-1 if self.held is not None or self.phase=='close' else 1.;error=None
  if self.phase=='view_floor':
   if self.age==1:self.view_target=plan_floor_view(self.root,q,cams)
   if self.view_target is None:self.change('search')
   else:
    action[15:22]=s[28:35]+np.clip(self.view_target-s[28:35],-.014,.014)
    if np.max(abs(self.view_target-s[28:35]))<.035 or self.age>240:self.change('search')
  elif self.phase=='search':
   # M14: finish the initial RGB-D panorama before grasping a rose when the
   # vase has not been observed. The global step budget expires only once.
   parent_ready=self.task!='make_rose_centerpieces' or self.parent in self.poses or self.step>=540
   if parent_ready and self.child in self.poses and self.step-self.seen[self.child]<30 and self.plan_navigation(self.center(self.child)):self.change('navigate')
   elif self.parent in self.poses and self.age>300 and self.plan_navigation(self.center(self.parent),inspect=True):self.change('inspect_parent')
   else:self.explore(action)
  elif self.phase=='inspect_parent':
   if self.child in self.poses and self.step-self.seen[self.child]<30 and self.plan_navigation(self.center(self.child)):self.change('navigate')
   elif self.drive(action) or self.age>320:self.inspection_positions.append(self.odom[:2].copy());self.nav=None;self.change('search')
  elif self.phase=='navigate':
   if self.age%180==1 and not (hasattr(self,'_route_progress_ok') and self._route_progress_ok()):self.plan_navigation(self.center(self.child))
   if self.drive(action):self.grasp=None;self.change('grasp_plan' if self.parked else 'park_arm')
   elif self.age>(900 if self.task=='make_rose_centerpieces' and self.held is None else 700):self.nav=None;self.change('search')
  elif self.phase=='park_arm':
   if self.age==1:self.park_target=plan_right_parking(self.root,q,cams)
   if self.park_target is None:self.parked=True;self.change('grasp_plan')
   else:
    action[15:22]=s[28:35]+np.clip(self.park_target-s[28:35],-.014,.014)
    if np.max(abs(self.park_target-s[28:35]))<.035 or self.age>220:self.parked=True;self.change('grasp_plan')
  elif self.phase=='grasp_plan':
   freshness_limit=60 if self.child=='rose' else 720 if not self.grasp_trials else 90
   if self.age%30==1 and self.step-self.seen.get(self.child,-1000)<freshness_limit and self.choose_grasp(q,base,cams):self.change('pregrasp');self.servo.target=self.grasp_path.pop(0).copy();self.servo.synchronized=True
   elif self.age>100:
    self.retries+=1;self.nav=None;self.change('search')
  elif self.phase in ['pregrasp','descend','close','lift']:
   point=self.grasp['point'].copy();rot=self.grasp['rotation']
   if 'world_pose' in self.grasp:
    current_grasp=np.linalg.inv(base)@self.grasp['world_pose'];point=current_grasp[:3,3];rot=current_grasp[:3,:3]
   if self.phase=='pregrasp':
    point+=rot[:,2]*.14
    if self.servo.target is None and self.grasp_path:self.servo.target=self.grasp_path.pop(0).copy()
   elif self.phase=='descend':
    if self.servo.target is None and self.descend_path:self.servo.target=self.descend_path.pop(0).copy()
   elif self.phase=='lift':point=(np.linalg.inv(base)@self.lift_world_goal)[:3] if hasattr(self,'lift_world_goal') else self.lift_goal
   if self.phase=='lift' and self.child in ['scrub_brush','pipe_cleaner'] and self.held is not None and self.age%15==0:
    predicted=self.servo.pose(q)@self.held
    fit=refine_held(self.perception.models[self.child],make_cameras(data,size=640),predicted)
    if fit is not None:
     self.held=np.linalg.inv(self.servo.pose(q))@fit[0];self.held_tracking_debug=fit[1]
   if self.phase!='close':
    self.replan_stalled(q,point,rot,cams,base)
    error=self.servo.step(action,q,point,rot)
    if (error[0]<(.045 if self.phase=='pregrasp' else .025) and error[1]<(.35 if self.phase=='pregrasp' else .25)):
     if self.phase=='pregrasp':self.change('descend');self.servo.target=self.descend_path.pop(0).copy() if self.descend_path else None
     elif self.phase=='descend':self.change('close')
     elif self.phase=='lift':
      observed=self.child in self.poses and self.step-self.seen.get(self.child,0)<45
      risen=observed and self.center(self.child)[2]-self.grasp['initial_height']>.055
      hand=base@self.servo.pose(q)
      near_hand=observed and np.linalg.norm(self.center(self.child)-hand[:3,3])<.20
      if self.child in ['scrub_brush','pipe_cleaner'] and hasattr(self,'held_tracking_debug') and float(s[24:26].sum())>.008:
       self.change('carry')
      elif risen and near_hand and self.child not in ['scrub_brush','pipe_cleaner'] and (self.child!='rose' or getattr(self,'pickup_ready',True)):
       self.held=np.linalg.inv(hand)@self.poses[self.child];self.change('carry')
      elif False:pass  # Aperture alone cannot verify pickup.
      elif self.age>180:self.held=None;self.retries+=1;self.change('search')
    if self.age>(750 if self.phase=='pregrasp' else 450):self.held=None;self.retries+=1;self.nav=None;self.change('search')
   elif self.age>160 and self.child=='rose' and not getattr(self,'close_ready',False):
    region=self.grasp.get('region','bounding_box');self.grasp_trials[region]=self.grasp_trials.get(region,0)+1
    self.held=None;self.retries+=1;self.nav=None;self.change('search')
   elif self.age>45 and (self.child!='rose' or getattr(self,'close_ready',False)):
    region=self.grasp.get('region','bounding_box');self.grasp_trials[region]=self.grasp_trials.get(region,0)+1
    if hasattr(self,'held_tracking_debug'):del self.held_tracking_debug
    close=self.servo.pose(q);self.held=np.linalg.inv(base@close)@self.poses[self.child];self.lift_goal=close[:3,3]+np.array([-.04,0,.18]);self.lift_world_goal=base@np.r_[self.lift_goal,1.];self.change('lift')
  elif self.phase=='carry':
   current=self.servo.pose(q)@self.held;current[:3,3]=[.45,.05,getattr(self,'carry_height',.85)];goal=current@np.linalg.inv(self.held)
   error=self.servo.step(action,q,goal[:3,3],goal[:3,:3])
   if error[0]<.055 or self.age>220:self.nav=None;self.change('search_target')
  elif self.phase=='search_target':
   target=self.center(self.parent) if self.parent in self.poses else self.place
   if self.mode=='floor':
    target=self.coverage.choose(self.odom,self.map)
    if target is not None and self.plan_navigation(target):self.change('target_navigate')
    else:
     if target is not None:self.coverage.reject()
     self.explore(action)
   elif target is not None and self.plan_navigation(target):self.change('target_navigate')
   else:self.explore(action)
  elif self.phase=='target_navigate':
   if self.drive(action):
    self.waypoints=[]
    if self.make_task_path(base,q,cams):self.change('execute')
   elif self.age>700:self.nav=None;self.change('search_target')
  elif self.phase=='execute':
   if self.waypoint>=len(self.waypoints):
    if self.mode in ['attach','place_toggle']:self.change('release')
    else:self.change('wipe_retract')
   else:
    pose,wait=self.waypoints[self.waypoint];goal=inverse@pose@np.linalg.inv(self.held)
    if self.age==1:
     self.task_joint_path=[]
     baseline=self.robot.margins(self.servo.kin.forward(q)[0])
     planned=self.servo.solve(q,goal[:3,3],goal[:3,:3],validator=lambda trial:self.robot.valid(self.servo.kin.forward(trial)[0],baseline))
     if planned is not None:self.servo.target=planned;self.servo.synchronized=True
     else:self.age=361
    if self.servo.target is None and self.task_joint_path:self.servo.target=self.task_joint_path.pop(0)
    self.replan_stalled(q,goal[:3,3],goal[:3,:3],cams,base)
    # Keep the final rose waypoint IK command until waypoint progression.
    # Intermediate joint path nodes retain the original clearing behavior.
    self.servo.retain_target=self.task=='make_rose_centerpieces' and not bool(self.task_joint_path)
    try:error=self.servo.step(action,q,goal[:3,3],goal[:3,:3])
    finally:self.servo.retain_target=False
    self.stable=self.stable+1 if error[0]<.026 and error[1]<.20 else 0
    if self.stable>=wait:self.waypoint+=1;self.stable=0;self.age=0
    elif self.age>360:
     if self.mode=='floor':self.coverage.reject();self.change('wipe_retract')
     else:self.task_failed_stances.append(self.odom[:2].copy());self.nav=None;self.change('carry')
  elif self.phase=='wipe_retract':
   held=base@self.servo.pose(q)@self.held
   if self.age==1:self.retract=held.copy();self.retract[2,3]+=.10
   goal=inverse@self.retract@np.linalg.inv(self.held);error=self.servo.step(action,q,goal[:3,3],goal[:3,:3])
   if error[0]<.03:
    if self.mode=='wipe':self.waypoints=[];self.make_task_path(base,q,cams);self.change('execute')
    else:self.coverage.finish(self.odom);self.nav=None;self.change('carry')
  elif self.phase=='release':
   action[14]=1.
   if self.age>50:
    self.held=None
    self.change('press' if self.mode=='place_toggle' else 'observe_result')
  elif self.phase=='press':
   if self.child in self.poses and self.step-self.seen[self.child]<60:
    meta=self.perception.models[self.child]['metadata'];button=link_pose(meta,'0');target=inverse@self.poses[self.child]@button
    if self.age<60:target[2,3]+=.04
    else:target[2,3]-=.004
    error=self.servo.step_position(action,q,target[:3,3]);action[14]=-1
    if self.age>220:self.change('observe_result')
   else:action[2]=.08
  elif self.phase=='observe_result':
   if self.age>180:self.change('done_hold')
  self.recovery.apply(self,action,s)
  # A bounded joint step must not worsen a known self-collision.
  trial=q.copy();trial[self.servo.indices]=np.r_[action[3:7],action[7:14]]
  baseline=self.robot.margins(self.servo.kin.forward(q)[0])
  if not all(self.robot.valid(self.servo.kin.forward(q+alpha*(trial-q))[0],baseline) for alpha in [.33,.67,1.]):
   action[3:7]=s[53:57];action[7:14]=s[3:10];self.blocked+=1
  if self.step==1 or (os.environ.get('POLICY_SAVE_OBSERVATIONS','1')=='1' and (self.step%180==0 or self.phase!=getattr(self,'_last_archived_phase',None))):np.savez_compressed(self.output/f'observation_{self.step:05d}.npz',**data)
  self._last_archived_phase=self.phase
  if self.step%30==0:
   record=dict(delivery_mode=getattr(self,'delivery_mode',None),attempted_releases=getattr(self,'completed_releases',[]),active_child=self.child,active_parent=self.parent,held_tool_tracking=getattr(self,'held_tracking_debug',None),grasp_regions=getattr(self,'grasp_region_debug',[]),floor_marks=getattr(self,'floor_mark_debug',[]),floor_target=None if self.coverage.target is None else self.coverage.target.tolist(),floor_target_is_mark=self.coverage.target_is_mark,step=self.step,phase=self.phase,age=self.age,odom=self.odom.tolist(),seen=self.seen,perception=self.stats,error=error,retries=self.retries,waypoint=self.waypoint,waypoints=len(self.waypoints),self_collision_holds=self.blocked,search_debug=self.searcher.debug,visual_odometry=self.vo.stats,stalled_joint_replan=getattr(self,'stall_debug',{}),base_recovery=self.recovery.debug,level_transform=self.level.tolist(),held_transform=None if self.held is None else self.held.tolist(),servo_target=None if self.servo.target is None else self.servo.target.tolist(),remaining_grasp_path=len(getattr(self,'grasp_path',[])),remaining_descent_path=len(getattr(self,'descend_path',[])),active_task_goal=None if self.waypoint>=len(self.waypoints) else self.waypoints[self.waypoint][0].tolist(),estimated_object_poses={k:v.tolist() for k,v in self.poses.items()},grasp=None if self.grasp is None else {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in self.grasp.items()})
   with (self.output/'trace.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  assert np.isfinite(action).all();self.action=action
  return torch.as_tensor(action,dtype=torch.float32)[None]
