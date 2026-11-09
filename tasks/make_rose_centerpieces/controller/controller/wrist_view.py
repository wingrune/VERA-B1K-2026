"""Aim the free wrist camera using allowed calibration and public kinematics."""
import numpy as np
from scipy.optimize import least_squares
from controller import ArmServo
from robot_collision import RobotCollision
from scene_collision import ObservedSceneCollision
from view_visibility import sight_line

def plan_floor_view(assets,q,cams,diagnostics=None,look_points=None,payload=None,payload_bounds=None):
 servo=ArmServo(assets,'right');robot=RobotCollision(assets/'robot_collision_spheres.json');ids=servo.indices[4:]
 start=servo.kin.forward(q)[0];grip=servo.pose(q);camera=cams['right_wrist'];actual=np.eye(4);actual[:3,:3]=camera['R'];actual[:3,3]=camera['t'];relative=np.linalg.inv(grip)@actual
 lower,upper=([0,0,0],[0,0,0]) if payload_bounds is None else payload_bounds
 scene=ObservedSceneCollision(cams,robot,start,payload,lower,upper);limits=np.array(servo.spec['limits'])[:,ids];choices=[]
 ground=cams['head'].get('ground_from_body',np.eye(4));inverse=np.linalg.inv(ground)
 for floor_point in ([[.7,-.12,0.],[1.2,-.3,0.],[.55,-.65,0.]] if look_points is None else look_points):
  look=(inverse@np.r_[floor_point,1.])[:3]
  anchor=actual[:3,3]
  def residual(x):
   trial=q.copy();trial[ids]=x;t=servo.pose(trial)@relative
   direction=look-t[:3,3];direction/=np.linalg.norm(direction)
   return np.r_[1.5*(t[:3,2]-direction),.4*(t[:3,3]-anchor),.025*(x-q[ids])]
  seeds=[q[ids],np.array([-1.3,-.4,0.,-.8,0.,.4,0.]),np.array([1.3,.4,0.,.8,0.,-.4,0.])]
  # Move the sensor in front of or beside the body when orientation alone
  # cannot provide a clear ray. Every candidate still passes the full sweep.
  offsets=[np.zeros(3)] if look_points is None else [np.zeros(3),np.array([.20,0.,.08]),np.array([.32,0.,.08]),np.array([.20,-.18,.08]),np.array([.12,-.28,.08])]
  for offset,seed in [(offset,seed) for offset in offsets for seed in seeds]:
   anchor=actual[:3,3]+offset
   fit=least_squares(residual,np.clip(seed,limits[0]+1e-4,limits[1]-1e-4),bounds=(limits[0]+1e-4,limits[1]-1e-4),max_nfev=100)
   error=residual(fit.x);trial=q.copy();trial[ids]=fit.x
   camera_pose=servo.pose(trial)@relative
   row=dict(axis_error=float(np.linalg.norm(error[:3])/1.5),camera_displacement=float(np.linalg.norm(camera_pose[:3,3]-actual[:3,3])),floor_point=floor_point,anchor_offset=offset.tolist(),reason='ik')
   if diagnostics is not None:diagnostics.append(row)
   if row['axis_error']>.12 or row['camera_displacement']>.45:continue
   trial=q.copy();trial[ids]=fit.x;row['reason']='collision'
   if not all(scene.valid(servo.kin.forward(q+a*(trial-q))[0],payload) for a in np.linspace(0,1,30)):continue
   camera_pose=servo.pose(trial)@relative
   visibility=sight_line(robot,servo.kin.forward(trial)[0],scene,camera_pose[:3,3],look,payload)
   row['visibility']=visibility
   if not visibility['clear']:
    row['reason']=visibility['reason'];continue
   row['reason']='accepted';choices.append((np.linalg.norm(fit.x-q[ids]),fit.x))
 return min(choices,key=lambda x:x[0])[1] if choices else None
