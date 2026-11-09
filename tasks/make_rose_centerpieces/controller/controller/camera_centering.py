"""Bounded open-hand base-yaw camera centering from allowed RGB-D only."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize_scalar
from navigation import wrap
from pose_navigation import PosePlanner
from scene_collision import ObservedSceneCollision

def rim_centering_yaw(head,target,max_turn=.9):
    target=np.asarray(target,float);width=head['rgb'].shape[1];height=head['rgb'].shape[0];center=head['K'][:2,2]
    def projection(delta):
        turn=R.from_euler('z',delta).as_matrix();p=(target-turn@head['t'])@(turn@head['R'])
        if p[2]<=.06:return None
        uv=head['K']@p;return uv[:2]/uv[2]
    def cost(delta):
        uv=projection(delta)
        return 1e9 if uv is None else float(abs(uv[0]-center[0])+.02*abs(delta))
    candidates=np.linspace(-max_turn,max_turn,61);best=min(candidates,key=cost)
    found=minimize_scalar(cost,bounds=(max(-max_turn,best-.04),min(max_turn,best+.04)),method='bounded');turn=float(found.x);uv=projection(turn)
    if uv is None or not .08*width<uv[0]<.92*width or not .08*height<uv[1]<.92*height:return None,{'reason':'rim_outside_centerable_fov'}
    return turn,{'reason':'rim_centerable','predicted_rim_uv':uv.tolist(),'delta_yaw_rad':turn}

def start_camera_centering(policy):
    if policy.held is not None:return None,{'reason':'object_still_held'}
    if 'head' not in policy.current_cameras:return None,{'reason':'missing_head_camera'}
    if policy.parent not in policy.poses or policy.step-policy.seen.get(policy.parent,-1000)>90:return None,{'reason':'stale_parent_pose'}
    body=np.linalg.inv(policy.base())@policy.poses[policy.parent]
    xyz=policy.perception.models[policy.parent]['xyz']@body[:3,:3].T+body[:3,3];rim=np.median(xyz[xyz[:,2]>xyz[:,2].max()-.015],axis=0)
    delta,debug=rim_centering_yaw(policy.current_cameras['head'],rim)
    if delta is None:return None,debug
    q=policy.current_q;poses=policy.servo.kin.forward(q)[0]
    scene=ObservedSceneCollision(policy.current_cameras,policy.robot,poses,None,*policy.bounds(policy.child))
    for alpha in np.linspace(0,1,101):
        turn=np.eye(4);turn[:3,:3]=R.from_euler('z',alpha*delta).as_matrix();future={k:turn@v for k,v in poses.items()}
        if not scene.valid(future,None):return None,dict(debug,reason='observed_depth_rotation_rejected',rejected_delta_rad=float(alpha*delta))
    planner=PosePlanner(policy.map,policy.robot,policy.servo,q,policy.level,None,None)
    baseline=planner.margins(policy.odom[:2],policy.odom[2])
    threshold=np.minimum(0.,baseline-.004)
    for alpha in np.linspace(0,1,101):
        margins=planner.margins(policy.odom[:2],policy.odom[2]+alpha*delta)
        if np.any(margins<threshold):return None,dict(debug,reason='fixed_xy_map_rotation_rejected',rejected_delta_rad=float(alpha*delta))
    goal=policy.odom.copy();goal[2]=wrap(goal[2]+delta)
    route=planner.plan(policy.odom,goal,seconds=.4)
    if route is None:return None,dict(debug,reason='map_rotation_plan_rejected',planner=planner.stats)
    if any(np.linalg.norm(p[:2]-goal[:2])>.08 for p in route):return None,dict(debug,reason='map_requires_translation',planner=planner.stats)
    previous=getattr(policy,'action',None)
    if previous is None or np.asarray(previous).shape!=(23,) or not np.isfinite(previous).all():return None,dict(debug,reason='missing_safe_motor_command')
    state={'heading':float(goal[2]),'start_step':policy.step,'arm_command':np.array(previous[3:14],copy=True),'right_command':np.array(previous[15:22],copy=True)}
    return state,dict(debug,reason='rotation_preflight_passed',planner=planner.stats,checked_scene_samples=101,checked_fixed_xy_map_samples=101)

def step_camera_centering(policy,state,action,proprio):
    action=action.copy();action[:3]=0.;action[14]=1.
    action[3:7]=proprio[53:57];action[7:14]=proprio[3:10];action[15:22]=proprio[28:35]
    if state is None:return action,True,{'reason':'centering_unavailable'}
    measured=np.r_[proprio[53:57],proprio[3:10]]
    action[3:14]=np.clip(state['arm_command'],measured-.12,measured+.12)
    action[15:22]=np.clip(state['right_command'],proprio[28:35]-.12,proprio[28:35]+.12)
    error=float(wrap(state['heading']-policy.odom[2]));elapsed=policy.step-state['start_step'];done=abs(error)<.05 or elapsed>=180
    if not done:action[2]=float(np.clip(error*.8,-.20,.20))
    return action,done,{'reason':'centered' if abs(error)<.05 else 'centering_timeout' if done else 'turning','heading_error_rad':error,'elapsed_steps':elapsed}
