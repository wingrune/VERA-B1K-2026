"""Draft helpers for task56; integrate before generic execute->carry timeout.

Start on timeout while vase is still upright. During retreat, run normal allowed
observation acquisition, then replace motors with step_vertical_retreat(), and
apply the existing policy.check_live_action() to every returned motor action.
Only change to carry when this helper returns complete=True. Failure keeps the
base stationary, grasp closed and arm held; do not fall through to lateralcarry.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scene_collision import ObservedSceneCollision

def start_vertical_retreat(policy):
    if policy.held is None or policy.parent not in policy.poses:
        return {'error':'missing_held_or_parent_pose'}
    if policy.child not in policy.perception.models or policy.parent not in policy.perception.models:
        return {'error':'missing_static_model'}
    rose=policy.base()@policy.servo.pose(policy.current_q)@policy.held
    vase=policy.poses[policy.parent]
    vp=policy.perception.models[policy.parent]['xyz']@vase[:3,:3].T+vase[:3,3]
    rp=policy.perception.models[policy.child]['xyz']@rose[:3,:3].T+rose[:3,3]
    rim=float(vp[:,2].max());bottom=float(rp[:,2].min())
    target=rose.copy();target[2,3]+=max(0.,rim+.07-bottom)
    policy.servo.target=None;policy.servo.synchronized=False
    return dict(start=rose,target=target,rim=rim,blocked=0,complete_count=0)

def step_vertical_retreat(policy,state,action,proprio):
    q=policy.current_q;action=action.copy();action[:3]=0.;action[14]=-1.
    # Erase insertion/carry commands on the same timeout transition frame.
    action[3:7]=proprio[53:57];action[7:14]=proprio[3:10];action[15:22]=proprio[28:35]
    if state is None or 'error' in state or policy.held is None:
        return action,False,{'reason':'missing_retreat_geometry_or_hold','error':None if state is None else state.get('error')}
    current=policy.base()@policy.servo.pose(q)@policy.held
    points=policy.perception.models[policy.child]['xyz']@current[:3,:3].T+current[:3,3]
    clearance=float(points[:,2].min()-state['rim'])
    lateral_error=float(np.linalg.norm(current[:2,3]-state['start'][:2,3]))
    angle=float(R.from_matrix(current[:3,:3].T@state['start'][:3,:3]).magnitude())
    cleared=clearance>=.05 and lateral_error<.02 and angle<.12
    state['complete_count']=state['complete_count']+1 if cleared else 0
    if state['complete_count']>=12:
        return action,True,dict(clearance_m=clearance,lateral_error_m=lateral_error,rotation_error_rad=angle)
    if cleared:
        return action,False,dict(clearance_m=clearance,lateral_error_m=lateral_error,rotation_error_rad=angle,confirmation_count=state['complete_count'])
    goal=state['target'].copy();goal[2,3]=min(goal[2,3],current[2,3]+.015)
    hand=np.linalg.inv(policy.base())@goal@np.linalg.inv(policy.held)
    poses=policy.servo.kin.forward(q)[0]
    scene=ObservedSceneCollision(policy.current_cameras,policy.robot,poses,policy.servo.pose(q)@policy.held,*policy.bounds(policy.child))
    baseline=policy.robot.margins(poses)
    def valid(trial):
        p=policy.servo.kin.forward(trial)[0]
        return policy.robot.valid(p,baseline) and scene.valid(p,policy.servo.pose(trial)@policy.held)
    solution=policy.servo.solve(q,hand[:3,3],hand[:3,:3],validator=valid)
    if solution is None:
        action[3:7]=proprio[53:57];action[7:14]=proprio[3:10]
        state['blocked']+=1
    else:
        policy.servo.target=solution;policy.servo.synchronized=True
        policy.servo.step(action,q,hand[:3,3],hand[:3,:3]);state['blocked']=0
    return action,False,dict(clearance_m=clearance,lateral_error_m=lateral_error,rotation_error_rad=angle,blocked=state['blocked'])
