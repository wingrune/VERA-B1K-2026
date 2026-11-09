"""Open-gripper vertical view clearance from allowed proprioception/RGB-D only."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from .skills import ObservedSceneCollision

def start_release_view(policy):
    if policy.held is not None:
        return {'error':'release_not_complete'}
    start=policy.base()@policy.servo.pose(policy.current_q)
    if not np.isfinite(start).all():return {'error':'invalid_hand_pose'}
    goal=start.copy();goal[2,3]+=.16
    policy.servo.target=None;policy.servo.synchronized=False
    return dict(start=start,goal=goal,confirmed=0,blocked=0,hold_command=None)

def step_release_view(policy,state,action,proprio):
    q=policy.current_q;action=action.copy();action[:3]=0.;action[14]=1.
    action[3:7]=proprio[53:57];action[7:14]=proprio[3:10];action[15:22]=proprio[28:35]
    if state is None or 'error' in state or policy.held is not None:
        return action,False,{'reason':'release_view_unavailable','error':None if state is None else state.get('error')}
    current=policy.base()@policy.servo.pose(q)
    upward=float(current[2,3]-state['start'][2,3]);lateral=float(np.linalg.norm(current[:2,3]-state['start'][:2,3]))
    angle=float(R.from_matrix(current[:3,:3].T@state['start'][:3,:3]).magnitude())
    cleared=upward>=.12 and lateral<.025 and angle<.15
    state['confirmed']=state['confirmed']+1 if cleared else 0
    debug=dict(upward_m=upward,lateral_error_m=lateral,rotation_error_rad=angle,confirmed=state['confirmed'],blocked=state['blocked'])
    if cleared:
        if state.get('hold_command') is None:
            previous=getattr(policy,'action',None)
            state['hold_command']=action[3:14].copy() if previous is None else np.asarray(previous[3:14]).copy()
        measured=np.r_[proprio[53:57],proprio[3:10]]
        action[3:14]=np.clip(state['hold_command'],measured-.12,measured+.12)
        debug['latched_previously_issued_command']=True
        return action,state['confirmed']>=12,debug
    state['hold_command']=None
    bodygoal=np.linalg.inv(policy.base())@state['goal'];poses=policy.servo.kin.forward(q)[0]
    scene=ObservedSceneCollision(policy.current_cameras,policy.robot,poses,None,*policy.bounds(policy.child))
    baseline=policy.robot.margins(poses)
    def valid(trial):
        p=policy.servo.kin.forward(trial)[0]
        return policy.robot.valid(p,baseline) and scene.valid(p,None)
    if policy.servo.target is None:
        solution=policy.servo.solve(q,bodygoal[:3,3],bodygoal[:3,:3],validator=valid)
        if solution is None:
            state['blocked']+=1;debug['blocked']=state['blocked'];return action,False,debug
        policy.servo.target=solution;policy.servo.synchronized=True
    # Retain this single final target until measured world-EE clearance passes.
    previous=getattr(policy.servo,'retain_target',False);policy.servo.retain_target=True
    try:policy.servo.step(action,q,bodygoal[:3,3],bodygoal[:3,:3])
    finally:policy.servo.retain_target=previous
    state['blocked']=0
    return action,False,debug
