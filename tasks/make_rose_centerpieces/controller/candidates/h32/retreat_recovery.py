"""Draft helpers for task56; integrate before generic execute->carry timeout.

Start on timeout while vase is still upright. During retreat, run normal allowed
observation acquisition, then replace motors with step_vertical_retreat(), and
apply the existing policy.check_live_action() to every returned motor action.
Only change to carry when this helper returns complete=True. Failure keeps the
base stationary, grasp closed and arm held; do not fall through to lateralcarry.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from .skills import ObservedSceneCollision

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
    return dict(start=rose,target=target,rim=rim,blocked=0,complete_count=0,hold_command=None,last_target_command=None,vase_points=vp.copy(),vase_pose=vase.copy(),exit=None)

def step_vertical_retreat(policy,state,action,proprio):
    if state is not None and state.get('exit') is not None:
        return step_checked_exit(policy,state,action,proprio)
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
    if cleared:
        if state.get('hold_command') is None:
            state['hold_command']=(state.get('last_target_command') if state.get('last_target_command') is not None else action).copy()
        action[3:14]=state['hold_command'][3:14]
    else:
        state['hold_command']=None
    if state['complete_count']>=12:
        state['exit']=dict(route=None,index=0,blocked=0,attempts=0,stable=0,last_plan_step=-100)
        return step_checked_exit(policy,state,action,proprio)
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
        state['last_target_command']=action.copy()
    return action,False,dict(clearance_m=clearance,lateral_error_m=lateral_error,rotation_error_rad=angle,blocked=state['blocked'])


# Additional uncertainty budget, above the original 8/35mm geometry guards.
# A 5mm translation and 2deg orientation perturbation are bounded by r*angle.
POSE_TRANSLATION_M = .005
POSE_ANGLE_RAD = np.deg2rad(2.)
CARRY_ROOT = np.array([.45,.05,.85])
RIGHT_IDS = [11,13,15,17,19,21,23]
EXIT_POSITION_TOLERANCE = .012
EXIT_ROTATION_TOLERANCE = .08


def exit_pose_errors(policy,q,hand):
    actual=policy.servo.pose(q)
    return (float(np.linalg.norm(actual[:3,3]-hand[:3,3])),
            float(R.from_matrix(hand[:3,:3]@actual[:3,:3].T).magnitude()))


def strict_exit_solution(policy,q,hand,valid,base):
    # Use the original IK solver, with an FK acceptance margin below the
    # execution thresholds. Preserve the complete original geometry validator.
    def admissible(trial):
        position,rotation=exit_pose_errors(policy,trial,hand)
        return (position<EXIT_POSITION_TOLERANCE*.75 and
                rotation<EXIT_ROTATION_TOLERANCE*.75 and valid(trial,base))
    return policy.servo.solve(q,hand[:3,3],hand[:3,:3],validator=admissible)


def exit_geometry(policy,state,world_pose):
    points=policy.perception.models[policy.child]['xyz']@world_pose[:3,:3].T+world_pose[:3,3]
    vase=state['vase_points']
    clearance=float(points[:,2].min()-vase[:,2].max())
    separation=float(np.maximum(points[:,:2].min(0)-vase[:,:2].max(0),vase[:,:2].min(0)-points[:,:2].max(0)).max())
    return clearance,separation


def commanded_q(policy,action,q):
    result=q.copy();result[policy.servo.indices]=action[3:14];result[RIGHT_IDS]=action[15:22]
    return result


def exit_checker(policy,state,scene,base,baseline):
    """Original guards plus persistent accepted vase CAD with pose tolerance.

    Callable accepts a full joint state and optional candidate world base.
    Scene points remain in the current sensor/body frame for moving-base checks.
    """
    tree=cKDTree(state['vase_points']);inverse=np.linalg.inv(base)
    lo,hi=policy.bounds(policy.child)
    axes=[np.linspace(a,b,max(2,int(np.ceil((b-a)/.025))+1)) for a,b in zip(lo,hi)]
    samples=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    vase_local=policy.perception.models[policy.parent]['xyz']
    vase_radius=float(np.linalg.norm(vase_local,axis=1).max())
    flower_radius=float(np.linalg.norm(samples,axis=1).max())
    vase_error=POSE_TRANSLATION_M+vase_radius*POSE_ANGLE_RAD
    payload_error=POSE_TRANSLATION_M+flower_radius*POSE_ANGLE_RAD
    def valid(q,future_base=None):
        future_base=base if future_base is None else future_base
        poses=policy.servo.kin.forward(q)[0];payload=policy.servo.pose(q)@policy.held
        motion=inverse@future_base
        observed={name:motion@pose for name,pose in poses.items()}
        if not policy.robot.valid(poses,baseline) or not scene.valid(observed,motion@payload):return False
        world=future_base@payload;clearance,separation=exit_geometry(policy,state,world)
        if clearance<.05 and separation<.05:return False
        centers=policy.robot.centers(poses)@future_base[:3,:3].T+future_base[:3,3]
        if np.any(tree.query(centers)[0]-policy.robot.radii<.008+vase_error):return False
        points=samples@world[:3,:3].T+world[:3,3]
        return bool(tree.query(points)[0].min()>.035+vase_error+payload_error)
    return valid


def checked_sweep(valid,start,end,start_base=None,end_base=None):
    count=max(3,int(np.ceil(float(np.max(abs(end-start)))/.01))+1)
    if start_base is not None:
        count=max(count,int(np.ceil(np.linalg.norm(end_base[:3,3]-start_base[:3,3])/.01))+1)
    for alpha in np.linspace(0.,1.,count):
        base=None
        if start_base is not None:
            base=start_base.copy();base[:3,3]+=alpha*(end_base[:3,3]-start_base[:3,3])
        if not valid(start+alpha*(end-start),base):return False
    return True


def plan_checked_exit(policy,state,scene):
    """Return only a complete raise/exit/lower route ending at ordinary carry.

    Raised prefixes are never returned as complete plans. Base alternatives
    translate with a latched arm; every whole-robot/payload segment is checked.
    """
    base=policy.base().copy();q=policy.current_q.copy();held=policy.held
    current=base@policy.servo.pose(q)@held
    valid=exit_checker(policy,state,scene,base,policy.robot.margins(policy.servo.kin.forward(q)[0]))
    trials=[]
    if not valid(q,base):return None,dict(reason='initial_exit_configuration_lacks_pose_margin',trials=trials)
    previous=policy.action
    if previous is not None and not checked_sweep(valid,commanded_q(policy,previous,q),q,base,base):
        return None,dict(reason='outstanding_motor_command_lacks_pose_margin',trials=trials)
    def segment(route,trial,start_pose,target_pose,at_base,stage):
        start=trial.copy();length=np.linalg.norm(target_pose[:3,3]-start_pose[:3,3]);rotvec=R.from_matrix(target_pose[:3,:3]@start_pose[:3,:3].T).as_rotvec();count=max(1,int(np.ceil(length/.04)),int(np.ceil(np.linalg.norm(rotvec)/.10)))
        for alpha in np.linspace(0.,1.,count+1)[1:]:
            world=start_pose.copy();world[:3,3]+=alpha*(target_pose[:3,3]-start_pose[:3,3]);world[:3,:3]=R.from_rotvec(alpha*rotvec).as_matrix()@start_pose[:3,:3]
            hand=np.linalg.inv(at_base)@world@np.linalg.inv(held)
            solution=strict_exit_solution(policy,start,hand,valid,at_base)
            if solution is None:return None,stage+'_ik_node_'+str(len(route))
            target=start.copy();target[policy.servo.indices]=solution
            if not checked_sweep(valid,start,target,at_base,at_base):return None,stage+'_sweep_node_'+str(len(route))
            route.append(dict(stage=stage,q=target.copy(),base=at_base.copy(),world=world.copy()))
            start=target
        return start,None
    # Multiple approaches: straight or lateral detour, each at three heights.
    for clearance in (.10,.16,.22):
        raised=current.copy();raised[2,3]+=max(0.,clearance-exit_geometry(policy,state,current)[0])
        prefix=[];liftq,reason=segment(prefix,q,current,raised,base,'raise')
        if reason:
            trials.append(dict(clearance=clearance,kind='raise',rejected=reason));continue
        carry=base@policy.servo.pose(liftq)@held;carry[:3,3]=(base@np.r_[CARRY_ROOT,1.])[:3]
        if exit_geometry(policy,state,carry)[1]>=.075:
            for side in (0.,-.12,.12):
                route=list(prefix);trial=liftq.copy();start=raised.copy()
                lateral=raised.copy();lateral[:2,3]=carry[:2,3]
                targets=[]
                if side:
                    middle=raised.copy();middle[:2,3]=(raised[:2,3]+lateral[:2,3])/2+base[:2,1]*side;targets.append(('detour',middle))
                targets.extend([('lateral',lateral),('lower',carry)])
                for stage,target in targets:
                    trial,reason=segment(route,trial,start,target,base,stage)
                    if reason:break
                    start=target
                trials.append(dict(clearance=clearance,kind='arm',side=side,rejected=reason))
                if reason is None:return route,dict(reason='complete_checked_egress',kind='arm',nodes=len(route),trials=trials,complete_route=True)
        # Back-away alternatives are world directions from public geometry only.
        away=base[:2,3]-state['vase_pose'][:2,3];away/=max(np.linalg.norm(away),1e-9)
        back=-base[:2,0]
        for direction in (back,away,(back+away)/max(np.linalg.norm(back+away),1e-9)):
            for distance in (.20,.35,.50,.65):
                finish=base.copy();finish[:2,3]+=direction*distance
                final=finish@policy.servo.pose(liftq)@held;final[:3,3]=(finish@np.r_[CARRY_ROOT,1.])[:3]
                if exit_geometry(policy,state,final)[1]<.075:continue
                route=list(prefix);previous_base=base.copy();reason=None
                for alpha in np.linspace(0,1,max(2,int(np.ceil(distance/.025))+1))[1:]:
                    waypoint=base.copy();waypoint[:2,3]+=alpha*direction*distance
                    if not checked_sweep(valid,liftq,liftq,previous_base,waypoint):reason='base_sweep';break
                    route.append(dict(stage='backaway',q=liftq.copy(),base=waypoint.copy(),world=waypoint@policy.servo.pose(liftq)@held))
                    previous_base=waypoint
                if reason is None:
                    moved=finish@policy.servo.pose(liftq)@held
                    # Once the entire payload clears the vase horizontally,
                    # lower before reaching inward. The elevated inward pose
                    # can be unreachable even when both endpoints are valid.
                    if exit_geometry(policy,state,moved)[1]>=.075:
                        # Carry at a fixed payload orientation can intersect the
                        # robot itself. Select a complete collision-free carry
                        # orientation only after backing fully clear of the vase.
                        reason='no_checked_carry_orientation'
                        rotations=[np.zeros(3),[0,0,-.35],[0,0,.35],[0,0,-.7],[0,0,.7],[.35,0,0],[-.35,0,0],[.7,0,0],[0,0,-1.2]]
                        for vector in rotations:
                            goal=final.copy();goal[:3,:3]=finish[:3,:3]@R.from_rotvec(vector).as_matrix()@finish[:3,:3].T@final[:3,:3]
                            hand=np.linalg.inv(finish)@goal@np.linalg.inv(held)
                            solution=strict_exit_solution(policy,liftq,hand,valid,finish)
                            if solution is None:continue
                            target=liftq.copy();target[policy.servo.indices]=solution
                            if checked_sweep(valid,liftq,target,finish,finish):
                                route.append(dict(stage='reorient_carry',q=target,base=finish.copy(),world=goal))
                                reason=None;break
                    else:reason='backaway_payload_xy_not_clear'
                trials.append(dict(clearance=clearance,kind='backaway',distance=distance,direction=direction.tolist(),rejected=reason))
                if reason is None:return route,dict(reason='complete_checked_egress',kind='backaway',nodes=len(route),trials=trials,complete_route=True)
    return None,dict(reason='no_complete_checked_egress',trials=trials,complete_route=False)


def check_egress_action(policy,state,action,previous=None):
    """Public adapter hook: check FINAL post-live-guard command, not just proposal."""
    q=policy.current_q;base=policy.base();poses=policy.servo.kin.forward(q)[0]
    scene=ObservedSceneCollision(policy.current_cameras,policy.robot,poses,policy.servo.pose(q)@policy.held,*policy.bounds(policy.child))
    valid=exit_checker(policy,state,scene,base,policy.robot.margins(poses))
    goal=commanded_q(policy,action,q);future=base.copy();future[:2,3]+=base[:2,:2]@action[:2]/30.
    # Egress never needs yaw; reject unexpected fallback yaw at the boundary.
    if abs(float(action[2]))>1e-8:return False,dict(reason='unexpected_egress_yaw')
    measured=checked_sweep(valid,q,goal,base,future)
    issued=True if previous is None else checked_sweep(valid,commanded_q(policy,previous,q),goal,base,future)
    return measured and issued,dict(measured_sweep=measured,previous_issued_sweep=issued)


def step_checked_exit(policy,state,action,proprio):
    exit_state=state['exit'];q=policy.current_q;base=policy.base();previous=policy.action
    action=action.copy();action[:3]=0.;action[14]=-1.
    if previous is not None:action[3:14]=previous[3:14];action[15:22]=previous[15:22]
    else:action[3:7]=proprio[53:57];action[7:14]=proprio[3:10];action[15:22]=proprio[28:35]
    current=base@policy.servo.pose(q)@policy.held;clearance,separation=exit_geometry(policy,state,current)
    debug=dict(clearance_m=clearance,xy_separation_m=separation,waypoint=exit_state['index'],blocked=exit_state['blocked'])
    poses=policy.servo.kin.forward(q)[0]
    scene=ObservedSceneCollision(policy.current_cameras,policy.robot,poses,policy.servo.pose(q)@policy.held,*policy.bounds(policy.child))
    if exit_state['route'] is None:
        if exit_state['attempts']>=2 or int(policy.step)-exit_state['last_plan_step']<30:
            return action,False,dict(debug,reason='no_complete_egress_wait',plan=exit_state.get('plan_debug'))
        route,receipt=plan_checked_exit(policy,state,scene)
        exit_state.update(route=route,plan_debug=receipt,last_plan_step=int(policy.step),attempts=exit_state['attempts']+1)
        if route is None:return action,False,dict(debug,reason='no_complete_egress',plan=receipt)
    route=exit_state['route'];index=exit_state['index']
    if index>=len(route):
        complete=separation>=.06 and np.linalg.norm((np.linalg.inv(base)@current)[:3,3]-CARRY_ROOT)<.025
        accepted,receipt=check_egress_action(policy,state,action,previous)
        exit_state['stable']=exit_state['stable']+1 if complete and accepted else 0
        return action,exit_state['stable']>=12,dict(debug,reason='complete_egress_settle',confirmation_count=exit_state['stable'],command_check=receipt)
    node=route[index]
    if node['stage']=='backaway':
        # Original motor targets remain latched; only translate the base.
        action[3:14]=node['q'][policy.servo.indices];action[15:22]=node['q'][RIGHT_IDS]
        delta=base[:2,:2].T@(node['base'][:2,3]-base[:2,3]);action[:2]=delta*min(1.8,.06/max(np.linalg.norm(delta),1e-9))
        reached=np.linalg.norm(delta)<.009 and np.max(abs(q[policy.servo.indices]-node['q'][policy.servo.indices]))<.04
    else:
        hand=np.linalg.inv(base)@node['world']@np.linalg.inv(policy.held)
        position_error,rotation_error=exit_pose_errors(policy,q,hand)
        joint_error=float(np.max(abs(q[policy.servo.indices]-node['q'][policy.servo.indices])))
        debug.update(position_error_m=position_error,rotation_error_rad=rotation_error,joint_error_rad=joint_error)
        # A latched joint target cannot follow base drift. Once joints have
        # settled, re-solve the unchanged world waypoint at the measured base.
        # Bound work to three attempts per waypoint, at least 30 controls apart.
        if (joint_error<.04 and
                (position_error>=EXIT_POSITION_TOLERANCE or rotation_error>=EXIT_ROTATION_TOLERANCE) and
                node.get('corrections',0)<3 and
                int(policy.step)-node.get('last_correction_step',-100)>=30):
            node['last_correction_step']=int(policy.step)
            node['corrections']=node.get('corrections',0)+1
            valid=exit_checker(policy,state,scene,base,policy.robot.margins(poses))
            solution=strict_exit_solution(policy,q,hand,valid,base)
            debug['correction']='no_admissible_ik'
            if solution is not None:
                target=q.copy();target[policy.servo.indices]=solution
                measured=checked_sweep(valid,q,target,base,base)
                issued=previous is None or checked_sweep(valid,commanded_q(policy,previous,q),target,base,base)
                if measured and issued:
                    node['q']=target;node['base']=base.copy()
                    debug['correction']='accepted_checked_target'
                else:debug['correction']='sweep_rejected'
        debug['correction_attempts']=node.get('corrections',0)
        policy.servo.target=node['q'][policy.servo.indices].copy();policy.servo.synchronized=True;policy.servo.retain_target=True
        try:errors=policy.servo.step(action,q,hand[:3,3],hand[:3,:3])
        finally:policy.servo.retain_target=False
        reached=errors[0]<EXIT_POSITION_TOLERANCE and errors[1]<EXIT_ROTATION_TOLERANCE and np.max(abs(q[policy.servo.indices]-node['q'][policy.servo.indices]))<.04
    accepted,receipt=check_egress_action(policy,state,action,previous)
    if not accepted:
        action[:3]=0.;action[3:7]=proprio[53:57];action[7:14]=proprio[3:10];action[15:22]=proprio[28:35]
        exit_state['blocked']+=1
        return action,False,dict(debug,reason='egress_command_rejected',stage=node['stage'],command_check=receipt)
    if reached:exit_state['index']+=1
    return action,False,dict(debug,reason='egress_motion',stage=node['stage'],command_check=receipt)
