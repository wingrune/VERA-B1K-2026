"""Observed insertion geometry, release evidence and arrangement memory."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from .skills import transform
from . import skills

def recover_blocked_pregrasp(p,action,s):
    """Bounded, fully guarded empty-hand retreat before retrying an approach."""
    state=p.release_debug.get('blocked_pregrasp_retry')
    if state is None:
        state=dict(count=0,attempts=0,last_step=None,active=False)
        p.release_debug['blocked_pregrasp_retry']=state
    eligible=(p.child=='rose' and p.phase=='pregrasp' and p.held is None
              and not p.dual.active and not p.paired.active
              and p.dual.right_held is None and p.age>=60)
    if not eligible:
        state['count']=0;state['active']=False
        return action
    hand=(p.base()@p.servo.pose(p.current_q))[:3,3]
    if state['active']:
        distance=float(np.linalg.norm(hand-np.asarray(state['anchor'])))
        state['measured_retreat_m']=distance
        if distance>=.04 or p.step-state['started']>=180:
            state['active']=False;state['count']=0
            state['finish_reason']='measured_retreat' if distance>=.04 else 'bounded_timeout'
            p.servo.target=None
            p.change('grasp_plan' if p.step-p.seen.get('rose',-1000)<25 else 'search')
        action[:3]=0.;action[3:7]=s[53:57];action[7:14]=s[3:10];action[15:22]=s[28:35]
        return action
    prior=p.release_debug.get('insertion_previous_issued')
    if prior is None or prior.get('action') is None:return action
    previous=np.asarray(prior['action']);measured=p.current_q[p.servo.indices]
    blocked=(p.stationary_taskspace and prior.get('control_step')==p.step
             and np.max(abs(previous[3:14]-measured))<.005
             and np.max(abs(action[3:14]-measured))>.01)
    if state['last_step']==p.step:return action
    continuous=state['last_step'] is not None and p.step==state['last_step']+1
    state['count']=(state['count']+1 if continuous else 1) if blocked else 0
    state['last_step']=int(p.step)
    if state['count']>=45 and state['attempts']<2 and 'vase' in p.poses:
        state.update(active=True,anchor=hand.tolist(),started=int(p.step),count=0,
                     attempts=state['attempts']+1)
    return action

def make_task_path(policy,base,q,cams):
    if policy.held is None or policy.parent not in policy.poses or policy.step-policy.seen.get(policy.parent,-1000)>90:return False
    target=policy.poses[policy.parent];lo,hi=policy.bounds(policy.parent)
    model=policy.perception.models[policy.child];xyz=model['xyz']
    if policy.child=='rose':
        # Object-local principal axis, signed toward the flower by stem color.
        center=xyz.mean(0);_,_,v=np.linalg.svd(xyz-center,full_matrices=False);axis=v[0]
        rgb=model['rgb'].astype(float);green=(rgb[:,1]>rgb[:,0]*1.15)&(rgb[:,1]>rgb[:,2]*1.15)
        if green.sum()>20 and (xyz[~green].mean(0)-xyz[green].mean(0))@axis<0:axis=-axis
        rotation=R.align_vectors([[0.,0.,1.]],[axis])[0].as_matrix()
        vase_xyz=policy.perception.models['vase']['xyz']@target[:3,:3].T+target[:3,3]
        vase_tree=cKDTree(vase_xyz)
        rim=float(vase_xyz[:,2].max());bottom=float(vase_xyz[:,2].min())
        vase_center=np.median(vase_xyz[:,:2],axis=0)
        angle=2*np.pi*len(policy.rose_releases)/3
        offsets=[r*np.array([np.cos(angle+a),np.sin(angle+a)])
                 for r,a in ((0.,0.),(.015,0.),(.025,0.),(.015,np.pi/3),(.015,-np.pi/3))]
        current=policy.servo.kin.forward(q)[0]
        baseline=policy.robot.margins(current)
        from .skills import ObservedSceneCollision
        scene=ObservedSceneCollision(cams,policy.robot,current,policy.servo.pose(q)@policy.held,*policy.bounds(policy.child))
        chosen=None;attempted=0
        # Rotation about the vertical stem preserves insertion geometry, while
        # changing the wrist orientation and hand position implied by held.
        for depth_scale,offset,yaw in ((d,o,y) for d in (1.,.8,.6,.4,.2) for o in offsets for y in (0.,np.pi/4,-np.pi/4,np.pi/2,-np.pi/2,3*np.pi/4,-3*np.pi/4,np.pi)):
            # Preserve the deepest nominal placement whenever it is feasible.
            turn=np.array([[np.cos(yaw),-np.sin(yaw),0.],[np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]])
            candidate_rotation=turn@rotation
            rotated=xyz@candidate_rotation.T
            height=np.ptp(rotated[:,2])
            nominal_insertion=min((rim-bottom)*.65,height*.70)
            insertion=max(0.,nominal_insertion-.024)*depth_scale
            if insertion<.03:continue
            position=np.r_[vase_center+offset,rim-insertion-rotated[:,2].min()]
            position[:2]-=(rotated[:,:2].min(0)+rotated[:,:2].max(0))*.5
            goal=transform(candidate_rotation,position)
            above=goal.copy();above[2,3]+=insertion+.10
            middle=goal.copy();middle[2,3]+=.04
            points=[(above,10),(middle,10),(goal,35)]
            trial=q.copy();feasible=True;attempted+=1
            for index,(pose,_) in enumerate(points):
                hand=np.linalg.inv(base)@pose@np.linalg.inv(policy.held)
                def valid(candidate):
                    poses=policy.servo.kin.forward(candidate)[0]
                    if not policy.robot.valid(poses,baseline) or (index==0 and not scene.valid(poses,policy.servo.pose(candidate)@policy.held)):return False
                    count=max(3,int(np.ceil(float(np.max(abs(candidate-trial)))/.01))+1)
                    for alpha in np.linspace(0.,1.,count):
                        fk=policy.servo.kin.forward(trial+alpha*(candidate-trial))[0]
                        centers=policy.robot.centers(fk)@base[:3,:3].T+base[:3,3]
                        if np.any(vase_tree.query(centers)[0]-policy.robot.radii<.013):return False
                    return True
                solution=policy.servo.solve(trial,hand[:3,3],hand[:3,:3],validator=valid)
                if solution is None:feasible=False;break
                trial[policy.servo.indices]=solution
            if feasible:chosen=points;break
        policy.release_debug=dict(insertion_preflight=chosen is not None,attempted_rotations=attempted,chosen_depth_scale=None if chosen is None else float(depth_scale),insertion_depth_m=None if chosen is None else float(insertion),chosen_offset_xy=None if chosen is None else offset.tolist())
        if chosen is None:return False
        policy.waypoints=chosen
    else:
        return policy.plan_vase_placement(base,q,cams)
    policy.release_pose=policy.waypoints[-1][0].copy();policy.waypoint=0;policy.stable=0
    return True

def release_observed(policy):
        if policy.child not in policy.poses or policy.parent not in policy.poses:return False
        if policy.step-policy.seen.get(policy.child,-1000)>45:return False
        child=policy.poses[policy.child];parent=policy.poses[policy.parent]
        points=policy.perception.models[policy.child]['xyz']@child[:3,:3].T+child[:3,3]
        if policy.child=='rose':return False  # Placement verification disabled by user.
        table=policy.perception.models['coffee_table']['xyz']@parent[:3,:3].T+parent[:3,3]
        error=abs(float(points[:,2].min()-np.quantile(table[:,2],.99)))
        center=points.mean(0);inside=np.all(center[:2]>table[:,:2].min(0)+.05) and np.all(center[:2]<table[:,:2].max(0)-.05)
        policy.release_debug={'support_height_error_m':error,'within_table':bool(inside)}
        return error<.035 and inside

def record_rose_release(policy,world):
        # Record the issued release, without observing or asserting placement.
        if world is None or np.asarray(world).shape!=(4,4) or not np.isfinite(world).all():
            raise RuntimeError('Missing allowed FK release pose')
        world=np.asarray(world).copy();point=world[:3,3].copy()
        policy.rose_releases.append(point.tolist());policy.perception.excluded.append(point)
        # Keep subsequent table acquisition outside the observed vase column.
        # This is an exclusion zone, not evidence that a release succeeded.
        target=policy.poses['vase']
        vase_xyz=policy.perception.models['vase']['xyz']@target[:3,:3].T+target[:3,3]
        center=np.median(vase_xyz[:,:2],axis=0)
        for z in np.linspace(vase_xyz[:,2].min()+.12,vase_xyz[:,2].max()+.20,6):
            policy.perception.excluded.append(np.r_[center,z])
        policy.rose_assembly.append(np.linalg.inv(policy.poses['vase'])@world)
        policy.commanded_release_world=None;policy.left_pickup_confirmed=False
        if policy.dual is not None:
            tracks=policy.dual.perception
            if tracks.hands['left'] is not None:tracks.mark_released('left',world)
            elif not any(np.array_equal(old,world) for old in tracks.released):tracks.released.append(world.copy())
