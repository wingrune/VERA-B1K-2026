"""Task sequence and vase/table placement from visual geometry and memory."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from .skills import transform

def plan_vase_placement(policy,base,q,cams):
    if policy.held is None or policy.parent not in policy.poses or policy.step-policy.seen.get(policy.parent,-1000)>90:return False
    target=policy.poses[policy.parent];lo,hi=policy.bounds(policy.parent)
    model=policy.perception.models[policy.child];xyz=model['xyz']
    # Place the vase and its contents on the visually localized table top.
    table_xyz=policy.perception.models['coffee_table']['xyz']@target[:3,:3].T+target[:3,3]
    top=np.quantile(table_xyz[:,2],.99)
    center=np.median(table_xyz[table_xyz[:,2]>top-.03,:2],axis=0)
    current=base@policy.servo.pose(q)@policy.held
    rotation=current[:3,:3]
    bottom=(xyz@rotation.T)[:,2].min()
    goal=transform(rotation,np.r_[center,float(top+.008-bottom)])
    above=goal.copy();above[2,3]+=.14
    policy.waypoints=[(above,15),(goal,45)]
    policy.release_pose=policy.waypoints[-1][0].copy();policy.waypoint=0;policy.stable=0
    return True

def resume_after_rose_release(policy):
        policy.perception.active=None;policy.poses.pop('rose',None);policy.seen.pop('rose',None)
        policy.nav=None;policy.pose_route=None;policy.held=None
        if policy.dual is not None and policy.dual.right_held is not None:
            if policy.dual.begin_right_place():return
            policy.change('done_hold');return
        if len(policy.rose_releases)>=3:
            # The requested endpoint is three released roses in the original vase.
            # This callback runs after the releasing hand has withdrawn; do not
            # grasp or transport the vase. Release counts are issued commands,
            # not proof of placement: evaluation must verify all three roses.
            policy.stage='roses_complete'
            policy.change('done_hold')
            return
        policy.change('search')
