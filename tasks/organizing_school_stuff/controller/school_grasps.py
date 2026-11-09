"""Local corner grasps for flat objects wider than the native gripper opening.

Do not relax the robot's jaw span. The open fingers must lie outside the object
and an actual assisted-grasp ray must cross its local bounding volume.
"""
import numpy as np
from art_skills import upright,transform


def corner_candidates(ctx,category,obj,base=None,coarse=False):
    bounds=ctx.models[category]['bounds'];lo,hi=bounds;centre=bounds.mean(0)
    world=ctx.base_matrix() if base is None else base
    yaw=float(np.arctan2(obj[1,0],obj[0,0]))
    corners=[(sx,sy) for sx in (-1.,1.) for sy in (-1.,1.)]
    corners.sort(key=lambda s:np.linalg.norm((obj@np.r_[np.where(np.array(s)>0,hi[:2],lo[:2]),centre[2],1.])[:2]-world[:2,3]))
    mesh=ctx.gripper[ctx.arm+'_collision']-ctx.ik.tool
    rays=ctx.gripper[ctx.arm+'_ag']-ctx.ik.tool
    support=float(np.quantile(ctx.models[category]['xyz']@obj[2,:3]+obj[2,3],.002))
    inverse=np.linalg.inv(obj)
    for sx,sy in corners:
        for depth in (.012,.018,.024):
            point=centre.copy();point[:2]=np.where(np.array([sx,sy])>0,hi[:2],lo[:2])-depth*np.array([sx,sy])
            # Closing along the corner diagonal intersects a short chord, even
            # though the entire object's projection is wider than the gripper.
            azimuth=yaw+float(np.arctan2(sy,sx))
            for reverse in (0.,np.pi):
                rotation=upright(azimuth+reverse)
                target=(obj@np.r_[point,1.])[:3]
                target[2]=support+.0008-float(np.min(mesh@rotation[2]))
                actual=mesh@rotation.T+target
                # Keep the open fingers above the actual sloping support; object
                # minimum Z alone can refer to a different part of that surface.
                target[2]+=max(0.,float(np.max(ctx.scene.support_heights(actual)+.0008-actual[:,2])))
                goal=transform(rotation,target)
                actual=mesh@rotation.T+target;local=(actual-obj[:3,3])@obj[:3,:3]
                # Open fingers must not already penetrate the target. The pad's
                # intended contact only happens when the native joint closes.
                penetration=np.all(local>lo+.0005,axis=1)&np.all(local<hi-.0005,axis=1)
                if penetration.any():continue
                # Native AG rays use paired finger points. Require a ray segment
                # to enter the object, rather than grasping empty space at a corner.
                ray_local=(rays@rotation.T+target-obj[:3,3])@obj[:3,:3]
                hits=False
                for z in np.unique(np.round(rays[:,2],5)):
                    group=ray_local[abs(rays[:,2]-z)<.0001]
                    if len(group)<2:continue
                    samples=np.linspace(group[0],group[-1],41)
                    if np.any(np.all(samples>=lo,axis=1)&np.all(samples<=hi,axis=1)):hits=True;break
                if hits:yield goal
