"""Shape-based pinches from fruit surface/collision models and estimated 6D poses."""
import numpy as np

def candidates(ctx,category,obj,base=None,coarse=False):
    from compost_skills import transform,upright
    model=ctx.models[category]['xyz']
    world=model@obj[:3,:3].T+obj[:3,3]
    support=float(np.quantile(world[:,2],.002))
    if category=='banana':
        # Curved banana: contacts on local convex body segments, not its global box centre.
        from pathlib import Path
        geometry=np.load(Path(__file__).parent/'priors/banana/collision.npz')
        patches=[geometry[k] for k in geometry.files if k.endswith('|v')]
        patches=sorted(patches,key=lambda p:np.ptp(p,axis=0).prod(),reverse=True)[:3]
        locations=[]
        for patch in patches:
            centre=np.median(patch,axis=0);delta=patch[:,:2]-centre[:2]
            _,vectors=np.linalg.eigh(delta.T@delta);axis=vectors[:,-1]
            direction=obj[:3,:3]@np.r_[axis,0.]
            locations.append((obj[:3,:3]@centre+obj[:3,3],float(np.arctan2(direction[1],direction[0]))))
        heights=(.002,.006) if not coarse else (.003,)
    else:
        centre=np.median(world,axis=0)
        axis=obj[:3,0];yaw=float(np.arctan2(axis[1],axis[0]))
        locations=[(centre,yaw+a) for a in (0.,np.pi/4,np.pi/2,3*np.pi/4)]
        heights=(.012,.022) if not coarse else (.015,)
    goals=[]
    for centre,yaw in locations:
        for flip in (0.,np.pi):
            rotation=upright(yaw+flip)
            local=(world-centre)@rotation
            section=model.shape[0]>0
            section=(abs(local[:,0])<.016)&(world[:,2]>support+.004)
            if section.sum()<15:continue
            band=local[section,1];low,high=np.quantile(band,[.01,.99]);width=float(high-low)
            if width<.012 or width>.092:continue
            xy=centre.copy()+rotation[:,1]*float((low+high)/2)
            floor_delta=float(np.min((ctx.gripper[ctx.arm+'_collision']-ctx.ik.tool)@rotation[2]))
            for clearance in heights:
                point=xy.copy();point[2]=support+clearance-floor_delta
                goals.append(transform(rotation,point))
    yield from goals
