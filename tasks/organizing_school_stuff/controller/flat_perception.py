"""RGB-D plane refinement for flat school objects; no simulator state."""
import numpy as np
from scipy.optimize import least_squares
from surface_icp import SurfaceObjective
from art_skills import upright,transform


def refine_flat_pose(points,model,tree,initial,score_pose=None):
    if len(points)<30:return None
    rng=np.random.default_rng(7);best=None
    for _ in range(80):
        a,b,c=points[rng.choice(len(points),3,replace=False)]
        normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
        if length<1e-5:continue
        normal/=length
        if normal[2]<0:normal=-normal
        if normal[2]<.94:continue
        keep=abs((points-a)@normal)<.0025
        count=int(keep.sum())
        if best is None or count>best[0]:best=(count,keep)
    if best is None or best[0]<max(25,.45*len(points)):return None
    surface=points[best[1]];centre=surface.mean(0)
    _,singular,v=np.linalg.svd(surface-centre,full_matrices=False)
    if singular[1]<.03:return None
    normal=v[-1]
    if normal[2]<0:normal=-normal
    if normal[2]<.94:return None
    x=np.array([1.,0.,0.]);x-=normal*(x@normal);x/=np.linalg.norm(x)
    frame=np.column_stack([x,np.cross(normal,x),normal])
    cloud=points@frame;seed=initial[:3,3]@frame
    seed[2]=float(np.median(surface@normal)-model['bounds'][1,2])
    initial_x=frame.T@initial[:3,0];yaw=float(np.arctan2(initial_x[1],initial_x[0]))
    objective=SurfaceObjective(cloud,tree);found=None
    for offset in (0.,np.pi,-np.pi/2,np.pi/2):
        angle=yaw+offset;start=np.r_[seed,angle]
        fit=least_squares(objective.fun,start,jac=objective.jac,max_nfev=45,loss='soft_l1',f_scale=.006,
            bounds=(np.r_[seed[:2]-.065,seed[2]-.003,angle-.50],np.r_[seed[:2]+.065,seed[2]+.003,angle+.50]))
        rotation=frame@upright(fit.x[3]);position=frame@fit.x[:3]
        distances=tree.query((points-position)@rotation)[0];score=float(np.quantile(distances,.75))
        if score_pose is not None:score+=float(score_pose(transform(rotation,position)))
        if found is None or score<found[0]:found=(score,transform(rotation,position))
    return found
