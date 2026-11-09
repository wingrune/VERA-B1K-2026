"""CUDA classification against canonical robot hulls, with CPU boundary fallback."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from . import cuda_collision_kernels as kernels


def point_mask(gpu,robot,points,poses,padding,tree,original):
    mask=np.zeros(len(points),bool)
    if not len(points): return mask
    tree=cKDTree(points) if tree is None else tree
    hulls=[h for h in robot.hulls if h['link'] in poses]
    if hulls:
        bounds=np.asarray([h['bounds'] for h in hulls])
        transforms=np.asarray([poses[h['link']] for h in hulls])
        centers=(bounds[:,0]+bounds[:,1])/2
        world=np.asarray([center@pose[:3,:3].T+pose[:3,3] for center,pose in zip(centers,transforms)])
        radii=np.asarray([np.linalg.norm((hi-lo)/2+padding)+1e-8 for lo,hi in bounds])
        candidates=tree.query_ball_point(world,radii,return_sorted=False)
        ids=np.concatenate([np.asarray(x,dtype=np.int64) for x in candidates])
        owners=np.repeat(np.arange(len(hulls)),[len(x) for x in candidates])
        if len(ids):
            cache_key=tuple(id(h) for h in hulls)
            cached=getattr(robot,'_cuda_hull_planes',None)
            if cached is None or cached[0]!=cache_key:
                width=max(len(h['equations']) for h in hulls)
                planes=np.zeros((len(hulls),width,4));planes[:,:,3]=-np.inf
                for i,h in enumerate(hulls): planes[i,:len(h['equations'])]=h['equations']
                cached=(cache_key,gpu.tensor(planes));robot._cuda_hull_planes=cached
            planes=cached[1];cloud=gpu.tensor(points);ts=gpu.tensor(transforms);bs=gpu.tensor(bounds)
            pair_ids=torch.as_tensor(ids,device=gpu.device,dtype=torch.int32)
            pair_owners=torch.as_tensor(owners,device=gpu.device,dtype=torch.int32)
            result=torch.zeros(len(points),device=gpu.device,dtype=torch.int32)
            uncertain=torch.zeros(1,device=gpu.device,dtype=torch.int32)
            kernels.launch(kernels.hull_union,[cloud,ts,bs,planes,pair_ids,pair_owners,result,uncertain],
                           len(ids),(float(padding),planes.shape[1]))
            mask=gpu.numpy(result).astype(bool)
            uncertain=bool(uncertain[0])
            if uncertain:
                return original(robot,points,poses,padding=padding,tree=tree)
    # Preserve the original wheel/steer strict norm comparison on CPU.
    for name,c,r in zip(robot.names,robot.centers(poses),robot.radii):
        if name.startswith(('wheel_motor','steer_motor')):
            ids=np.asarray(tree.query_ball_point(c,r+.025+1e-8),dtype=int)
            mask[ids]|=np.linalg.norm(points[ids]-c,axis=1)<r+.025
    return mask


def sphere_mask(gpu,points,centers,radii,tree):
    """Same closed sphere union as CPU KD queries; exact CPU fallback near edges."""
    result=np.zeros(len(points),bool)
    if not len(points) or not len(centers):return result
    cloud=gpu.tensor(points);cs=gpu.tensor(centers);rs=gpu.tensor(radii)
    output=torch.zeros(len(points),device=gpu.device,dtype=torch.int32)
    uncertain=torch.zeros(1,device=gpu.device,dtype=torch.int32)
    kernels.launch(kernels.sphere_union,[cloud,cs,rs,output,uncertain],len(points),(len(centers),))
    result=gpu.numpy(output).astype(bool)
    uncertain=bool(uncertain[0])
    if uncertain:
        result.fill(False)
        for ids in tree.query_ball_point(centers,radii): result[ids]=True
    return result
