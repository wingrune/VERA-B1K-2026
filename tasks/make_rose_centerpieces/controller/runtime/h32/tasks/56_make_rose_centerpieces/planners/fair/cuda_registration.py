"""CUDA pose fits with original RGB/geometry acceptance gates and priors."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def rose_candidates(gpu, original, model, cameras, level):
    red,green = original.masks(model['rgb'][None])
    mr,mg = model['xyz'][red[0]],model['xyz'][green[0]]
    if min(len(mr),len(mg))<20:
        return []
    mr,mg = mr[::max(1,len(mr)//350)],mg[::max(1,len(mg)//500)]
    rmodel,gmodel = gpu.pose_tensor(mr),gpu.pose_tensor(mg)
    choices = []
    yaw = np.linspace(0,2*np.pi,24,endpoint=False)
    for role,c in cameras.items():
        rm,gm = original.masks(c['rgb'])
        rp,gp = original.pointcloud(c,rm)[0],original.pointcloud(c,gm)[0]
        rp,gp = rp@level[:3,:3].T+level[:3,3],gp@level[:3,:3].T+level[:3,3]
        rp = rp[(rp[:,2]>.35)&(rp[:,2]<1.6)&(np.linalg.norm(rp[:,:2],axis=1)>.35)]
        gp = gp[(gp[:,2]>.35)&(gp[:,2]<1.6)]
        for cluster in original.groups(rp,.025):
            if len(cluster)<10 or np.linalg.norm(np.ptp(cluster,axis=0))>.14:
                continue
            flower = cluster.mean(0);leaves = gp[np.linalg.norm(gp-flower,axis=1)<.32]
            if len(leaves)<15:
                continue
            origins = flower-Rotation.from_euler('z',yaw[:,None]).as_matrix()@mr.mean(0)
            x0 = np.column_stack((origins,yaw));origin = gpu.pose_tensor(origins)
            reds,greens = gpu.pose_tensor(cluster),gpu.pose_tensor(leaves)
            def components(x):
                rot = gpu.yaw(x[:,3])
                return gpu.nearest(rmodel[None]@rot.transpose(-1,-2)+x[:,None,:3],reds), \
                    gpu.nearest(gmodel[None]@rot.transpose(-1,-2)+x[:,None,:3],greens)
            def residual(x):
                dr,dg = components(x)
                origins_batch = origin.repeat(x.shape[0]//len(origin),1)
                return torch.cat((dr.clamp_max(.04),dg.clamp_max(.04),.15*(x[:,:3]-origins_batch)),1)
            fit = gpu.optimize(residual,x0,x0-[.055,.055,.055,.4],x0+[.055,.055,.055,.4],35)
            dr,dg = components(fit);values = gpu.numpy(fit)
            rc = gpu.numpy((dr<.012).to(gpu.dtype).mean(1));gc = gpu.numpy((dg<.018).to(gpu.dtype).mean(1))
            errors = gpu.numpy(torch.quantile(torch.cat((dr,dg),1),.65,dim=1))
            candidates = []
            for x,r,g,error in zip(values,rc,gc,errors):
                if r<.40 or g<.35 or error>.018:
                    continue
                pose = np.eye(4);pose[:3,:3] = Rotation.from_euler('z',x[3]).as_matrix();pose[:3,3] = x[:3]
                pose = np.linalg.inv(level)@pose
                stats = dict(method='rose_color_canonical_icp',camera=role,residual=float(error),
                             red_coverage=float(r),green_coverage=float(g),red_points=len(cluster),green_points=len(leaves),compute_backend='cuda')
                candidates.append((error+.012*(1-g)+.008*(1-r),pose,stats))
            if candidates:
                choices.append(min(candidates,key=lambda x:x[0]))
    unique = []
    for _,pose,stats in sorted(choices,key=lambda x:x[0]):
        if not any(np.linalg.norm(pose[:3,3]-p[:3,3])<.065 for p,_ in unique):
            unique.append((pose,stats))
    return unique


def vase_fit(gpu, original, model, cameras, level):
    import cv2
    xyz = gpu.pose_tensor(model['xyz'][::max(1,len(model['xyz'])//2500)])
    lo,hi = model['bounds'];extent = hi-lo;choices = []
    for role,c in cameras.items():
        hsv = cv2.cvtColor(c['rgb'],cv2.COLOR_RGB2HSV)
        p = original.pointcloud(c,(hsv[...,1]<55)&(hsv[...,2]>100),stride=2)[0]
        p = p@level[:3,:3].T+level[:3,3]
        p = p[(p[:,2]>.35)&(p[:,2]<1.6)&(np.linalg.norm(p[:,:2],axis=1)>.35)]
        for cloud in original.groups(p,.018):
            span = np.ptp(cloud,axis=0)
            if not .50*extent[2]<span[2]<1.20*extent[2] or max(span[:2])>max(extent[:2])*1.20 or max(span[:2])<min(extent[:2])*.75:
                continue
            cloud = cloud[::max(1,len(cloud)//700)];origin = (cloud.min(0)+cloud.max(0))/2-(lo+hi)/2
            x0 = np.array([np.r_[origin,angle] for angle in (0.,np.pi/2)])
            points = gpu.pose_tensor(cloud)
            def residual(x):
                local = (points[None]-x[:,None,:3])@gpu.yaw(x[:,3])
                return gpu.nearest(local,xyz).clamp_max(.06)
            fit = gpu.optimize(residual,x0,x0-[.12,.12,.055,.8],x0+[.12,.12,.055,.8],70)
            errors = gpu.numpy(torch.quantile(residual(fit),.75,dim=1))
            for x,error in zip(gpu.numpy(fit),errors):
                if error>.014:
                    continue
                pose = np.eye(4);pose[:3,:3] = Rotation.from_euler('z',x[3]).as_matrix();pose[:3,3] = x[:3]
                pose = np.linalg.inv(level)@pose;support = gpu.support(model,cameras,pose)
                if support['matched_fraction']<.65:
                    continue
                choices.append((error,pose,dict(method='vase_white_canonical_icp',camera=role,residual=float(error),points=len(cloud),visible_support=support,compute_backend='cuda')))
    if choices:
        _,pose,stats = min(choices,key=lambda x:x[0]);return pose,stats
    return None


def guided_single(gpu, original, model, cameras, predicted, translation_limit=.075, red_min=.4,
                  green_min=.35, visible_only=False, prepared=None, deadline=None,
                  initial_delta=None, rotation_limit=.30):
    prepared = original._prepare_guided(model,cameras) if prepared is None else prepared
    mr,mg,red,green = original._guided_clouds(prepared,predicted,model,translation_limit)
    if min(len(mr),len(mg))<20 or min(len(red),len(green))<10:
        return None
    mr,mg,red,green = (gpu.pose_tensor(p) for p in (mr,mg,red,green))
    origin,rotation = gpu.pose_tensor(predicted[:3,3]),gpu.pose_tensor(predicted[:3,:3])
    def components(x):
        rot = gpu.rotation(x[:,3:])@rotation
        if visible_only:
            return gpu.nearest((red[None]-origin-x[:,None,:3])@rot,mr), \
                gpu.nearest((green[None]-origin-x[:,None,:3])@rot,mg)
        return gpu.nearest(mr[None]@rot.transpose(-1,-2)+origin+x[:,None,:3],red), \
            gpu.nearest(mg[None]@rot.transpose(-1,-2)+origin+x[:,None,:3],green)
    def residual(x):
        dr,dg = components(x);cap = .12 if visible_only else .04
        return torch.cat((dr.clamp_max(cap),dg.clamp_max(cap),.10*x[:,:3],.015*x[:,3:]),1)
    initial = np.zeros((1,6))
    if initial_delta is not None:
        initial[0,:3] = initial_delta
    limits = np.array([translation_limit]*3+[rotation_limit]*3)
    fit = gpu.optimize(residual,initial,-limits,limits,55 if visible_only else 50,robust=visible_only)
    dr,dg = components(fit);x = gpu.numpy(fit)[0]
    rc,gc = float((dr<.014).to(gpu.dtype).mean()),float((dg<.018).to(gpu.dtype).mean())
    error = float(torch.quantile(torch.cat((dr,dg),1),.65))
    if rc<red_min or gc<green_min or error>.018 or np.max(abs(x[:3]))>translation_limit-.003 or np.max(abs(x[3:]))>rotation_limit-.01:
        return None
    pose = predicted.copy();pose[:3,:3] = Rotation.from_rotvec(x[3:]).as_matrix()@predicted[:3,:3];pose[:3,3] += x[:3]
    return pose,dict(method='kinematically_gated_rose_color_icp',correspondence_direction='observed_to_model' if visible_only else 'model_to_observed',
                     observed_red_points=len(red),observed_green_points=len(green),residual=error,red_coverage=rc,green_coverage=gc,
                     translation_correction=x[:3].tolist(),rotation_correction=x[3:].tolist(),compute_backend='cuda')
