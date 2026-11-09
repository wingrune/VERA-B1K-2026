"""Fine map alignment from observed RGB-D tabletop boundaries and a static table prior."""
import numpy as np
import cv2
from scipy.optimize import least_squares
from art_perception import CAMERAS,quaternion_matrix
from art_skills import upright
from manip import pose2

def align_desk(observer,base,excluded=()):
    desk=observer.scene.meta['target_surface'];height=float(desk['aabb_hi'][2]);boxes=observer.scene.boxes
    plates=[b for b in boxes if b[5]>height-.004 and b[5]<=height+.004 and b[3]-b[0]>.4 and b[4]-b[1]>.3 and b[5]-b[2]<.10]
    if not plates:return None,dict(reason='no tabletop prior')
    plate=max(plates,key=lambda b:(b[3]-b[0])*(b[4]-b[1]));edges=np.r_[plate[:2],plate[3:5]]
    frame=pose2(*base);points=[];labels=[]
    for camera,(link,slot,focal) in CAMERAS.items():
        depth=observer.data[f'robot_r1::robot_r1:{link}:Camera:0::depth_linear'];h,w=depth.shape;v,u=np.mgrid[:h,:w]
        valid=np.isfinite(depth)&(depth>.12)&(depth<4.)
        safe=np.where(valid,depth,0.);pc=np.stack([(u+.5-w/2)/(focal*w)*safe,(v+.5-h/2)/(focal*w)*safe,safe],-1)
        rel=observer.data['robot_r1::cam_rel_poses'].reshape(3,7)[slot];rot=quaternion_matrix(rel[3:])@np.diag([1.,-1.,-1.])
        body=pc@rot.T+rel[:3];world=body@frame[:3,:3].T+frame[:3,3]
        # Restrict the boundary to the thin horizontal tabletop plane. The old
        # 45 mm outside threshold skipped front edges adjoining a short fascia,
        # leaving almost no observations of one translation axis.
        above=world[...,2]-height;outer=valid&(abs(above)>.008)
        mask=valid&(abs(above)<.004)&cv2.dilate(outer.astype('uint8'),np.ones((3,3),'uint8')).astype(bool)
        mask &= np.all(world[...,:2]>=plate[:2]-.06,axis=-1)&np.all(world[...,:2]<=plate[3:5]+.06,axis=-1)
        for category,obj in excluded:
            bounds=observer.models[category]['bounds'];centre=(obj@np.r_[bounds.mean(0),1.])[:2]
            half=abs(obj[:2,:3])@(bounds[1]-bounds[0])/2+.02
            mask &= ~np.all(abs(world[...,:2]-centre)<=half,axis=-1)
        pts=body[mask];mapped=world[mask]
        if not len(pts):continue
        distances=abs(mapped[:,[0,1,0,1]]-edges);nearest=distances.argmin(1)
        keep=distances.min(1)<.06;pts=pts[keep];nearest=nearest[keep]
        if len(pts)>600:ids=np.linspace(0,len(pts)-1,600).astype(int);pts=pts[ids];nearest=nearest[ids]
        points.append(pts[:,:2]);labels.append(nearest)
    if not points:return None,dict(reason='no observed boundary')
    points=np.concatenate(points);labels=np.concatenate(labels);count=len(points)
    if count<30 or np.linalg.norm(np.ptp(points,axis=0))<.35:return None,dict(reason='insufficient boundary',points=count)
    axes=labels%2;base=np.asarray(base,float);prior=.002*np.sqrt(count)
    def residual(p):
        mapped=points@upright(p[2])[:2,:2].T+p[:2]
        errors=mapped[np.arange(count),axes]-edges[labels]
        return np.r_[errors,prior*(p-base)/[.04,.04,.03]]
    fit=least_squares(residual,base,bounds=(base-[.045,.045,.035],base+[.045,.045,.035]),loss='soft_l1',f_scale=.003,max_nfev=40)
    before=float(np.median(abs(residual(base)[:count])));after=float(np.median(abs(residual(fit.x)[:count])))
    info=dict(points=count,edges=np.bincount(labels,minlength=4).tolist(),before=before,after=after,delta=(fit.x-base).tolist())
    if after>.003 or before-after<.002 or after>.7*before:return None,dict(info,reason='weak alignment')
    return fit.x,info
