"""Resolve cleaner direction by matching bristle geometry to observed gray surfaces."""
import cv2,numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from perception import pointcloud


def resolve_pipe(model,cameras,blue,rotation,colored_template,capsules):
    hsv=cv2.cvtColor(model['rgb'][None].astype(np.uint8),cv2.COLOR_RGB2HSV)[0]
    white_template=model['xyz'][(hsv[:,1]<70)&(hsv[:,2]>60)][::5]
    if len(white_template)<30:return None
    center=blue.mean(0);white=[]
    for camera in cameras.values():
        h=cv2.cvtColor(camera['rgb'],cv2.COLOR_RGB2HSV)
        points,_,_=pointcloud(camera,mask=(h[...,1]<75)&(h[...,2]>50))
        delta=points-center
        keep=(np.linalg.norm(delta,axis=1)<.45)&(abs(delta[:,2])<.095)
        points=points[keep];keep=np.ones(len(points),bool)
        for a,b,radius in capsules:
            axis=b-a;alpha=np.clip((points-a)@axis/max(float(axis@axis),1e-10),0,1)
            keep&=np.linalg.norm(points-a-alpha[:,None]*axis,axis=1)>radius+.01
        points=points[keep]
        if len(points):white.append(points[::max(1,len(points)//1500)])
    if not white:return None
    observed=np.concatenate(white);tree=cKDTree(observed)
    template=colored_template[::max(1,len(colored_template)//500)];blue_tree=cKDTree(template)
    blue=blue[::max(1,len(blue)//500)];colored_center=colored_template.mean(0);options=[]
    yaw=np.arctan2(rotation[1,0],rotation[0,0])
    for sign in [0.,np.pi]:
        initial=np.r_[np.zeros(3),yaw+sign]
        def geometry(x):
            rot=R.from_euler('z',x[3]).as_matrix();origin=center-rot@colored_center+x[:3]
            return rot,origin
        def residual(x):
            rot,origin=geometry(x)
            bd=blue_tree.query((blue-origin)@rot)[0]
            wd=tree.query(white_template@rot.T+origin)[0]
            return np.r_[np.minimum(bd,.04),np.minimum(wd,.08),.5*x[:3]]
        fit=least_squares(residual,initial,bounds=(initial-[.035,.035,.035,.35],initial+[.035,.035,.035,.35]),max_nfev=35,diff_step=.001)
        rot,origin=geometry(fit.x);distance=tree.query(white_template@rot.T+origin)[0]
        coverage=float(np.mean(distance<.04));score=float(np.mean(np.minimum(distance,.08)))
        if coverage>.45 and score<.035:
            pose=np.eye(4);pose[:3,:3]=rot;pose[:3,3]=origin
            options.append((score,pose,dict(bristle_coverage=coverage,bristle_residual=score,orientation_resolved=True)))
    if options:
        _,pose,stats=min(options,key=lambda x:x[0]);return pose,stats
    return None
