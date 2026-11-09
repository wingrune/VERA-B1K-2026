"""Refine a carried tool's translation from RGB-D near its kinematic prediction."""
import cv2,numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from perception import pointcloud


def refine_held(model,cameras,predicted,robot_mask=None):
    colors=model['rgb'].astype(np.uint8);hsv=cv2.cvtColor(colors[None],cv2.COLOR_RGB2HSV)[0]
    colorful=hsv[:,1]>100
    if colorful.sum()<50:return None
    bins=np.bincount((hsv[colorful,0]//6).astype(int),minlength=30);hue=(np.argmax(bins)+.5)*6
    delta=np.minimum(abs(hsv[:,0].astype(float)-hue),180-abs(hsv[:,0].astype(float)-hue))
    template=model['xyz'][colorful&(delta<12)][::2];tree=cKDTree(template)
    rotation=predicted[:3,:3];origin=predicted[:3,3];lo,hi=model['bounds'];observations=[]
    for role,c in cameras.items():
        h=cv2.cvtColor(c['rgb'],cv2.COLOR_RGB2HSV);d=np.minimum(abs(h[...,0].astype(float)-hue),180-abs(h[...,0].astype(float)-hue))
        points,_,_=pointcloud(c,mask=(h[...,1]>100)&(d<12))
        if robot_mask is not None:points=points[~robot_mask(points)]
        local=(points-origin)@rotation
        keep=((local>lo-.07)&(local<hi+.07)).all(axis=1)
        points=points[keep]
        if len(points)>=20:observations.append(points[::max(1,len(points)//700)])
    if not observations:return None
    points=np.concatenate(observations);local=(points-origin)@rotation
    if len(points)<40 or np.linalg.norm(np.ptp(local,axis=0))<np.linalg.norm(hi-lo)*.3:return None
    def residual(translation):
        distance,_=tree.query((points-origin-translation)@rotation)
        return np.r_[np.minimum(distance,.04),.7*translation]
    fit=least_squares(residual,np.zeros(3),bounds=(-.065,.065),max_nfev=60,diff_step=.001)
    error=float(np.quantile(residual(fit.x)[:len(points)],.80))
    if error>.014 or np.max(abs(fit.x))>.062:return None
    result=predicted.copy();result[:3,3]+=fit.x
    return result,dict(method='kinematically_gated_color_icp',points=len(points),residual=error,correction_m=fit.x.tolist())
