"""Bounded partial vase fitting seeded only by a previous RGB-D observation."""
import time
import cv2,numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from perception import pointcloud
from rose_geometry import visible_model_support

def guided_vase_fit(model,cameras,predicted,robot=None,robot_poses=None):
    started=time.monotonic();lo,hi=model['bounds'];parts=[]
    for c in cameras.values():
        hsv=cv2.cvtColor(c['rgb'],cv2.COLOR_RGB2HSV);white=(hsv[...,1]<55)&(hsv[...,2]>100)
        points,_,_=pointcloud(c,white,stride=2);local=(points-predicted[:3,3])@predicted[:3,:3]
        points=points[((local>lo-.06)&(local<hi+.06)).all(axis=1)]
        if robot is not None and len(points):points=points[~robot.point_self_mask(points,robot_poses)]
        if len(points)>20:parts.append(points[::max(1,int(np.ceil(len(points)/600)))])
    if not parts:return None
    observed=np.concatenate(parts)
    if len(observed)<60:return None
    xyz=model['xyz'][::max(1,int(np.ceil(len(model['xyz'])/2500)))];tree=cKDTree(xyz,copy_data=True)
    origin=predicted[:3,3];rotation=predicted[:3,:3]
    def distances(x):
        rot=R.from_rotvec([0,0,x[3]]).as_matrix()@rotation
        return tree.query((observed-origin-x[:3])@rot)[0]
    def residual(x):return np.r_[np.minimum(distances(x),.06),.1*x[:3],.01*x[3]]
    fit=least_squares(residual,np.zeros(4),bounds=([-.06]*3+[-.3],[.06]*3+[.3]),max_nfev=50,diff_step=.001,loss='soft_l1',f_scale=.008)
    errors=distances(fit.x);coverage=float((errors<.012).mean());error=float(np.quantile(errors,.65))
    if coverage<.75 or error>.012 or np.max(abs(fit.x[:3]))>.055:return None
    pose=predicted.copy();pose[:3,3]+=fit.x[:3];pose[:3,:3]=R.from_rotvec([0,0,fit.x[3]]).as_matrix()@rotation
    support=visible_model_support(model,cameras,pose)
    if support['matched_fraction']<.65:return None
    return pose,dict(method='bounded_partial_white_vase_icp',white_points=len(observed),white_coverage=coverage,residual=error,translation_correction=fit.x[:3].tolist(),visible_support=support,fit_seconds=time.monotonic()-started)
