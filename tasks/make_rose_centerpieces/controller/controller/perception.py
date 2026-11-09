"""Known-object RGB-D localization from onboard images, in the current robot frame."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from scipy.optimize import minimize

def red_mask(rgb):
 a=rgb.astype(float);return (a[...,0]>85)&(a[...,0]>2.5*a[...,1]+20)&(a[...,0]>2.5*a[...,2]+20)

def make_cameras(obs,size=256):
 rel=obs['robot_r1::cam_rel_poses'].reshape(3,7);cams={}
 for i,(role,link,focal) in enumerate([('left_wrist','left_realsense_link',388.6639/480),('right_wrist','right_realsense_link',388.6639/480),('head','zed_link',306/720)]):
  prefix=f'robot_r1::robot_r1:{link}:Camera:0::';im=obs[prefix+'rgb'][...,:3];h,w=im.shape[:2];width=w if size is None else size
  depth=obs.get(prefix+'depth_linear');depth=None if depth is None else cv2.resize(depth.squeeze(),(width,width),interpolation=cv2.INTER_NEAREST)
  im=cv2.resize(im,(width,width),interpolation=cv2.INTER_AREA);K=np.array([[focal*width,0,width/2],[0,focal*width,width/2],[0,0,1.]])
  rot=R.from_quat(rel[i,3:]).as_matrix()@np.diag([1,-1,-1]);cams[role]={'rgb':im,'depth':depth,'K':K,'R':rot,'t':rel[i,:3]}
 return cams

def pointcloud(cam,mask=None,stride=1):
 d=cam['depth'];yy,xx=np.indices(d.shape);valid=np.isfinite(d)&(d>.06)&(d<5)
 if mask is not None:valid &=mask
 y,x=np.where(valid);y,x=y[::stride],x[::stride];z=d[y,x]
 p=np.c_[(x+.5-cam['K'][0,2])/cam['K'][0,0]*z,(y+.5-cam['K'][1,2])/cam['K'][1,1]*z,z]
 return p@cam['R'].T+cam['t'],cam['rgb'][y,x],np.c_[x,y]

