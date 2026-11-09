"""Geometry verification for small texture-poor objects on observed support planes."""
import cv2,numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from perception import pointcloud,red_mask
def shape_fit(model,cams,level,cat):
 if cat not in ['digital_camera','facsimile','keyboard','scrub_brush']:return None
 lo,hi=model['bounds'];ext=hi-lo;center=(lo+hi)/2;xyz=model['xyz'][::4];tree=cKDTree(xyz);choices=[]
 for name,c in cams.items():
  hsv=cv2.cvtColor(c['rgb'],cv2.COLOR_RGB2HSV)
  mask=(hsv[...,1]<85)&(hsv[...,2]<(205 if cat=='facsimile' else 80))
  pp,_,uv=pointcloud(c,mask=mask);leveled=pp@level[:3,:3].T+level[:3,3]
  if cat=='keyboard':valid=(leveled[:,2]>.5)&(leveled[:,2]<1.2)
  else:valid=(leveled[:,2]>(.006 if cat=='scrub_brush' else .014))&(leveled[:,2]<ext[2]+.035)
  valid &=np.linalg.norm(pp[:,:2],axis=1)>.42
  clean=np.zeros(mask.shape,np.uint8);uv=uv[valid];clean[uv[:,1],uv[:,0]]=1
  clean=cv2.morphologyEx(clean,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
  n,labels,stats,_=cv2.connectedComponentsWithStats(clean)
  for i in range(1,n):
   if stats[i,cv2.CC_STAT_AREA]<10:continue
   points,_,_=pointcloud(c,mask=labels==i);p=points@level[:3,:3].T+level[:3,3]
   if cat!='keyboard':p=p[(p[:,2]>.008)&(p[:,2]<ext[2]+.04)]
   if len(p)<10:continue
   p=p[::max(1,len(p)//180)]
   xy=cv2.minAreaRect(p[:,:2].astype(np.float32));span=np.sort(xy[1]);expected=np.sort(ext[:2])
   if np.any(span<expected*np.array([.38,.55])) or np.any(span>expected*1.3):continue
   if np.ptp(p[:,2])>max(.045,ext[2]*1.25):continue
   markers=[]
   marker_local=None
   if cat in ['digital_camera','facsimile']:
    links=model.get('metadata',{}).get('links',{})
    toggle=[v for v in links.values() if v['meta_link_id']=='0']
    if toggle:
     marker_local=np.array(toggle[0]['position'])
     rp,_,_=pointcloud(c,red_mask(c['rgb']));rp=rp@level[:3,:3].T+level[:3,3]
     rp=rp[(np.linalg.norm(rp[:,:2]-p[:,:2].mean(0),axis=1)<max(ext[:2]))&(rp[:,2]<.18)]
     if len(rp)>=2:markers=[np.median(rp,axis=0)]
   if cat in ['digital_camera','facsimile'] and not markers:continue
   candidates=[]
   for yaw in np.linspace(0,2*np.pi,12,endpoint=False):
    rot=R.from_euler('z',yaw).as_matrix();position=np.mean(p,axis=0)-rot@xyz.mean(0)
    if cat!='keyboard':position[2]=-lo[2]+.003
    else:position[2]=np.median(p[:,2])-hi[2]
    def residual(x):
     rot=R.from_euler('z',x[3]).as_matrix();local=(p-x[:3])@rot;dist,ids=tree.query(local)
     error=np.minimum(dist,.04)
     if markers:error=np.r_[error,2.*(rot@marker_local+x[:3]-markers[0])]
     return error
    x0=np.r_[position,yaw];lower=x0-np.array([.07,.07,.008 if cat!='keyboard' else .015,.65]);upper=x0+np.array([.07,.07,.008 if cat!='keyboard' else .015,.65])
    fit=least_squares(residual,x0,bounds=(lower,upper),max_nfev=35,diff_step=.001)
    residuals=residual(fit.x);quality=float(np.quantile(residuals[:len(p)],.75))+(float(np.linalg.norm(residuals[-3:]))*.1 if markers else 0.)
    candidates.append((quality,fit.x))
   error,x=min(candidates,key=lambda a:a[0])
   if error>.012:continue
   marker_error=None
   if markers:
    marker_error=float(np.linalg.norm(R.from_euler('z',x[3]).as_matrix()@marker_local+x[:3]-markers[0]))
    if marker_error>.025:continue
   t=np.eye(4);t[:3,:3]=R.from_euler('z',x[3]).as_matrix();t[:3,3]=x[:3];t=np.linalg.inv(level)@t
   choices.append((error-.00001*len(p),t,dict(method='rgbd_shape',camera=name,residual=error,points=len(p),orientation_ambiguous=not bool(markers),marker_error_m=marker_error)))
 if choices:
  _,t,stats=min(choices,key=lambda a:a[0]);return t,stats
 return None

