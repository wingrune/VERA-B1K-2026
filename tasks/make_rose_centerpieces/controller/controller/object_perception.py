"""RGB-D object pose estimation. No simulator, state, segmentation, or goal access."""
import json,cv2,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from perception import pointcloud,red_mask
def rigid(a,b):
 ca=a.mean(0);cb=b.mean(0);u,s,v=np.linalg.svd((a-ca).T@(b-cb));rot=v.T@np.diag([1,1,np.linalg.det(v.T@u.T)])@u.T
 t=cb-rot@ca
 return rot,t
def unproject(camera,pixels):
 xy=np.rint(pixels).astype(int);h,w=camera['depth'].shape;xy[:,0]=xy[:,0].clip(0,w-1);xy[:,1]=xy[:,1].clip(0,h-1)
 d=camera['depth'][xy[:,1],xy[:,0]]
 p=np.c_[(xy[:,0]+.5-camera['K'][0,2])*d/camera['K'][0,0],(xy[:,1]+.5-camera['K'][1,2])*d/camera['K'][1,1],d]
 return p@camera['R'].T+camera['t'],np.isfinite(d)&(d>.08)&(d<4.5)
class ObjectPerception:
 def __init__(self,folder,categories):
  self.level=np.eye(4);self.models={};self.fit_calls={};self.sift=cv2.SIFT_create(nfeatures=1600,contrastThreshold=.012)
  for cat in categories:
   p=Path(folder)/cat;m=np.load(p/'model.npz');f=np.load(p/'features.npz')
   self.models[cat]=dict(xyz=m['xyz'],rgb=m['rgb'],bounds=m['bounds'],feature_xyz=f['xyz'],des=f['descriptors'].astype(np.float32),metadata=json.loads((p/'metadata.json').read_text()))
 def features(self,cams):
  out={}
  for name,c in cams.items():
   mask=(np.isfinite(c['depth'])&(c['depth']>.15)&(c['depth']<4.5)).astype(np.uint8)*255
   kp,des=self.sift.detectAndCompute(c['rgb'],mask)
   if des is not None:out[name]=(np.array([k.pt for k in kp]),des)
  return out
 def fit(self,cat,cams,features=None):
  if cat=='keyboard':
   from keyboard_perception import fit_keyboard
   return fit_keyboard(self.models[cat],cams,self.level)
  m=self.models[cat];choices=[];features=self.features(cams) if features is None else features
  for name,(pixels,des) in features.items():
   c=cams[name];raw=cv2.BFMatcher().knnMatch(m['des'],des,k=2);unique={}
   for pair in raw:
    if len(pair)!=2:continue
    a,b=pair
    if a.distance<.72*b.distance and (a.trainIdx not in unique or a.distance<unique[a.trainIdx].distance):unique[a.trainIdx]=a
   good=list(unique.values())
   if len(good)<6:continue
   a=m['feature_xyz'][[p.queryIdx for p in good]].astype(float);b,valid=unproject(c,pixels[[p.trainIdx for p in good]]);a=a[valid];b=b[valid]
   if len(a)<6:continue
   rng=np.random.default_rng(839);best=None
   for _ in range(min(700,len(a)*30)):
    ii=rng.choice(len(a),3,replace=False);aa=a[ii];bb=b[ii]
    if np.linalg.norm(np.cross(aa[1]-aa[0],aa[2]-aa[0]))<.00015:continue
    if np.max(np.abs(np.linalg.norm(aa[:,None]-aa,axis=2)-np.linalg.norm(bb[:,None]-bb,axis=2)))>.035:continue
    rot,t=rigid(aa,bb);err=np.linalg.norm(a@rot.T+t-b,axis=1);mask=err<.018
    score=int(mask.sum())
    if score>=6 and (best is None or score>best[0]):best=(score,mask)
   if best is None:continue
   ii=best[1];rot,t=rigid(a[ii],b[ii]);err=np.linalg.norm(a@rot.T+t-b,axis=1);ii=err<.016
   if ii.sum()<6 or np.linalg.norm(np.ptp(a[ii],axis=0))<.045:continue
   # Reject a geometrically degenerate or implausibly distant correspondence set.
   if np.linalg.svd(a[ii]-a[ii].mean(0),compute_uv=False)[1]<.012:continue
   rot,t=rigid(a[ii],b[ii]);res=float(np.median(np.linalg.norm(a[ii]@rot.T+t-b[ii],axis=1)))
   T=np.eye(4);T[:3,:3]=rot;T[:3,3]=t
   marker_error=None
   if cat in ['digital_camera','facsimile']:
    link=next(v for v in m['metadata']['links'].values() if v['meta_link_id']=='0')
    expected=T[:3,3]+T[:3,:3]@np.array(link['position'])
    rp,_,_=pointcloud(c,red_mask(c['rgb']))
    marker_error=float(np.min(np.linalg.norm(rp-expected,axis=1))) if len(rp) else 1.
    if marker_error>.04:continue
   choices.append((int(ii.sum())-100*res,T,dict(method='rgbd_features',camera=name,inliers=int(ii.sum()),residual=res,marker_error_m=marker_error)))
  if choices:
   _,T,stats=max(choices,key=lambda x:x[0]);return T,stats
  if cat=='camera_tripod':
   self.fit_calls[cat]=self.fit_calls.get(cat,0)+1
   if self.fit_calls[cat]%10==1:
    from tripod_perception import fit_tripod
    return fit_tripod(m,{'head':cams['head']},self.level)
   return None
  if cat in ['pipe_cleaner','scrub_brush']:
   found=self.color_fit(cat,cams)
   if found is not None:return found
  from shape_perception import shape_fit
  return shape_fit(m,cams,self.level,cat)
 def color_fit(self,cat,cams):
  m=self.models[cat];hsv=cv2.cvtColor(m['rgb'][None].astype(np.uint8),cv2.COLOR_RGB2HSV)[0];sat=hsv[:,1]>100
  if sat.sum()<100:return None
  bins=np.bincount((hsv[sat,0]//6).astype(int),minlength=30);hue=(np.argmax(bins)+.5)*6;choices=[]
  for name,c in cams.items():
   h=cv2.cvtColor(c['rgb'],cv2.COLOR_RGB2HSV);dh=np.abs(h[...,0].astype(float)-hue);mask=(np.minimum(dh,180-dh)<12)&(h[...,1]>95)&(h[...,2]>35)
   n,lab,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
   for i in range(1,n):
    if stats[i,cv2.CC_STAT_AREA]<8:continue
    points,_,_=pointcloud(c,lab==i)
    if len(points)<8:continue
    leveled=points@self.level[:3,:3].T+self.level[:3,3]
    if cat=='scrub_brush' and getattr(self,'held',None)!=cat and not -.025<float(np.median(leveled[:,2]))<.055:continue
    if np.linalg.norm(np.ptp(points,axis=0))>np.linalg.norm(np.ptp(m['bounds'],axis=0))*1.4:continue
    center=np.median(points,axis=0);points=points[np.linalg.norm(points-center,axis=1)<max(np.ptp(m['bounds'],axis=0))*.8]
    if len(points)<8:continue
    _,s,v=np.linalg.svd(points-points.mean(0),full_matrices=False);axis=v[0];axis[2]=0;axis/=max(np.linalg.norm(axis),1e-8)
    if axis[0]<0:axis=-axis
    rot=np.column_stack([axis,np.cross([0,0,1],axis),[0,0,1.]])
    ext=np.ptp(m['bounds'],axis=0);observed=(points-center)@rot
    if np.ptp(observed[:,0])<ext[0]*.3 or np.ptp(observed[:,0])>ext[0]*1.25:continue
    center=(points.min(0)+points.max(0))/2
    # Colored bristles occupy only part of the tool. Align the observed colored
    # point centroid to the training color subset, with the two yaw hypotheses.
    subset=m['xyz'][sat & (np.minimum(abs(hsv[:,0].astype(float)-hue),180-abs(hsv[:,0].astype(float)-hue))<12)]
    if len(subset)<10:continue
    if cat=='pipe_cleaner':
     from pipe_orientation import resolve_pipe
     resolved=resolve_pipe(m,cams,points,rot,subset,getattr(self,'robot_capsules',[]))
     if resolved is not None:
      pose,extra=resolved;choices.append((len(points),pose,dict(method='tool_color_shape',camera=name,points=len(points),yaw_ambiguous=False,**extra)))
     continue
    T=np.eye(4);T[:3,:3]=rot;T[:3,3]=points.mean(0)-rot@subset.mean(0)
    choices.append((len(points),T,dict(method='tool_color_shape',camera=name,points=len(points),yaw_ambiguous=True)))
  if choices:
   _,T,s=max(choices,key=lambda x:x[0]);return T,s
  return None
