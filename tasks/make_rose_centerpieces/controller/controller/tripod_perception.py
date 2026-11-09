"""Upright tripod localization from its observed head, stem and leg geometry."""
import numpy as np,cv2
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from perception import pointcloud
def pieces(points):
 if len(points)<5:return []
 pairs=cKDTree(points).query_pairs(.035,output_type='ndarray')
 if not len(pairs):return []
 graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(points),len(points)))
 n,labels=connected_components(graph,directed=False)
 return [points[labels==i] for i in range(n) if np.sum(labels==i)>=5]
def fit_tripod(model,cams,level):
 xyz=model['xyz'][::18];lo,hi=model['bounds'];height=hi[2]-lo[2];choices=[]
 head=model['xyz'][(model['xyz'][:,2]>hi[2]-.22)&(model['rgb'].max(1)<130)];head_center=head.mean(0);head_span=np.linalg.norm(np.ptp(head[:,:2],axis=0))
 for role,c in cams.items():
  width=c['rgb'].shape[1]
  if width>480:
   c=c.copy();c['rgb']=cv2.resize(c['rgb'],(480,480),interpolation=cv2.INTER_AREA);c['depth']=cv2.resize(c['depth'],(480,480),interpolation=cv2.INTER_NEAREST);c['K']=c['K'].copy();c['K'][:2]*=480/width
  hsv=cv2.cvtColor(c['rgb'],cv2.COLOR_RGB2HSV);dark=(hsv[...,2]<95)&(hsv[...,1]<90)
  points,_,uv=pointcloud(c,mask=dark);p=points@level[:3,:3].T+level[:3,3]
  keep=(p[:,2]>height-.20)&(p[:,2]<height+.04)&(np.linalg.norm(p[:,:2],axis=1)>.5)
  mask=np.zeros(dark.shape,np.uint8);valid=uv[keep];mask[valid[:,1],valid[:,0]]=1;mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
  n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
  allp,colors,_=pointcloud(c,stride=3);allp=allp@level[:3,:3].T+level[:3,3]
  for label in sorted(range(1,n),key=lambda i:-stats[i,cv2.CC_STAT_AREA])[:8]:
   if stats[label,cv2.CC_STAT_AREA]<8:continue
   pts,_,_=pointcloud(c,mask=labels==label);pts=pts@level[:3,:3].T+level[:3,3];pts=pts[(pts[:,2]>height-.21)&(pts[:,2]<height+.05)]
   for pts in pieces(pts):
    if len(pts)<5 or np.linalg.norm(np.ptp(pts[:,:2],axis=0))>head_span*1.25:continue
    center=np.median(pts,axis=0);near=allp[(np.linalg.norm(allp[:,:2]-center[:2],axis=1)<.75)&(allp[:,2]>.09)&(allp[:,2]<height+.05)]
    if len(near)<100 or np.ptp(near[:,2])<.8:continue
    tree=cKDTree(near);best=None
    for yaw in np.linspace(0,2*np.pi,12,endpoint=False):
     start=np.r_[center[:2]-(R.from_euler('z',yaw).as_matrix()@head_center)[:2],yaw]
     def residual(x):
      rot=R.from_euler('z',x[2]).as_matrix();world=xyz@rot.T+np.r_[x[:2],-lo[2]]
      distance=tree.query(world)[0]
      return np.minimum(distance,.08)
     fit=least_squares(residual,start,bounds=(start-[.12,.12,.4],start+[.12,.12,.4]),max_nfev=18,diff_step=.002)
     dist=residual(fit.x);score=float(np.quantile(dist,.65))
     if best is None or score<best[0]:best=(score,fit.x,float(np.mean(dist<.025)))
    error,x,coverage=best
    if error>.024 or coverage<.58:continue
    pose=np.eye(4);pose[:3,:3]=R.from_euler('z',x[2]).as_matrix();pose[:3,3]=np.r_[x[:2],-lo[2]];pose=np.linalg.inv(level)@pose
    choices.append((error,pose,dict(method='tripod_geometry',camera=role,residual=error,coverage=coverage,orientation_ambiguous=True)))
 if choices:
  _,pose,stats=min(choices,key=lambda x:x[0]);return pose,stats
 return None
