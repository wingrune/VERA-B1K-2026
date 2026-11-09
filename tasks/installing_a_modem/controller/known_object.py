"""Rendered known-object RGB-D fitting; only canonical object geometry is stored."""
from pathlib import Path
import cv2,numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from perception import pointcloud

def blue_mask(rgb):
    im=rgb.astype(float);return (im[...,2]>50)&(im[...,2]>1.25*im[...,0])&(im[...,2]>1.15*im[...,1])
class ModemPose:
    def __init__(self,path):
        x=np.load(path);self.xyz=x['xyz'][::3];self.rgb=x['rgb'][::3];self.blue=blue_mask(self.rgb);self.last=None
    def render(self,p,cam):
        rot=R.from_euler('xyz',[p[4],p[5],p[3]]).as_matrix();pts=(self.xyz@rot.T+p[:3]-cam['t'])@cam['R'];uv=pts@cam['K'].T;xy=np.floor(uv[:,:2]/np.maximum(uv[:,2:],1e-8)).astype(int);h,w=cam['rgb'].shape[:2]
        ids=np.flatnonzero((pts[:,2]>.05)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h));pix=xy[ids,1]*w+xy[ids,0];order=np.lexsort((pts[ids,2],pix));ids=ids[order];pix=pix[order]
        if len(ids):unique=np.r_[True,pix[1:]!=pix[:-1]];ids=ids[unique];pix=pix[unique]
        im=np.zeros((h*w,3),np.uint8);depth=np.zeros(h*w);im[pix]=self.rgb[ids];depth[pix]=pts[ids,2];return im.reshape(h,w,3),depth.reshape(h,w)
    def fit(self,cameras):
        choices=[]
        for role,cam in cameras.items():
            mask=blue_mask(cam['rgb']);n,labels,stats,_=cv2.connectedComponentsWithStats(cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8)))
            for label in np.argsort(stats[1:,cv2.CC_STAT_AREA])[::-1][:5]+1:
                cloud,_,_=pointcloud(cam,mask&(labels==label))
                if len(cloud)<3:continue
                center=np.median(cloud,axis=0);cloud=cloud[np.linalg.norm(cloud-center,axis=1)<.14]
                if len(cloud)<3 or not .25<center[2]<.95 or np.linalg.norm(np.ptp(cloud,axis=0))>.20:continue
                truth=mask&(labels==label);cloud=cloud[::max(1,len(cloud)//120)];base=self.xyz[self.blue]
                def loss(p):
                    if np.linalg.norm(p[:3]-center)>.19:return 100.
                    image,depth=self.render(p,cam);visible=(depth>0)&np.isfinite(cam['depth'])&(cam['depth']>.05)&(cam['depth']>=depth-.025);pred=blue_mask(image)&visible;iou=(pred&truth).sum()/max(1,(pred|truth).sum())
                    if visible.sum()<.25*np.count_nonzero(depth):return 100.
                    if visible.sum()<12:return 100.
                    depth_error=np.minimum(np.abs(depth[visible]-cam['depth'][visible]),.12).mean();photo=np.abs(image.astype(float)-cam['rgb'].astype(float))[visible].mean()/255
                    return 1-iou+.6*photo+9*depth_error
                candidates=[]
                if self.last is not None and np.linalg.norm(self.last[:3]-center)<.2:candidates.append((loss(self.last),self.last.copy()))
                for angle in np.linspace(-np.pi,np.pi,48,endpoint=False):
                    rot=R.from_euler('z',angle).as_matrix();model=base@rot.T;pos=center-model.mean(0);tree=cKDTree(model)
                    for _ in range(4):
                        _,ids=tree.query(cloud-pos);pos=np.median(cloud-model[ids],axis=0)
                    p=np.r_[pos,angle,0.,0.];candidates.append((loss(p),p))
                best=sorted(candidates,key=lambda x:x[0])[:3]
                for step,angular in [(.008,.10),(.003,.04),(.0015,.015)]:
                    tests=best.copy()
                    for _,p in best:
                        for axis,scale in enumerate([step,step,step,angular,angular,angular]):
                            for sign in [-1,1]:
                                trial=p.copy();trial[axis]+=sign*scale;tests.append((loss(trial),trial))
                    best=sorted(tests,key=lambda x:x[0])[:3]
                score,p=best[0]
                if score<1.10:choices.append((score,p,role,len(cloud)))
        if not choices:return None
        score,p,role,count=min(choices,key=lambda x:x[0]);cam=cameras[role];im=cam['rgb'].astype(float);white=(im.min(-1)>95)&((im.max(-1)-im.min(-1))<65);cloud,_,_=pointcloud(cam,white)
        near=(np.linalg.norm(cloud[:,:2]-p[:2],axis=1)<.11)&(cloud[:,2]>p[2]-.045)&(cloud[:,2]<p[2]+.055);plane=cloud[near]
        if len(plane)>16:
            _,_,axes=np.linalg.svd(plane[:,:2]-plane[:,:2].mean(0),full_matrices=False);normal=axes[-1];old=R.from_euler('z',p[3]).as_matrix()[:2,1]
            if normal@old<0:normal=-normal
            angle=np.arctan2(-normal[0],normal[1]);difference=(angle-p[3]+np.pi)%(2*np.pi)-np.pi
            if abs(difference)<.6:p[3]=angle
        self.last=p.copy();return dict(position=p[:3],rotation=R.from_euler('xyz',[p[4],p[5],p[3]]).as_matrix(),score=float(score),view=role,matches=count,depth_error=None)
