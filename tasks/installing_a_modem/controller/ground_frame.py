"""Estimate gravity and ground clearance from the visible RGB-D floor, not simulator pose."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from perception import pointcloud
class GroundFrameEstimator:
    def __init__(self):self.transform=np.eye(4);self.stats={'valid':False}
    def update(self,camera):
        points,_,_=pointcloud(camera,stride=5);radius=np.linalg.norm(points[:,:2],axis=1);points=points[(radius>.5)&(radius<4)&(points[:,2]>-.9)&(points[:,2]<.55)]
        points=points[::max(1,len(points)//1800)]
        if len(points)<60:return self.transform.copy()
        rng=np.random.default_rng(905);best=None
        for _ in range(240):
            a,b,c=points[rng.choice(len(points),3,replace=False)];normal=np.cross(b-a,c-a);norm=np.linalg.norm(normal)
            if norm<.02:continue
            normal/=norm
            if normal[2]<0:normal=-normal
            if normal[2]<.80:continue
            offset=-normal@a;intercept=-offset/normal[2]
            if not -.45<intercept<.15:continue
            mask=np.abs(points@normal+offset)<.012;count=mask.sum()
            if count<100:continue
            spread=np.linalg.norm(np.ptp(points[mask,:2],axis=0))
            if spread<.9:continue
            score=count-200*max(intercept,0)
            if best is None or score>best[0]:best=(score,mask)
        if best is None:return self.transform.copy()
        selected=points[best[1]];center=selected.mean(0);_,_,axes=np.linalg.svd(selected-center,full_matrices=False);normal=axes[-1]
        if normal[2]<0:normal=-normal
        residual=float(np.median(np.abs((selected-center)@normal)))
        if residual>.003:return self.transform.copy()
        cross=np.cross(normal,[0.,0.,1.]);sine=np.linalg.norm(cross);rot=np.eye(3) if sine<1e-8 else R.from_rotvec(cross/sine*np.arctan2(sine,normal[2])).as_matrix();height=-np.median(selected@rot[2])
        transform=np.eye(4);transform[:3,:3]=rot;transform[2,3]=height;self.transform=transform;self.stats={'valid':True,'normal_body':normal.tolist(),'height_m':float(height),'inliers':len(selected),'residual_m':float(np.median(np.abs((selected-center)@normal)))};return transform.copy()

def level_cameras(cameras,transform):
    result={}
    for name,camera in cameras.items():
        out=camera.copy();out['R']=transform[:3,:3]@camera['R'];out['t']=transform[:3,:3]@camera['t']+transform[:3,3];result[name]=out
    return result
