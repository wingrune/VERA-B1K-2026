"""Observed desk support enclosed by two cubicle-like raised panels."""
import numpy as np
from scipy.spatial import cKDTree
from perception import pointcloud
from ground_frame import level_cameras


def carrel_support(cameras,level):
    clouds=[];colors=[]
    for camera in level_cameras(cameras,level).values():
        p,rgb,_=pointcloud(camera,stride=3)
        keep=(p[:,2]>.50)&(p[:,2]<1.4)&(np.linalg.norm(p[:,:2],axis=1)<3.)
        clouds.append(p[keep]);colors.append(rgb[keep])
    if not clouds:return None
    points=np.concatenate(clouds);rgb=np.concatenate(colors).astype(float)
    if len(points)<200:return None
    neutral=(rgb.max(1)-rgb.min(1)<75)&(rgb.mean(1)>105)
    bins=np.rint(points[:,2]/.01);values,counts=np.unique(bins[(points[:,2]>.6)&(points[:,2]<1.)],return_counts=True)
    candidates=[]
    for height in values[np.argsort(counts)[-5:]][::-1]*.01:
        plane=points[(abs(points[:,2]-height)<.008)&neutral]
        if len(plane)<120:continue
        for center in plane[::max(1,len(plane)//35)]:
            patch=plane[(abs(plane[:,0]-center[0])<.17)&(abs(plane[:,1]-center[1])<.22)]
            if len(patch)<60:continue
            xy=patch[:,:2]-center[:2]
            if not all(np.sum((xy[:,0]*a>0)&(xy[:,1]*b>0))>=10 for a in [-1,1] for b in [-1,1]):continue
            above=points[(np.linalg.norm(points[:,:2]-center[:2],axis=1)<.19)&(points[:,2]>height+.035)]
            if len(above)>8:continue
            boundary=points[neutral&(points[:,2]>height+.12)&(points[:,2]<height+.55)&(np.linalg.norm(points[:,:2]-center[:2],axis=1)<1.)]
            if len(boundary)<70:continue
            xy=boundary[:,:2];rng=np.random.default_rng(37);lines=[]
            for _ in range(100):
                a,b=xy[rng.choice(len(xy),2,replace=False)];axis=b-a;length=np.linalg.norm(axis)
                if length<.3:continue
                axis/=length;normal=np.array([-axis[1],axis[0]]);distance=abs((xy-a)@normal)
                inside=distance<.025
                if inside.sum()<30 or np.ptp((xy[inside]-a)@axis)<.5:continue
                offset=abs((center[:2]-a)@normal)
                if not .20<offset<.85:continue
                if any(abs(normal@other[0])>.94 for other in lines):continue
                lines.append((normal,float(offset),int(inside.sum())))
            if not any(abs(a[0]@b[0])<.45 for i,a in enumerate(lines) for b in lines[i+1:]):continue
            candidates.append((np.linalg.norm(center[:2]),center.copy(),dict(method='cubicle_panel_geometry',height=float(height),panel_lines=len(lines),support_points=len(patch))))
    if candidates:
        _,point,stats=min(candidates,key=lambda x:x[0]);return point,stats
    return None
