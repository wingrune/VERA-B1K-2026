"""Find a keyboard above its observed support, rejecting flat desk mats."""
import cv2,numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from perception import pointcloud


def fit_keyboard(model,cameras,level):
    lo,hi=model['bounds'];extent=hi-lo;xyz=model['xyz'][::5];tree=cKDTree(xyz);choices=[]
    for role,c in cameras.items():
        points,rgb,uv=pointcloud(c);p=points@level[:3,:3].T+level[:3,3]
        valid=(p[:,2]>.45)&(p[:,2]<1.15)&(np.linalg.norm(p[:,:2],axis=1)>.45)
        if valid.sum()<100:continue
        height=np.full(c['depth'].shape,np.nan);height[uv[:,1],uv[:,0]]=p[:,2]
        # Horizontal support surfaces form peaks in measured height.
        smooth=np.isfinite(height)&(abs(height-np.roll(height,2,axis=0))<.009)&(abs(height-np.roll(height,2,axis=1))<.009)
        flat=valid&smooth[uv[:,1],uv[:,0]]
        bins=np.rint(p[flat,2]/.005).astype(int);values,counts=np.unique(bins,return_counts=True)
        peaks=[]
        for i in np.argsort(counts)[::-1]:
            z=values[i]*.005
            if counts[i]<100:break
            if any(abs(z-old)<.04 for old in peaks):continue
            peaks.append(z)
            if len(peaks)==3:break
        hsv=cv2.cvtColor(c['rgb'],cv2.COLOR_RGB2HSV);dark=(hsv[...,1]<95)&(hsv[...,2]<125)
        for support in peaks:
            mask=(dark&np.isfinite(height)&(height>support+.012)&(height<support+.075)).astype(np.uint8)
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
            n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
            for label in range(1,n):
                if stats[label,cv2.CC_STAT_AREA]<35:continue
                keep=(labels[uv[:,1],uv[:,0]]==label)&(p[:,2]>support+.010)&(p[:,2]<support+.075)
                sample=p[keep]
                if len(sample)<35:continue
                rect=cv2.minAreaRect(sample[:,:2].astype(np.float32));span=np.sort(rect[1]);expected=np.sort(extent[:2])
                if np.any(span<expected*.72) or np.any(span>expected*1.20) or np.ptp(sample[:,2])>.045:continue
                sample=sample[::max(1,len(sample)//500)];candidates=[]
                for yaw in np.linspace(0,2*np.pi,12,endpoint=False):
                    rot=R.from_euler('z',yaw).as_matrix();position=np.r_[rect[0],np.median(sample[:,2])-hi[2]]-rot@np.r_[(lo[:2]+hi[:2])/2,0.]
                    def residual(x):
                        rot=R.from_euler('z',x[3]).as_matrix();dist,_=tree.query((sample-x[:3])@rot)
                        return np.minimum(dist,.035)
                    initial=np.r_[position,yaw]
                    fit=least_squares(residual,initial,bounds=(initial-[.04,.04,.012,.6],initial+[.04,.04,.012,.6]),max_nfev=25,diff_step=.001)
                    rot=R.from_euler('z',fit.x[3]).as_matrix();local=(sample-fit.x[:3])@rot
                    coverage=float(np.mean(cKDTree(local).query(xyz)[0]<.023));error=float(np.quantile(residual(fit.x),.75))
                    if coverage>.55 and error<.012:candidates.append((error-.01*coverage,fit.x,coverage,error))
                if not candidates:continue
                score,x,coverage,error=min(candidates,key=lambda x:x[0]);pose=np.eye(4);pose[:3,:3]=R.from_euler('z',x[3]).as_matrix();pose[:3,3]=x[:3]
                choices.append((score,np.linalg.inv(level)@pose,dict(method='raised_keyboard_geometry',camera=role,support_height=float(support),coverage=coverage,residual=error,points=len(sample),orientation_ambiguous=True)))
    if choices:
        _,pose,stats=min(choices,key=lambda x:x[0]);return pose,stats
    return None
