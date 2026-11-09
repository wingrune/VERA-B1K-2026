"""Object-local feature priors + onboard RGB-D. No simulator imports or state access."""
from pathlib import Path
import cv2,numpy as np
from scipy.spatial.transform import Rotation

class PosterTracker:
    def __init__(self,assets):
        self.templates=[];self.sift=cv2.SIFT_create(nfeatures=2400,contrastThreshold=.014)
        for sign in [1,-1]:
            src=np.load(Path(assets)/f'poster_features_{sign:+d}.npz');image=src['image'];rows=[];descs=[]
            for scale in [1.,.5,.25]:
                im=cv2.resize(image,None,fx=scale,fy=scale);kp,des=self.sift.detectAndCompute(im,None)
                if des is None:continue
                uv=np.array([k.pt for k in kp])/scale
                xyz=np.c_[np.full(len(uv),.006814 if sign==1 else -.003186),sign*(uv[:,0]-310)/2000.,-(uv[:,1]-445)/2000.]
                rows.append(xyz);descs.append(des)
            self.templates.append((sign,np.concatenate(rows),np.concatenate(descs)))
    def fit(self,cameras):
        candidates=[]
        for role,cam in cameras.items():
            image=cv2.cvtColor(cam['rgb'],cv2.COLOR_RGB2GRAY);kp,des=self.sift.detectAndCompute(image,None)
            if des is None or len(kp)<8:continue
            for sign,xyz,reference in self.templates:
                matches=cv2.BFMatcher().knnMatch(reference,des,k=2)
                matches=[pair[0] for pair in matches if len(pair)==2 and pair[0].distance<.72*pair[1].distance]
                unique={}
                for m in matches:
                    if m.trainIdx not in unique or m.distance<unique[m.trainIdx].distance:unique[m.trainIdx]=m
                matches=list(unique.values())
                if len(matches)<7:continue
                object_points=np.array([xyz[m.queryIdx] for m in matches]);pixels=np.array([kp[m.trainIdx].pt for m in matches])
                ok,rvec,tvec,inliers=cv2.solvePnPRansac(object_points,pixels,cam['K'],None,iterationsCount=180,reprojectionError=3.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
                if not ok or inliers is None or len(inliers)<7:continue
                ids=inliers.ravel();op=object_points[ids];ip=pixels[ids]
                if np.ptp(op[:,1])<.08 or np.ptp(op[:,2])<.10:continue
                rvec,tvec=cv2.solvePnPRefineLM(op,ip,cam['K'],None,rvec,tvec)
                rot=cv2.Rodrigues(rvec)[0];pos=tvec.ravel()
                # Depth supports pose and rejects false image matches / reflected appearances.
                xy=np.rint(ip).astype(int);xy[:,0]=xy[:,0].clip(0,image.shape[1]-1);xy[:,1]=xy[:,1].clip(0,image.shape[0]-1)
                z=cam['depth'][xy[:,1],xy[:,0]];pred=op@rot.T+pos
                good=np.isfinite(z)&(z>.08)&(np.abs(z-pred[:,2])<.10)
                if good.sum()<6:continue
                errors=z[good]-pred[good,2];depth_error=float(np.median(np.abs(errors)))
                if depth_error>.055:continue
                # Correct the dominant axial PnP uncertainty with measured depth.
                pos[2]+=np.median(errors)
                br=cam['R']@rot;bp=cam['R']@pos+cam['t']
                if not .15<bp[2]<2.5:continue
                facing=sign*br[:,0]@(cam['t']-bp)/max(np.linalg.norm(cam['t']-bp),1e-6)
                if facing<.1:continue
                repro=cv2.projectPoints(op,rvec,tvec,cam['K'],None)[0].reshape(-1,2);rerr=float(np.median(np.linalg.norm(repro-ip,axis=1)))
                score=len(ids)/(1+20*depth_error+2*rerr)
                candidates.append((score,{'position':bp,'rotation':br,'visible_side':sign,'matches':len(ids),'depth_error':depth_error,'pixel_error':rerr,'view':role,'pixels':ip}))
        return max(candidates,key=lambda x:x[0])[1] if candidates else None

class FeatureObjectTracker:
    """PnP + depth validation with static multiview object keypoints."""
    def __init__(self,path):
        model=np.load(path);self.xyz=model['xyz'];self.des=model['descriptors'];self.sift=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.012)
    def fit(self,cameras):
        choices=[]
        for role,cam in cameras.items():
            kp,des=self.sift.detectAndCompute(cv2.cvtColor(cam['rgb'],cv2.COLOR_RGB2GRAY),None)
            if des is None:continue
            raw=cv2.BFMatcher().knnMatch(self.des,des,k=2);good=[p[0] for p in raw if len(p)==2 and p[0].distance<.74*p[1].distance];unique={}
            for m in good:
                if m.trainIdx not in unique or m.distance<unique[m.trainIdx].distance:unique[m.trainIdx]=m
            good=list(unique.values())
            if len(good)<7:continue
            xyz=np.array([self.xyz[m.queryIdx] for m in good]);uv=np.array([kp[m.trainIdx].pt for m in good]);ok,rv,tv,ids=cv2.solvePnPRansac(xyz,uv,cam['K'],None,iterationsCount=220,reprojectionError=3,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
            if not ok or ids is None or len(ids)<7:continue
            ids=ids.ravel();xyz=xyz[ids];uv=uv[ids]
            if np.linalg.norm(np.ptp(xyz,axis=0))<.06:continue
            rv,tv=cv2.solvePnPRefineLM(xyz,uv,cam['K'],None,rv,tv);rot=cv2.Rodrigues(rv)[0];pos=tv.ravel();pixels=np.rint(uv).astype(int);pixels[:,0]=pixels[:,0].clip(0,cam['rgb'].shape[1]-1);pixels[:,1]=pixels[:,1].clip(0,cam['rgb'].shape[0]-1);z=cam['depth'][pixels[:,1],pixels[:,0]];pred=xyz@rot.T+pos;valid=np.isfinite(z)&(z>.08)&(np.abs(z-pred[:,2])<.08)
            if valid.sum()<6:continue
            err=float(np.median(np.abs(z[valid]-pred[valid,2])))
            if err>.035:continue
            pos[2]+=np.median(z[valid]-pred[valid,2]);bp=cam['R']@pos+cam['t'];br=cam['R']@rot
            if not .15<bp[2]<1.7 or br[2,2]<.5:continue
            choices.append((len(ids)/(1+30*err),dict(position=bp,rotation=br,matches=len(ids),depth_error=err,view=role)))
        return max(choices,key=lambda x:x[0])[1] if choices else None

class TelevisionDetector:
    """RGB rectangle + metric depth plane, including observed support-surface candidates."""
    def fit(self,cam):
        image=cam['rgb'];gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);edges=cv2.Canny(gray,50,130);contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);choices=[]
        for c in contours:
            if cv2.contourArea(c)<900:continue
            poly=cv2.approxPolyDP(c,.02*cv2.arcLength(c,True),True)
            if len(poly)!=4 or not cv2.isContourConvex(poly):continue
            mask=np.zeros(gray.shape,np.uint8);cv2.fillConvexPoly(mask,poly,1);inner=cv2.erode(mask,np.ones((7,7),np.uint8)).astype(bool)
            if inner.sum()<700 or np.median(gray[inner])<130:continue
            from perception import pointcloud
            pts,_,_=pointcloud(cam,inner,stride=8)
            if len(pts)<60:continue
            center=np.median(pts,axis=0);_,_,axes=np.linalg.svd(pts-center,full_matrices=False);normal=axes[-1]
            if abs(normal[2])>.3 or np.median(np.abs((pts-center)@normal))>.012 or not .5<center[2]<1.7:continue
            if normal@(cam['t']-center)<0:normal=-normal
            up=np.array([0.,0.,1.]);right=np.cross(up,normal);right/=np.linalg.norm(right);up=np.cross(normal,right);coords=(pts-center)@np.column_stack([right,up]);width,height=np.ptp(coords,axis=0)
            if not .40<width<.95 or not .24<height<.65 or not 1.1<width/height<2.2:continue
            choices.append((mask.sum(),dict(position=center,rotation=np.column_stack([normal,right,up]),width=float(width),height=float(height))))
        return max(choices,key=lambda x:x[0])[1] if choices else None
