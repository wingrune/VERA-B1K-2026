"""Floor-supported bin geometry from RGB-D, checked against canonical priors."""
import cv2,numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from perception import pointcloud


def depth_coverage(model,pose,camera):
    world=model['xyz']@pose[:3,:3].T+pose[:3,3]
    local=(world-camera['t'])@camera['R']
    valid=(local[:,2]>.1)&(local[:,2]<4.5)
    local=local[valid]
    uv=(local[:,:2]/local[:,2,None])*np.diag(camera['K'])[:2]+camera['K'][:2,2]-.5
    uv=np.rint(uv).astype(int);height,width=camera['depth'].shape
    valid=(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
    uv,local=uv[valid],local[valid]
    expected=np.full(height*width,np.inf)
    np.minimum.at(expected,uv[:,1]*width+uv[:,0],local[:,2])
    modeled=np.isfinite(expected);observed=camera['depth'].ravel()
    finite=modeled&np.isfinite(observed)&(observed>.1)&(observed<4.5)
    visible=finite&(observed>=expected-.05)
    matched=visible&(abs(observed-expected)<.035)
    return dict(visible_pixels=int(visible.sum()),modeled_pixels=int(modeled.sum()),
                depth_coverage=float(matched.sum()/max(visible.sum(),1)))

def fit_bin(model,cameras,level,predicted=None):
    lo,hi=model['bounds'];extent=hi-lo;choices=[];diagnostics=[]
    # Fit the lower body, which is unchanged when the lid opens. A closed
    # upper surface is neither an identity check nor a placement surface.
    body_height=.8*extent[2]
    body_xyz=model['xyz'][model['xyz'][:,2]<lo[2]+body_height]
    tree=cKDTree(body_xyz[::2]);body_model=dict(xyz=body_xyz)
    for name,camera in cameras.items():
        hsv=cv2.cvtColor(camera['rgb'],cv2.COLOR_RGB2HSV)
        mask=(hsv[...,2]<100)&(hsv[...,1]<110)
        points,_,uv=pointcloud(camera,mask)
        leveled=points@level[:3,:3].T+level[:3,3]
        valid=(leveled[:,2]>.025)&(leveled[:,2]<extent[2]+.08)&(np.linalg.norm(points[:,:2],axis=1)>.48)
        clean=np.zeros(mask.shape,np.uint8);clean[uv[valid,1],uv[valid,0]]=1
        clean=cv2.morphologyEx(clean,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        # Split touching image silhouettes at measured depth discontinuities.
        depth=camera['depth'];edges=np.zeros(depth.shape,bool)
        dx=np.abs(depth[:,1:]-depth[:,:-1])>.07;dy=np.abs(depth[1:]-depth[:-1])>.07
        edges[:,1:] |= dx;edges[:,:-1] |= dx;edges[1:] |= dy;edges[:-1] |= dy
        clean[edges]=0
        count,labels,stats,_=cv2.connectedComponentsWithStats(clean)
        for i in range(1,count):
            if stats[i,cv2.CC_STAT_AREA]<60:continue
            pts,_,_=pointcloud(camera,labels==i);p=pts@level[:3,:3].T+level[:3,3]
            p=p[(p[:,2]>.025)&(p[:,2]<extent[2]+.08)]
            if len(p)<60:continue
            row=dict(camera=name,points=len(p),span=np.ptp(p,axis=0).tolist(),minimum_height=float(p[:,2].min()),reason='floor_and_height')
            diagnostics.append(row)
            partial=p[:,2].min()>.12 or np.ptp(p[:,2])<extent[2]*.8
            if partial and (np.ptp(p[:,2])<.08 or p[:,2].max()<extent[2]-.14):continue
            row['partial_visibility']=bool(partial)
            fitting=p[p[:,2]<body_height]
            if len(fitting)<45:row['reason']='body_points';continue
            rectangle=cv2.minAreaRect(fitting[:,:2].astype(np.float32))
            span=np.sort(rectangle[1]);expected=np.sort(extent[:2]);row.update(observed_xy_span=span.tolist(),reason='width')
            if np.any(span<expected*(.25 if partial else .35)) or np.any(span>expected*1.4):continue
            fitting=fitting[::max(1,len(fitting)//450)];fits=[]
            for yaw in np.linspace(0,2*np.pi,16,endpoint=False):
                rot=R.from_euler('z',yaw).as_matrix();translation=np.r_[rectangle[0],-lo[2]]-rot@np.r_[model['bounds'].mean(0)[:2],0.]
                def residual(x):
                    rot=R.from_euler('z',x[2]).as_matrix();t=np.r_[x[:2],-lo[2]]
                    return np.minimum(tree.query((fitting-t)@rot)[0],.07)
                x0=np.r_[translation[:2],yaw];fit=least_squares(residual,x0,bounds=(x0-[.10,.10,.5],x0+[.10,.10,.5]),max_nfev=40,diff_step=.001)
                error=float(np.quantile(residual(fit.x),.8));fits.append((error,fit.x))
            error,x=min(fits,key=lambda a:a[0]);row.update(residual=error,reason='residual')
            if error>.014:continue
            pose=np.eye(4);pose[:3,:3]=R.from_euler('z',x[2]).as_matrix();pose[:3,3]=np.r_[x[:2],-lo[2]];pose=np.linalg.inv(level)@pose
            if predicted is not None and np.linalg.norm(pose[:3,3]-predicted[:3,3])>.30:
                row['reason']='confirmed_bin_association';continue
            coverage=depth_coverage(body_model,pose,camera);row.update(coverage);row['reason']='coverage'
            if coverage['visible_pixels']<(90 if partial else 60) or coverage['depth_coverage']<(.75 if partial else .65):continue
            row['reason']='accepted';choices.append((error,pose,dict(method='floor_supported_bin_body',camera=name,residual=error,points=len(fitting),orientation_ambiguous=True,**coverage)))
    if choices:
        _,pose,stats=min(choices,key=lambda x:x[0]);return (pose,stats),diagnostics
    return None,diagnostics
