"""Color and canonical-geometry fallbacks using current RGB-D only."""
import time
import cv2,numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from perception import pointcloud

def groups(points,radius=.025):
    if len(points)<8:return []
    _,ids=np.unique(np.floor(points/.006).astype(int),axis=0,return_index=True)
    points=points[ids];tree=cKDTree(points);pairs=tree.query_pairs(radius,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(points),len(points)))
    count,labels=connected_components(graph,directed=False)
    return [points[labels==i] for i in range(count) if (labels==i).sum()>=8]

def masks(rgb):
    hsv=cv2.cvtColor(rgb.astype(np.uint8),cv2.COLOR_RGB2HSV)
    red=((hsv[...,0]<12)|(hsv[...,0]>170))&(hsv[...,1]>80)&(hsv[...,2]>30)
    green=(hsv[...,0]>30)&(hsv[...,0]<95)&(hsv[...,1]>55)&(hsv[...,2]>12)
    return red,green

def rose_candidates(model,cameras,level):
    red,green=masks(model['rgb'][None]);mr=model['xyz'][red[0]];mg=model['xyz'][green[0]]
    if len(mr)<20 or len(mg)<20:return []
    mr=mr[::max(1,len(mr)//350)];mg=mg[::max(1,len(mg)//500)]
    rt=cKDTree(mr);gt=cKDTree(mg);choices=[]
    local_flower=mr.mean(0)
    for role,c in cameras.items():
        rm,gm=masks(c['rgb']);rp,_,_=pointcloud(c,rm);gp,_,_=pointcloud(c,gm)
        rp=rp@level[:3,:3].T+level[:3,3];gp=gp@level[:3,:3].T+level[:3,3]
        rp=rp[(rp[:,2]>.35)&(rp[:,2]<1.6)&(np.linalg.norm(rp[:,:2],axis=1)>.35)]
        gp=gp[(gp[:,2]>.35)&(gp[:,2]<1.6)]
        for cluster in groups(rp,.025):
            if len(cluster)<10 or np.linalg.norm(np.ptp(cluster,axis=0))>.14:continue
            flower=cluster.mean(0);leaves=gp[np.linalg.norm(gp-flower,axis=1)<.32]
            if len(leaves)<15:continue
            # Multiple nearby stems can be clutter. Template-to-observation color
            # distances find the supported branch while robust penalties cap clutter.
            obs_red=cKDTree(cluster);obs_green=cKDTree(leaves)
            candidates=[]
            for yaw in np.linspace(0,2*np.pi,24,endpoint=False):
                rot=R.from_euler('z',yaw).as_matrix();origin=flower-rot@local_flower
                def residual(x):
                    rot=R.from_euler('z',x[3]).as_matrix()
                    redp=mr@rot.T+x[:3];greenp=mg@rot.T+x[:3]
                    dr=obs_red.query(redp)[0];dg=obs_green.query(greenp)[0]
                    return np.r_[np.minimum(dr,.04),np.minimum(dg,.04),.15*(x[:3]-origin)]
                x0=np.r_[origin,yaw];bounds=(x0-[.055,.055,.055,.4],x0+[.055,.055,.055,.4])
                fit=least_squares(residual,x0,bounds=bounds,max_nfev=35,diff_step=.001)
                rot=R.from_euler('z',fit.x[3]).as_matrix();dr=obs_red.query(mr@rot.T+fit.x[:3])[0];dg=obs_green.query(mg@rot.T+fit.x[:3])[0]
                red_coverage=float((dr<.012).mean());green_coverage=float((dg<.018).mean());error=float(np.quantile(np.r_[dr,dg],.65))
                if red_coverage<.40 or green_coverage<.35 or error>.018:continue
                t=np.eye(4);t[:3,:3]=rot;t[:3,3]=fit.x[:3];t=np.linalg.inv(level)@t
                score=error+.012*(1-green_coverage)+.008*(1-red_coverage)
                candidates.append((score,t,dict(method='rose_color_canonical_icp',camera=role,residual=error,red_coverage=red_coverage,green_coverage=green_coverage,red_points=len(cluster),green_points=len(leaves))))
            if candidates:choices.append(min(candidates,key=lambda x:x[0]))
    unique=[]
    for score,pose,stats in sorted(choices,key=lambda x:x[0]):
        if not any(np.linalg.norm(pose[:3,3]-p[:3,3])<.065 for p,s in unique):unique.append((pose,stats))
    return unique

def vase_fit(model,cameras,level):
    xyz=model['xyz'][::max(1,len(model['xyz'])//2500)];tree=cKDTree(xyz);lo,hi=model['bounds'];extent=hi-lo;choices=[]
    for role,c in cameras.items():
        hsv=cv2.cvtColor(c['rgb'],cv2.COLOR_RGB2HSV);mask=(hsv[...,1]<55)&(hsv[...,2]>100)
        p,_,_=pointcloud(c,mask,stride=2);p=p@level[:3,:3].T+level[:3,3]
        p=p[(p[:,2]>.35)&(p[:,2]<1.6)&(np.linalg.norm(p[:,:2],axis=1)>.35)]
        for cloud in groups(p,.018):
            span=np.ptp(cloud,axis=0)
            if not .50*extent[2]<span[2]<1.20*extent[2] or max(span[:2])>max(extent[:2])*1.20 or max(span[:2])<min(extent[:2])*.75:continue
            cloud=cloud[::max(1,len(cloud)//700)];origin=(cloud.min(0)+cloud.max(0))/2-(lo+hi)/2
            for yaw in (0.,np.pi/2):
                def residual(x):
                    rot=R.from_euler('z',x[3]).as_matrix();local=(cloud-x[:3])@rot
                    return np.minimum(tree.query(local)[0],.06)
                x0=np.r_[origin,yaw];fit=least_squares(residual,x0,bounds=(x0-[.12,.12,.055,.8],x0+[.12,.12,.055,.8]),max_nfev=70,diff_step=.001)
                error=float(np.quantile(residual(fit.x),.75))
                if error>.014:continue
                t=np.eye(4);t[:3,:3]=R.from_euler('z',fit.x[3]).as_matrix();t[:3,3]=fit.x[:3];t=np.linalg.inv(level)@t
                support=visible_model_support(model,cameras,t)
                if support['matched_fraction']<.65:continue
                choices.append((error,t,dict(method='vase_white_canonical_icp',camera=role,residual=error,points=len(cloud),visible_support=support)))
    if choices:
        _,t,s=min(choices,key=lambda x:x[0]);return t,s
    return None

def table_fit(model,cameras,level):
    """Recognize a low rectangular support from depth and canonical dimensions."""
    lo,hi=model['bounds'];expected=np.sort((hi-lo)[:2]);choices=[]
    for role,c in cameras.items():
        points,_,_=pointcloud(c,stride=3);p=points@level[:3,:3].T+level[:3,3]
        p=p[(p[:,2]>.22)&(p[:,2]<.65)&(np.linalg.norm(p[:,:2],axis=1)>.45)]
        if len(p)<80:continue
        bins=np.floor(p[:,2]/.008).astype(int);values,counts=np.unique(bins,return_counts=True)
        used=[]
        for height_bin in values[np.argsort(counts)[::-1]][:12]:
            height=(height_bin+.5)*.008
            if any(abs(height-old)<.018 for old in used):continue
            used.append(height);surface=p[abs(p[:,2]-height)<.008]
            for cloud in groups(surface,.035):
                if len(cloud)<80:continue
                rect=cv2.minAreaRect(cloud[:,:2].astype(np.float32));span=np.sort(rect[1])
                if np.any(span<expected*[.60,.60]) or np.any(span>expected*1.12):continue
                # Require broad planar support and matching rectangle orientation;
                # small sofa patches cannot pass the canonical dimensions.
                angle=np.deg2rad(rect[2]);rot=R.from_euler('z',angle).as_matrix()
                if (rect[1][0]>rect[1][1]) != ((hi-lo)[0]>(hi-lo)[1]):rot=rot@R.from_euler('z',np.pi/2).as_matrix()
                local=cloud@rot
                # Partial views cannot certify an accurate center. Only promote
                # a support when both observed dimensions cover>=85% of template.
                coverage=span/expected
                if np.min(coverage)<.85:continue
                center=np.r_[rect[0],float(np.median(cloud[:,2]))]
                origin=center-rot@np.r_[(lo[:2]+hi[:2])/2,hi[2]]
                t=np.eye(4);t[:3,:3]=rot;t[:3,3]=origin;t=np.linalg.inv(level)@t
                residual=float(np.median(abs(cloud[:,2]-center[2])))
                choices.append((residual,t,dict(method='table_depth_rectangle',camera=role,residual=residual,points=len(cloud),dimension_coverage=coverage.tolist())))
    if choices:
        _,t,s=min(choices,key=lambda x:x[0]);return t,s
    return None

class FitTimeLimit(Exception):
    pass

def _prepare_guided(model,cameras):
    red,green=masks(model['rgb'][None])
    mr=np.array(model['xyz'][red[0]][::3],dtype=np.float64,copy=True)
    mg=np.array(model['xyz'][green[0]][::3],dtype=np.float64,copy=True)
    clouds={'red':[],'green':[]}
    for c in cameras.values():
        rm,gm=masks(c['rgb'])
        for mask,key in ((rm,'red'),(gm,'green')):
            points=pointcloud(c,mask)[0]
            clouds[key].append(np.array(points,dtype=np.float64,copy=True))
    return mr,mg,clouds

def _guided_clouds(prepared,predicted,model,translation_limit):
    mr,mg,clouds=prepared;lo,hi=model['bounds'];origin=predicted[:3,3];rotation=predicted[:3,:3]
    groups={}
    for color in ('red','green'):
        parts=[]
        for points in clouds[color]:
            local=(points-origin)@rotation
            keep=((local>=lo-translation_limit-.015)&(local<=hi+translation_limit+.015)).all(axis=1)
            points=points[keep]
            if len(points)>10:parts.append(points[::max(1,int(np.ceil(len(points)/400)))])
        groups[color]=np.concatenate(parts) if parts else np.empty((0,3))
    return mr,mg,groups['red'],groups['green']

def _guided_rose_fit_single(model,cameras,predicted,translation_limit=.075,red_min=.4,green_min=.35,visible_only=False,prepared=None,deadline=None,initial_delta=None,rotation_limit=.30):
    prepared=_prepare_guided(model,cameras) if prepared is None else prepared
    mr,mg,red_observed,green_observed=_guided_clouds(prepared,predicted,model,translation_limit)
    if min(len(mr),len(mg))<20 or min(len(red_observed),len(green_observed))<10:return None
    origin=predicted[:3,3];rotation=predicted[:3,:3]
    rt=cKDTree(red_observed,copy_data=True);gt=cKDTree(green_observed,copy_data=True)
    rmodel=cKDTree(mr,copy_data=True);gmodel=cKDTree(mg,copy_data=True)
    def components(x):
        if deadline is not None and time.monotonic()>deadline:raise FitTimeLimit()
        rot=R.from_rotvec(x[3:]).as_matrix()@rotation
        if visible_only:
            return rmodel.query((red_observed-origin-x[:3])@rot)[0],gmodel.query((green_observed-origin-x[:3])@rot)[0]
        return rt.query(mr@rot.T+origin+x[:3])[0],gt.query(mg@rot.T+origin+x[:3])[0]
    def residual(x):
        dr,dg=components(x)
        return np.r_[np.minimum(dr,.12 if visible_only else .04),np.minimum(dg,.12 if visible_only else .04),.10*x[:3],.015*x[3:]]
    initial=np.zeros(6)
    if initial_delta is not None:initial[:3]=initial_delta
    fit=least_squares(residual,initial,bounds=([-translation_limit]*3+[-rotation_limit]*3,[translation_limit]*3+[rotation_limit]*3),max_nfev=55 if visible_only else 50,diff_step=.001,loss='soft_l1' if visible_only else 'linear',f_scale=.012)
    dr,dg=components(fit.x);rc=float((dr<.014).mean());gc=float((dg<.018).mean());error=float(np.quantile(np.r_[dr,dg],.65))
    if rc<red_min or gc<green_min or error>.018 or np.max(abs(fit.x[:3]))>translation_limit-.003 or np.max(abs(fit.x[3:]))>rotation_limit-.01:return None
    t=predicted.copy();t[:3,:3]=R.from_rotvec(fit.x[3:]).as_matrix()@rotation;t[:3,3]+=fit.x[:3]
    return t,dict(method='kinematically_gated_rose_color_icp',correspondence_direction='observed_to_model' if visible_only else 'model_to_observed',observed_red_points=len(red_observed),observed_green_points=len(green_observed),residual=error,red_coverage=rc,green_coverage=gc,translation_correction=fit.x[:3].tolist(),rotation_correction=fit.x[3:].tolist())

def guided_rose_fit(model,cameras,predicted,translation_limit=.075,red_min=.4,green_min=.35,multistart=False,visible_only=False):
    started=time.monotonic();deadline=started+3.0;prepared=_prepare_guided(model,cameras)
    seeds=[np.zeros(3)]
    if multistart:
        for axis in np.eye(3):seeds.extend([axis*.45,-axis*.45])
        for first in range(3):
            for second in range(first+1,3):
                for a in (-1,1):
                    for b in (-1,1):
                        seed=np.zeros(3);seed[first]=a*.35;seed[second]=b*.35;seeds.append(seed)
    ranked=[]
    if multistart:
        rtree=cKDTree(prepared[0],copy_data=True);gtree=cKDTree(prepared[1],copy_data=True)
        for vec in seeds:
            pose=predicted.copy();pose[:3,:3]=R.from_rotvec(vec).as_matrix()@pose[:3,:3]
            mr,mg,red,green=_guided_clouds(prepared,pose,model,translation_limit)
            if min(len(red),len(green))<10:continue
            delta=np.clip(np.median(red,axis=0)-pose[:3,:3]@np.median(mr,axis=0)-pose[:3,3],-translation_limit+.004,translation_limit-.004)
            dr=rtree.query((red-pose[:3,3]-delta)@pose[:3,:3])[0]
            dg=gtree.query((green-pose[:3,3]-delta)@pose[:3,:3])[0]
            ranked.append((float(np.quantile(np.r_[dr,dg],.65)),vec,delta))
        selected=[(vec,delta) for _,vec,delta in sorted(ranked,key=lambda x:x[0])[:5]]
        if not visible_only:selected=[(vec,None) for vec in seeds[:7]]
    else:selected=[(seeds[0],None)]
    results=[];attempts=0
    for vec,delta in selected:
        if time.monotonic()>deadline:break
        pose=predicted.copy();pose[:3,:3]=R.from_rotvec(vec).as_matrix()@pose[:3,:3]
        attempts+=1
        try:fit=_guided_rose_fit_single(model,cameras,pose,translation_limit,red_min,green_min,visible_only,prepared,deadline,delta)
        except FitTimeLimit:break
        if fit is None:continue
        pose,stats=fit;angle=float(R.from_matrix(predicted[:3,:3].T@pose[:3,:3]).magnitude())
        if angle>(.85 if multistart else .52):continue
        stats=dict(stats,rotation_seed=vec.tolist(),total_rotation_correction_rad=angle)
        score=stats['residual']-.005*(stats['red_coverage']+stats['green_coverage'])
        results.append((score,pose,stats))
    if not results:return None
    _,pose,stats=min(results,key=lambda x:x[0]);stats.update(fit_seconds=time.monotonic()-started,optimized_candidates=attempts,ranked_candidates=len(ranked));return pose,stats

def visible_model_support(model,cameras,pose):
    """Check predicted visible depth; prevent partial white patches posing as a vase."""
    xyz=model['xyz'][::3];best={'matched_fraction':0.,'unoccluded_fraction':0.,'pixels':0}
    for role,c in cameras.items():
        points=xyz@pose[:3,:3].T+pose[:3,3];points=(points-c['t'])@c['R'];good=points[:,2]>.06;points=points[good]
        if not len(points):continue
        uv=points@c['K'].T;uv=uv[:,:2]/uv[:,2,None];uv=np.rint(uv-.5).astype(int);height,width=c['depth'].shape
        good=(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height);uv=uv[good];z=points[good,2]
        pixels=uv[:,1]*width+uv[:,0];front=np.full(height*width,np.inf);np.minimum.at(front,pixels,z);ids=np.where(np.isfinite(front))[0]
        if len(ids)<60:continue
        observed=c['depth'].ravel()[ids];expected=front[ids];finite=np.isfinite(observed)&(observed>.06)
        visible=finite&(observed>=expected-.025);matches=visible&(abs(observed-expected)<.025)
        unoccluded=float(visible.mean());fraction=float(matches.sum()/max(1,visible.sum()))
        if fraction>best['matched_fraction'] and unoccluded>.50:best={'camera':role,'matched_fraction':fraction,'unoccluded_fraction':unoccluded,'pixels':len(ids)}
    return best
