"""Task23 RGB-D registration against canonical books and box, no simulator inputs."""
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from art_perception import Observer as SchoolObserver,quaternion_matrix,CAMERAS
from art_skills import transform,upright
from surface_icp import SurfaceObjective
from manip import pose2
from localizer import true_velocity,command_velocity,turn_rate
from kinematics import from_proprio

class Observer(SchoolObserver):
    def __init__(self,root,scene):
        super().__init__(root,scene)
        for category in self.models:
            if category.startswith('book') and self.signatures[category]['colour_fraction']<.20:
                xyz=self.models[category]['xyz'][::2];self.colours[category]=(None,xyz);self.trees[category]=cKDTree(xyz)

    def image_cloud(self,camera,base):
        link,slot,focal=CAMERAS[camera];prefix=f'robot_r1::robot_r1:{link}:Camera:0::'
        depth=self.data[prefix+'depth_linear'];rgb=self.data[prefix+'rgb'][...,:3].astype(np.uint8)
        h,w=depth.shape;v,u=np.mgrid[:h,:w];pc=np.stack([(u+.5-w/2)/focal/w*depth,(v+.5-h/2)/focal/w*depth,depth],-1)
        rel=self.data['robot_r1::cam_rel_poses'].reshape(3,7)[slot]
        body=pc@(quaternion_matrix(rel[3:])@np.diag([1.,-1.,-1.])).T+rel[:3]
        return body@base[:3,:3].T+base[:3,3],body,cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV),np.isfinite(depth)&(depth>.08)&(depth<3.8)

    def refine_cabinet_base(self):
        # Bounded RGB-D registration of fixed cabinet surfaces; no movable poses or simulator state.
        if self.loc.pose is None:return False
        from pathlib import Path
        if not hasattr(self,'cabinet_tree'):
            self.cabinet_tree=cKDTree(np.load(Path(__file__).parent/'priors/bookcase_surfaces.npz')['xyz'])
        current=np.asarray(self.loc.pose).copy();base=pose2(*current)
        world,body,hsv,valid=self.image_cloud('head',base)
        valid &= (hsv[...,1]<60)&(hsv[...,2]>70)&(world[...,2]>.16)&(world[...,2]<1.10)
        valid &= (world[...,0]>-6.05)&(world[...,0]<-2.8)&(world[...,1]>-.95)&(world[...,1]<-.40)
        pts=body[valid][::max(1,int(valid.sum())//5000)]
        if len(pts)<100:return False
        mapped=pts@base[:3,:3].T+base[:3,3];distance=self.cabinet_tree.query(mapped)[0];pts=pts[distance<.08]
        if len(pts)<100:return False
        start=current.copy()
        def error(x):
            r=upright(x[2]);world=pts@r.T+[x[0],x[1],.0053];distance,ids=self.cabinet_tree.query(world)
            delta=world-self.cabinet_tree.data[ids];weight=np.minimum(1.,.018/(distance+1e-9))
            return (delta*weight[:,None]).ravel()
        initial=float(np.mean(abs(error(start))))
        fit=least_squares(error,start,bounds=(start-[.08,.08,.08],start+[.08,.08,.08]),max_nfev=20,loss='soft_l1',f_scale=.008)
        final=float(np.mean(abs(error(fit.x))))
        if final>.009 or final>initial*.95:return False
        self.loc.pose=fit.x;self.debug['cabinet_registration']=dict(before=start.tolist(),after=fit.x.tolist(),residual_before=initial,residual_after=final,points=len(pts))
        return True

    def confirm_book(self,category,base,pose):
        # Current-depth support for a recent track; never infer a new object from an occluded patch.
        evidence={cam:self.depth_agreement(category,cam,base,pose,tolerance=.012,occlusion=.010) for cam in CAMERAS}
        good=[cam for cam,d in evidence.items() if d['points']>=80 and d['visible_fraction']>=.12 and d['agreement']>=.82 and d['cost']<=.009]
        bad=[cam for cam,d in evidence.items() if d['points']>=150 and d['visible_fraction']>=.25 and d['agreement']<.45 and d['cost']>.025]
        self.debug['recent_'+category]=evidence
        if not good or bad:return None
        return dict(source='current_RGBD_confirmed_recent_book',depth=evidence,refitted=False)

    def colour_agreement(self,category,camera,base,pose):
        hue,model=self.colours[category]
        if hue is None:return dict(points=0,agreement=1.)
        link,slot,focal=CAMERAS[camera];prefix=f'robot_r1::robot_r1:{link}:Camera:0::'
        depth=self.data[prefix+'depth_linear'];rgb=self.data[prefix+'rgb'][...,:3].astype(np.uint8);h,w=depth.shape
        model=model[::max(1,len(model)//1800)]
        world=model@pose[:3,:3].T+pose[:3,3];body=(world-base[:3,3])@base[:3,:3]
        rel=self.data['robot_r1::cam_rel_poses'].reshape(3,7)[slot]
        pc=(body-rel[:3])@(quaternion_matrix(rel[3:])@np.diag([1.,-1.,-1.]));pc=pc[pc[:,2]>.08]
        if not len(pc):return dict(points=0,agreement=0.)
        uv=np.rint(pc[:,:2]/pc[:,2,None]*(focal*w)+[w/2-.5,h/2-.5]).astype(int)
        keep=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);pc=pc[keep];uv=uv[keep]
        delta=depth[uv[:,1],uv[:,0]]-pc[:,2];keep=np.isfinite(delta)&(abs(delta)<.015);uv=uv[keep]
        if not len(uv):return dict(points=0,agreement=0.)
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[uv[:,1],uv[:,0]];d=abs(hsv[:,0].astype(float)-hue)
        match=(np.minimum(d,180-d)<16)&(hsv[:,1]>45)&(hsv[:,2]>12)
        return dict(points=len(uv),agreement=float(match.mean()))

    def box_rim_agrees(self,camera,base,pose):
        link,slot,focal=CAMERAS[camera];depth=self.data[f'robot_r1::robot_r1:{link}:Camera:0::depth_linear'];h,w=depth.shape
        model=self.models['box']['xyz'];model=model[model[:,2]>.025][::3]
        world=model@pose[:3,:3].T+pose[:3,3];body=(world-base[:3,3])@base[:3,:3]
        rel=self.data['robot_r1::cam_rel_poses'].reshape(3,7)[slot]
        pc=(body-rel[:3])@(quaternion_matrix(rel[3:])@np.diag([1.,-1.,-1.]));pc=pc[pc[:,2]>.08]
        if not len(pc):return False
        uv=np.rint(pc[:,:2]/pc[:,2,None]*(focal*w)+[w/2-.5,h/2-.5]).astype(int)
        keep=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);pc=pc[keep];uv=uv[keep]
        if not len(pc):return False
        pixels=uv[:,1]*w+uv[:,0];order=np.lexsort((pc[:,2],pixels));_,first=np.unique(pixels[order],return_index=True);chosen=order[first];pc=pc[chosen];uv=uv[chosen]
        delta=depth[uv[:,1],uv[:,0]]-pc[:,2];visible=np.isfinite(delta)&(delta>-.02)
        return visible.sum()>20 and np.mean(abs(delta[visible])<.015)>.65 and np.mean(delta[visible]>.04)<.1

    def fits(self,category,base,support=None,cameras=('head','left','right'),prior=None,**kwargs):
        is_box=category=='box';model=self.models[category]['xyz'];bounds=self.models[category]['bounds']
        hue=self.colours[category][0];tree=self.trees[category] if hue is not None else self.geometry_trees[category];fit_model=self.colours[category][1] if hue is not None else model;results=[];debug=[]
        frames=self.body.fk(from_proprio(self.data['robot_r1::proprio'],self.body.names))[0]
        robot=self.collision.centers(frames);radii=self.collision.radii+.008
        cabinet=None if is_box else self.scene.meta['bookcases'][self.scene.meta['bookcase_assignment'][category]]
        for camera in cameras:
            world,body,hsv,valid=self.image_cloud(camera,base)
            if is_box:
                valid &= (world[...,2]>.012)&(world[...,2]<.17)
                floor=self.scene.meta['scope']['floor.n.01_1']
                valid &= np.all(world[...,:2]>=np.asarray(floor['aabb_lo'][:2]),axis=-1)&np.all(world[...,:2]<=np.asarray(floor['aabb_hi'][:2]),axis=-1)
                colour=hsv[...,1]<100
            else:
                lo=np.asarray(cabinet['aabb_lo']);hi=np.asarray(cabinet['aabb_hi'])
                valid &= (world[...,2]>.16)&np.all(world>=lo-[.12,.12,.02],axis=-1)&np.all(world<=hi+[.12,.12,.12],axis=-1)
                if prior is not None:
                    valid &= np.linalg.norm(world-prior[:3,3],axis=-1)<.45
                if hue is None:
                    median=self.signatures[category]['median_value']
                    colour=(hsv[...,2]>100)&(hsv[...,1]<110) if median>180 else (hsv[...,2]<160)
                else:
                    delta=abs(hsv[...,0].astype(float)-hue)
                    colour=(np.minimum(delta,180-delta)<14)&(hsv[...,1]>55)&(hsv[...,2]>15)
            if not is_box and hue is None:
                # White covers must be separated from the white cabinet using the
                # observed horizontal top and its clearance above the fixed shelf.
                du=world[1:-1,2:]-world[1:-1,:-2];dv=world[2:,1:-1]-world[:-2,1:-1]
                normal=np.cross(du,dv);length=np.linalg.norm(normal,axis=-1)
                horizontal=np.zeros(valid.shape,bool);horizontal[1:-1,1:-1]=(abs(normal[...,2])>.90*length)&(length>1e-9)
                ids=np.flatnonzero(valid&horizontal);raised=np.zeros(valid.shape,bool)
                if len(ids):
                    pts=world.reshape(-1,3)[ids];support=self.scene.support_heights(pts+[0.,0.,.025]);height=pts[:,2]-support
                    raised.reshape(-1)[ids]=(height>.006)&(height<.09)
                valid &= raised
            roi=valid.copy()
            valid &= colour
            # Carve fixed furniture from depth using the same canonical collision geometry.
            ids=np.flatnonzero(valid)
            if len(ids):
                pts=world.reshape(-1,3)[ids]
                gap=self.scene.contact_clearance(pts,np.zeros(len(pts)),ignore_below=np.full(len(pts),.008))
                valid.reshape(-1)[ids[gap<.002]]=False
            ids=np.flatnonzero(valid);pts=body.reshape(-1,3)[ids];keep=np.ones(len(ids),bool)
            for centre,radius in zip(robot,radii):keep &= np.sum((pts-centre)**2,axis=1)>radius**2
            valid.reshape(-1)[ids[~keep]]=False
            mask=valid&colour
            # Small gaps between printed regions do not split a book's identity.
            connected=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
            n,labels,stats,_=cv2.connectedComponentsWithStats(connected,8)
            clouds=[]
            for label in sorted(range(1,n),key=lambda k:stats[k,4],reverse=True)[:10]:
                chosen=(labels==label)&mask
                if chosen.sum()<(30 if is_box else 8):continue
                cloud=world[chosen];span=np.ptp(cloud,axis=0)
                if is_box and (span[2]<.035 or max(span[:2])<.25):continue
                if max(span[:2])>(.78 if is_box else .42) or span[2]>(.17 if is_box else .12):continue
                clouds.append((cloud,chosen))
            debug.append(dict(camera=camera,valid=int(valid.sum()),coloured=int(mask.sum()),components=len(clouds)))
            for pts,seed_mask in clouds[:4]:
                pts=pts[::max(1,len(pts)//650)];best=[]
                # Task books rest on shallow tilted shelves; yaw and height are observed.
                stable=np.eye(3)
                if not is_box:
                    xy=pts[:,:2].mean(0)
                    surfaces=[s for s in self.scene.meta['shelf_surfaces'] if np.all(xy>=np.array(s['bounds'])[:2]-.15) and np.all(xy<=np.array(s['bounds'])[3:5]+.15)]
                    if surfaces:
                        candidates=[]
                        for surface in surfaces:
                            planes=np.array(surface['planes']);heights=-(planes[:,:2]@xy+planes[:,3])/planes[:,2]
                            index=int(np.argmin(heights));height=float(heights[index])
                            if height<float(np.median(pts[:,2]))+.012:candidates.append((height,planes[index,:3]))
                        normal=max(candidates,key=lambda item:item[0])[1].copy() if candidates else np.array([0.,0.,1.]);normal/=np.linalg.norm(normal)
                        x=np.array([1.,0.,0.]);x-=normal*(normal@x);x/=np.linalg.norm(x)
                        stable=np.column_stack([x,np.cross(normal,x),normal])
                # Grow colour seeds over their actual planar top in depth. Printed patches
                # alone underconstrain the centre of a flat rectangular book.
                active_tree=tree;shape=fit_model
                if not is_box:
                    normal=stable[:,2];height=float(np.quantile(pts@normal,.70))
                    planar=roi&(abs(world@normal-height)<.006)
                    count,plane_labels=cv2.connectedComponents(planar.astype(np.uint8),connectivity=8)
                    ids,counts=np.unique(plane_labels[seed_mask&planar],return_counts=True)
                    good=ids!=0
                    if np.any(good):
                        label=ids[good][np.argmax(counts[good])];expanded=world[plane_labels==label]
                        if len(expanded)>len(pts) and np.linalg.norm(np.ptp(expanded[:,:2],axis=0))<1.30*np.linalg.norm(bounds[1,:2]-bounds[0,:2]):
                            pts=expanded[::max(1,len(expanded)//900)];active_tree=self.geometry_trees[category];shape=model
                # Transform points to the shelf's level frame; canonical book is flat there.
                level=pts@stable
                yaws=np.linspace(-np.pi,np.pi,16,endpoint=False)
                if prior is not None:yaws=np.r_[np.arctan2(*(stable.T@prior[:3,0])[1::-1]),yaws]
                for yaw in yaws:
                    rot=upright(yaw);xy=level[:,:2].mean(0)-(shape.mean(0)@rot.T)[:2]
                    z=-bounds[0,2]+.001 if is_box else float(np.quantile(level[:,2],.8)-bounds[1,2])
                    initial=np.r_[xy,z,yaw];objective=SurfaceObjective(level,active_tree)
                    variation=np.array([.20,.20,.002 if is_box else .035,.40])
                    fit=least_squares(objective.fun,initial,jac=objective.jac,max_nfev=40,loss='soft_l1',f_scale=.006,bounds=(initial-variation,initial+variation))
                    pose=transform(stable@upright(fit.x[3]),stable@fit.x[:3])
                    distances=active_tree.query((pts-pose[:3,3])@pose[:3,:3])[0];score=float(np.quantile(distances,.75))
                    best.append((score,pose))
                for score,pose in sorted(best,key=lambda row:row[0])[:3]:
                    if score>(.025 if is_box else .012):continue
                    if not is_box:
                        lo=np.asarray(cabinet['aabb_lo']);hi=np.asarray(cabinet['aabb_hi'])
                        if np.any(pose[:2,3]<lo[:2]+.015) or np.any(pose[:2,3]>hi[:2]-.015):continue
                    if is_box:
                        elevated=model[model[:,2]>.018][::4]@pose[:3,:3].T+pose[:3,3]
                        if np.mean(self.scene.contact_clearance(elevated,np.zeros(len(elevated)))<.002)>.05:continue
                        if not self.box_rim_agrees(camera,base,pose):continue
                    agreement=self.depth_agreement(category,camera,base,pose,tolerance=.018 if is_box else .010,occlusion=.014)
                    if agreement['points']<(20 if is_box else 8) or agreement['agreement']<(.55 if is_box else .60) or agreement['visible_fraction']<.18:continue
                    if prior is not None and np.linalg.norm(pose[:3,3]-prior[:3,3])>.25:continue
                    colour_check=self.colour_agreement(category,camera,base,pose) if not is_box else dict(points=0,agreement=1.)
                    if hue is not None and (colour_check['points']<20 or colour_check['agreement']<.48):continue
                    cost=score+.5*agreement['cost']+.012*(1.-colour_check['agreement'])+.002*(1.-agreement['visible_fraction'])
                    info=dict(camera=camera,cost=cost,residual=score,points=len(pts),depth=agreement,colour=colour_check)
                    results.append((pose,info))
        results.sort(key=lambda row:row[1]['cost']);self.debug[category]=dict(frames=debug,fits=[dict(i,pose=p.tolist()) for p,i in results[:3]])
        return results

class Estimator:
    def __init__(self,observer,scene):
        self.ob=observer;self.scene=scene;self.odom=np.zeros(3);self.tracks={};self.step=0;self.last=np.zeros(23);self.bag_on_table=False
    def base(self):return tuple(self.ob.loc.pose)
    def predict(self,action,s):
        self.step+=1;self.last=np.asarray(action).copy();v=true_velocity(s[:2]);commanded=command_velocity(action[:2]);speed=np.linalg.norm(commanded)
        if .008<speed<.12 and np.linalg.norm(v)<.5*speed:v=commanded
        if np.linalg.norm(action[:2])<.005 and np.linalg.norm(s[:2])<.025:v=np.zeros(2)
        wz=turn_rate(action[2],driving=np.linalg.norm(action[:2])>.06);rot=upright(self.odom[2]+wz/60.)
        self.odom[:2]+=rot[:2,:2]@v/30.;self.odom[2]+=wz/30.;self.ob.loc.predict(action[:3],measured=np.r_[v,0.])
    def remember(self,category,obj,**info):
        base=pose2(*self.base());self.tracks[category]=dict(world=obj.copy(),anchor=pose2(*self.odom)@np.linalg.inv(base)@obj,step=self.step,info=info)
    def object(self,category,max_age=150):
        track=self.tracks.get(category)
        if track is None or self.step-track['step']>max_age:return None
        if category=='box' or self.step-track['step']>180:return track['world'].copy()
        return pose2(*self.base())@np.linalg.inv(pose2(*self.odom))@track['anchor']
    def refresh(self,category,**kwargs):
        prior=self.object(category,max_age=600)
        if category.startswith('book') and prior is not None:
            from book_rectification import rectangle_poses
            rectangles=rectangle_poses(self.ob,category,pose2(*self.base()),prior)
            if rectangles:
                obj,info=rectangles[0];self.remember(category,obj,**info);return obj
            confirmed=self.ob.confirm_book(category,pose2(*self.base()),prior)
            if confirmed is not None:self.remember(category,prior,**confirmed);return prior
        if category.startswith('book'):self.ob.refine_cabinet_base()
        prior=self.object(category,max_age=10000)
        fits=self.ob.fits(category,pose2(*self.base()),prior=prior)
        if not fits:return None
        obj,info=fits[0]
        if category.startswith('book'):
            from book_rectification import rectangle_poses
            rectangles=rectangle_poses(self.ob,category,pose2(*self.base()),obj)
            if rectangles:obj,info=rectangles[0]
        self.remember(category,obj,**info);return obj
    def clear(self,category):self.tracks.pop(category,None)
