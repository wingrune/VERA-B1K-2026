"""RGB-D canonical colour/geometry registration; inputs are onboard arrays only."""
from pathlib import Path
import json
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from surface_icp import SurfaceObjective
from localizer import Localizer, camera_points, command_velocity, true_velocity,turn_rate
from compost_skills import upright, transform
from manip import pose2
from wbik import Body
from robot_collision import RobotCollision
from kinematics import from_proprio

CAMERAS = {'head': ('zed_link', 2, 306./720.),
           'left': ('left_realsense_link', 0, 388.66/480.),
           'right': ('right_realsense_link', 1, 388.66/480.)}


def quaternion_matrix(q):
    x,y,z,w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


class Observer:

    def __init__(self, root, scene):
        self.scene=scene; self.loc=Localizer(scene); self.data=None; self.debug={}
        self.support_grid=dict(np.load(Path(root)/'priors/support_geometry.npz'))
        grid=self.support_grid;valid=np.isfinite(grid['height'])
        ij=np.argwhere(valid);self.support_xy=grid['origin']+ij*float(grid['resolution'])
        self.support_values=grid['height'][valid];self.support_tree=cKDTree(self.support_xy)
        self.body=Body(root);self.collision=RobotCollision(Path(root)/'robot_collision_spheres.json')
        self.models={}; self.colours={}; self.trees={};self.geometry_trees={};self.signatures={}
        for p in (Path(root)/'priors').glob('*/model.npz'):
            category=p.parent.name; model=dict(np.load(p)); self.models[category]=model
            hsv=cv2.cvtColor(model['rgb'][None].astype(np.uint8),cv2.COLOR_RGB2HSV)[0]
            if category=='marker' or np.count_nonzero(hsv[:,1]>80)<max(20,len(hsv)//20):
                selected=np.ones(len(hsv),bool); hue=None
            else:
                hist=np.bincount((hsv[hsv[:,1]>80,0]//6).astype(int),minlength=30)
                hue=(int(hist.argmax())+.5)*6
                delta=np.abs(hsv[:,0].astype(float)-hue)
                selected=(np.minimum(delta,180-delta)<12)&(hsv[:,1]>80)
                if category=='tote':selected=np.ones(len(hsv),bool)
            xyz=model['xyz'][selected][::2]; self.colours[category]=(hue,xyz)
            self.trees[category]=cKDTree(xyz)
            self.geometry_trees[category]=cKDTree(model['xyz'][::2])
            self.signatures[category]=dict(colour_fraction=float(((hsv[:,1]>75)&(hsv[:,2]>25)).mean()),
                hue_fraction=float(selected.mean()),median_value=float(np.median(hsv[:,2])))

        hinge=json.loads((Path(root)/'priors/ashcan_lid/joint_geometry.json').read_text())['j_link_0']
        # Exclude the articulated lid/hinge band from rigid-body component
        # extraction when the full-height component merges with background.
        self.ashcan_body_fit_top=-float(self.models['ashcan']['bounds'][0,2])+float(hinge['physics:localPos0'][2])-.025

    def desk_support(self, points):
        """Local upper surfaces of the same static desk mesh boxes used by collision checks."""
        grid=self.support_grid
        ij=np.rint((points[...,:2]-grid['origin'])/float(grid['resolution'])).astype(int)
        height=grid['height'];inside=np.all(ij>=0,axis=-1)&np.all(ij<height.shape,axis=-1)
        sampled=height[np.clip(ij[...,0],0,height.shape[0]-1),np.clip(ij[...,1],0,height.shape[1]-1)]
        return np.where(inside,sampled,-np.inf)

    def frame(self, data, payload=None):self.data=data;self.payload=payload

    def depth_agreement(self,category,camera,base,pose,tolerance=.030,occlusion=.015):
        """Reject a surface fit whose predicted visible geometry lies in observed free space."""
        link,slot,focal=CAMERAS[camera];depth=self.data[f'robot_r1::robot_r1:{link}:Camera:0::depth_linear']
        h,w=depth.shape;model=self.models[category]['xyz'];model=model[::max(1,len(model)//1600)]
        upper=model[:,2]>.205
        world=model@pose[:3,:3].T+pose[:3,3];body=(world-base[:3,3])@base[:3,:3]
        rel=self.data['robot_r1::cam_rel_poses'].reshape(3,7)[slot]
        pc=(body-rel[:3])@(quaternion_matrix(rel[3:])@np.diag([1.,-1.,-1.]))
        selected=pc[:,2]>.08;pc=pc[selected];upper=upper[selected]
        uv=np.rint(pc[:,:2]/pc[:,2,None]*(focal*w)+[w/2-.5,h/2-.5]).astype(int)
        selected=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        pc=pc[selected];uv=uv[selected];upper=upper[selected]
        if not len(pc):return dict(points=0,agreement=0.,cost=.15,projected_points=0,visible_fraction=0.,upper_points=0,upper_agreement=0.)
        pixels=uv[:,1]*w+uv[:,0];order=np.lexsort((pc[:,2],pixels));_,first=np.unique(pixels[order],return_index=True)
        chosen=order[first];pc=pc[chosen];uv=uv[chosen];upper=upper[chosen];measured=depth[uv[:,1],uv[:,0]]
        delta=measured-pc[:,2]
        # A nearer sensed surface may occlude the model; a farther surface cannot.
        visible=np.isfinite(measured)&(measured>.08)&(delta>-occlusion)
        upper_delta=delta[visible&upper];delta=delta[visible]
        return dict(points=len(delta),agreement=float(np.mean(abs(delta)<tolerance)) if len(delta) else 0.,
                    cost=float(np.mean(np.clip(delta,0.,.15))) if len(delta) else .15,
                    projected_points=len(pc),visible_fraction=float(visible.mean()),
                    upper_points=len(upper_delta),upper_agreement=float(np.mean(abs(upper_delta)<.030)) if len(upper_delta) else 0.)

    def scan(self):
        link,slot,focal=CAMERAS['head']; prefix=f'robot_r1::robot_r1:{link}:Camera:0::'
        full=from_proprio(self.data['robot_r1::proprio'],self.body.names)
        poses=self.body.fk(full)[0]
        centres=self.collision.centers(poses);radii=self.collision.radii
        # m05 excludes its estimated carried fax from scan matching. Reuse that method:
        # a carried tote must not be mistaken for a static wall during localization.
        if self.payload is not None:
            link,points,radius=self.payload;t=poses[link]
            centres=np.vstack([centres,points@t[:3,:3].T+t[:3,3]])
            radii=np.r_[radii,np.full(len(points),radius)]
        spheres=(centres,radii)
        return self.loc.scan(camera_points(self.data[prefix+'depth_linear'],
                      self.data['robot_r1::cam_rel_poses'].reshape(3,7)[slot],focal,stride=3),exclude=spheres)

    def image_cloud(self,camera,base):
        link,slot,focal=CAMERAS[camera];prefix=f'robot_r1::robot_r1:{link}:Camera:0::'
        depth=self.data[prefix+'depth_linear'];rgb=self.data[prefix+'rgb'][...,:3].astype(np.uint8)
        h,w=depth.shape;v,u=np.mgrid[:h,:w];f=focal*w
        pc=np.stack([(u+.5-w/2)/f*depth,(v+.5-h/2)/f*depth,depth],-1)
        rel=self.data['robot_r1::cam_rel_poses'].reshape(3,7)[slot]
        body=pc@(quaternion_matrix(rel[3:])@np.diag([1.,-1.,-1.])).T+rel[:3]
        world=body@base[:3,:3].T+base[:3,3]
        valid=np.isfinite(depth)&(depth>.08)&(depth<4.)
        return world,body,cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV),valid

    def fits(self,category,base,support=None,cameras=('head','left','right'),prior=None,rigid_body_only=False):
        """Fit canonical surfaces to raised RGB-D components, without simulator labels."""
        from scipy.spatial.transform import Rotation
        model=self.models[category]['xyz'];bounds=self.models[category]['bounds']
        is_bin=category=='ashcan';results=[];debug=[]
        surface=self.scene.meta['target_surface'];height=float(surface['aabb_hi'][2])
        full=from_proprio(self.data['robot_r1::proprio'],self.body.names);poses=self.body.fk(full)[0]
        centres=self.collision.centers(poses);radii=self.collision.radii+.008
        hue=self.colours[category][0]
        for camera in cameras:
            world,body,hsv,valid=self.image_cloud(camera,base)
            if is_bin:
                valid &= (world[...,2]>.06)&(world[...,2]<(self.ashcan_body_fit_top if rigid_body_only else .76))
                room=self.scene.meta['scope']['floor.n.01_1']
                lo=np.array(room['aabb_lo']);hi=np.array(room['aabb_hi'])
                valid &= np.all(world[...,:2]>lo[:2]-.12,axis=-1)&np.all(world[...,:2]<hi[:2]+.12,axis=-1)
                colour=hsv[...,2]<155
            else:
                valid &= (world[...,2]>height+.006)&(world[...,2]<height+.16)
                valid &= np.all(world[...,:2]>=np.array(surface['aabb_lo'][:2])-.08,axis=-1)&np.all(world[...,:2]<=np.array(surface['aabb_hi'][:2])+.08,axis=-1)
                if hue is None:colour=hsv[...,2]>40
                else:
                    delta=abs(hsv[...,0].astype(float)-hue)
                    colour=(np.minimum(delta,180-delta)<16)&(hsv[...,1]>60)&(hsv[...,2]>20)
            # Remove visible robot geometry using permitted proprioception and static robot models.
            ids=np.flatnonzero(valid);points=body.reshape(-1,3)[ids]
            keep=np.ones(len(ids),bool)
            for centre,radius in zip(centres,radii):keep &= np.sum((points-centre)**2,axis=1)>radius**2
            valid.reshape(-1)[ids[~keep]]=False
            if is_bin:
                ids=np.flatnonzero(valid);pts=world.reshape(-1,3)[ids]
                gap=self.scene.contact_clearance(pts,np.zeros(len(pts)),static_only=True,ignore_below=np.full(len(pts),.06))
                valid.reshape(-1)[ids[gap<.025]]=False
            n,labels,stats,_=cv2.connectedComponentsWithStats(valid.astype(np.uint8),8)
            clouds=[]
            for label in sorted(range(1,n),key=lambda j:stats[j,4],reverse=True)[:12]:
                if stats[label,4]<(30 if is_bin else 5):continue
                chosen=labels==label;fraction=float(colour[chosen].mean())
                if fraction<(.30 if is_bin else .12):continue
                pts=world[chosen];span=np.ptp(pts,axis=0)
                if np.linalg.norm(span[:2])>(.65 if is_bin else .24):continue
                if is_bin and (span[2]<.18 or max(span[:2])<.12):continue
                clouds.append((pts,fraction))
            debug.append(dict(camera=camera,components=len(clouds),valid_pixels=int(valid.sum())))
            for pts,fraction in clouds[:4]:
                pts=pts[::max(1,len(pts)//700)];best=[]
                # The cut fruit can rest with either cut-face normal. Refine all six pose DOFs.
                orientations=(np.eye(3),Rotation.from_euler('x',np.pi).as_matrix()) if not is_bin else (np.eye(3),)
                for stable in orientations:
                    shaped=model@stable.T;tree=cKDTree(shaped[::2])
                    z=(-float(shaped[:,2].min())+.001)+(0. if is_bin else height)
                    yaws=np.linspace(-np.pi,np.pi,16,endpoint=False)
                    if prior is not None:yaws=np.r_[np.arctan2(prior[1,0],prior[0,0]),yaws]
                    for yaw in yaws:
                        rotation=upright(yaw);xy=pts[:,:2].mean(0)-(shaped.mean(0)@rotation.T)[:2]
                        objective=SurfaceObjective(pts,tree)
                        initial=np.r_[xy,z,yaw]
                        fit=least_squares(objective.fun,initial,jac=objective.jac,max_nfev=32,loss='soft_l1',f_scale=.008,
                            bounds=(initial+[-.18,-.18,-.004,-.4],initial+[.18,.18,.004,.4]))
                        rot=upright(fit.x[3])@stable;pose=transform(rot,fit.x[:3]);distance=self.geometry_trees[category].query((pts-pose[:3,3])@rot)[0]
                        score=float(np.quantile(distance,.75));best.append((score,pose))
                for score,pose in sorted(best,key=lambda x:x[0])[:3]:
                    if score>(.030 if is_bin else .014):continue
                    if not is_bin:
                        # Small 6D tilt is determined from current depth; support contact bounds its height.
                        reference=pose.copy();tree=self.geometry_trees[category]
                        def residual(x):
                            rot=Rotation.from_rotvec(x[3:]).as_matrix()@reference[:3,:3]
                            local=(pts-reference[:3,3]-x[:3])@rot
                            _,idx=tree.query(local)
                            predicted_z=(model[::3]@rot.T)[:,2]+reference[2,3]+x[2]
                            return np.r_[(local-tree.data[idx]).ravel(),8*(predicted_z.min()-height-.001)]
                        fit=least_squares(residual,np.zeros(6),max_nfev=30,loss='soft_l1',f_scale=.006,
                            bounds=([-.02,-.02,-.025,-.22,-.22,-.20],[.02,.02,.025,.22,.22,.20]))
                        pose=transform(Rotation.from_rotvec(fit.x[3:]).as_matrix()@reference[:3,:3],reference[:3,3]+fit.x[:3])
                        score=float(np.quantile(tree.query((pts-pose[:3,3])@pose[:3,:3])[0],.75))
                    agreement=self.depth_agreement(category,camera,base,pose,tolerance=.020 if is_bin else .009,occlusion=.014)
                    if agreement['points']<(25 if is_bin else 5) or agreement['agreement']<(.60 if is_bin else .65):continue
                    if agreement['visible_fraction']<.30:continue
                    if score>(.022 if is_bin else .009):continue
                    if prior is not None and np.linalg.norm(pose[:2,3]-prior[:2,3])>(.25 if is_bin else .16):continue
                    cost=score+.5*agreement['cost']
                    info=dict(camera=camera,residual=score,cost=cost,points=len(pts),colour_fraction=fraction,depth=agreement)
                    results.append((pose,info))
        results.sort(key=lambda pair:pair[1]['cost'])
        self.debug[category]=dict(frames=debug,fits=[dict(info,pose=pose.tolist()) for pose,info in results[:5]])
        return results

class Estimator:
    """Observation-only map tracking and relative-odometry-anchored object memory."""
    def __init__(self,observer,scene):
        self.ob=observer;self.scene=scene;self.odom=np.zeros(3);self.tracks={};self.step=0;self.last=np.zeros(23)
    def base(self):return tuple(self.ob.loc.pose)
    def predict(self,action,s):
        self.step+=1;self.last=np.asarray(action).copy();v=true_velocity(s[:2]);commanded=command_velocity(action[:2]);speed=np.linalg.norm(commanded)
        if .008<speed<.12 and np.linalg.norm(v)<.5*speed:v=commanded
        if np.linalg.norm(action[:2])<.005 and np.linalg.norm(s[:2])<.025:v=np.zeros(2)
        wz=turn_rate(action[2],driving=np.linalg.norm(action[:2])>.06);rotation=upright(self.odom[2]+wz/60.)
        self.odom[:2]+=rotation[:2,:2]@v/30.;self.odom[2]+=wz/30.
        self.ob.loc.predict(action[:3],measured=np.r_[v,0.])
    def remember(self,category,obj,**info):
        base=pose2(*self.base())
        self.tracks[category]=dict(world=obj.copy(),anchor=pose2(*self.odom)@np.linalg.inv(base)@obj,step=self.step,info=info)
    def object(self,category,max_age=150):
        track=self.tracks.get(category)
        if track is None or self.step-track['step']>max_age:return None
        if category=='ashcan':return track['world'].copy()
        return pose2(*self.base())@np.linalg.inv(pose2(*self.odom))@track['anchor']
    def refresh(self,category,**kwargs):
        prior=self.object(category,max_age=1500);fits=self.ob.fits(category,pose2(*self.base()),prior=prior)
        if not fits and category=='ashcan':
            fits=self.ob.fits(category,pose2(*self.base()),prior=prior,rigid_body_only=True)
            for _,info in fits:info['body_only_recovery']=True
        if not fits:return None
        obj,info=fits[0];self.remember(category,obj,**info);return obj
