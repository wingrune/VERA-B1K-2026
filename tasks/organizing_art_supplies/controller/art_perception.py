"""RGB-D canonical colour/geometry registration; inputs are onboard arrays only."""
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from surface_icp import SurfaceObjective
from localizer import Localizer, camera_points, command_velocity, true_velocity,turn_rate
from art_skills import upright, transform
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
        self.body=Body(root);self.collision=RobotCollision(Path(root)/'robot_collision_spheres.json')
        self.models={}; self.colours={}; self.trees={};self.geometry_trees={};self.signatures={}
        for p in (Path(root)/'priors').glob('*/model.npz'):
            category=p.parent.name; model=dict(np.load(p)); self.models[category]=model
            hsv=cv2.cvtColor(model['rgb'][None].astype(np.uint8),cv2.COLOR_RGB2HSV)[0]
            if category=='marker':
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
                hue_fraction=float(selected.mean()))

    def desk_support(self, points):
        """Local upper surfaces of the same static desk mesh boxes used by collision checks."""
        height=np.full(points.shape[:-1],-np.inf)
        for box in self.scene.boxes:
            size=box[3:]-box[:3]
            # Desk side panels/rods are not horizontal support plates. Their narrow AABB
            # tops otherwise turn attached dark furniture into false marker proposals.
            # Lower shelves are retained; this is a geometry filter, not a fixed height.
            if size[2]>.10 or min(size[:2])<.07:continue
            inside=np.all(points[...,:2]>=box[:2]-.004,axis=-1)&np.all(points[...,:2]<=box[3:5]+.004,axis=-1)
            # OnTop excludes a desk surface above the item; use the topmost exposed
            # horizontal plate at XY. This rejects attached parts underneath the desktop.
            height=np.where(inside,np.maximum(height,box[5]),height)
        return height

    def frame(self, data, payload=None):self.data=data;self.payload=payload

    def depth_agreement(self,category,camera,base,pose):
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
        visible=np.isfinite(measured)&(measured>.08)&(delta>-.015)
        upper_delta=delta[visible&upper];delta=delta[visible]
        return dict(points=len(delta),agreement=float(np.mean(abs(delta)<.030)) if len(delta) else 0.,
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

    def table_obstacles(self,base):
        """Observed small raised geometry reserves space even without a category detection."""
        clouds=[]
        for link,slot,focal in CAMERAS.values():
            prefix=f'robot_r1::robot_r1:{link}:Camera:0::'
            clouds.append(camera_points(self.data[prefix+'depth_linear'],self.data['robot_r1::cam_rel_poses'].reshape(3,7)[slot],focal,stride=6,max_range=3.))
        body=np.concatenate(clouds);world=body@base[:3,:3].T+base[:3,3];support=self.desk_support(world)
        desk=self.scene.meta['scope']['desk.n.01_1']
        keep=np.isfinite(support)&(world[:,2]>support+.004)&(world[:,2]<support+.07)
        keep &= np.all(world[:,:2]>=np.asarray(desk['aabb_lo'][:2])-.01,axis=1)&np.all(world[:,:2]<=np.asarray(desk['aabb_hi'][:2])+.01,axis=1)
        body=body[keep];world=world[keep]
        if not len(body):return np.zeros((0,3))
        full=from_proprio(self.data['robot_r1::proprio'],self.body.names);poses=self.body.fk(full)[0]
        centres=self.collision.centers(poses);radii=self.collision.radii+.02
        if self.payload is not None:
            link,points,radius=self.payload;t=poses[link]
            centres=np.vstack([centres,points@t[:3,:3].T+t[:3,3]]);radii=np.r_[radii,np.full(len(points),radius+.025)]
        visible=np.ones(len(body),bool)
        for centre,radius in zip(centres,radii):visible &= np.sum((body-centre)**2,axis=1)>radius**2
        return world[visible][::max(1,int(visible.sum())//1500)]

    def fits(self, category, base, support, cameras=('head','left','right'),prior=None,reacquire=False):
        """Return ranked (map pose, metadata) fits. A known support constrains Z, never instance XY."""
        results=[]; rejected=[];hue, model=self.colours[category]; tree=self.trees[category]
        height=-float(self.models[category]['bounds'][0,2])+(.003 if category=='tote' else .001)
        for camera in cameras:
            link,slot,focal=CAMERAS[camera]; prefix=f'robot_r1::robot_r1:{link}:Camera:0::'
            rgb=self.data[prefix+'rgb'][...,:3].astype(np.uint8); depth=self.data[prefix+'depth_linear']
            hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
            if hue is None: mask=(hsv[...,2]<75)|((hsv[...,1]<110)&(hsv[...,2]<195))
            else:
                delta=np.abs(hsv[...,0].astype(float)-hue)
                tolerance=6 if category=='glue_stick' else 12
                # A placed tote can shadow the eraser's green coating down to V~11.
                # Retain its saturated hue there; raised-component and canonical
                # geometry checks below still establish the object identity.
                minimum_value=8 if category=='rubber_eraser' else 25
                mask=(np.minimum(delta,180-delta)<tolerance)&(hsv[...,1]>75)&(hsv[...,2]>minimum_value)
            h,w=depth.shape; v,u=np.mgrid[:h,:w]; f=focal*w
            valid=np.isfinite(depth)&(depth>(.15 if camera=='head' else .08))&(depth<4.5)
            pc=np.stack([(u+.5-w/2)/f*depth,(v+.5-h/2)/f*depth,depth],-1)
            rel=self.data['robot_r1::cam_rel_poses'].reshape(3,7)[slot]
            rot=quaternion_matrix(rel[3:])@np.diag([1.,-1.,-1.])
            points=pc@rot.T+rel[:3]; world=points@base[:3,:3].T+base[:3,3]
            local_support=np.full((h,w),support) if category=='tote' else self.desk_support(world)
            above=world[...,2]-local_support
            # Table texture is not a tool. Requiring the upper part of the known thickness
            # prevents tiny wooden-table patches from fitting a cylindrical template's underside.
            min_height={'tote':.045,'glue_stick':.009,'rubber_eraser':.0055,'marker':.005,'paintbrush':.0015}[category]
            valid &= above<(.53 if category=='tote' else .065)
            if category=='tote' and support>.55:
                desk=self.scene.meta['scope']['desk.n.01_1']
                valid &= np.all(world[...,:2]>np.asarray(desk['aabb_lo'][:2])-.08,axis=-1)&np.all(world[...,:2]<np.asarray(desk['aabb_hi'][:2])+.08,axis=-1)
            if category!='tote':
                desk=self.scene.meta['scope']['desk.n.01_1']
                valid &= np.all(world[...,:2]>np.asarray(desk['aabb_lo'][:2])-.06,axis=-1)
                valid &= np.all(world[...,:2]<np.asarray(desk['aabb_hi'][:2])+.06,axis=-1)
            binary=(mask&valid&(above>min_height)).astype(np.uint8)
            raised_valid=valid&(above>(min_height if category=='tote' else .0015))
            # Classify a whole raised depth component, not a coloured patch on another tool.
            _,raised_labels=cv2.connectedComponents(raised_valid.astype(np.uint8),connectivity=8)
            n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
            candidates=sorted(range(1,n),key=lambda i:stats[i,4],reverse=True)[:24]
            clouds=[];used_components=set()
            for label in candidates:
                if stats[label,4]<(20 if category=='tote' else 4):continue
                selected=labels==label
                if category!='tote':
                    ids,counts=np.unique(raised_labels[selected],return_counts=True)
                    component=int(ids[counts.argmax()])
                    if component in used_components:continue
                    used_components.add(component);selected=(raised_labels==component)&raised_valid
                    palette=hsv[selected]
                    if category=='marker':
                        coloured=float(((palette[:,1]>140)&(palette[:,2]>25)).mean())
                        if coloured>max(.40,self.signatures[category]['colour_fraction']+.30):
                            rejected.append(dict(camera=camera,reason='component colour inconsistent with marker',colour_fraction=coloured));continue
                    else:
                        fraction=float(mask[selected].mean())
                        minimum=max(.10,min(.35,.45*self.signatures[category]['hue_fraction']))
                        if fraction<minimum:
                            rejected.append(dict(camera=camera,reason='only a small colour patch, not the target object',hue_fraction=fraction));continue
                clouds.append(world[selected].astype(float))
            # Printed colour bands split a single tool into disconnected image components.
            # Fitting one band alone can match either end and shift the root by several cm.
            # Merge nearby coloured fragments using the canonical object's physical scale.
            if category!='tote':
                gap=max(.015,(.80 if category=='glue_stick' else .30)*float(np.max(np.ptp(self.models[category]['bounds'],axis=0))))
                joined=[]
                while clouds:
                    pts=clouds.pop(0);changed=True
                    while changed:
                        changed=False
                        for i,other in enumerate(clouds):
                            separation=np.maximum(np.maximum(pts.min(0)-other.max(0),other.min(0)-pts.max(0)),0.)
                            if np.linalg.norm(separation)<gap:
                                pts=np.vstack([pts,clouds.pop(i)]);changed=True;break
                    joined.append(pts)
                clouds=joined
            for pts in clouds[:8]:
                if category!='tote':
                    extent=float(np.linalg.norm(np.ptp(pts[:,:2],axis=0)))
                    maximum=1.08*float(np.sum(self.models[category]['bounds'][1,:2]-self.models[category]['bounds'][0,:2]))+.005
                    minimum=.50*float(np.max(self.models[category]['bounds'][1,:2]-self.models[category]['bounds'][0,:2])) if category=='marker' else 0.
                    if extent<minimum:
                        rejected.append(dict(camera=camera,reason='component too short to identify marker',extent=extent,minimum=minimum));continue
                    if extent>maximum:
                        rejected.append(dict(camera=camera,reason='raised component too large for target',extent=extent,maximum=maximum));continue
                partial=bool(category=='tote' and (np.ptp(pts[:,2])<.15 or np.linalg.norm(np.ptp(pts[:,:2],axis=0))<.20))
                if category=='tote' and prior is not None:partial=True
                if partial and (prior is None or np.ptp(pts[:,2])<.035):continue
                if len(pts)>700: pts=pts[np.linspace(0,len(pts)-1,700).astype(int)]
                centre=pts[:,:2].mean(0)
                coloured=None
                if category in ('glue_stick','paintbrush'):
                    samples=world[mask&raised_valid]
                    near=np.all(samples>=pts.min(0)-.003,axis=1)&np.all(samples<=pts.max(0)+.003,axis=1)
                    coloured=samples[near][::max(1,int(near.sum())//400)]
                group_support=support if category=='tote' else float(np.median(self.desk_support(pts)))
                z=group_support+height
                fit_tree=tree if category=='tote' else self.geometry_trees[category]
                # Coarse yaw hypotheses, then trimmed nearest-surface fitting. The observed side
                # need not include the full silhouette; canonical model colour points set the root.
                best=None
                prior_yaw=None if prior is None else float(np.arctan2(prior[1,0],prior[0,0]))
                yaws=prior_yaw+np.array([-.15,0.,.15]) if partial else np.linspace(-np.pi,np.pi,16,endpoint=False)
                for yaw in yaws:
                    rotation=upright(yaw); start=centre-(model.mean(0)@rotation.T)[:2]
                    if partial:start=prior[:2,3].copy()
                    objective=SurfaceObjective(pts,fit_tree)
                    if category in ('tote','glue_stick','paintbrush'):
                        def residual(x):return objective.fun(np.r_[x[:2],z,x[2]])
                        def jacobian(x):return objective.jac(np.r_[x[:2],z,x[2]])[:,[0,1,3]]
                        initial=np.r_[start,yaw]
                        radius=.20 if partial and reacquire else .07 if partial else .15
                        limits=(np.r_[start-radius,yaw-(.25 if partial else .45)],
                                np.r_[start+radius,yaw+(.25 if partial else .45)])
                    else:
                        residual=objective.fun;jacobian=objective.jac;initial=np.r_[start,z,yaw]
                        limits=(np.r_[start-.15,z-.003,yaw-.45],np.r_[start+.15,z+.003,yaw+.45])
                    fit=least_squares(residual,initial,jac=jacobian,max_nfev=40,
                                      loss='soft_l1',f_scale=.015,bounds=limits)
                    fit_z=z if category in ('tote','glue_stick','paintbrush') else float(fit.x[2])
                    rotation=upright(fit.x[-1]);position=np.r_[fit.x[:2],fit_z];local=(pts-position)@rotation
                    distances=fit_tree.query(local)[0]; score=float(np.quantile(distances,.75))
                    if category in ('glue_stick','paintbrush') and len(coloured)>=8:
                        # Canonical painted-band geometry breaks a cylinder's 180-degree
                        # ambiguity, which otherwise changes reachable pinch approaches.
                        colour_distance=tree.query((coloured-position)@rotation)[0]
                        score+=.6*float(np.quantile(colour_distance,.75))
                    if prior is not None and category!='tote':
                        angle=abs((fit.x[-1]-prior_yaw+np.pi)%(2*np.pi)-np.pi)
                        score+=.0007*angle+.015*float(np.linalg.norm(position[:2]-prior[:2,3]))
                    if partial and not reacquire:score+=.15*float(np.linalg.norm(fit.x[:2]-prior[:2,3]))
                    if category=='tote':
                        agreement=self.depth_agreement(category,camera,base,transform(rotation,position))
                        if agreement['points']<20 or agreement['agreement']<.45:continue
                        # A few coincident points cannot identify a bag whose remaining
                        # model is hidden inside a larger foreground furnishing.
                        if agreement['visible_fraction']<(.20 if prior is not None else .40):continue
                        if prior is None and (agreement['upper_points']<4 or agreement['upper_agreement']<.50):continue
                        score+=.35*agreement['cost']
                    if best is None or score<best[0]:best=(score,transform(rotation,position))
                if best is not None and category=='tote' and support<.2 and self.scene.clearance(best[1][:2,3],static_only=True)<.06:best=None
                if best is not None and category=='tote' and prior is not None:
                    shift=float(np.linalg.norm(best[1][:2,3]-prior[:2,3]))
                    # The observed floor recipient has not been commanded to move. A new brown
                    # furniture face metres away must not replace its persistent identity.
                    if shift>.20:best=None
                    elif not reacquire:best=(best[0]+.04*shift,best[1])
                if best is None or best[0]>(.022 if category=='tote' else .010):
                    rejected.append(dict(camera=camera,points=len(pts),span=np.ptp(pts,axis=0).tolist(),
                        residual=None if best is None else best[0],pose=None if best is None else best[1].tolist()))
                    continue
                span=float(np.linalg.norm(np.ptp(pts[:,:2],axis=0)))
                length=float(np.max(self.models[category]['bounds'][1,:2]-self.models[category]['bounds'][0,:2]))
                coverage=min(1.,span/max(length,1e-3))
                info=dict(camera=camera,residual=best[0],points=len(pts),range=float(np.linalg.norm(best[1][:2,3]-base[:2,3])),tracked_partial=partial,
                          coverage=coverage,span=span)
                results.append((best[1],info))
        results.sort(key=lambda r:r[1]['residual']+.001*r[1]['range']+(.004*(1.-r[1]['coverage']) if category!='tote' else 0.))
        self.debug[category]=[r[1] for r in results]+[dict(rejected=r) for r in rejected]
        return results

    def insertion_evidence(self, category, base, bag):
        """RGB-D evidence relative to the observed bag. This cannot certify a BDDL predicate."""
        counts=dict(inside_points=0,outside_points=0,confidence='unobserved')
        hue,_=self.colours[category]
        for camera,(link,slot,focal) in CAMERAS.items():
            prefix=f'robot_r1::robot_r1:{link}:Camera:0::';rgb=self.data[prefix+'rgb'][...,:3].astype(np.uint8)
            depth=self.data[prefix+'depth_linear'];h,w=depth.shape;hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
            if hue is None:mask=hsv[...,2]<60
            else:
                delta=np.abs(hsv[...,0].astype(float)-hue)
                mask=(np.minimum(delta,180-delta)<(6 if category=='glue_stick' else 8))&(hsv[...,1]>90)
            v,u=np.nonzero(mask&np.isfinite(depth)&(depth>.15)&(depth<4.))
            d=depth[v,u];f=focal*w;pc=np.c_[(u+.5-w/2)/f*d,(v+.5-h/2)/f*d,d]
            rel=self.data['robot_r1::cam_rel_poses'].reshape(3,7)[slot]
            world=(pc@(quaternion_matrix(rel[3:])@np.diag([1.,-1.,-1.])).T+rel[:3])@base[:3,:3].T+base[:3,3]
            support=float(bag[2,3]+self.models['tote']['bounds'][0,2]-.003)
            raised=world[:,2]-support>(.006 if category=='glue_stick' else .0035)
            world=world[raised]
            local=(world-bag[:3,3])@bag[:3,:3]
            inside=(abs(local[:,0])<.13)&(abs(local[:,1])<.08)&(local[:,2]>-.16)&(local[:,2]<.15)
            near=(np.linalg.norm(local[:,:2],axis=1)<.35)&(local[:,2]>-.22)&(local[:,2]<.19)
            counts['inside_points']+=int(inside.sum());counts['outside_points']+=int((near&~inside).sum())
        if counts['inside_points']>=3:counts['confidence']='visual_points_inside_prior_volume'
        elif counts['outside_points']>=8:counts['confidence']='visual_points_outside'
        return counts


class Estimator:
    """Map localisation plus fresh object poses anchored in relative odometry for contact."""
    def __init__(self, observer, scene):
        self.ob=observer; self.scene=scene; self.odom=np.zeros(3); self.tracks={}; self.step=0
        self.last=np.zeros(23); self.table_height=float(scene.meta['scope']['desk.n.01_1']['aabb_hi'][2])
        self.bag_on_table=False
        self.last_desk_alignment=-1000

    def base(self): return tuple(self.ob.loc.pose)

    def predict(self, action, s):
        self.step+=1
        self.last=np.asarray(action).copy()
        v=true_velocity(s[:2])
        if np.linalg.norm(action[:2])<.005 and np.linalg.norm(s[:2])<.025:v=np.zeros(2)
        wz=turn_rate(action[2],driving=np.linalg.norm(action[:2])>.06)
        rotation=upright(self.odom[2]+wz/60.)
        self.odom[:2]+=(rotation[:2,:2]@v)/30.; self.odom[2]+=wz/30.
        self.ob.loc.predict(action[:3],measured=np.r_[v,0.])

    def refine_desk(self):
        if self.ob.loc.pose is None or self.step-self.last_desk_alignment<60:return None
        if self.ob.payload is not None:return None
        if np.linalg.norm(self.ob.data['robot_r1::proprio'][:2])>.02 or np.linalg.norm(self.last[:3])>.02:return None
        self.last_desk_alignment=self.step
        from desk_alignment import align_desk
        excluded=[(k,self.object(k,max_age=10000)) for k in self.tracks]
        pose,info=align_desk(self.ob,np.asarray(self.base()),excluded)
        if pose is not None:self.ob.loc.pose=np.asarray(pose)
        return dict(info,accepted=pose is not None)

    def refresh(self, category):
        base=pose2(*self.base())
        support=self.table_height if category!='tote' or self.bag_on_table else 0.
        prior=self.object(category,max_age=10000 if category=='tote' else 300)
        track=self.tracks.get(category)
        # Long task phases do not erase an untouched object's identity. For stale
        # tracks use the last observed map pose as a weak association prior, avoiding
        # accumulated odometry drift. Existing geometry and colour tests still decide.
        if prior is None and track is not None:
            prior=track['world'].copy()
        stationary=(np.linalg.norm(self.ob.data['robot_r1::proprio'][:2])<.025
                    and np.linalg.norm(self.last[:3])<.02)
        if (prior is not None and category!='tote' and track is not None
            and self.step-track['step']<=180 and stationary and self.ob.payload is None):
            from pose_confirmation import confirm_tool
            confirmation=confirm_tool(self.ob,category,base,prior)
            if confirmation is not None:
                self.tracks[category]=dict(anchor=pose2(*self.odom)@np.linalg.inv(base)@prior,
                    world=prior,step=self.step,info=confirmation)
                self.ob.debug[category]=[confirmation]
                return prior
        fits=self.ob.fits(category,base,support,prior=prior)
        if not fits and category=='tote' and prior is not None:
            from recipient_reacquisition import corroborated_recipient
            fits=corroborated_recipient(self.ob,base,support,prior)
        from object_association import distinct_tool_fits
        fits=distinct_tool_fits(self.ob,category,base,fits,self.tracks)
        if not fits:return None
        obj,info=fits[0]
        self.tracks[category]=dict(anchor=pose2(*self.odom)@np.linalg.inv(base)@obj,
                                   world=obj,step=self.step,info=info)
        return obj

    def object(self, category, max_age=120):
        track=self.tracks.get(category)
        if track is None or self.step-track['step']>max_age:return None
        # Recipient memory spans long navigation and repeated turns. Keep its last observed
        # map pose; the independent odometry anchor accumulated 10–25 cm while it stayed still.
        if category=='tote':return track['world'].copy()
        return pose2(*self.base())@np.linalg.inv(pose2(*self.odom))@track['anchor']

    def clear(self, category): self.tracks.pop(category,None)

    def remember(self, category, pose, source):
        """Action-derived pose prediction, subsequently corrected by RGB-D observations."""
        obj=np.asarray(pose).copy();base=pose2(*self.base())
        self.tracks[category]=dict(anchor=pose2(*self.odom)@np.linalg.inv(base)@obj,
                                   world=obj,step=self.step,info=dict(source=source))
