"""Local six-DOF recovery of an observed bed recipient, with multi-view depth gates."""
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from art_skills import transform

SOURCE='multiview_tilted_recipient'
def tilted_recipient(observer,base,support,prior):
    from art_perception import CAMERAS,quaternion_matrix
    if prior is None or support<.35:return []
    model=observer.models['tote']['xyz'];tree=observer.trees['tote']
    hue=observer.colours['tote'][0];seed=prior[:3,3].copy();retained=[]
    known_tilt=np.arccos(np.clip(prior[2,2],-1.,1.))>.14
    old_angles=Rotation.from_matrix(prior[:3,:3]).as_euler('xyz')
    if known_tilt:
        seeds=[old_angles+np.array([x,y,z]) for x,y,z in [(0,0,0),(.1,0,0),(-.1,0,0),(0,.1,0),(0,-.1,0),(0,0,.15),(0,0,-.15)]]
    else:
        seeds=[np.array([x,y,yaw]) for yaw in np.linspace(-np.pi,np.pi,8,endpoint=False)
               for x,y in [(0.,0.),(.45,0.),(-.45,0.),(0.,.45),(0.,-.45)]]
    for camera,(link,slot,focal) in CAMERAS.items():
        prefix=f'robot_r1::robot_r1:{link}:Camera:0::'
        rgb=observer.data[prefix+'rgb'][...,:3].astype(np.uint8);depth=observer.data[prefix+'depth_linear']
        h,w=depth.shape;v,u=np.mgrid[:h,:w];rel=observer.data['robot_r1::cam_rel_poses'].reshape(3,7)[slot]
        rot=quaternion_matrix(rel[3:])@np.diag([1.,-1.,-1.])
        pc=np.stack([(u+.5-w/2)/(focal*w)*depth,(v+.5-h/2)/(focal*w)*depth,depth],-1)
        world=(pc@rot.T+rel[:3])@base[:3,:3].T+base[:3,3]
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);delta=abs(hsv[...,0].astype(float)-hue)
        mask=np.isfinite(depth)&(depth>.08)&(depth<4.5)&(np.minimum(delta,180-delta)<12)&(hsv[...,1]>75)&(hsv[...,2]>25)
        mask&=(world[...,2]>support+.04)&(world[...,2]<support+.65)&(np.linalg.norm(world[...,:2]-prior[:2,3],axis=2)<.50)
        points=world[mask]
        if len(points)<100:continue
        points=points[::max(1,len(points)//900)]
        def residual(x):
            rotation=Rotation.from_euler('xyz',x[3:]).as_matrix()
            return np.minimum(tree.query((points-x[:3])@rotation)[0],.20)
        proposals=[]
        for angles in seeds:
            yaw=angles[2]
            initial=np.r_[seed,np.clip(angles[:2],-.74,.74),yaw]
            fit=least_squares(residual,initial,
                bounds=(np.r_[seed[:2]-.25,seed[2]-.14,-.75,-.75,yaw-.65],
                        np.r_[seed[:2]+.25,seed[2]+.10,.75,.75,yaw+.65]),
                max_nfev=80,loss='soft_l1',f_scale=.01,diff_step=1e-4)
            pose=transform(Rotation.from_euler('xyz',fit.x[3:]).as_matrix(),fit.x[:3])
            tilt=float(np.arccos(np.clip(pose[2,2],-1.,1.)))
            if tilt<.10 or tilt>.70 or np.linalg.norm(pose[:2,3]-prior[:2,3])>.30:continue
            distances=tree.query((points-pose[:3,3])@pose[:3,:3])[0]
            residual75=float(np.quantile(distances,.75))
            if residual75>.018:continue
            bottom=float(np.quantile(model@pose[2,:3]+pose[2,3],.001))
            if not support-.025<=bottom<=support+.025:continue
            evidence={cam:observer.depth_agreement('tote',cam,base,pose,tolerance=.025) for cam in CAMERAS}
            visible=[cam for cam,e in evidence.items() if e['points']>=60 and e['visible_fraction']>=.18]
            good=[cam for cam in visible if evidence[cam]['agreement']>=.80 and evidence[cam]['cost']<=.025]
            if len(good)<2 or any(evidence[cam]['agreement']<.60 for cam in visible):continue
            score=residual75+.50*float(np.mean([evidence[cam]['cost'] for cam in visible]))
            proposals.append((score,pose,dict(source=SOURCE,camera=camera,residual=residual75,tilt=float(tilt),
                corroborating_cameras=good,depth=evidence,range=float(np.linalg.norm(pose[:2,3]-base[:2,3])),
                coverage=1.,points=len(points),support_bottom=bottom)))
        if proposals:
            proposals.sort(key=lambda x:x[0])
            # A well-observed head proposal verified by the wrists avoids three
            # independent expensive fits; every accepted pose still passes two views.
            _,pose,info=proposals[0];retained.append((pose,info))
            if camera=='head' and all(info['depth'][cam]['agreement']>.90 for cam in info['corroborating_cameras']):break
    retained.sort(key=lambda row:row[1]['residual']+.50*np.mean([e['cost'] for e in row[1]['depth'].values() if e['points']>=60]))
    observer.debug['tilted_recipient']=[info for _,info in retained]
    return retained

