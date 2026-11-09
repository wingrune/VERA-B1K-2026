"""Local flat-object pose proposals from RGB-D top-plane rectangle boundaries."""
import numpy as np
import cv2
from art_skills import upright,transform


def depth_contradictions(observer,category,base,pose):
    """Current depth may occlude a target, but cannot expose its predicted solid volume."""
    from art_perception import CAMERAS
    evidence={camera:observer.depth_agreement(category,camera,base,pose,tolerance=.008,occlusion=.006)
              for camera in CAMERAS}
    bad=[camera for camera,d in evidence.items() if d['points']>=120 and d['visible_fraction']>=.25
         and d['agreement']<.35 and d['cost']>.035]
    return bad,evidence


def rectangle_poses(observer,category,base,prior):
    from art_perception import CAMERAS,quaternion_matrix
    if category not in ('notebook','folder') or prior is None:return []
    bounds=observer.models[category]['bounds'];extent=bounds[1]-bounds[0]
    hue=observer.colours[category][0];proposals=[]
    if hue is None:return []
    for camera,(link,slot,focal) in CAMERAS.items():
        prefix=f'robot_r1::robot_r1:{link}:Camera:0::'
        rgb=observer.data[prefix+'rgb'][...,:3].astype(np.uint8);depth=observer.data[prefix+'depth_linear']
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);delta=abs(hsv[...,0].astype(float)-hue)
        mask=(np.minimum(delta,180-delta)<12)&(hsv[...,1]>75)&(hsv[...,2]>25)&np.isfinite(depth)&(depth>.08)&(depth<4.5)
        v,u=np.nonzero(mask)
        if len(v)<100:continue
        h,w=depth.shape;z=depth[v,u];pc=np.c_[(u+.5-w/2)/(focal*w)*z,(v+.5-h/2)/(focal*w)*z,z]
        rel=observer.data['robot_r1::cam_rel_poses'].reshape(3,7)[slot]
        rotation=quaternion_matrix(rel[3:])@np.diag([1.,-1.,-1.])
        world=(pc@rotation.T+rel[:3])@base[:3,:3].T+base[:3,3]
        local=(world-prior[:3,3])@prior[:3,:3]
        near=np.all(local[:,:2]>=bounds[0,:2]-.045,axis=1)&np.all(local[:,:2]<=bounds[1,:2]+.045,axis=1)
        near&=(local[:,2]>bounds[1,2]-.020)&(local[:,2]<bounds[1,2]+.020)
        points=world[near]
        if len(points)<150:continue
        points=points[np.linspace(0,len(points)-1,min(2400,len(points))).astype(int)]
        rng=np.random.default_rng(27);best=None
        for _ in range(50):
            p0,p1,p2=points[rng.choice(len(points),3,replace=False)]
            normal=np.cross(p1-p0,p2-p0);length=np.linalg.norm(normal)
            if length<1e-6:continue
            normal/=length
            if normal[2]<0:normal=-normal
            if normal[2]<.94:continue
            inliers=abs((points-p0)@normal)<.0025
            if best is None or inliers.sum()>best.sum():best=inliers
        if best is None or best.sum()<max(100,.65*len(points)):continue
        plane=points[best];mean=plane.mean(0);_,_,vectors=np.linalg.svd(plane-mean,full_matrices=False)
        normal=vectors[-1]
        if normal[2]<0:normal=-normal
        if normal[2]<.94:continue
        x=np.array([1.,0.,0.]);x-=normal*(x@normal);x/=np.linalg.norm(x)
        frame=np.column_stack([x,np.cross(normal,x),normal]);projected=plane@frame
        rect=cv2.minAreaRect(projected[:,:2].astype(np.float32));corners=cv2.boxPoints(rect).astype(float)
        center=corners.mean(0);height=float(np.median(projected[:,2]))-bounds[1,2]
        for edge in (corners[1]-corners[0],corners[2]-corners[1]):
            angle=float(np.arctan2(edge[1],edge[0]))
            for half_turn in (0.,np.pi):
                yaw=angle+half_turn;rot2=upright(yaw);aligned=(corners-center)@rot2[:2,:2]
                span=np.ptp(aligned,axis=0)
                # An occluded front strip cannot determine the complete rectangle.
                if np.any(abs(span-extent[:2])>np.maximum(.008,.09*extent[:2])):continue
                position=frame@np.r_[center-rot2[:2,:2]@bounds.mean(0)[:2],height]
                pose=transform(frame@rot2,position)
                if np.linalg.norm(position[:2]-prior[:2,3])>.060:continue
                depth={cam:observer.depth_agreement(category,cam,base,pose,tolerance=.008,occlusion=.006) for cam in CAMERAS}
                visible=[cam for cam,d in depth.items() if d['points']>=100 and d['visible_fraction']>=.10]
                good=[cam for cam in visible if depth[cam]['agreement']>=.92 and depth[cam]['cost']<=.006]
                if not good:continue
                if any(depth[cam]['points']>=150 and depth[cam]['visible_fraction']>=.25 and depth[cam]['agreement']<.65 for cam in visible):continue
                score=min(depth[cam]['cost'] for cam in good)+.02*float(np.linalg.norm(span-extent[:2]))
                # Rectangle orientation is half-turn ambiguous; use the prior label only as a tie-break.
                prior_yaw=np.arctan2(prior[1,0],prior[0,0]);label_cost=abs((np.arctan2(pose[1,0],pose[0,0])-prior_yaw+np.pi)%(2*np.pi)-np.pi)
                proposals.append((score+.00005*label_cost,pose,dict(source='RGBD_top_rectangle',camera=camera,points=len(plane),span=span.tolist(),depth=depth,residual=score,coverage=1.,range=float(np.linalg.norm(position[:2]-base[:2,3])))))
    proposals.sort(key=lambda row:row[0])
    return [(pose,info) for _,pose,info in proposals]


def confirm_contact_pose(observer,category,base,pose):
    """Keep a recently planned contact only when current colour and depth support it."""
    from pose_confirmation import confirm_tool
    confirmation=confirm_tool(observer,category,base,pose,allow_flat_edge=True)
    if confirmation is None:return None
    bad,depth=depth_contradictions(observer,category,base,pose)
    good=[camera for camera,d in depth.items() if d['points']>=100 and d['visible_fraction']>=.10
          and d['agreement']>=.92 and d['cost']<=.006]
    contradictory=[camera for camera,d in depth.items() if d['points']>=150 and d['visible_fraction']>=.25
                   and d['agreement']<.65]
    if bad or contradictory or not good:return None
    return dict(source='current_RGBD_contact_confirmation',colour=confirmation,depth=depth,refitted=False)
