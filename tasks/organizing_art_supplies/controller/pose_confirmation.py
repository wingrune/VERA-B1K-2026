"""Verify a recent stationary tool track with current RGB-D without refitting an occluded silhouette."""
import numpy as np
import cv2


def confirm_tool(observer,category,base,pose,diagnostic=None):
    from art_perception import CAMERAS,quaternion_matrix
    hue,_=observer.colours[category]
    if hue is None or category=='tote':return None
    bounds=observer.models[category]['bounds'];extent=bounds[1]-bounds[0]
    tree=observer.trees[category];evidence=[];views=[];tolerances=[]
    for camera,(link,slot,focal) in CAMERAS.items():
        prefix=f'robot_r1::robot_r1:{link}:Camera:0::'
        rgb=observer.data[prefix+'rgb'][...,:3].astype(np.uint8);depth=observer.data[prefix+'depth_linear']
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);delta=abs(hsv[...,0].astype(float)-hue)
        # Hue is usable only above a few RGB quantization levels; this admits shadowed
        # saturated coatings without inferring a new object from a dark furniture patch.
        chroma=np.max(rgb,axis=-1).astype(int)-np.min(rgb,axis=-1).astype(int)
        mask=(np.minimum(delta,180-delta)<(6 if category=='glue_stick' else 12))&(hsv[...,1]>75)&(chroma>=3)
        mask &= np.isfinite(depth)&(depth>.08)&(depth<4.5)
        v,u=np.nonzero(mask)
        if not len(v):continue
        h,w=depth.shape;d=depth[v,u];f=focal*w
        pc=np.c_[(u+.5-w/2)/f*d,(v+.5-h/2)/f*d,d]
        rel=observer.data['robot_r1::cam_rel_poses'].reshape(3,7)[slot]
        body=pc@(quaternion_matrix(rel[3:])@np.diag([1.,-1.,-1.])).T+rel[:3]
        world=body@base[:3,:3].T+base[:3,3];local=(world-pose[:3,3])@pose[:3,:3]
        # Tolerance follows asset thickness and a single projected RGB-D pixel.
        centre_body=(pose[:3,3]-base[:3,3])@base[:3,:3]
        pixel=float(np.linalg.norm(centre_body-rel[:3])/f)
        tolerance=.20*float(np.min(extent))+pixel
        nearby=np.all(local>=bounds[0]-2*tolerance,axis=1)&np.all(local<=bounds[1]+2*tolerance,axis=1)
        local=local[nearby]
        if len(local)<4:continue
        distance=tree.query(local)[0];good=distance<tolerance
        geometry_distance=observer.geometry_trees[category].query(local)[0]
        evidence.append(local[good]);tolerances.append(tolerance)
        views.append(dict(camera=camera,nearby_points=len(local),inlier_points=int(good.sum()),
                          inlier_fraction=float(good.mean()),tolerance=tolerance,
                          colour_distance_quantiles=np.quantile(distance,[.25,.5,.75]).tolist(),
                          geometry_distance_quantiles=np.quantile(geometry_distance,[.25,.5,.75]).tolist()))
    if diagnostic is not None:diagnostic.update(views=views)
    if not evidence:return None
    points=np.concatenate(evidence)
    total=sum(view['nearby_points'] for view in views)
    if diagnostic is not None:diagnostic.update(points=len(points),total=total,
        span=None if not len(points) else np.ptp(points[:,:2],axis=0).tolist(),extent=extent.tolist())
    if len(points)<12 or len(points)/total<.75:return None
    span=np.ptp(points[:,:2],axis=0);long=int(np.argmax(extent[:2]));short=1-long
    # Contact requires evidence along the item, rather than a single end/painted band.
    if span[long]<.65*extent[long] or span[short]<.40*extent[short]:return None
    return dict(source='current_rgbd_confirmed_recent_track',points=len(points),span=span.tolist(),
                views=views,refitted=False)
