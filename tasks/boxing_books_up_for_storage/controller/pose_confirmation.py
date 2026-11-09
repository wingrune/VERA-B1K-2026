"""Verify a recent stationary tool track with current RGB-D without refitting an occluded silhouette."""
import numpy as np
import cv2


def confirm_tool(observer,category,base,pose,diagnostic=None,allow_flat_edge=False):
    from art_perception import CAMERAS,quaternion_matrix
    hue,_=observer.colours[category]
    if category=='tote':return None
    bounds=observer.models[category]['bounds'];extent=bounds[1]-bounds[0]
    tree=observer.trees[category];evidence=[];views=[];tolerances=[]
    for camera,(link,slot,focal) in CAMERAS.items():
        prefix=f'robot_r1::robot_r1:{link}:Camera:0::'
        rgb=observer.data[prefix+'rgb'][...,:3].astype(np.uint8);depth=observer.data[prefix+'depth_linear']
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);delta=None if hue is None else abs(hsv[...,0].astype(float)-hue)
        # Hue is usable only above a few RGB quantization levels; this admits shadowed
        # saturated coatings without inferring a new object from a dark furniture patch.
        chroma=np.max(rgb,axis=-1).astype(int)-np.min(rgb,axis=-1).astype(int)
        if hue is None:
            mask=((hsv[...,1]<85)&(hsv[...,2]>100)) if observer.signatures[category]['median_value']>180 else ((hsv[...,1]<110)&(hsv[...,2]<195))
        else:
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
    if span[long]<.65*extent[long] or span[short]<.40*extent[short]:
        if not allow_flat_edge or category not in ('notebook','folder'):return None
        # At a horizontal edge approach the wrist sees a thin front strip.
        # Confirm only a recent pregrasp pose, with colour/geometry evidence
        # along most of an edge plus model-to-depth agreement. Never use this
        # branch to update a pose during pushing or discover a new object.
        if len(points)<80 or len(points)/total<.95:return None
        broad_partial=(span[long]>=.60*extent[long] and span[short]>=.90*extent[short])
        if span[long]<.65*extent[long] and not broad_partial:return None
        # A shallow edge approach can expose nearly the whole short edge while
        # occluding the last few millimetres of the long dimension. Require
        # strong two-axis coverage for this case, plus current depth agreement.
        geometry_limit=.005 if broad_partial else .004
        if min(view['geometry_distance_quantiles'][2] for view in views)>geometry_limit:return None
        agreements={view['camera']:observer.depth_agreement(category,view['camera'],base,pose,
                     tolerance=.008,occlusion=.006) for view in views}
        if not any(x['points']>=80 and x['agreement']>=.65 for x in agreements.values()):return None
        return dict(source='current_rgbd_confirmed_recent_flat_edge',points=len(points),span=span.tolist(),
                    views=views,depth=agreements,refitted=False)
    return dict(source='current_rgbd_confirmed_recent_track',points=len(points),span=span.tolist(),
                views=views,refitted=False)
