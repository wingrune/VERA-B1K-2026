"""Conservative upright neutral appliance cues from current RGB-D.

These are route hints, not verified object identities or manipulation targets.
No instance positions or semantic simulator observations are used.
"""
import cv2
import numpy as np
from perception import pointcloud


def fridge_candidates(camera,level,base,counter=None,origin=None):
    hsv=cv2.cvtColor(camera['rgb'],cv2.COLOR_RGB2HSV)
    mask=((hsv[...,1]<45)&(hsv[...,2]>65)&(hsv[...,2]<215)).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    # Color alone joins the silver appliance to gray walls and ceiling.
    # Split at depth discontinuities and retain locally vertical surfaces.
    points,_,pixels=pointcloud(camera)
    leveled=points@level[:3,:3].T+level[:3,3]
    grid=np.full(camera['depth'].shape+(3,),np.nan)
    grid[pixels[:,1],pixels[:,0]]=leveled
    dy=grid[1:,:-1]-grid[:-1,:-1];dx=grid[:-1,1:]-grid[:-1,:-1]
    normals=np.cross(dy,dx);length=np.linalg.norm(normals,axis=-1)
    vertical=np.zeros(mask.shape,bool)
    vertical[:-1,:-1]=(length>1e-8)&(abs(normals[...,2])<.20*length)&(np.linalg.norm(dx,axis=-1)<.06)&(np.linalg.norm(dy,axis=-1)<.06)
    mask &= vertical.astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    proposals=[]
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]<80:continue
        fill=float(stats[i,cv2.CC_STAT_AREA]/max(1,stats[i,cv2.CC_STAT_WIDTH]*stats[i,cv2.CC_STAT_HEIGHT]))
        # A door surround is a thin frame around a different-colored panel;
        # the appliance cue should occupy most of its rectangular silhouette.
        if fill<.55:continue
        points,_,_=pointcloud(camera,labels==i)
        if len(points)<80:continue
        leveled=points@level[:3,:3].T+level[:3,3]
        keep=(leveled[:,2]>.85)&(leveled[:,2]<2.15)
        points=points[keep];leveled=leveled[keep]
        if len(points)<60 or np.ptp(leveled[:,2])<.42:continue
        center=np.median(points,axis=0)
        points=points[np.linalg.norm(points-center,axis=1)<1.15]
        if len(points)<60:continue
        _,_,axes=np.linalg.svd(points-points.mean(0),full_matrices=False)
        normal=level[:3,:3]@axes[-1]
        if abs(normal[2])>.20:continue
        residual=np.abs((points-points.mean(0))@axes[-1])
        if np.quantile(residual,.8)>.025:continue
        leveled=points@level[:3,:3].T+level[:3,3]
        width=float(np.max(np.ptp(leveled[:,:2],axis=0)))
        if not .20<width<1.1:continue
        # A plain gray wall is not an appliance cue. Require a compact
        # vertical dark handle/trim region on the same observed surface.
        dark=(hsv[...,2]<85)&(hsv[...,1]<100)
        dark_points,_,dark_pixels=pointcloud(camera,dark)
        dark_level=dark_points@level[:3,:3].T+level[:3,3]
        lo,hi=leveled.min(0),leveled.max(0)
        near=(np.abs((dark_points-points.mean(0))@axes[-1])<.06)
        near &= np.all((dark_level>=lo-.04)&(dark_level<=hi+.04),axis=1)
        dark_mask=np.zeros(mask.shape,np.uint8)
        dark_mask[dark_pixels[near,1],dark_pixels[near,0]]=1
        count,dark_labels,_,_=cv2.connectedComponentsWithStats(dark_mask)
        handle=False
        for j in range(1,count):
            patch,_,_=pointcloud(camera,dark_labels==j)
            if len(patch)<8:continue
            patch=patch@level[:3,:3].T+level[:3,3]
            span=np.ptp(patch,axis=0)
            if .12<span[2]<.65 and max(span[:2])<.10:
                handle=True;break
        if not handle:continue
        position=base[:3,:3]@np.median(points,axis=0)+base[:3,3]
        if counter is not None:
            if np.linalg.norm(position[:2]-counter[:2])>3.:continue
            direction=counter[:2]-origin
            direction/=max(np.linalg.norm(direction),1e-6)
            if (position[:2]-counter[:2])@direction<.10:continue
        proposals.append(dict(position=position,width=width,points=len(points),
                              residual=float(np.quantile(residual,.8)),handle_region=True,image_fill=fill))
    return sorted(proposals,key=lambda p:(-p['points'],p['residual']))
