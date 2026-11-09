"""Conservative route-grid reserve and observed movable-recipient occupancy."""
import copy
import numpy as np
from scipy.ndimage import distance_transform_edt

def planning_scene(ctx,reserve=.025):
    scene=copy.copy(ctx.scene)
    scene.chassis_distance=ctx.scene.chassis_distance-reserve
    movable=ctx.scene.chassis_movable_distance.copy()
    if len(ctx.guard):
        gx=scene.x0+(np.arange(scene.shape[0])+.5)*scene.res
        gy=scene.y0+(np.arange(scene.shape[1])+.5)*scene.res
        x,y=np.meshgrid(gx,gy,indexing='ij');points=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
        frame=getattr(ctx.guard,'frame',None)
        local=points if frame is None else (points-frame[:3,3])@frame[:3,:3]
        mask=np.zeros(x.size,bool)
        for box in ctx.guard.boxes if hasattr(ctx.guard,'boxes') else ctx.guard:
            top=box[5]+(0. if frame is None else frame[2,3]);bottom=box[2]+(0. if frame is None else frame[2,3])
            if top<.06 or bottom>.43:continue
            mask|=np.all(local[:,:2]>=box[:2],axis=1)&np.all(local[:,:2]<=box[3:5],axis=1)
        if mask.any():movable=np.minimum(movable,distance_transform_edt(~mask.reshape(scene.shape))*scene.res)
    scene.chassis_movable_distance=movable-reserve
    return scene
