"""Pinch a raised collision-hull section above the calculator's lower terrace.

This avoids treating a non-antipodal outer corner as a stable parallel-jaw grip.
Only fixed object-local hulls, RGB-D pose and the unchanged native hand are used.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from art_skills import upright,transform
from flat_contact import segment_inside


@lru_cache(maxsize=1)
def hulls():
    with np.load(Path(__file__).parent/'priors/calculator_collision_hulls.npz') as data:
        return [data[k].copy() for k in sorted(data.files)]


def section_interval(point,direction):
    intervals=[]
    for planes in hulls():
        slope=planes[:,:3]@direction;value=planes[:,:3]@point+planes[:,3]
        flat=abs(slope)<1e-9
        if np.any(value[flat]>0.):continue
        positive=slope>1e-9;negative=slope<-1e-9
        upper=float(np.min(-value[positive]/slope[positive])) if positive.any() else np.inf
        lower=float(np.max(-value[negative]/slope[negative])) if negative.any() else -np.inf
        if upper>=lower:intervals.append((lower,upper))
    if not intervals:return None
    return min(a for a,b in intervals),max(b for a,b in intervals)


def raised_section_candidates(ctx,obj):
    bounds=ctx.models['calculator']['bounds'];lo,hi=bounds;centre=bounds.mean(0)
    mesh=ctx.gripper[ctx.arm+'_collision']-ctx.ik.tool
    rays=ctx.gripper[ctx.arm+'_ag']-ctx.ik.tool
    low=rays[abs(rays[:,2]-rays[:,2].min())<1e-4];ray_centre=low.mean(0)
    yaw=float(np.arctan2(obj[1,0],obj[0,0]))
    for z in np.linspace(lo[2]+.55*(hi[2]-lo[2]),hi[2]-.003,5):
        for y in (centre[1],centre[1]-.022,centre[1]+.022):
            point=np.array([0.,y,z]);section=section_interval(point,np.array([1.,0.,0.]))
            if section is None or not .015<=section[1]-section[0]<=.074:continue
            point[0]=(section[0]+section[1])/2
            for reverse in (0.,np.pi):
                rotation=upright(yaw-np.pi/2+reverse)
                target=obj[:3,:3]@point+obj[:3,3]-rotation@ray_centre
                goal=transform(rotation,target);world=mesh@rotation.T+target
                if np.any(world[:,2]<ctx.scene.support_heights(world)+.0008):continue
                local=(world-obj[:3,3])@obj[:3,:3]
                if any(np.any(np.max(local@planes[:,:3].T+planes[:,3],axis=1)<-.0005) for planes in hulls()):continue
                ray_local=(low@rotation.T+target-obj[:3,3])@obj[:3,:3]
                if len(ray_local)<2:continue
                if any(segment_inside(ray_local[0],ray_local[-1],planes,.001) for planes in hulls()):yield goal
