"""Canonical flat-object contact checks with a bounded RGB-D pose envelope.

Only object-local, fixed asset collision hulls and estimated poses are used.
The envelope is a development hypothesis, not a statistical confidence bound.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np


@lru_cache(maxsize=1)
def collision_hulls():
    with np.load(Path(__file__).parent/'priors/flat_collision_hulls.npz') as data:
        return {category: [data[key].copy() for key in data.files
                           if key.startswith(category+'_planes_')]
                for category in ('folder','notebook')}


def segment_inside(a, b, planes, margins):
    """Clip a segment against inward-offset convex halfspaces."""
    value=planes[:,:3]@a+planes[:,3]+margins
    slope=planes[:,:3]@(b-a)
    flat=abs(slope)<1e-10
    if np.any(value[flat]>0.):return False
    positive=slope>1e-10;negative=slope<-1e-10
    upper=min(1.,float(np.min(-value[positive]/slope[positive]))) if positive.any() else 1.
    lower=max(0.,float(np.max(-value[negative]/slope[negative]))) if negative.any() else 0.
    return upper>=lower


def robust_edge_rays(ctx, category, obj, goal):
    """Require native AG-ray contact under XY4mm/Z2mm/yaw3deg perturbations.

    Both half-turn pose labels are checked because these rectangular silhouettes
    are ambiguous. Native collision geometry can be slightly asymmetric.
    Translational uncertainty is handled conservatively by eroding each hull.
    """
    hulls=collision_hulls()[category]
    rays=ctx.gripper[ctx.arm+'_ag']-ctx.ik.tool
    world=rays@goal[:3,:3].T+goal[:3,3]
    centre=ctx.models[category]['bounds'].mean(0)
    centre_world=obj[:3,:3]@centre+obj[:3,3]
    for half_turn in (0.,np.pi):
        for angle in (-np.deg2rad(3.),0.,np.deg2rad(3.)):
            c,s=np.cos(half_turn+angle),np.sin(half_turn+angle)
            rotation=obj[:3,:3]@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
            local=(world-centre_world)@rotation+centre
            contact=False
            for planes in hulls:
                margin=.001+.004*np.linalg.norm(planes[:,:2],axis=1)+.002*abs(planes[:,2])
                if any(segment_inside(a,b,planes,margin) for a in local[:2] for b in local[2:]):
                    contact=True;break
            if not contact:return False
    return True


def open_gripper_clear(ctx,category,obj,goal):
    """The open jaw envelope must remain outside both flat-pose hull labels."""
    mesh=ctx.gripper[ctx.arm+'_collision']-ctx.ik.tool
    world=mesh@goal[:3,:3].T+goal[:3,3]
    centre=ctx.models[category]['bounds'].mean(0)
    centre_world=obj[:3,:3]@centre+obj[:3,3]
    for sign in (1.,-1.):
        rotation=obj[:3,:3]@np.diag([sign,sign,1.])
        local=(world-centre_world)@rotation+centre
        for planes in collision_hulls()[category]:
            if np.any(np.max(local@planes[:,:3].T+planes[:,3],axis=1)<-.001):return False
    return True
