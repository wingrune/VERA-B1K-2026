"""Protect an observed manipulation target during free-space hand approaches."""
from contextlib import contextmanager
import numpy as np


def vertices(bounds):
    return np.array(np.meshgrid(*zip(*bounds),indexing='ij')).reshape(3,-1).T


@contextmanager
def protect_object(ctx,category,obj):
    previous=ctx.guard;boxes=[]
    frame=getattr(previous,'frame',None)
    for box in previous:
        if frame is None:boxes.append(np.asarray(box).copy())
        else:
            world=vertices(np.asarray(box).reshape(2,3))@frame[:3,:3].T+frame[:3,3]
            boxes.append(np.r_[world.min(0),world.max(0)])
    points=vertices(ctx.models[category]['bounds'])@obj[:3,:3].T+obj[:3,3]
    boxes.append(np.r_[points.min(0),points.max(0)])
    # World AABBs conservatively contain every previous oriented recipient
    # box as well as the target. Existing guard margin and swept checks apply.
    ctx.guard=np.asarray(boxes)
    try:yield
    finally:ctx.guard=previous
