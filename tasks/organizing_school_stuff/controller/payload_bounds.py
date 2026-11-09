"""Conservative oriented bounds against recipient boxes using separating axes."""
import numpy as np


def recipient_bounds_clear(bounds, pose, guards, clearance=.015):
    """Inflate each object half-extent by clearance, then test every guard box.

    The inflated box contains the object's Euclidean clearance neighborhood.
    Unlike a sampled lattice with radius, it does not add a second discretization
    allowance to exact asset bounds. All 15 OBB separating axes are checked.
    """
    if not len(guards):return True
    frame=getattr(guards,'frame',np.eye(4))
    rotation=frame[:3,:3].T@pose[:3,:3]
    centre=frame[:3,:3].T@(pose[:3,3]-frame[:3,3])+rotation@bounds.mean(0)
    half=(bounds[1]-bounds[0])/2+clearance
    identity=np.eye(3)
    axes=np.concatenate([identity,rotation.T,np.cross(identity[:,None],rotation.T[None]).reshape(-1,3)])
    axes=axes[np.linalg.norm(axes,axis=1)>1e-8]
    guard_centre=(guards[:,:3]+guards[:,3:])/2
    guard_half=(guards[:,3:]-guards[:,:3])/2
    separation=abs((centre-guard_centre)@axes.T)
    extent=guard_half@abs(axes).T+(abs(axes@rotation)@half)[None]
    return bool(np.all(np.any(separation>extent+1e-8,axis=1)))
