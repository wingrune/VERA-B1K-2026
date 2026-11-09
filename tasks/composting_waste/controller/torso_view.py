"""Use the existing checked torso joint motion when chassis rotation is blocked."""
import numpy as np
from manip import move_to

def observe_with_torso(ctx, est):
    from compost_pipeline import observe_bin
    known=est.object('ashcan',max_age=20000)
    if known is not None:
        points=[known[:2,3]]
    elif getattr(ctx.guard,'frame',None) is None:
        points=[(box[:2]+box[3:5])/2 for box in ctx.guard]
        points.sort(key=lambda point:np.linalg.norm(point-np.asarray(est.base())[:2]))
    else:
        points=[ctx.guard.frame[:2,3]]
    for point in points[:2]:
        base=np.asarray(est.base());bearing=float(np.arctan2(point[1]-base[1],point[0]-base[0]))
        relative=(bearing-base[2]+np.pi)%(2*np.pi)-np.pi
        targets=[]
        for offset in (0.,2*np.pi,-2*np.pi):
            target=float(np.clip(relative+offset,ctx.ik.lower[3]+.02,ctx.ik.upper[3]-.02))
            error=abs((base[2]+target-bearing+np.pi)%(2*np.pi)-np.pi)
            if error<.20 and not any(abs(target-old)<.05 for old in targets):targets.append(target)
        targets.sort(key=lambda target:abs(target-ctx.chain[3]))
        for target in targets:
            goal=ctx.chain.copy();goal[3]=target
            status=yield from move_to(ctx,goal,rate=.022,max_steps=250,label='recipient_torso_view',max_load=1.10)
            ctx.event('recipient_torso_view',status=status,torso_yaw=target,bearing=bearing)
            if status!='done':continue
            recipient=yield from observe_bin(ctx,est)
            if recipient is not None:return recipient
    return None


def prefer_stationary_chassis(ctx):
    """Coarse observed floor guards near the chassis turning envelope warrant a torso view first."""
    if not len(ctx.guard):return False
    poses=ctx.body.fk(ctx.q)[0];centres=ctx.collision.centers(poses)[ctx.base_part]
    radius=float(np.max(np.linalg.norm(centres[:,:2],axis=1)+ctx.collision.radii[ctx.base_part]))
    point=ctx.base_matrix()[:3,3];frame=getattr(ctx.guard,'frame',None)
    if frame is not None:point=(point-frame[:3,3])@frame[:3,:3]
    boxes=ctx.guard.boxes if hasattr(ctx.guard,'boxes') else ctx.guard
    near=np.clip(point[:2],boxes[:,:2],boxes[:,3:5])
    # Coarse depth components show only visible surfaces. Reserve8cm for
    # unseen geometry before committing the chassis to a complete rotation.
    return bool(np.any(np.linalg.norm(near-point[:2],axis=1)<radius+.08))
