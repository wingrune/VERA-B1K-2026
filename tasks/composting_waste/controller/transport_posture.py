"""Select a carry posture that clears the next full-body swept route, not just its start."""
import numpy as np
from navigation_geometry import planning_scene
from manip import pose2,TALL_TRUNK,move_to,hold
from compost_skills import transform,upright

def prepare_route(ctx,stance,height):
    if ctx.held is None:return True
    from base_clearance import restore_clearance
    if not (yield from restore_clearance(ctx)):return False
    arm,category,relation=ctx.held;ctx.arm=arm;start=np.array(ctx.est.base());yaw=float(stance[2])
    current_poses=ctx.body.fk(ctx.q)[0]
    # Only the active arm changes during posture search. The other arm, torso,
    # head and chassis must clear the chosen route for every candidate.
    fixed=np.array([not str(name).startswith(arm+'_') for name in ctx.collision.names])
    fixed_full=ctx.q.copy();fixed_full[ctx.ik.columns[:4]]=TALL_TRUNK
    current_poses=ctx.body.fk(fixed_full)[0]
    chassis_centres=ctx.collision.centers(current_poses)[fixed];radii=ctx.collision.radii[fixed]
    fixed_base=ctx.base_part[fixed]
    margins=np.where(fixed_base,.020,.035)
    ignore=np.where(fixed_base,.06,0.)
    def chassis_clear(xy):
        base=pose2(*xy,yaw);points=chassis_centres@base[:3,:3].T+base[:3,3]
        if np.any(ctx.scene.contact_clearance(points,radii,ignore_below=ignore)<margins):return False
        if len(ctx.guard):
            frame=getattr(ctx.guard,'frame',None);local=points if frame is None else (points-frame[:3,3])@frame[:3,:3]
            nearest=np.clip(local[None],ctx.guard[:,None,:3],ctx.guard[:,None,3:])
            if np.min(np.linalg.norm(local[None]-nearest,axis=2)-radii[None])<.025:return False
        return True
    from checked_route import path as checked_path
    # The inherited raster planner accepts footprint OR exact geometry. Here
    # every connector and A* edge must pass the actual chassis geometry.
    path=checked_path(start[:2],stance[:2],chassis_clear)
    if path is None:
        ctx.event('route_carry',ok=False,reason='no fixed-body route',goal=list(stance))
        return False
    samples=[]
    for a,b in zip(path,path[1:]):samples.extend(np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/.01))+1)))
    turn=(yaw-start[2]+np.pi)%(2*np.pi)-np.pi
    bases=[pose2(*start[:2],a) for a in np.linspace(start[2],start[2]+turn,max(2,int(np.ceil(abs(turn)/.04))+1))]
    bases += [pose2(*xy,yaw) for xy in samples]
    base0=ctx.base_matrix();side=1. if arm=='left' else -1.
    points=((.35,.20*side),(.25,.12*side),(.25,0.),(.20,.18*side),(.30,-.08*side))
    for angle in (0.,np.pi/2,-np.pi/2,np.pi):
      for xy in points:
        yield ctx.action()
        obj=base0@transform(upright(angle),[xy[0],xy[1],height-.012])
        tip=np.linalg.inv(base0)@obj@np.linalg.inv(relation)
        locked={i:float(v) for i,v in enumerate(TALL_TRUNK)}
        found=ctx.ik.solve(ctx.q,tip[:3,3],tip[:3,:3],count=10,validator=ctx.validator(max_load=1.05),all_solutions=True,locked=locked)
        for solution in found:
            full=ctx.q.copy();full[ctx.ik.columns]=solution;poses=ctx.body.fk(full)[0]
            if not all(ctx.validator(base=base,max_load=1.05)(full,poses) for base in bases):continue
            status=yield from move_to(ctx,solution,label='route_carry',rate=.025,max_steps=450)
            if status!='done':continue
            yield from hold(ctx,8)
            # Measured post-servo configuration is checked before any base motion.
            full=ctx.q;poses=ctx.body.fk(full)[0]
            if not all(ctx.validator(base=base,max_load=1.05)(full,poses) for base in bases):continue
            retained=float(np.sum(ctx.fingers()))>.003
            ctx.prepared_route=dict(start=np.array(ctx.est.base()),goal=np.array(stance),path=path)
            ctx.event('route_carry',planning_reserve=.025,ok=retained,arm=arm,xy=list(xy),object_yaw=angle,height=height,waypoints=[p.tolist() for p in path])
            if not retained:ctx.held=None
            return retained
    ctx.event('route_carry',ok=False,arm=arm,height=height,reason='no posture clears swept route')
    return False
