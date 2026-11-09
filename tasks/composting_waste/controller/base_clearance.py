"""Observation-based egress when scan matching exposes insufficient chassis clearance."""
import numpy as np
from manip import pose2,drive,hold,BASE_LINEAR_GAIN

def _restore_once(ctx,reserve=.025):
    initial=ctx.q.copy();poses=ctx.body.fk(initial)[0];start=np.array(ctx.est.base());base0=pose2(*start)
    radii=ctx.collision.radii;ignore=np.where(ctx.base_part,.06,0.)
    required=np.where(ctx.hand,-.012,np.where(ctx.base_part,0.,.03))
    def gaps(full,base):
        points=ctx.collision.centers(ctx.body.fk(full)[0])@base[:3,:3].T+base[:3,3]
        return ctx.scene.contact_clearance(points,radii,ignore_below=ignore)
    initial_gaps=gaps(initial,base0)
    if np.min(initial_gaps[ctx.base_part])>=reserve and ctx.validator()(initial,poses):return True
    floor=np.minimum(initial_gaps,required)
    def safe(xy,full=initial,terminal=False):
        base=pose2(*xy,start[2]);frames=ctx.body.fk(full)[0];clear=gaps(full,base)
        # Strict non-worsening egress, including each individual sphere. No
        # relaxation of the original collision margins at the endpoint.
        if np.any(clear<floor-1e-6):return False
        if terminal:
            return np.min(clear[ctx.base_part])>=reserve and ctx.validator(base=base)(full,frames)
        return ctx.validator(base=base,relax_from=initial,baseline_base=base0)(full,frames)
    candidates=[]
    for radius in (.04,.06,.08,.10,.12):
        for angle in np.arange(16)*np.pi/8:
            xy=start[:2]+radius*np.array([np.cos(angle),np.sin(angle)])
            if not safe(xy,terminal=True):continue
            if all(safe(point) for point in np.linspace(start[:2],xy,int(np.ceil(radius/.005))+1)):
                candidates.append(xy)
        if candidates:break
    if not candidates:
        ctx.event('base_clearance',ok=False,reason='no strictly non-worsening egress',minimum=float(initial_gaps[ctx.base_part].min()));return False
    goal=candidates[0];ctx.event('base_clearance',phase='egress',start=start.tolist(),goal=goal.tolist(),minimum=float(initial_gaps[ctx.base_part].min()))
    motion=drive(ctx,goal,start[2],direct=True,face_travel=False,speed=.15,min_speed=.03,tolerance=.006,yaw_tolerance=.02,max_steps=140)
    while True:
        try:action=next(motion)
        except StopIteration as done:
            if done.value!='arrived':
                full=ctx.q;frames=ctx.body.fk(full)[0]
                if not ctx.validator()(full,frames) or gaps(full,ctx.base_matrix())[ctx.base_part].min()<reserve-.005:return False
            break
        now=np.array(ctx.est.base());local=np.asarray(action[:2])*BASE_LINEAR_GAIN/30.
        predicted=now[:2]+np.array([[np.cos(now[2]),-np.sin(now[2])],[np.sin(now[2]),np.cos(now[2])]])@local
        if not safe(predicted,ctx.q):
            yield ctx.action();ctx.event('base_clearance',ok=False,reason='updated observation invalidates egress');return False
        yield action
    yield from hold(ctx,6)
    full=ctx.q;valid=ctx.validator()(full,ctx.body.fk(full)[0])
    ctx.event('base_clearance',ok=valid,pose=list(ctx.est.base()));return valid


def restore_clearance(ctx,reserve=.025):
    """Recompute a non-worsening egress after a fresh observation invalidates it."""
    for attempt in range(3):
        count=len(ctx.events)
        if (yield from _restore_once(ctx,reserve=reserve)):return True
        new=ctx.events[count:]
        failure=next((e for e in reversed(new) if e.get('event')=='base_clearance' and e.get('ok') is False),None)
        # A completed short egress can also expose an endpoint margin deficit
        # after measured pose/joints update. Recompute from that observed pose;
        # no-route and planning failures are not blindly repeated.
        if failure is None:return False
        if failure.get('reason')!='updated observation invalidates egress' and 'pose' not in failure:return False
        if attempt==2:return False
        yield from hold(ctx,8)
        ctx.event('egress_reanchor',attempt=attempt+1,basis='new permitted observation; unchanged per-sphere non-worsening and strict endpoint checks')
    return False
