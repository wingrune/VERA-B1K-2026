"""School task adapters for the reused checked motion and grasp stack."""
import time
import numpy as np
from art_skills import tucked,plan_pinch,pinch,tuck,pick_surface
from manip import pose2,hold
from carried_navigation import checked_carried_drive


def pick_floor(ctx,category,obj,refresh=None):
    nominal=tucked(ctx);previous=ctx.arm;target=np.asarray(obj)
    bounds=ctx.models[category]['bounds']
    exclusion=.30+float(np.max(np.linalg.norm(bounds[:,:2],axis=1)))
    centre=target[:2,3];candidates=[]
    for radius in ((.62,.75,.50) if category=='tote' else (.43,.50,.60,.70)):
        if radius<exclusion:continue
        for angle in np.linspace(-np.pi,np.pi,16,endpoint=False):
            xy=centre+radius*np.array([np.cos(angle),np.sin(angle)])
            yaw=float(np.arctan2(*(centre-xy)[::-1]));stance=(*map(float,xy),yaw)
            if not ctx.scene.base_free(xy):continue
            path=ctx.scene.path(ctx.est.base()[:2],xy,keepout=((centre,exclusion),))
            if path is None:continue
            length=sum(np.linalg.norm(b-a) for a,b in zip(path,path[1:]))
            candidates.append((length,stance,path))
    candidates.sort(key=lambda row:row[0] if category=='tote' else
                    row[0]+3.*np.linalg.norm(np.asarray(row[1][:2])-centre))
    ctx.event('floor_stances',category=category,candidates=len(candidates),exclusion=exclusion)
    attempts=0;planned=0
    for _,stance,path in candidates[:24]:
        for arm in ('left','right'):
            yield ctx.action()
            ctx.arm=arm;world=pose2(*stance)
            if not ctx.validator(base=world)(nominal,ctx.body.fk(nominal)[0]):continue
            plan=plan_pinch(ctx,category,target,base=world,q=nominal,count=12,candidate_limit=24)
            if plan is None:continue
            planned+=1
            ctx.event('floor_plan',category=category,stance=list(stance),arm=arm,candidate=plan['candidate'])
            # Earlier attempts may have moved the robot. The path above is only
            # a ranking estimate from the initial pose, not an executable route.
            status=yield from checked_carried_drive(ctx,stance,max_steps=1000,keepout=((centre,exclusion),))
            ctx.event('floor_route',category=category,status=status)
            if status!='arrived':continue
            yield from tuck(ctx);yield from hold(ctx,8)
            current=refresh(category) if refresh is not None else target
            if current is None:continue
            attempts+=1
            if (yield from pinch(ctx,category,current,refresh=refresh,preferred_candidate=plan['candidate'])):return True
            yield from tuck(ctx)
            if attempts>=3:
                ctx.arm=previous;return False
    ctx.event('floor_pick',category=category,ok=False,reason='bounded stance/IK search exhausted',planned=planned,attempts=attempts)
    ctx.arm=previous;return False


def pick_bed(ctx,category,obj,refresh=None):
    return (yield from pick_surface(ctx,category,obj,refresh=refresh,allow_fallback=False))
