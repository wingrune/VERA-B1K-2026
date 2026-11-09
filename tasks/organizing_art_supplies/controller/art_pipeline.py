"""Bounded, observation-driven task state machine expressed as per-step generators."""
import numpy as np
from manip import hold,drive,pose2
from art_skills import (floor_stance,pinch,tuck,carry,place_bag,pick_surface,insert,bag_guards,ITEMS,reorient_on_surface,surface_drive,hybrid_bag_slots,carried_drive)
from recipient_view import observe_recipient
from recipient_access import accessible_bag_slots


def phase(ctx, name, **kwargs):ctx.event('phase',phase=name,**kwargs)


def tool_order(ctx,poses):
    """Handle geometrically constrained supplies before open tabletop items.

    Canonical perimeter probes just above each item measure its available approach
    corridor using the existing static and observed-recipient collision geometry.
    Physical stance, IK and navigation validation remain in pick_surface.
    """
    rows=[]
    for category,obj in poses.items():
        bounds=ctx.models[category]['bounds'];extent=bounds[1]-bounds[0]
        x,y=np.meshgrid(bounds[:,0],bounds[:,1],indexing='ij')
        points=np.c_[x.ravel(),y.ravel(),np.full(4,bounds[1,2])]
        points=points@obj[:3,:3].T+obj[:3,3];points[:,2]+=.03
        gap=float(np.min(ctx.scene.contact_clearance(points,np.zeros(len(points)))))
        guard=ctx.guard
        if len(guard):
            frame=getattr(guard,'frame',None)
            local=points if frame is None else (points-frame[:3,3])@frame[:3,:3]
            nearest=np.clip(local[None],guard[:,None,:3],guard[:,None,3:])
            gap=min(gap,float(np.min(np.linalg.norm(local[None]-nearest,axis=2))))
        rows.append((gap,float(extent[2]),category))
    rows.sort()
    order=[category for _,_,category in rows]
    ctx.event('tool_order',order=order,approach_clearances={category:gap for gap,_,category in rows})
    return order


def localize(ctx, est):
    phase(ctx,'localize');scans=[];start_yaw=float(est.odom[2])
    for n in range(360):
        if n%30==0:scans.append((est.ob.scan(),float(est.odom[2]-start_yaw)))
        yield ctx.action(base=(0.,0.,.58))
    yield from hold(ctx,10)
    pose,score,margin=est.ob.loc.localize(scans)
    ctx.event('localize',ok=pose is not None,score=score,margin=margin)
    if pose is None or score<.35:return False
    pose[2]+=float(est.odom[2]-start_yaw);est.ob.loc.start_tracking(pose)
    return True


def look(ctx, est, categories, limit=180):
    phase(ctx,'observe',categories=list(categories));found={}
    for n in range(limit):
        if n%30==0:
            for category in categories:
                obj=est.refresh(category)
                if obj is not None:found[category]=obj
            if len(found)==len(categories):break
        yield ctx.action(base=(0.,0.,.42))
    yield from hold(ctx,8)
    return found


def search_bag(ctx, est):
    yield from look(ctx,est,('tote',),limit=480)
    bag=est.object('tote',max_age=400)
    if bag is not None:return bag
    floor=ctx.scene.meta['scope']['floor.n.01_1'];lo=np.array(floor['aabb_lo']);hi=np.array(floor['aabb_hi'])
    candidates=[]
    for x in np.linspace(lo[0]+.6,hi[0]-.6,5):
        for y in np.linspace(lo[1]+.6,hi[1]-.6,5):
            if ctx.scene.base_free((x,y)):
                candidates.append((float(x),float(y)))
    candidates.sort(key=lambda p:np.linalg.norm(np.array(p)-ctx.est.base()[:2]))
    visited=[]
    for xy in candidates:
        if any(np.linalg.norm(np.array(xy)-p)<.8 for p in visited):continue
        visited.append(np.array(xy))
        if len(visited)>4:break
        status=yield from drive(ctx,xy,ctx.est.base()[2],max_steps=700)
        if status!='arrived':continue
        yield from look(ctx,est,('tote',),limit=480)
        bag=est.object('tote',max_age=400)
        if bag is not None:return bag
    return None


def inventory(ctx,est):
    """Find supply poses from onboard views; these become observation history, never initial-state priors."""
    found={};desk=ctx.scene.meta['scope']['desk.n.01_1'];centre=(np.asarray(desk['aabb_lo'])+desk['aabb_hi'])/2
    for attempt in range(3):
        phase(ctx,'inventory',attempt=attempt)
        view=(float(centre[0]+(0.,.25,-.25)[attempt]),float(desk['aabb_hi'][1]+(.65,.46,.28)[attempt]),-np.pi/2)
        status=yield from surface_drive(ctx,view)
        if status!='arrived':continue
        yield from hold(ctx,8);info=est.refine_desk()
        if info is not None:ctx.event('desk_alignment',**info)
        for category in ITEMS:
            obj=est.refresh(category)
            if obj is not None:found[category]=obj
        if len(found)==4:break
        yield from look(ctx,est,tuple(k for k in ITEMS if k not in found),limit=240)
        for category in ITEMS:
            obj=est.object(category,max_age=400)
            if obj is not None:found[category]=obj
        if len(found)==4:break
    # A floor recipient can block every central survey stance. Expand the survey
    # along the known desk edge before treating an unreachable view as missing tools.
    # Existing routes and whole-body / recipient collision checks remain mandatory.
    if len(found)<len(ITEMS):
        lo=np.asarray(desk['aabb_lo']);hi=np.asarray(desk['aabb_hi'])
        views=[(float(x),float(hi[1]+.80),-np.pi/2)
               for x in np.linspace(lo[0]+.15,hi[0]-.15,7)]
        views.sort(key=lambda view:np.linalg.norm(np.asarray(view[:2])-ctx.est.base()[:2]))
        reached=0
        for view in views:
            if reached>=2:break
            phase(ctx,'inventory_reobserve',stance=list(view))
            status=yield from surface_drive(ctx,view)
            ctx.event('inventory_view_route',status=status,stance=list(view))
            if status!='arrived':continue
            reached+=1
            yield from hold(ctx,8);info=est.refine_desk()
            if info is not None:ctx.event('desk_alignment',**info)
            for category in ITEMS:
                obj=est.refresh(category)
                if obj is not None:found[category]=obj
            if len(found)==len(ITEMS):break
        if not found:
            # Even when every survey route is blocked, the current camera may see
            # the supplies. These coarse observations are refreshed before contact.
            for category in ITEMS:
                obj=est.refresh(category)
                if obj is not None:found[category]=obj
    ctx.event('inventory',categories=list(found),complete=len(found)==4)
    return found


def fetch_and_insert(ctx,est,category,bag,delivered,initial=None):
    """Bounded pickup/reposition/regrasp, refreshed recipient and checked release."""
    for attempt in range(3):
        if ctx.steps>8400:return False
        phase(ctx,'find_tool',category=category,attempt=attempt)
        info=est.refine_desk()
        if info is not None:ctx.event('desk_alignment',**info)
        target=est.refresh(category)
        if target is None and attempt==0:target=initial
        if target is None:
            yield from look(ctx,est,(category,),limit=180);target=est.object(category,max_age=300)
        if target is None:continue
        phase(ctx,'pick_tool',category=category,estimated_pose=target.tolist())
        if not (yield from pick_surface(ctx,category,target,refresh=est.refresh)):
            est.clear(category);continue
        if ctx.requires_regrasp:
            occupied=[]
            for other in ITEMS:
                if other==category or other in delivered:continue
                obj=est.refresh(other)
                if obj is not None:occupied.append(obj[:3,3])
            phase(ctx,'reorient_tool',category=category)
            if (yield from reorient_on_surface(ctx,occupied)) is None:return False
            est.clear(category);yield from tuck(ctx);continue
        if not (yield from carry(ctx)):return False
        fresh=est.refresh('tote')
        if fresh is not None:bag=fresh
        else:
            remembered=est.object('tote',max_age=10000)
            if remembered is not None:bag=remembered
        ctx.guard=bag_guards(bag)
        phase(ctx,'insert_into_table_bag' if est.bag_on_table else 'insert_into_floor_bag',category=category)
        if not (yield from insert(ctx,category,bag,len(delivered),refresh=est.refresh,
                                observe=lambda recipient:observe_recipient(ctx,est,recipient))):return False
        est.clear(category);yield from tuck(ctx)
        evidence=est.ob.insertion_evidence(category,pose2(*est.base()),bag)
        # Raw coloured pixels also occur on wooden floors and furniture. They are logged
        # as weak evidence, rather than overturning a completed geometric release.
        ctx.event('visual_insertion',category=category,**evidence)
        delivered.append(category);return True
    return False


def transfer_one_item_bag(ctx,est,known,first,preferred):
    ctx.guard=np.zeros((0,6));failed=[]
    for attempt in range(4):
        phase(ctx,'find_one_item_carryall',attempt=attempt)
        bag=yield from search_bag(ctx,est)
        if bag is None:continue
        stance=floor_stance(ctx,bag,exclude=failed)
        if stance is None:continue
        ctx.arm=stance['arm'];failed.append(stance['base'])
        status=yield from drive(ctx,stance['base'][:2],stance['base'][2],tolerance=.015,keepout=((bag[:2,3],.48),),max_steps=1200)
        if status!='arrived':continue
        bag=est.refresh('tote')
        if bag is None:continue
        if not (yield from pinch(ctx,'tote',bag,refresh=est.refresh)):
            yield from tuck(ctx);est.clear('tote');continue
        phase(ctx,'carry_one_item_carryall')
        if not (yield from carry(ctx)):return None
        desk=ctx.scene.meta['scope']['desk.n.01_1'];centre=(np.asarray(desk['aabb_lo'])+desk['aabb_hi'])/2
        survey=(float(centre[0]),float(desk['aabb_hi'][1]+.65),-np.pi/2)
        yield from carried_drive(ctx,survey,max_steps=1000)
        remaining={k:v for k,v in known.items() if k!=first}
        for category in ITEMS:
            if category==first:continue
            obj=est.refresh(category)
            if obj is not None:remaining[category]=obj
        occupied=est.ob.table_obstacles(pose2(*est.base()))
        goals=accessible_bag_slots(ctx,remaining,preferred,occupied_points=occupied)
        ctx.event('reserved_table_footprint',candidates=len(goals),observed_objects=list(remaining),unclassified_points=len(occupied))
        phase(ctx,'place_one_item_carryall')
        placed=yield from place_bag(ctx,goals=goals)
        if placed is None:return None
        # Protect the newly placed recipient before either empty arm is parked.
        # The commanded upright release settles vertically onto the known support;
        # this prediction uses the executed action and canonical geometry only.
        settled=placed.copy();settled[2,3]=float(desk['aabb_hi'][2]-ctx.models['tote']['bounds'][0,2]+.003)
        ctx.guard=bag_guards(settled)
        est.bag_on_table=True;est.clear('tote')
        est.remember('tote',settled,source='executed_table_placement_prediction')
        yield from tuck(ctx)
        yield from look(ctx,est,('tote',),limit=150)
        bag=est.object('tote',max_age=200)
        if bag is not None:ctx.guard=bag_guards(bag)
        return bag
    return None


def task(ctx,est):
    if not (yield from localize(ctx,est)):return 'localization_failed'
    yield from tuck(ctx);bag=yield from search_bag(ctx,est)
    if bag is None:return 'carryall_not_found'
    ctx.guard=bag_guards(bag);known=yield from inventory(ctx,est)
    order=tool_order(ctx,known)
    if not order:return 'supplies_not_found'
    delivered=[];first=None
    desk=ctx.scene.meta['scope']['desk.n.01_1']
    preferred=(np.asarray(desk['aabb_lo'])[:2]+np.asarray(desk['aabb_hi'])[:2])/2
    occupied=est.ob.table_obstacles(pose2(*est.base()))
    empty_slots=hybrid_bag_slots(ctx,known,preferred,occupied_points=occupied)
    # The baseline makes room with a first floor insertion. If RGB-D already shows
    # room for an empty recipient without covering any supply, establish the table
    # recipient first and avoid transporting contents or dropping into a floor bag.
    if len(known)==len(ITEMS) and empty_slots:
        phase(ctx,'empty_recipient_first',free_footprints=len(empty_slots))
        bag=yield from transfer_one_item_bag(ctx,est,known,None,preferred)
        if bag is None:return 'empty_bag_transfer_failed'
    else:
        phase(ctx,'hybrid_first_tool',free_footprints=len(empty_slots))
        for category in order[:3]:
            if (yield from fetch_and_insert(ctx,est,category,bag,delivered,initial=known[category])):
                first=category;break
            if ctx.held is not None:return 'first_tool_unreleased:'+category
        if first is None:return 'first_tool_failed'
        bag=yield from transfer_one_item_bag(ctx,est,known,first,known[first][:2,3])
        if bag is None:return 'one_item_bag_transfer_failed'
    remaining={category:obj for category,obj in known.items() if category!=first}
    for category in tool_order(ctx,remaining):
        if not (yield from fetch_and_insert(ctx,est,category,bag,delivered,initial=known.get(category))):
            ctx.event('tool_attempts_exhausted',category=category)
            if ctx.held is not None:return 'table_tool_unreleased:'+category
            continue
        fresh=est.refresh('tote')
        if fresh is not None:bag=fresh;ctx.guard=bag_guards(bag)
    phase(ctx,'final_inspection',delivered=delivered)
    yield from look(ctx,est,('tote',),limit=120)
    for category in ITEMS:ctx.event('final_visual_insertion',category=category,**est.ob.insertion_evidence(category,pose2(*est.base()),bag))
    return 'hybrid_actions_completed_visual_goals_unconfirmed'

