"""RGB-D School Stuff task controller using the existing checked planning stack.

Delivered items are action/visual beliefs, never simulator goal predicates.
All movable-object poses enter through Estimator and onboard observations.
"""
import numpy as np
from art_pipeline import localize,look,phase
from art_skills import (ITEMS,tuck,carry,pinch,insert,place_bag,bag_guards,
                        surface_drive,surface_stances,hybrid_bag_slots)
from school_skills import pick_floor,pick_bed
from school_push import push_to_edge,pick_exposed
from school_insertion import insert_flat
from recipient_view import observe_recipient
from manip import hold,pose2


def bounded(ctx,script,actions,label):
    for _ in range(actions):
        try:action=next(script)
        except StopIteration as done:return done.value
        yield action
    script.close()
    ctx.event('stage_budget',stage=label,actions=actions)
    yield from hold(ctx,8)
    return None


def survey(ctx,est,categories):
    found=yield from look(ctx,est,categories,limit=240)
    bag=est.object('tote',max_age=20000)
    if bag is not None:ctx.guard=bag_guards(bag)
    if len(found)==len(categories):return found
    surface=ctx.scene.meta['target_surface'];lo=np.asarray(surface['aabb_lo']);hi=np.asarray(surface['aabb_hi'])
    centre=(lo+hi)/2;views=[]
    for axis in (0,1):
        for sign in (-1,1):
            xy=centre[:2].copy();xy[axis]=(lo[axis] if sign<0 else hi[axis])+sign*.65
            if ctx.scene.base_free(xy):views.append((*xy,float(np.arctan2(*(centre[:2]-xy)[::-1]))))
    views.sort(key=lambda p:np.linalg.norm(np.asarray(p[:2])-est.base()[:2]))
    for view in views[:3]:
        phase(ctx,'survey_stance',stance=list(view))
        if (yield from surface_drive(ctx,view))!='arrived':continue
        yield from hold(ctx,8)
        found.update((yield from look(ctx,est,tuple(k for k in categories if k not in found),limit=180)))
        if len(found)==len(categories):break
    ctx.event('school_inventory',categories=list(found),poses={k:v.tolist() for k,v in found.items()})
    return found


def find_item(ctx,est,category):
    obj=est.refresh(category)
    if obj is not None:return obj
    remembered=est.object(category,max_age=20000)
    if remembered is not None:
        # Use RGB-D memory to plan the approach when a target is outside camera
        # range. Both pick adapters require a fresh observation before contact.
        ctx.event('reacquire_at_approach',category=category,remembered_pose=remembered.tolist())
        return remembered
    if category in ('notebook','folder','pencil'):
        yield from survey(ctx,est,(category,))
    else:
        yield from look(ctx,est,(category,),limit=180)
    return est.object(category,max_age=200)


def observe_bed_item(ctx,est,category,target):
    """Use remembered RGB-D geometry only to reach a fresh full view."""
    yield from tuck(ctx)
    fresh=est.refresh(category,force_refit=True)
    if fresh is not None:return fresh
    stances=surface_stances(ctx,target[:3,3])
    stances.sort(key=lambda p:np.linalg.norm(np.asarray(p[:2])-est.base()[:2])+.3*np.linalg.norm(np.asarray(p[:2])-target[:2,3]))
    for stance in stances[:10]:
        status=yield from surface_drive(ctx,stance)
        if status!='arrived':continue
        yield from hold(ctx,8)
        fresh=est.refresh(category,force_refit=True)
        ctx.event('flat_view_approach',category=category,stance=list(stance),observed=fresh is not None)
        if fresh is not None:return fresh
    return None


def fetch(ctx,est,category,bag,delivered):
    target=yield from find_item(ctx,est,category)
    if target is None:return False
    phase(ctx,'pick_school_item',category=category,pose=target.tolist())
    if category in ('pen','calculator'):
        ok=yield from bounded(ctx,pick_floor(ctx,category,target,refresh=est.refresh),1900,'floor_pick')
    elif category in ('folder','notebook'):
        target=yield from bounded(ctx,observe_bed_item(ctx,est,category,target),1400,'flat_view')
        if target is None:return False
        # Direct pinches did not retain these broad objects in paired native
        # tests. Use thickness pinching after measured edge exposure.
        refresh=lambda key:est.refresh(key,force_refit=True)
        exposed=yield from bounded(ctx,push_to_edge(ctx,category,target,refresh),3600,'push_to_edge')
        ok=exposed is not None and (yield from bounded(ctx,pick_exposed(ctx,category,exposed,refresh),1800,'edge_grasp'))
    else:
        ok=yield from bounded(ctx,pick_bed(ctx,category,target,refresh=est.refresh),1500,'bed_pick')
    if not ok or ctx.held is None:return False
    if not (yield from carry(ctx)):return False
    fresh=est.refresh('tote')
    if fresh is not None:bag=fresh
    ctx.guard=bag_guards(bag)
    if category in ('notebook','folder'):
        ok=yield from bounded(ctx,insert_flat(ctx,category,bag,refresh=est.refresh),2000,'flat_insert')
    else:
        ok=yield from bounded(ctx,insert(ctx,category,bag,len(delivered),refresh=est.refresh,
            observe=lambda recipient:observe_recipient(ctx,est,recipient)),2000,'insert')
    if not ok:return False
    est.clear(category);delivered.append(category)
    ctx.event('delivery_belief',category=category,delivered=list(delivered),
              visual=est.ob.insertion_evidence(category,pose2(*est.base()),bag))
    yield from tuck(ctx)
    return True


def observe_floor_recipient(ctx,est,bag):
    """Get a nearer full silhouette before choosing a narrow grasp corridor."""
    from carried_navigation import checked_carried_drive
    if np.linalg.norm(np.asarray(est.base()[:2])-bag[:2,3])>1.4:
        candidates=[];full=ctx.q.copy();poses=ctx.body.fk(full)[0]
        for radius in (1.05,.90,1.20):
            for angle in np.linspace(-np.pi,np.pi,24,endpoint=False):
                xy=bag[:2,3]+radius*np.array([np.cos(angle),np.sin(angle)])
                if not ctx.scene.base_free(xy):continue
                yaw=float(np.arctan2(*(bag[:2,3]-xy)[::-1]));stance=(*xy,yaw)
                if not ctx.validator(base=pose2(*stance))(full,poses):continue
                route=ctx.scene.path(est.base()[:2],xy,keepout=((bag[:2,3],.65),))
                if route is None:continue
                cost=sum(np.linalg.norm(b-a) for a,b in zip(route,route[1:]))+.3*radius
                candidates.append((cost,stance))
        for _,stance in sorted(candidates,key=lambda x:x[0])[:6]:
            yield ctx.action()
            status=yield from checked_carried_drive(ctx,stance,max_steps=900,keepout=((bag[:2,3],.65),))
            ctx.event('recipient_staging',status=status,stance=list(stance))
            if status!='arrived':continue
            observed=yield from look(ctx,est,('tote',),limit=120,force_refit=True)
            if 'tote' in observed:return observed['tote']
    observed=yield from look(ctx,est,('tote',),limit=240,force_refit=True)
    return observed.get('tote')


def relocate_bag(ctx,est,bag,remaining):
    if ctx.held is not None:return None
    ctx.guard=bag_guards(bag)
    fresh=yield from bounded(ctx,observe_floor_recipient(ctx,est,bag),1800,'recipient_staging')
    if fresh is not None:
        ctx.event('recipient_before_pick',shift=float(np.linalg.norm(fresh[:3,3]-bag[:3,3])),pose=fresh.tolist());bag=fresh
    ctx.guard=np.zeros((0,6))
    if not (yield from bounded(ctx,pick_floor(ctx,'tote',bag,refresh=est.refresh),1700,'bag_pick')):
        ctx.guard=bag_guards(bag);return None
    if not (yield from carry(ctx)):return None
    surface=ctx.scene.meta['target_surface'];centre=(np.asarray(surface['aabb_lo'])+surface['aabb_hi'])/2
    occupied=est.ob.table_obstacles(pose2(*est.base()))
    preferred=np.clip(np.asarray(est.base()[:2]),np.asarray(surface['aabb_lo'][:2])+.25,np.asarray(surface['aabb_hi'][:2])-.25)
    goals=hybrid_bag_slots(ctx,remaining,preferred,occupied_points=occupied)
    ctx.event('bed_slot_preference',preferred=preferred.tolist(),candidate_count=len(goals))
    # Reuse the proven placement routine; only reserve space using RGB-D objects.
    placed=yield from bounded(ctx,place_bag(ctx,objects=[p[:3,3] for p in remaining.values()],
        goals=goals if goals else None),2000,'bag_place')
    if placed is None:return None
    settled=placed.copy()
    bottom=ctx.models['tote']['bounds'][0,2]
    settled[2,3]=float(ctx.scene.support_heights(settled[None,:3,3])[0]-bottom+.003)
    est.bag_on_table=True;est.clear('tote');est.remember('tote',settled,source='executed_bed_placement')
    ctx.guard=bag_guards(settled)
    yield from tuck(ctx);yield from hold(ctx,15)
    fresh=est.refresh('tote')
    if fresh is not None:settled=fresh
    ctx.guard=bag_guards(settled)
    return settled


def retry_tools(ctx,est,bag,delivered):
    """One bounded second pass over missing small-tool delivery beliefs."""
    if ctx.held is not None or not est.bag_on_table:return bag
    for category in ('pencil','pen'):
        # Leave enough native horizon for the existing pick/carry/insert budgets.
        # Delivered items are excluded even if their container obscures them.
        if category in delivered or ctx.steps>17000:continue
        fresh=est.object('tote',max_age=20000)
        if fresh is not None:bag=fresh
        ctx.guard=bag_guards(bag)
        phase(ctx,'retry_school_tool',category=category,delivered=list(delivered))
        ok=yield from fetch(ctx,est,category,bag,delivered)
        ctx.event('tool_retry_result',category=category,ok=bool(ok),held=ctx.held is not None)
        if ctx.held is not None:break
    return bag


def task(ctx,est):
    if not (yield from localize(ctx,est)):return 'localization_failed'
    phase(ctx,'initial_floor_inventory')
    known=yield from look(ctx,est,('tote','pen','calculator'),limit=480)
    bag=est.object('tote',max_age=20000)
    if bag is not None:ctx.guard=bag_guards(bag)
    yield from tuck(ctx)
    missing=tuple(k for k in ('tote',)+ITEMS if k not in known)
    if missing:known.update((yield from survey(ctx,est,missing)))
    bag=est.object('tote',max_age=20000)
    if bag is None:return 'carryall_not_found'
    ctx.guard=bag_guards(bag);delivered=[]
    # Verified calculator/folder floor insertions make bounded prefill viable.
    # Clear floor-route blockers and the broad bed object before reserving a
    # bag footprint. Additional contents may stabilize the lightweight tote;
    # this is a task-order hypothesis, not a change to physical parameters.
    for category in ('calculator','pen','folder'):
        if ctx.steps>10000:break
        phase(ctx,'prefill_floor_recipient',category=category,delivered=list(delivered))
        yield from fetch(ctx,est,category,bag,delivered)
        if ctx.held is not None:return 'held_item_requires_safe_recovery'
        fresh=est.object('tote',max_age=20000)
        if fresh is not None:bag=fresh
    remaining={k:v for k,v in known.items() if k!='tote' and k not in delivered}
    phase(ctx,'transfer_carryall_to_bed',delivered=list(delivered))
    placed=yield from relocate_bag(ctx,est,bag,remaining)
    if ctx.held is not None:return 'held_carryall_requires_safe_recovery'
    if placed is not None:bag=placed
    for category in ('pencil','pen','folder','notebook','calculator'):
        if category in delivered or ctx.steps>19500:continue
        phase(ctx,'remaining_school_item',category=category,delivered=list(delivered))
        yield from fetch(ctx,est,category,bag,delivered)
        if ctx.held is not None:return 'held_item_requires_safe_recovery'
        fresh=est.object('tote',max_age=20000)
        if fresh is not None:bag=fresh
    bag=yield from retry_tools(ctx,est,bag,delivered)
    if ctx.held is not None:return 'held_item_requires_safe_recovery'
    yield from hold(ctx,90)
    ctx.event('school_complete',delivered_beliefs=delivered,bag_on_bed_belief=est.bag_on_table)
    return 'bounded_task_complete'
