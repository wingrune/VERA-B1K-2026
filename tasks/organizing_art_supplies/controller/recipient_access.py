"""Rank table recipient footprints against cached static-checked approaches to remaining supplies."""
import numpy as np
from art_skills import tucked,plan_pinch,surface_stances,bag_guards,hybrid_bag_slots
from manip import pose2


def accessible_bag_slots(ctx,remaining,preferred_xy,occupied_points=()):
    goals=hybrid_bag_slots(ctx,remaining,preferred_xy,occupied_points)
    previous_arm,previous_guard,previous_held=ctx.arm,ctx.guard,ctx.held
    cached={};nominal=tucked(ctx)
    try:
        # Predict empty-arm configurations after recipient release. No action is emitted
        # during this comparison; physical navigation/manipulation retain every guard.
        ctx.guard=np.zeros((0,6));ctx.held=None
        for category,obj in remaining.items():
            plans=[]
            for stance in surface_stances(ctx,obj[:3,3])[:6]:
                world=pose2(*stance)
                for arm in ('left','right'):
                    ctx.arm=arm
                    if not ctx.validator(base=world)(nominal,ctx.body.fk(nominal)[0]):continue
                    plan=plan_pinch(ctx,category,obj,base=world,q=nominal,count=8,candidate_limit=40)
                    if plan is None:continue
                    centers=[]
                    for chain in [nominal[ctx.ik.columns],plan['low']]+plan['up']:
                        full=nominal.copy();full[ctx.ik.columns]=chain
                        centers.append(ctx.collision.centers(ctx.body.fk(full)[0])@world[:3,:3].T+world[:3,3])
                    plans.append((np.concatenate(centers),np.tile(ctx.collision.radii,len(centers))))
                if len(plans)>=4:break
            cached[category]=plans
    finally:
        ctx.arm,ctx.guard,ctx.held=previous_arm,previous_guard,previous_held
    ranked=[]
    for index,goal in enumerate(goals):
        settled=goal.copy();settled[2,3]=float(ctx.scene.meta['scope']['desk.n.01_1']['aabb_hi'][2]-ctx.models['tote']['bounds'][0,2]+.003)
        guard=bag_guards(settled);lost=[];accessible={}
        for category,plans in cached.items():
            count=0
            for centers,radii in plans:
                local=(centers-settled[:3,3])@settled[:3,:3]
                nearest=np.clip(local[None],guard[:,None,:3],guard[:,None,3:])
                if np.min(np.linalg.norm(local[None]-nearest,axis=2)-radii[None])>=.015:count+=1
            accessible[category]=count
            if plans and not count:lost.append(category)
        ranked.append((len(lost),index,goal,accessible))
    ranked.sort(key=lambda r:(r[0],r[1]))
    ctx.event('recipient_access',cached_plans={k:len(v) for k,v in cached.items()},
              candidates=len(goals),best_lost=None if not ranked else ranked[0][0],
              best_compatible={} if not ranked else ranked[0][3],
              scope='prospective approach clearance; physical route and release checked separately')
    return [row[2] for row in ranked]


def relocate_table_recipient(ctx,est,remaining):
    """Short upright re-placement when the current recipient blocks a remaining approach."""
    from art_skills import pick_surface,move_tote_upright,place_object,tuck
    bag=est.refresh('tote')
    if bag is None:return False
    goals=accessible_bag_slots(ctx,remaining,bag[:2,3])
    if not goals:return False
    # This is the existing intentional tote-handle pickup operation. Its static/self
    # checks stay enabled; the recipient becomes the protected carried payload on grasp.
    ctx.guard=np.zeros((0,6))
    if not (yield from pick_surface(ctx,'tote',bag,refresh=est.refresh)):
        ctx.guard=bag_guards(bag);return False
    current=ctx.base_matrix()@ctx.tip()@ctx.held[2]
    lifted=current.copy();lifted[2,3]+=.065
    if not (yield from move_tote_upright(ctx,lifted)):return False
    for goal in goals[:3]:
        high=goal.copy();high[2,3]=max(high[2,3],lifted[2,3])
        if not (yield from move_tote_upright(ctx,high)):continue
        if not (yield from place_object(ctx,goal,label='relocate_table_bag')):continue
        settled=goal.copy();settled[2,3]=float(ctx.scene.meta['scope']['desk.n.01_1']['aabb_hi'][2]-ctx.models['tote']['bounds'][0,2]+.003)
        ctx.guard=bag_guards(settled);est.clear('tote')
        est.remember('tote',settled,source='executed_table_replacement_prediction')
        yield from tuck(ctx)
        fresh=est.refresh('tote')
        if fresh is not None:ctx.guard=bag_guards(fresh)
        return True
    return False
