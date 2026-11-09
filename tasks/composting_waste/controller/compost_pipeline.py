"""Composting state machine. Development candidate; only implemented phases execute."""
import numpy as np
from manip import pose2,hold
from compost_skills import tuck,tucked,chassis_surface_drive,plan_pinch,pinch

def initialize(ctx,est):
    scan=est.ob.scan();pose,score,margin=est.ob.loc.localize([(scan,0.)])
    ctx.event('localize',score=score,margin=margin,pose=None if pose is None else pose.tolist())
    if pose is None or score<.65 or margin<.055:return False
    est.ob.loc.start_tracking(pose)
    if (yield from tuck(ctx))!='done':return False
    return True

def inventory(ctx,est,categories=('banana','pomegranate')):
    surface=ctx.scene.meta['target_surface'];lo=np.array(surface['aabb_lo']);hi=np.array(surface['aabb_hi'])
    found={}
    for shift in (0.,.35,-.35):
        # The view is based on static countertop/chair geometry, not fruit instance positions.
        stance=((lo[0]+hi[0])/2+shift,float(lo[1]-.73),np.pi/2)
        from compost_navigation import route_to
        from base_clearance import restore_clearance
        if not (yield from restore_clearance(ctx)):
            ctx.event('inventory_stance',stance=list(stance),status='clearance_recovery_failed');continue
        result=yield from route_to(ctx,stance,max_steps=1200)
        ctx.event('inventory_stance',stance=list(stance),status=result)
        if result!='arrived':continue
        yield from hold(ctx,12)
        for category in categories:
            obj=est.refresh(category)
            if obj is not None:found[category]=obj;ctx.event('observed_object',category=category,pose=obj.tolist(),evidence=est.tracks[category]['info'])
        if len(found)==len(categories):break
    return found

def pick(ctx,est,category,obj):
    surface=ctx.scene.meta['target_surface'];edge=float(surface['aabb_lo'][1]);nominal=tucked(ctx)
    for offset in (.72,.80,.90):
        for lateral in (0.,.22,-.22):
            stance=(float(obj[0,3])+lateral,edge-offset,np.pi/2)
            if not ctx.scene.footprint_free(*stance,(-.40,.24,.345),margin=.015,movable_margin=.02):continue
            for arm in ('left','right'):
                ctx.arm=arm
                if not ctx.validator(base=pose2(*stance))(nominal,ctx.body.fk(nominal)[0]):continue
                yield ctx.action()
                plan=plan_pinch(ctx,category,obj,base=pose2(*stance),q=nominal,count=8,candidate_limit=12,seconds=8.)
                if plan is None:continue
                result=yield from chassis_surface_drive(ctx,stance)
                if result!='arrived':continue
                yield from hold(ctx,10)
                fresh=est.refresh(category)
                if fresh is None:return False
                success=yield from pinch(ctx,category,fresh,refresh=est.refresh,preferred_candidate=plan['candidate'])
                if success:return True
                yield from tuck(ctx)
    return False

def task(ctx,est):
    if not (yield from initialize(ctx,est)):return 'localization_or_tuck_failed'
    remaining=['banana','pomegranate'];placed=[]
    found=yield from inventory(ctx,est,remaining)
    for category in list(remaining):
        if category not in found:continue
        ctx.event('phase',phase='pick',category=category)
        if not (yield from pick(ctx,est,category,found[category])):
            ctx.event('pick_failed',category=category);continue
        ctx.event('phase',phase='transport',category=category)
        if not (yield from visit_kitchen(ctx,est)):return 'transport_failed_with_held_object'
        recipient=yield from observe_bin(ctx,est)
        if recipient is None:return 'recipient_not_observed'
        if not (yield from place_in_bin(ctx,est,recipient,placed)):return 'placement_unreachable'
        placed.append(category);remaining.remove(category)
        if not remaining:break
        if ctx.steps>4400:return 'preserve_first_placement_near_horizon'
        if not (yield from return_counter(ctx,est)):return 'counter_return_failed'
        found=yield from inventory(ctx,est,remaining)
    yield from hold(ctx,90)
    return 'placements_completed:'+','.join(placed)


def transit_waypoints(ctx):
    """Geometry-derived entrance coordinates shared across every randomized instance."""
    surface=ctx.scene.meta['target_surface'];kitchen=ctx.scene.meta['scope']['floor.n.01_1']
    cabinets=[o for o in ctx.scene.meta['objects'] if o['category']=='bottom_cabinet' and o['aabb_lo'][0]>surface['aabb_hi'][0]+.5 and o['aabb_hi'][1]>kitchen['aabb_lo'][1]+.4]
    right=min(float(o['aabb_lo'][0]) for o in cabinets)
    x=right-.44;boundary=float(kitchen['aabb_lo'][1])
    return dict(retreat=(float(surface['aabb_hi'][0])+.10,float(surface['aabb_lo'][1])-.92,np.pi/2),
                gate=(x-.04,boundary-.23,0.),inside=(x,boundary+.65,0.))

def visit_kitchen(ctx,est):
    from compost_navigation import route_to
    from compost_skills import carry
    waypoints=transit_waypoints(ctx)
    if ctx.held is not None:
        if not (yield from carry(ctx,height=float(ctx.scene.meta['target_surface']['aabb_hi'][2])+.14)):return False
    for key in ('retreat','gate'):
        if key=='gate' and ctx.held is not None:
            from transport_posture import prepare_route
            if not (yield from prepare_route(ctx,waypoints[key],height=1.22)):return False
        result=yield from route_to(ctx,waypoints[key],max_steps=900)
        if result!='arrived':return False
    return (yield from enter_kitchen(ctx,est))

def enter_kitchen(ctx,est):
    from compost_navigation import route_to
    from compost_skills import carry
    from transport_posture import prepare_route
    nominal=transit_waypoints(ctx)['inside'];arrived=False
    # Unrecognized RGB-D floor-object guards can require a wider entry stance.
    # Search only collision-checked alternatives; no guard is removed to fit.
    for dx in (0.,.03,.06,.09):
        stance=(nominal[0]+dx,nominal[1],nominal[2])
        if ctx.held is not None and not (yield from prepare_route(ctx,stance,height=1.04)):continue
        if (yield from route_to(ctx,stance,max_steps=700))=='arrived':arrived=True;break
    if not arrived:return False
    if ctx.held is not None and not (yield from carry(ctx,height=1.25)):return False
    from torso_view import prefer_stationary_chassis,observe_with_torso
    if prefer_stationary_chassis(ctx):
        recipient=yield from observe_with_torso(ctx,est)
        if recipient is not None:return True
    for yaw in (np.pi,3*np.pi/4):
        x,y,_=est.base();result=yield from route_to(ctx,(x,y,yaw),max_steps=500)
        if result=='arrived':
            if yaw!=np.pi:
                # Turn part way where the coarse floor-object enclosure permits
                # it, translate with that safe heading, then face the recipient.
                x,y,_=est.base()
                for dx in (-.04,-.07,-.10):
                    if (yield from route_to(ctx,(x+dx,y,np.pi),max_steps=450))=='arrived':return True
            return True
        error=abs((est.base()[2]-yaw+np.pi)%(2*np.pi)-np.pi)
        ctx.event('view_stance_partial',yaw_error=error,status=result)
        if error<.20:return True
    from torso_view import observe_with_torso
    return (yield from observe_with_torso(ctx,est)) is not None

def observe_bin(ctx,est):
    from recipient_geometry import lid_from_rgbd,recipient_guards
    for attempt in range(3):
        yield from hold(ctx,12)
        obj=est.refresh('ashcan')
        if obj is None:continue
        lid=lid_from_rgbd(est.ob,obj,ctx.base_matrix())
        if lid is None:
            # A partially visible body can fit depth with a small yaw error that
            # prevents every lid angle from matching. Refine both jointly while
            # retaining the original body/lid acceptance thresholds.
            from joint_recipient_fit import refine
            yield ctx.action()
            refined=refine(est.ob,ctx.base_matrix(),obj,
                           lambda label,record:ctx.event('recipient_joint_fit',**record))
            if refined is not None:
                obj,lid=refined
                est.remember('ashcan',obj,source='joint_RGBD_body_lid_fit')
        ctx.event('observed_recipient',pose=obj.tolist(),lid=lid)
        if lid is None:continue
        ctx.guard=recipient_guards(obj,lid['angle']);return obj,lid['angle']
    return None

def place_in_bin(ctx,est,recipient,placed=(),_repositioned=False):
    from recipient_geometry import release_goals,recipient_guards
    from compost_skills import place_object
    obj,angle=recipient;category=ctx.held[1];ctx.guard=recipient_guards(obj,angle)
    from base_clearance import restore_clearance
    from compost_navigation import route_to
    if abs((est.base()[2]+ctx.chain[3]-np.pi+np.pi)%(2*np.pi)-np.pi)>.30:
        # After fitting the recipient, replace its coarse unknown-object box
        # with observed articulated geometry and seek a facing work stance.
        x,y,_=est.base()
        for dx in (-.04,-.07,-.10):
            if (yield from route_to(ctx,(x+dx,y,np.pi),max_steps=450))=='arrived':break
    if not (yield from restore_clearance(ctx)):return False
    occupied=[[-.052,-.035]] if placed else []
    for goal in release_goals(ctx,category,obj,angle,occupied):
        yield ctx.action()
        ctx.event('release_candidate',category=category,pose=goal.tolist())
        if (yield from place_object(ctx,goal,label='ashcan_insert')):
            yield from hold(ctx,60)
            ctx.event('placement_belief',category=category,basis='RGBD recipient pose, checked release, open gripper, gravity settling',inside_not_privileged=True)
            return True
    if not _repositioned:
        from recipient_stance import approach_recipient
        if (yield from approach_recipient(ctx,recipient,occupied)):
            fresh=yield from observe_bin(ctx,est)
            if fresh is not None:return (yield from place_in_bin(ctx,est,fresh,placed,_repositioned=True))
    return False


def return_counter(ctx,est):
    from compost_navigation import route_to
    from base_clearance import restore_clearance
    active=ctx.arm
    if (yield from tuck(ctx,arms=(active,'left' if active=='right' else 'right')))!='done':return False
    if not (yield from restore_clearance(ctx)):return False
    previous=float(getattr(ctx,'route_reserve',0.));ctx.route_reserve=.025
    try:
        waypoints=transit_waypoints(ctx)
        for key in ('gate','retreat'):
            goal=(*waypoints[key][:2],float(est.base()[2])) if key=='gate' else waypoints[key]
            result=yield from route_to(ctx,goal,max_steps=1000)
            if result!='arrived':return False
        return True
    finally:
        ctx.route_reserve=previous
