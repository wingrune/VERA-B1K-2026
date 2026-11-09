"""Bounded vertical insertion through the observed carryall opening."""
import numpy as np
from art_skills import (transform,upright,bag_guards,release_ik,place_object,
                        carry,leave_surface,surface_stances,carried_drive)
from manip import pose2,hold,move_to,follow
from skills import interpolate_cartesian


def flat_insertion_goals(ctx,category,bag):
    bounds=ctx.models[category]['bounds'];extent=bounds[1]-bounds[0];centre=bounds.mean(0)
    vertical=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    inverse=np.linalg.inv(ctx.held[2]);goals=[]
    for turn in (0.,np.pi):
        for flip in (1.,-1.):
            relative=upright(turn)@vertical@np.diag([flip,1.,flip])
            half=abs(relative)@extent/2
            if np.any(np.array([.14,.09])-half[:2]-.015<0.):continue
            rotation=bag[:3,:3]@relative
            # Partly insert the lower edge and release above the walls. Native
            # gravity completes the insertion; fingers need not enter the bag.
            for z in (.23,.18,.12):
                goal=transform(rotation,(bag@np.array([0.,0.,z,1.]))[:3]-rotation@centre)
                goals.append((float((goal@inverse)[2,3]),goal))
    goals.sort(key=lambda row:-row[0])
    baseline=[g for _,g in goals]
    if category=='folder':
        # The open gripper collision envelope can sit below the handles for a
        # corner grasp even when the canonical finger mesh clears them.
        # Derive the smallest extra height from the same conservative spheres
        # used by full-body collision checking, with every finger fully open.
        full=ctx.q.copy()
        fingers=[i for i,name in enumerate(ctx.body.names) if name.startswith(ctx.arm+'_gripper_finger')]
        full[fingers]=.05
        centers=ctx.collision.centers(ctx.body.fk(full)[0]);current=ctx.tip()
        selected=np.array([name.startswith(ctx.arm+'_gripper') for name in ctx.collision.names])
        local=(centers[selected]-current[:3,3])@current[:3,:3];radii=ctx.collision.radii[selected]
        handle_top=bag[2,3]+max(.295,float(ctx.models['tote']['bounds'][1,2]))
        expanded=[]
        for goal in baseline[:8]:
            expanded.append(goal)
            tip=goal@inverse
            lowest=float(np.min(local@tip[2,:3]+tip[2,3]-radii))
            extra=max(0.,handle_top+.020-lowest)
            if .001<extra<=.12:
                raised=goal.copy();raised[2,3]+=extra;expanded.append(raised)
        return expanded
    if category!='notebook':return baseline
    # A tilted thickness grasp can make the old vertical book orientation
    # unreachable. Rotate in the book's own plane, keeping the whole projected
    # footprint inside the same cavity margins. Preserve the old first choices.
    alternatives=[]
    for angle in (np.pi/4,-np.pi/4,np.pi/2):
        for turn in (0.,np.pi):
            for flip in (1.,-1.):
                relative=upright(turn)@vertical@upright(angle)@np.diag([flip,1.,flip])
                half=abs(relative)@extent/2
                if np.any(np.array([.14,.09])-half[:2]-.015<0.):continue
                rotation=bag[:3,:3]@relative
                goal=transform(rotation,(bag@np.array([0.,0.,.23,1.]))[:3]-rotation@centre)
                alternatives.append((float((goal@inverse)[2,3]),goal))
    alternatives.sort(key=lambda row:-row[0])
    ordered=baseline[:8]+[g for _,g in alternatives]+baseline[8:]
    # A thickness grasp may put open fingers beside the tote handles at the old
    # release height. Keep each established goal, then try a bounded higher
    # release derived from the actual open-hand collision geometry. Full robot,
    # payload, cavity and load checks remain mandatory; no guard is relaxed.
    expanded=[]
    hand=ctx.gripper[ctx.arm+'_collision']-ctx.ik.tool
    handle_top=bag[2,3]+max(.295,float(ctx.models['tote']['bounds'][1,2]))
    for goal in ordered[:20]:
        expanded.append(goal)
        tip=goal@inverse
        lowest=float(np.min(hand@tip[2,:3]+tip[2,3]))
        extra=max(0.,handle_top+.020-lowest)
        if .001<extra<=.12:
            raised=goal.copy();raised[2,3]+=extra;expanded.append(raised)
    return expanded



def place_flat_continuous(ctx, goal):
    """Stay on the current IK branch: traverse above the mouth, then descend."""
    if ctx.held is None:return False
    target=np.linalg.inv(ctx.base_matrix())@goal@np.linalg.inv(ctx.held[2])
    start=ctx.tip();rotation=target[:3,:3]
    orientation_error=float(np.linalg.norm(start[:3,:3]-rotation))
    if orientation_error>.08:
        ctx.event('flat_cartesian_plan',ok=False,reason='orientation not settled',error=orientation_error)
        return False
    for extra in (0.,.10):
        hover=target[:3,3].copy();hover[2]=max(start[2,3],target[2,3]+.20)+extra
        lift=start[:3,3].copy();lift[2]=hover[2]
        full=ctx.q.copy();point=start[:3,3].copy();path=[];failed=None
        check=ctx.validator(max_load=1.10)
        for label,waypoint in [('lift',lift),('traverse',hover),('descend',target[:3,3])]:
            segment=interpolate_cartesian(ctx.ik,full,point,waypoint,rotation,validator=check,step=.015)
            if segment is None:failed=label;break
            path.extend(segment);full[ctx.ik.columns]=segment[-1];point=waypoint.copy()
        ctx.event('flat_cartesian_plan',ok=failed is None,failed_segment=failed,waypoints=len(path),extra_height=extra)
        if failed is not None:continue
        status=yield from follow(ctx,path,rate=.015,max_steps=1200)
        yield from hold(ctx,20)
        error=float(np.linalg.norm(ctx.tip()[:3,3]-target[:3,3]))
        if status!='done' or error>.014:
            ctx.event('flat_cartesian_reach',ok=False,status=status,error=error);return False
        ctx.set_grip(1.);yield from hold(ctx,30);ctx.held=None
        up=interpolate_cartesian(ctx.ik,ctx.q,ctx.tip()[:3,3],ctx.tip()[:3,3]+[0.,0.,.15],ctx.tip()[:3,:3],validator=ctx.validator(),step=.02)
        if up is not None:yield from follow(ctx,up,rate=.022,max_steps=250)
        yield from hold(ctx,60)
        ctx.event('flat_cartesian_release',ok=True,error=error,retreat=up is not None)
        return True
    return False


def insert_flat(ctx,category,bag,refresh=None):
    if ctx.held is None:return False
    on_floor=bag[2,3]<.55
    ctx.guard=bag_guards(bag)
    # Update a tilted receiver before transport so its guard follows observed
    # geometry. Ordinary upright perception and search keep their old behavior.
    if not on_floor and refresh is not None:
        fresh=refresh('tote')
        track=getattr(ctx.est,'tracks',{}).get('tote',{})
        if fresh is not None and track.get('info',{}).get('source')=='multiview_tilted_recipient':
            bag=fresh;ctx.guard=bag_guards(bag)
            ctx.event('flat_tilted_recipient',pose=bag.tolist(),evidence=track['info'])
    if on_floor and not (yield from leave_surface(ctx)):return False
    if not on_floor and not (yield from carry(ctx,height=bag[2,3]+.41)):return False
    goals=flat_insertion_goals(ctx,category,bag)
    if not goals:return False
    if on_floor:
        candidates=[]
        for radius in (.62,.75,.52):
            for angle in np.linspace(-np.pi,np.pi,16,endpoint=False):
                xy=bag[:2,3]+radius*np.array([np.cos(angle),np.sin(angle)])
                if ctx.scene.base_free(xy):
                    candidates.append((*xy,float(np.arctan2(*(bag[:2,3]-xy)[::-1]))))
    else:candidates=surface_stances(ctx,bag[:3,3])
    # Nearest-only selection can repeatedly move a right-hand payload across
    # the torso. Rank the receiver in the working hand's workspace as well as
    # navigation distance. Endpoint, swept-body and Cartesian checks still decide.
    side=.20 if ctx.arm=='left' else -.20
    def stance_cost(stance):
        local=np.linalg.inv(pose2(*stance))@bag[:,3]
        return abs(local[1]-side)+abs(local[0]-.75)+.15*np.linalg.norm(np.asarray(stance[:2])-ctx.est.base()[:2])
    unique={}
    for stance in [ctx.est.base()]+candidates:
        unique.setdefault(tuple(np.round(stance,5)),stance)
    aligned=sorted(unique.values(),key=stance_cost)
    # Keep a proven current-stance insertion before trying the complete aligned
    # family. Interleaving nearby variants spent the whole physical budget on
    # the same unreachable cross-body geometry before reaching a valid stance.
    candidates=[ctx.est.base()]
    for stance in aligned:
        if np.linalg.norm(np.asarray(stance[:2])-np.asarray(ctx.est.base()[:2]))>=.08:candidates.append(stance)
    ctx.event('flat_insertion_stances',category=category,count=len(candidates),
              first=[dict(stance=list(st),cost=float(stance_cost(st))) for st in candidates[:4]])
    physical_attempts=0
    for stance in candidates[:12]:
        stance_attempts=0
        world=pose2(*stance)
        for index,goal in enumerate(goals[:40 if category=='notebook' else 16]):
            yield ctx.action()
            tip=np.linalg.inv(world)@goal@np.linalg.inv(ctx.held[2])
            if not release_ik(ctx,tip,base=world):continue
            ctx.event('flat_insertion_plan',category=category,stance=list(stance),candidate=index,goal=goal.tolist())
            keepout=((bag[:2,3],.50),) if on_floor else ()
            if (yield from carried_drive(ctx,stance,max_steps=1000,keepout=keepout))!='arrived':break
            if refresh is not None:
                fresh=refresh('tote')
                if fresh is None and hasattr(ctx.est,'refresh'):
                    from recipient_view import observe_recipient
                    fresh=yield from observe_recipient(ctx,ctx.est,bag)
                if fresh is None:
                    ctx.event('flat_recipient_confirmation',ok=False,reason='recipient not observed',stance=list(stance))
                    if on_floor:
                        # A reachable release behind the robot can still hide
                        # the receiver from all cameras. Restore transport and
                        # try the next checked stance facing the floor tote.
                        if not (yield from carry(ctx)):return False
                        break
                    return False
                shift=float(np.linalg.norm(fresh[:3,3]-bag[:3,3]))
                if shift>.025:
                    # A new view can correct a few centimetres of registration
                    # bias. Require a stable observed footprint before replanning.
                    yield from hold(ctx,12)
                    confirmed=refresh('tote')
                    mismatch=np.inf
                    if confirmed is not None:
                        bounds=ctx.models['tote']['bounds'];x,y=np.meshgrid(bounds[:,0],bounds[:,1])
                        corners=np.c_[x.ravel(),y.ravel()]
                        a=corners@fresh[:2,:2].T+fresh[:2,3]
                        b=corners@confirmed[:2,:2].T+confirmed[:2,3]
                        distances=np.linalg.norm(a[:,None]-b[None,:],axis=2)
                        mismatch=max(float(distances.min(0).max()),float(distances.min(1).max()))
                    track=getattr(ctx.est,'tracks',{}).get('tote',{})
                    max_shift=.30 if track.get('info',{}).get('source')=='multiview_tilted_recipient' else .20
                    stable=confirmed is not None and shift<=max_shift and mismatch<=.015 and abs(confirmed[2,3]-fresh[2,3])<=.015
                    ctx.event('flat_recipient_confirmation',ok=stable,shift=shift,footprint_mismatch=float(mismatch))
                    if not stable:return False
                    fresh=confirmed
                goal=fresh@np.linalg.inv(bag)@goal;bag=fresh;ctx.guard=bag_guards(bag)
            # Base yaw changes during navigation rotate a held flat object.
            # Reorient AFTER arrival to this exact release orientation; an early
            # free-space turn cannot be carried over as a world-frame invariant.
            current=ctx.base_matrix()@ctx.tip()@ctx.held[2]
            free=goal.copy();free[:3,3]=current[:3,3];free[2,3]=max(.80,current[2,3])
            tip=np.linalg.inv(ctx.base_matrix())@free@np.linalg.inv(ctx.held[2])
            oriented=False
            for low in release_ik(ctx,tip)[:2]:
                if (yield from move_to(ctx,low,label='flat_reorient',rate=.018,max_steps=350,max_load=1.10))=='done':
                    oriented=True;break
            ctx.event('flat_reorientation',category=category,ok=oriented,after_navigation=True,goal=free.tolist())
            if not oriented:continue
            physical_attempts+=1;stance_attempts+=1
            if (yield from place_flat_continuous(ctx,goal)):
                yield from hold(ctx,60)
                ctx.event('flat_insertion_release',category=category,goal=goal.tolist())
                return True
            if physical_attempts>=4:
                ctx.event('flat_insertion_failed',category=category,reason='checked physical attempt budget')
                return False
            if stance_attempts>=2:
                if not (yield from carry(ctx,height=bag[2,3]+.41)):return False
                break
    ctx.event('flat_insertion_failed',category=category,reason='bounded checked insertion search exhausted')
    return False
