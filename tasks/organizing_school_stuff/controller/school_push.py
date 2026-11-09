"""Closed-loop edge exposure from estimated poses and fixed support geometry.

No simulator imports. Callers provide refreshed RGB-D object poses. Diagnostic
callers may substitute their oracle outside the controller import boundary.
"""
import numpy as np
from art_skills import upright, transform, surface_stances, surface_drive, tuck
from manip import pose2, move_to, follow, hold
from skills import interpolate_cartesian

EDGE_GRIPPER_CLEARANCE = .008


def world_vertices(ctx, category, obj):
    bounds = ctx.models[category]['bounds']
    corners = np.array(np.meshgrid(*zip(*bounds), indexing='ij')).reshape(3, -1).T
    return corners @ obj[:3, :3].T + obj[:3, 3]


def support_margin(ctx,category,obj,normal):
    """Distance from the estimated centre to the local support-grid boundary."""
    centre=obj[:3,:3]@ctx.models[category]['bounds'].mean(0)+obj[:3,3]
    tangent=np.array([-normal[1],normal[0],0.])
    distances=np.arange(0.,.701,.0025)
    points=centre+distances[:,None]*normal
    samples=points[:,None]+np.array([-.01,0.,.01])[None,:,None]*tangent
    minimum=float(ctx.scene.meta['target_surface']['aabb_lo'][2])-.025
    supported=np.all(ctx.scene.support_heights(samples)>minimum,axis=1)
    missing=np.flatnonzero(~supported)
    return max(0.,float(distances[missing[0]])-.0025) if len(missing) else .70


def edge_options(ctx, category, obj):
    """Geometry-dependent travel to a stable, graspable overhang on each edge."""
    vertices = world_vertices(ctx, category, obj)
    surface = ctx.scene.meta['target_surface']
    lo, hi = np.asarray(surface['aabb_lo']), np.asarray(surface['aabb_hi'])
    result = []
    for axis in (0, 1):
        for sign in (-1., 1.):
            normal = np.zeros(3); normal[axis] = sign
            edge = float(sign * (hi[axis] if sign > 0 else lo[axis]))
            projected = vertices @ normal
            width = float(np.ptp(projected))
            # Positive exposure can only increase travel. Prune unreachable
            # distant edges before expensive contact geometry enumeration.
            if edge-float(projected.max())>.65:continue
            # Select exposure from the actual lower-finger / bed-frame geometry.
            # A mattress edge can overhang a wider frame below it, so a fixed
            # 3-4 cm exposure can still leave the lower jaw inside collision.
            overhang = None
            for exposure in np.arange(.015, min(.10, .35 * width) + .001, .005):
                trial=obj.copy()
                trial[:3,3]+=(edge+exposure-float(projected.max()))*normal
                if edge_is_clear(ctx,category,trial,normal):
                    overhang=float(exposure);break
            if overhang is None:
                continue
            travel = edge + overhang - float(projected.max())
            supported=obj.copy();supported[:3,3]+=max(0.,travel)*normal
            centre_margin = support_margin(ctx,category,supported,normal)
            if overhang < .015 or travel > .65:
                continue
            staging=centre_margin < max(.025,.15*width)
            if staging:
                # Far from the edge, a short safe staging push can improve the
                # observed orientation. This is not a promise of graspability.
                # Every executed segment still obeys the local support margin.
                current_margin=support_margin(ctx,category,obj,normal)
                if current_margin<.10:continue
                travel=min(travel,current_margin-.06)
            destination = obj[:3, 3] + max(0., travel) * normal
            result.append(dict(normal=normal, edge=edge, width=width, overhang=overhang,
                               travel=max(0., travel), destination=destination,
                               centre_margin=centre_margin,staging_only=staging))
    return sorted(result, key=lambda e: e['travel'])


def edge_is_clear(ctx,category,obj,normal):
    previous=ctx.arm
    try:
        for arm in ('left','right'):
            ctx.arm=arm
            for goal in edge_grasp_candidates(ctx,category,obj):
                if float(goal[:3,2]@normal)<.45:continue
                mesh=ctx.gripper[arm+'_collision']-ctx.ik.tool
                pts=mesh@goal[:3,:3].T+goal[:3,3]
                if float(ctx.scene.contact_clearance(pts,np.full(len(pts),.0004)).min())>=EDGE_GRIPPER_CLEARANCE:
                    return True
        return False
    finally:
        ctx.arm=previous


def edge_grasp_candidates(ctx, category, obj):
    # Preserve the physically verified central chord, then add exposed corners.
    from flat_contact import robust_edge_rays,open_gripper_clear
    # Tilt the lower jaw outwards to clear the mattress at a smaller overhang.
    # Every orientation retains native hull-ray and full open-finger checks.
    nominal=[goal for depth in (.026,.022,.018,.014) for section_chord in (False,True)
             for goal in _edge_grasp_candidates(ctx,category,obj,section_chord,depth)]
    for angle in (0.,-.2,-.4,-.6,-.8,-1.):
        c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
        for original in nominal:
            goal=original.copy();goal[:3,:3]=goal[:3,:3]@rotation
            if robust_edge_rays(ctx,category,obj,goal) and open_gripper_clear(ctx,category,obj,goal):yield goal

def _edge_grasp_candidates(ctx, category, obj, section_chord, depth):
    """Pinch thickness vertically with the lower finger outside the mattress."""
    vertices = world_vertices(ctx, category, obj)
    surface = ctx.scene.meta['target_surface']
    lo, hi = np.asarray(surface['aabb_lo']), np.asarray(surface['aabb_hi'])
    for axis in (0, 1):
        for sign in (-1., 1.):
            n = np.zeros(3); n[axis] = sign
            edge = sign * (hi[axis] if sign > 0 else lo[axis])
            protrusion = float((vertices @ n).max() - edge)
            if protrusion < .012:
                continue
            vertical = obj[:3,2]/np.linalg.norm(obj[:3,2])
            outward=n-vertical*float(n@vertical);outward/=np.linalg.norm(outward)
            rotation = np.column_stack([np.cross(vertical, outward), vertical, outward])
            tangent = rotation[:, 0]
            model = ctx.models[category]['xyz'] @ obj[:3, :3].T + obj[:3, 3]
            front = model[(model @ n) > edge + .012]
            if len(front) < 8:
                continue
            for lateral in (0.5, .25, .75, .10, .90, .03, .97):
                target = np.median(front, axis=0)
                target += (np.quantile(front @ tangent, lateral) - target @ tangent) * tangent
                # An oblique book corner's global AABB extremum lies on a
                # different cross-section. Use the actually selected finger chord.
                section=model[abs(model@tangent-float(target@tangent))<.008]
                front_edge=float((section@n).max()) if section_chord and len(section) else edge+protrusion
                target += (front_edge-depth-target@n)*n
                # The observed top can tilt as the overhang grows. Centre the
                # jaws on the physical thickness and close normal to that plane.
                middle=float(ctx.models[category]['bounds'][:,2].mean())
                target+=(middle-float((target-obj[:3,3])@vertical))*vertical
                for dz in (0., .004, -.004):
                    goal=transform(rotation,target+[0.,0.,dz])
                    rays=ctx.gripper[ctx.arm+'_ag']-ctx.ik.tool
                    local=(rays@goal[:3,:3].T+goal[:3,3]-obj[:3,3])@obj[:3,:3]
                    bounds=ctx.models[category]['bounds']
                    # Pad contact alone does not satisfy native assisted grasp.
                    # Require a ray to cross the object's interior with clearance.
                    intersects=False
                    for a in local[:2]:
                        for b in local[2:]:
                            line=np.linspace(a,b,65)
                            if np.any(np.all(line>bounds[0]+.002,axis=1)&np.all(line<bounds[1]-.002,axis=1)):
                                intersects=True
                    if intersects:yield goal


def contact_goal(ctx, category, obj, normal):
    """Closed fingers contact the trailing side, keeping their tips above support."""
    vertices = world_vertices(ctx, category, obj)
    rotation = upright(float(np.arctan2(normal[1], normal[0])))
    # Use the open envelope conservatively even though execution closes the jaws.
    mesh = (ctx.gripper[ctx.arm + '_collision'] - ctx.ik.tool) @ rotation.T
    target = obj[:3, 3].copy()
    target += ((vertices @ normal).min() - .003 - (mesh @ normal).max() - target @ normal) * normal
    footprint = mesh + target
    support = ctx.scene.support_heights(footprint)
    target[2] += float(np.max(support + .0015 - footprint[:, 2]))
    # Only geometry at the object's height can make pushing contact. The palm
    # protrudes farther than the fingertips but passes above a flat folder;
    # using its extremum leaves the actual pusher centimetres short.
    model=ctx.models[category]['xyz']@obj[:3,:3].T+obj[:3,3]
    tangent=np.array([-normal[1],normal[0],0.])
    section=model[abs((model-obj[:3,3])@tangent)<.012]
    if len(section)<8:section=model
    contact_band=(mesh[:,2]+target[2]>=float(model[:,2].min())-.002)&(
                  mesh[:,2]+target[2]<=float(model[:,2].max())+.002)
    pusher=mesh[contact_band]
    if len(pusher):
        target+=((section@normal).min()-.003-(pusher@normal).max()-target@normal)*normal
    return transform(rotation, target)


def plan_push_segment(ctx, goal, normal, distance, base=None, q=None):
    world = ctx.base_matrix() if base is None else base
    full = ctx.q if q is None else q
    tip = np.linalg.inv(world) @ goal
    delta = world[:3, :3].T @ (normal * distance)
    check = ctx.validator(base=base, max_load=1.05)
    mesh = ctx.gripper[ctx.arm + '_collision'] - ctx.ik.tool
    for alpha in np.linspace(0., 1., max(3, int(np.ceil(distance / .005)))):
        pts = mesh @ goal[:3, :3].T + goal[:3, 3] + alpha * normal * distance
        if np.min(ctx.scene.contact_clearance(pts, np.full(len(pts), .0004))) < -.0002:
            return None
    raw = ctx.ik.solve(full, tip[:3, 3], tip[:3, :3], count=12, validator=ctx.validator(base=base),
                      all_solutions=True, position_tolerance=.001, rotation_tolerance=.02,
                      rest_weight=.0005)
    def load_of(low):
        trial=full.copy();trial[ctx.ik.columns]=low
        return ctx.load(trial)[0]
    lows=[low for low in raw if load_of(low)<=1.05]
    if not lows and raw and min(map(load_of,raw))<1.45:
        # Search a different redundant torso/arm posture; retain native joint
        # limits, collision validation and the original static-load threshold.
        def balance(trial,poses):
            return np.array([.10*max(0.,ctx.load(trial)[0]-.92)])
        lows=ctx.ik.solve(full,tip[:3,3],tip[:3,:3],count=4,validator=check,extra=balance,
                          all_solutions=True,position_tolerance=.001,rotation_tolerance=.02,
                          rest_weight=.0005)
    for low in lows[:4]:
        start = full.copy(); start[ctx.ik.columns] = low
        up = interpolate_cartesian(ctx.ik, start, tip[:3, 3], tip[:3, 3] + [0., 0., .09],
                                   tip[:3, :3], validator=check, step=.015)
        slide = interpolate_cartesian(ctx.ik, start, tip[:3, 3], tip[:3, 3] + delta,
                                      tip[:3, :3], validator=check, step=.005)
        if up is not None and slide is not None:
            return dict(up=up, slide=slide, low=low)
    return None


def observe_flat(ctx,category,refresh):
    # A cropped wrist view can align a flat top at several lateral offsets.
    # Retract the empty arms before every closed-loop geometry update, giving
    # the head camera the complete silhouette and an independent depth view.
    if ctx.held is not None:return None
    status=yield from tuck(ctx)
    yield from hold(ctx,10)
    pose=refresh(category)
    ctx.event('flat_observation',category=category,tuck=status,observed=pose is not None)
    return pose


def exposed_grasp_ready(ctx,category,obj,normal):
    if not edge_is_clear(ctx,category,obj,normal):return False
    from art_skills import plan_pinch
    previous=ctx.arm;edge_mode=getattr(ctx,'edge_only',False);ctx.edge_only=True
    try:
        for arm in ('left','right'):
            ctx.arm=arm
            if plan_pinch(ctx,category,obj,count=8,candidate_limit=30) is not None:return True
        return False
    finally:ctx.arm=previous;ctx.edge_only=edge_mode


def push_to_edge(ctx, category, obj, refresh, max_segments=24):
    """Bounded observe-plan-push-observe loop; returns a fresh exposed pose or None."""
    if refresh is None or category not in ('notebook', 'folder') or ctx.held is not None:
        return None
    previous_arm = ctx.arm
    options = edge_options(ctx, category, obj)
    for option in options:
        n, edge = option['normal'], option['edge']
        ctx.event('push_edge_goal',category=category,normal=n.tolist(),staging_only=option['staging_only'],predicted_target_margin=option['centre_margin'])
        target = yield from observe_flat(ctx,category,refresh)
        if target is None:
            return None
        trial = contact_goal(ctx, category, target, n)
        stances = surface_stances(ctx, trial[:3, 3])
        stances = [s for s in stances if np.linalg.norm(np.array(s[:2]) - trial[:2, 3]) < 1.1]
        stances.sort(key=lambda st:2.*np.linalg.norm(np.asarray(st[:2])-trial[:2,3])
                     +.2*np.linalg.norm(np.asarray(st[:2])-ctx.est.base()[:2]))
        ready = False
        for stance in stances[:24]:
            for arm in ('left', 'right'):
                yield ctx.action()
                ctx.arm = arm
                from art_skills import tucked
                q = tucked(ctx); base = pose2(*stance)
                if not ctx.validator(base=base)(q, ctx.body.fk(q)[0]):
                    continue
                goal = contact_goal(ctx, category, target, n)
                if plan_push_segment(ctx, goal, n, .025, base=base, q=q) is None:
                    continue
                if (yield from surface_drive(ctx, stance)) != 'arrived':
                    continue
                yield from tuck(ctx)
                ready = True
                break
            if ready:
                break
        if not ready:
            ctx.event('push_edge_rejected', category=category, reason='no contact stance', normal=n.tolist())
            continue
        ctx.set_grip(-1.); yield from hold(ctx, 15)
        no_progress = 0; lateral_replans=0
        for segment in range(max_segments):
            current = yield from observe_flat(ctx,category,refresh)
            if current is None:
                ctx.event('push_stop', category=category, reason='object not observed')
                break
            verts = world_vertices(ctx, category, current)
            over = float((verts @ n).max() - edge)
            centre_margin = support_margin(ctx,category,current,n)
            ready_grasp=over>=.012 and exposed_grasp_ready(ctx,category,current,n)
            if ready_grasp:
                ctx.set_grip(1.); yield from hold(ctx, 12)
                ctx.event('push_exposed', category=category, overhang=over, segments=segment)
                return current
            if centre_margin < .025:
                ctx.event('push_stop',category=category,reason='local support stability margin',margin=centre_margin)
                break
            remaining=option['overhang']-over
            if remaining<=.004 and not ready_grasp:
                actual_width=float(np.ptp(verts@n))
                extra=min(.003,max(0.,.5*(centre_margin-.025)))
                if extra<.001 or over+extra>min(.10,.35*actual_width):
                    ctx.event('push_stop',category=category,reason='safe overhang has no lower-jaw clearance',overhang=over,width=actual_width,centre_margin=centre_margin)
                    break
                remaining=extra
            distance = min(.025, max(.001,remaining))+.003
            distance = min(distance,.5*max(0.,centre_margin-.025))
            if distance<.003:
                ctx.event('push_stop',category=category,reason='local support motion budget',margin=centre_margin)
                break
            goal = contact_goal(ctx, category, current, n)
            plan = plan_push_segment(ctx, goal, n, distance)
            if plan is None:
                ctx.event('push_stop', category=category, reason='segment collision or IK')
                break
            from approach_obstacles import protect_object
            with protect_object(ctx,category,current):
                approach=yield from move_to(ctx,plan['up'][-1],label='push_pre',max_steps=350)
            ctx.event('push_preapproach',category=category,status=approach,target_protected=True)
            if approach!='done':break
            if (yield from follow(ctx, plan['up'][::-1][1:] + [plan['low']], rate=.012, max_steps=220)) != 'done':
                break
            status = yield from follow(ctx, plan['slide'], rate=.004 if remaining<.012 else .006, max_steps=180)
            yield from hold(ctx, 10)
            # Retract vertically before asking for a new view; never blindly keep pushing.
            full = ctx.q; tip = ctx.tip(); check = ctx.validator(relax_from=full, max_load=1.05)
            retreat = interpolate_cartesian(ctx.ik, full, tip[:3, 3], tip[:3, 3] + [0., 0., .06],
                                            tip[:3, :3], validator=check, step=.015)
            if retreat is None:
                break
            yield from follow(ctx, retreat, rate=.012, max_steps=180)
            yield from hold(ctx, 8)
            fresh = yield from observe_flat(ctx,category,refresh)
            if fresh is None:
                break
            # A rectangular flat silhouette has a half-turn ambiguity. Measure
            # its physical centre and unoriented principal axis, so relabelling
            # equivalent corners cannot masquerade as a 180-degree collision.
            centre=ctx.models[category]['bounds'].mean(0)
            delta = (fresh[:3,:3]@centre+fresh[:3,3])-(current[:3,:3]@centre+current[:3,3])
            progress = float(delta @ n)
            normal_angle=float(np.arccos(np.clip(current[:3,2]@fresh[:3,2],-1.,1.)))
            axis_angle=float(np.arccos(np.clip(abs(current[:3,0]@fresh[:3,0]),0.,1.)))
            angle=max(normal_angle,axis_angle)
            ctx.event('push_segment', category=category, segment=segment, status=status,
                      progress=progress, lateral=float(np.linalg.norm(delta - progress * n)),
                      rotation=angle, displacement=delta.tolist())
            if status != 'done' or angle > .30 or abs(delta[2]) > .025:
                break
            lateral_error=float(np.linalg.norm(delta-progress*n))
            if lateral_error>.035:
                # A sideways slip is a new contact-planning problem, rather
                # than an automatic loss. Continue only while the physical
                # centre remains supported away from both adjacent edges.
                centre_world=fresh[:3,:3]@centre+fresh[:3,3]
                surface=ctx.scene.meta['target_surface'];side=1-int(np.argmax(abs(n[:2])))
                adjacent_margin=min(centre_world[side]-surface['aabb_lo'][side],surface['aabb_hi'][side]-centre_world[side])
                lateral_replans+=1
                safe=adjacent_margin>=.04 and lateral_replans<=3
                ctx.event('push_lateral_replan',category=category,ok=safe,lateral=lateral_error,
                          adjacent_edge_margin=float(adjacent_margin),attempt=lateral_replans)
                if not safe:break
                # The next iteration observes again and recomputes the full
                # contact, IK and swept geometry from the displaced object.

            no_progress = no_progress + 1 if progress < max(.0005,min(.003,.2*max(.001,distance-.003))) else 0
            if no_progress >= 2:
                break
        ctx.set_grip(1.); yield from hold(ctx, 12)
        yield from tuck(ctx)
    ctx.arm = previous_arm
    return None


def pick_exposed(ctx,category,obj,refresh):
    """Replan the grasping hand/stance independently of the pushing hand."""
    from art_skills import plan_pinch,pinch,tucked
    previous=ctx.arm;ctx.edge_only=True;attempts=0
    try:
        current=yield from observe_flat(ctx,category,refresh)
        if current is None:return False
        stances=[ctx.est.base()]+surface_stances(ctx,current[:3,3])
        seen=[]
        for stance in stances[:14]:
            xy=np.asarray(stance[:2])
            if any(np.linalg.norm(xy-p)<.07 for p in seen):continue
            seen.append(xy)
            here=np.linalg.norm(xy-np.asarray(ctx.est.base()[:2]))<.03
            nominal=ctx.q if here else tucked(ctx);world=pose2(*stance)
            for arm in ('left','right'):
                yield ctx.action()
                ctx.arm=arm
                if not ctx.validator(base=world)(nominal,ctx.body.fk(nominal)[0]):continue
                plan=plan_pinch(ctx,category,current,base=world,q=nominal,count=12,candidate_limit=30)
                if plan is None:continue
                ctx.event('exposed_grasp_plan',category=category,arm=arm,stance=list(stance),candidate=plan['candidate'])
                if not here:
                    if (yield from surface_drive(ctx,stance))!='arrived':continue
                    yield from tuck(ctx)
                target=yield from observe_flat(ctx,category,refresh)
                if target is None:continue
                attempts+=1
                if (yield from pinch(ctx,category,target,refresh=refresh,preferred_candidate=plan['candidate'])):return True
                yield from tuck(ctx)
                if attempts>=3:return False
        ctx.event('exposed_grasp_failed',category=category,reason='no checked edge grasp',attempts=attempts)
        return False
    finally:
        ctx.edge_only=False
        if ctx.held is None:ctx.arm=previous
