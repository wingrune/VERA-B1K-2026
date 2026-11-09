"""Heading-aware transport with separately checked translation and rotation."""
import time
import numpy as np
from manip import pose2,drive,hold,BASE_LINEAR_GAIN
from localizer import turn_rate


def intermediate_turn(path,start_yaw,goal_yaw,safe,seconds=8.):
    """Find an interior turn without relaxing whole-body or payload checks.

    Translation samples are <=10 mm apart, turn samples <=.05 rad.
    The prefix retains the measured heading, the suffix uses the requested one.
    """
    deadline=time.monotonic()+seconds
    points=[np.asarray(path[0],float)];original_indices=[0]
    for a,b in zip(path,path[1:]):
        points.extend(np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/.01))+1))[1:])
        original_indices.append(len(points)-1)
    prefix=-1
    for i,p in enumerate(points):
        if time.monotonic()>deadline or not safe(p,start_yaw):break
        prefix=i
    suffix=len(points)
    for i in range(len(points)-1,-1,-1):
        if time.monotonic()>deadline or not safe(points[i],goal_yaw):break
        suffix=i
    if suffix>prefix:return None
    candidates=list(range(suffix,prefix+1));middle=(suffix+prefix)/2
    candidates.sort(key=lambda i:abs(i-middle))
    delta=(goal_yaw-start_yaw+np.pi)%(2*np.pi)-np.pi
    angles=np.linspace(start_yaw,start_yaw+delta,max(2,int(np.ceil(abs(delta)/.05))+1))
    for i in candidates:
        if time.monotonic()>deadline:return None
        if all(safe(points[i],angle) for angle in angles):
            before=[points[j] for j in original_indices if j<i]+[points[i]]
            after=[points[i]]+[points[j] for j in original_indices if j>i]
            return before,after
    return None


def floor_tool_keepouts(ctx,start,keepout):
    """Protect remembered, unhandled floor tools from incidental chassis contact.

    Tracks are RGB-D beliefs. A caller's explicit target exclusion retains its
    tested approach distance. Starting inside a new disc permits only egress.
    """
    if not hasattr(ctx.est,'object'):return []
    discs=[]
    for category in ('pen','calculator'):
        if ctx.held is not None and ctx.held[1]==category:continue
        obj=ctx.est.object(category,max_age=20000)
        if obj is None or obj[2,3]>.12:continue
        centre=np.asarray(obj[:2,3],float)
        if any(np.linalg.norm(centre-np.asarray(point))<.10 for point,_ in keepout):continue
        radius=.40+float(np.max(np.linalg.norm(ctx.models[category]['bounds'][:,:2],axis=1)))+.04
        radius=min(radius,max(0.,float(np.linalg.norm(start[:2]-centre))-.02))
        if radius>.05:discs.append((centre.copy(),radius))
    return discs


def discs_clear(path,discs):
    for a,b in zip(path,path[1:]):
        a,b=np.asarray(a),np.asarray(b);delta=b-a
        for centre,radius in discs:
            t=np.clip(np.dot(centre-a,delta)/max(float(delta@delta),1e-12),0.,1.)
            if np.linalg.norm(a+t*delta-centre)<radius-.01:return False
    return True


def checked_carried_drive(ctx,stance,max_steps=1200,keepout=(),route=None):
    start=np.asarray(ctx.est.base(),float);goal=np.asarray(stance,float)
    explicit_keepout=tuple(keepout)
    tools=floor_tool_keepouts(ctx,start,keepout)
    if tools:
        # SceneMap.simplify accepts 3cm obstacle-disc slack. Reserve this in
        # the search so the executed sweep still obeys the unrelaxed tool disc.
        keepout=tuple(keepout)+tuple((point,radius+.04) for point,radius in tools)
        if any(np.linalg.norm(goal[:2]-centre)<radius for centre,radius in tools):
            ctx.event('floor_tool_route_guard',ok=False,reason='goal overlaps remembered floor tool',stance=goal.tolist())
            return 'no_route'
        if route is not None and not discs_clear(route,tools):route=None
    # Reserve clearance through transit corners before the detailed sweep.
    # A native307 route passed25mm samples but intersected the base model by
    # 2.4mm between them. The denser sweep below remains authoritative.
    path=ctx.scene.path(start[:2],goal[:2],margin=.06,keepout=keepout) if route is None else route
    if path is None and route is None:
        path=ctx.scene.path(start[:2],goal[:2],keepout=keepout)
    if path is None and tools and route is None:
        # The extra search reserve compensates for A* simplification, but may
        # close a narrow passage that still satisfies the actual tool guard.
        # Retry bounded reserves and accept only the unchanged analytic disc
        # check, followed by dense whole-body/payload sweeps and online checks.
        for reserve in (.02,0.):
            trial_keepout=explicit_keepout+tuple((point,radius+reserve) for point,radius in tools)
            for margin in (.06,.02):
                trial=ctx.scene.path(start[:2],goal[:2],margin=margin,keepout=trial_keepout)
                if trial is not None and discs_clear(trial,tools):
                    path=trial
                    ctx.event('floor_tool_search_fallback',reserve=reserve,scene_margin=margin,
                              unchanged_execution_guard=True)
                    break
            if path is not None:break
    if path is None:
        ctx.event('carried_route',ok=False,reason='no geometric route with observed tool guards')
        return 'no_route'
    if tools:
        if not discs_clear(path,tools):return 'no_route'
        ctx.event('floor_tool_route_guard',ok=True,discs=[dict(centre=p.tolist(),radius=r) for p,r in tools])
    initial=ctx.q.copy();poses=ctx.body.fk(initial)[0];baseline=ctx.base_matrix()
    cache={}
    def safe(point,yaw,full=initial,frames=poses):
        if any(np.linalg.norm(np.asarray(point)-centre)<radius-.01 for centre,radius in tools):return False
        key=(*map(float,point),float(yaw))
        if full is initial and key in cache:return cache[key]
        valid=ctx.validator(base=pose2(*point,yaw),relax_from=initial,baseline_base=baseline)(full,frames)
        if full is initial:cache[key]=valid
        return valid
    def turn_safe(point,a,b):
        delta=(b-a+np.pi)%(2*np.pi)-np.pi
        return all(safe(point,angle) for angle in np.linspace(a,a+delta,max(2,int(np.ceil(abs(delta)/.05))+1)))
    def path_safe(yaw):
        return all(safe(point,yaw) for a,b in zip(path,path[1:])
                   for point in np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/.01))+1)))
    if turn_safe(start[:2],start[2],goal[2]) and path_safe(goal[2]):
        strategy='turn_then_translate';segments=[(path,goal[2])]
    elif turn_safe(goal[:2],start[2],goal[2]) and path_safe(start[2]):
        strategy='translate_then_turn';segments=[(path,start[2])]
    else:
        split=intermediate_turn(path,start[2],goal[2],safe)
        if split is None:
            ctx.event('carried_route',ok=False,reason='no checked transport heading');return 'no_route'
        strategy='translate_turn_translate';segments=[(split[0],start[2]),(split[1],goal[2])]
    ctx.event('carried_route',ok=True,strategy=strategy,stance=goal.tolist(),waypoints=[p.tolist() for p in path])
    if strategy=='translate_turn_translate':
        ctx.event('intermediate_turn',point=segments[0][0][-1].tolist(),start_yaw=float(start[2]),goal_yaw=float(goal[2]))
    remaining=max_steps
    def guarded(action):
        x,y,yaw=ctx.est.base();local=np.asarray(action[:2])*BASE_LINEAR_GAIN/30.
        predicted=np.array([x+np.cos(yaw)*local[0]-np.sin(yaw)*local[1],y+np.sin(yaw)*local[0]+np.cos(yaw)*local[1]])
        angle=yaw+turn_rate(float(action[2]),driving=np.linalg.norm(action[:2])>.06)/30.
        full=ctx.q
        return safe(predicted,angle,full,ctx.body.fk(full)[0])
    def rotate(yaw):
        nonlocal remaining
        while remaining>0:
            delta=(yaw-ctx.est.base()[2]+np.pi)%(2*np.pi)-np.pi
            if abs(delta)<.015:
                yield from hold(ctx,4);remaining-=4;return 'arrived'
            turn=float(np.sign(delta)*max(.06,min(.5,1.6*abs(delta))))
            action=ctx.action(base=(0.,0.,turn))
            if not guarded(action):
                yield ctx.action();return 'updated_collision'
            remaining-=1;yield action
        return 'timeout'
    for segment_index,(segment,heading) in enumerate(segments):
        if strategy=='turn_then_translate' or segment_index>0:
            status=yield from rotate(heading)
            if status!='arrived':return status
        for point in segment[1:]:
            motion=drive(ctx,point,heading,direct=True,face_travel=False,
                         speed=.18 if ctx.held is not None and ctx.held[1]=='tote' else .35,
                         tolerance=.012,yaw_tolerance=.015,min_turn=.06,max_steps=remaining)
            while True:
                try:action=next(motion)
                except StopIteration as done:status=done.value;break
                if remaining<=0:return 'timeout'
                if not guarded(action):
                    yield ctx.action();ctx.event('carried_route',ok=False,reason='updated observation invalidates next step');return 'no_route'
                remaining-=1;yield action
            if status!='arrived':return status
    if strategy=='translate_then_turn':return (yield from rotate(goal[2]))
    return 'arrived'
