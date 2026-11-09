"""Heading-aware transport with distinct checked translation and rotation phases."""
import numpy as np
from manip import pose2,drive,hold,BASE_LINEAR_GAIN
from localizer import turn_rate

def checked_carried_drive(ctx,stance,max_steps=1200,keepout=()):
    start=np.asarray(ctx.est.base(),float);goal=np.asarray(stance,float)
    path=ctx.scene.path(start[:2],goal[:2],keepout=keepout)
    if path is None:return 'no_route'
    initial=ctx.q.copy();poses=ctx.body.fk(initial)[0];baseline=ctx.base_matrix()
    def safe(point,yaw,full=initial,frames=poses):
        return ctx.validator(base=pose2(*point,yaw),relax_from=initial,baseline_base=baseline)(full,frames)
    def turn_safe(point,a,b):
        delta=(b-a+np.pi)%(2*np.pi)-np.pi
        return all(safe(point,angle) for angle in np.linspace(a,a+delta,max(2,int(np.ceil(abs(delta)/.05))+1)))
    def path_safe(yaw):
        return all(safe(point,yaw) for a,b in zip(path,path[1:])
                   for point in np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/.025))+1)))
    # The same XY route may be safe only before reorienting the payload. Neither
    # branch weakens a collision margin or interpolates unchecked turn-and-drive poses.
    if turn_safe(start[:2],start[2],goal[2]) and path_safe(goal[2]):
        strategy='turn_then_translate';heading=goal[2]
    elif turn_safe(goal[:2],start[2],goal[2]) and path_safe(start[2]):
        strategy='translate_then_turn';heading=start[2]
    else:
        ctx.event('carried_route',ok=False,reason='no checked transport heading');return 'no_route'
    ctx.event('carried_route',ok=True,strategy=strategy,stance=goal.tolist(),waypoints=[p.tolist() for p in path])
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
    if strategy=='turn_then_translate':
        status=yield from rotate(heading)
        if status!='arrived':return status
    for point in path[1:]:
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
