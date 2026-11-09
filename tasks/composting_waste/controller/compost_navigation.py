"""Task-local routing composition of the proven whole-body carried navigator."""
import numpy as np
from manip import pose2,hold
from navigation_geometry import planning_scene
from carried_navigation import checked_carried_drive

def _route_once(ctx,stance,max_steps=1200):
    start=np.asarray(ctx.est.base());goal=np.asarray(stance)
    prepared=getattr(ctx,'prepared_route',None);ctx.prepared_route=None
    if prepared is not None and np.linalg.norm(prepared['start']-start)<.02 and np.linalg.norm(prepared['goal']-goal)<.001:
        return (yield from checked_carried_drive(ctx,goal,max_steps=max_steps,route=prepared['path']))
    # The shared navigator checks turning before, after, or along the route.
    # Rectangular route generation supplies candidates where its circular map is conservative.
    full=ctx.q.copy();poses=ctx.body.fk(full)[0];baseline=ctx.base_matrix()
    tried=[]
    axis=float(np.round(start[2]/(np.pi/2))*(np.pi/2))
    headings=(float(start[2]),float(goal[2]))
    if float(getattr(ctx,'route_reserve',0.))>0:headings+=(axis,)
    for yaw in headings:
        if any(abs((yaw-old+np.pi)%(2*np.pi)-np.pi)<.001 for old in tried):continue
        tried.append(yaw)
        reserve=float(getattr(ctx,'route_reserve',0.))
        centres=ctx.collision.centers(poses)[ctx.base_part];radii=ctx.collision.radii[ctx.base_part]
        def chassis_gap(point):
            base=pose2(*point,yaw);points=centres@base[:3,:3].T+base[:3,3]
            return float(np.min(ctx.scene.contact_clearance(points,radii,ignore_below=np.full(len(points),.06))))
        endpoints=[max(0.,chassis_gap(point)-.002) for point in (start[:2],goal[:2])] if reserve else [0.,0.]
        def valid(point):
            base=pose2(*point,yaw)
            if reserve:
                # The requested end stance may have less than the extra search
                # reserve. Taper only that extra reserve over its final centimetres;
                # the unchanged whole-body collision validator remains authoritative.
                required=min(reserve,endpoints[0]+np.linalg.norm(point-start[:2]),endpoints[1]+np.linalg.norm(point-goal[:2]))
                if chassis_gap(point)<required:return False
            return ctx.validator(base=base,relax_from=full,baseline_base=baseline)(full,poses)
        from checked_route import path as checked_path
        # The raster planner supplies fast long-room candidates. Its permissive
        # raster shortcut is never authoritative: check every <=1cm edge with
        # the same actual full-body/payload validator before accepting it.
        path=ctx.scene.chassis_path(start[:2],goal[:2],yaw,valid=valid)
        if path is not None:
            if not all(valid(point) for a,b in zip(path,path[1:])
                       for point in np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/.01))+1))):path=None
        if path is None:path=checked_path(start[:2],goal[:2],valid)
        if path is None and reserve:
            # A 30mm grid can miss a narrow connected region that still meets
            # the identical extra clearance reserve and whole-body checks.
            path=checked_path(start[:2],goal[:2],valid,resolution=.015,seconds=15.,max_expansions=12000)
            ctx.event('route_resolution_refinement',resolution=.015,reserve=reserve,found=path is not None)
        if path is None:continue
        canonical=abs((yaw-start[2]+np.pi)%(2*np.pi)-np.pi)>.001 and abs((yaw-goal[2]+np.pi)%(2*np.pi)-np.pi)>.001
        if canonical:
            # A slight measured yaw error can close a fixed-heading corridor.
            # Search the static scene's nearest orthogonal axis as a transit
            # heading, with exactly the existing swept turn/body checks.
            def turn_clear(point,a,b):
                delta=(b-a+np.pi)%(2*np.pi)-np.pi
                return all(ctx.validator(base=pose2(*point,angle),relax_from=full,baseline_base=baseline)(full,poses)
                           for angle in np.linspace(a,a+delta,max(2,int(np.ceil(abs(delta)/.04))+1)))
            if not turn_clear(start[:2],start[2],yaw) or not turn_clear(goal[:2],yaw,goal[2]):continue
            begin=ctx.steps;transit=goal.copy();transit[2]=yaw
            ctx.event('canonical_transit_heading',heading=yaw,goal=goal.tolist(),reserve=reserve)
            status=yield from checked_carried_drive(ctx,transit,max_steps=max_steps,route=path)
            if status=='arrived':
                remaining=max_steps-(ctx.steps-begin)
                if remaining<=0:return 'timeout'
                here=np.asarray(ctx.est.base())[:2]
                status=yield from checked_carried_drive(ctx,goal,max_steps=remaining,route=[here,goal[:2]])
        else:
            status=yield from checked_carried_drive(ctx,goal,max_steps=max_steps,route=path)
        ctx.event('composite_route',status=status,heading=yaw,goal=goal.tolist())
        if status=='arrived':return status
        # After actual motion, all subsequent geometry must use a new baseline.
        if np.linalg.norm(np.asarray(ctx.est.base())[:2]-start[:2])>.02:return status
    return 'no_route'


def route_to(ctx,stance,max_steps=1200):
    start=np.array(ctx.est.base());before=ctx.steps
    status=yield from _route_once(ctx,stance,max_steps)
    moved=np.linalg.norm(np.array(ctx.est.base())[:2]-start[:2])>.02
    remaining=max_steps-(ctx.steps-before)
    if status!='arrived' and moved and remaining>100:
        # A fresh observation can invalidate a precomputed sweep. Stop, settle and
        # replan once from the updated estimate; never execute the invalid edge.
        yield from hold(ctx,10)
        from base_clearance import restore_clearance
        if not (yield from restore_clearance(ctx)):return status
        ctx.event('route_replan',reason=status,start=start.tolist(),updated=list(ctx.est.base()))
        status=yield from _route_once(ctx,stance,remaining)
    return status
