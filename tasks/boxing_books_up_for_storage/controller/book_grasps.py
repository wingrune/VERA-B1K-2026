"""Book pinches using School s91 CPU IK, real hand geometry and checked shelf withdrawal."""
import time
import numpy as np
from art_skills import transform
from manip import pose2
from skills import interpolate_cartesian

def candidates(ctx,category,obj,base=None):
    model=ctx.models[category];lo,hi=model['bounds'];centre=(lo+hi)/2;size=hi-lo
    hand=ctx.gripper[ctx.arm+'_collision']-ctx.ik.tool
    rays=ctx.gripper[ctx.arm+'_ag']-ctx.ik.tool
    world=ctx.base_matrix() if base is None else base
    thin=int(np.argmin(size));closing=obj[:3,thin];options=[]
    # Fingers close across the true thickness; the palm remains outside the selected edge.
    for axis in range(3):
        if axis==thin:continue
        for sign in (-1.,1.):
            outward=sign*obj[:3,axis]
            if outward[2]<-.25:continue
            for reverse in (1.,-1.):
                y=closing*reverse;rot=np.column_stack([np.cross(y,outward),y,outward])
                lateral=next(i for i in range(3) if i not in (thin,axis))
                for fraction in (.5,.25,.75,.10,.90):
                    for depth in (.024,.016,.032):
                        local=centre.copy();local[axis]=(hi[axis] if sign>0 else lo[axis])-sign*depth
                        local[lateral]=lo[lateral]+fraction*size[lateral]
                        point=obj[:3,:3]@local+obj[:3,3];goal=transform(rot,point)
                        local_hand=(hand@rot.T+point-obj[:3,3])@obj[:3,:3]
                        if np.any(np.all(local_hand>lo+.0005,axis=1)&np.all(local_hand<hi-.0005,axis=1)):continue
                        local_rays=(rays@rot.T+point-obj[:3,3])@obj[:3,:3]
                        hit=any(np.any(np.all(np.linspace(a,b,41)>lo+.0005,axis=1)&np.all(np.linspace(a,b,41)<hi-.0005,axis=1))
                                for a in local_rays[:2] for b in local_rays[2:])
                        if not hit:continue
                        cost=np.linalg.norm(point[:2]-world[:2,3])+.08*abs(fraction-.5)
                        options.append((cost,goal))
    # Thick hardbacks may allow a direct top corner pinch before any pushing.
    from school_grasps import corner_candidates
    for original in corner_candidates(ctx,category,obj,base=base):
        for pitch in (0.,.45,.75,1.0,-.45,-.75,-1.0):
            c,s=np.cos(pitch),np.sin(pitch);goal=original.copy()
            goal[:3,:3]=original[:3,:3]@np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
            points=hand@goal[:3,:3].T+goal[:3,3]
            support=ctx.scene.support_heights(points)
            goal[2,3]+=max(0.,float(np.max(support+.0008-points[:,2])))
            local_hand=(hand@goal[:3,:3].T+goal[:3,3]-obj[:3,3])@obj[:3,:3]
            if np.any(np.all(local_hand>lo+.0005,axis=1)&np.all(local_hand<hi-.0005,axis=1)):continue
            lr=(rays@goal[:3,:3].T+goal[:3,3]-obj[:3,3])@obj[:3,:3]
            hit=any(np.any(np.all(np.linspace(a,b,41)>lo+.0005,axis=1)&np.all(np.linspace(a,b,41)<hi-.0005,axis=1)) for a in lr[:2] for b in lr[2:])
            if hit:
                length=0.
                for a,b in ((lr[0],lr[2]),(lr[1],lr[3])):
                    line=np.linspace(a,b,101);inside=np.all(line>lo+.001,axis=1)&np.all(line<hi-.001,axis=1)
                    length=max(length,float(inside.mean()*np.linalg.norm(a-b)))
                options.append((np.linalg.norm(goal[:2,3]-world[:2,3])+.02*(1.-abs(pitch))-.9*length,goal))
    for _,goal in sorted(options,key=lambda pair:pair[0]):yield goal

def plan(ctx,category,obj,base=None,q=None,count=12,candidate_limit=None,preferred_candidate=None,seconds=25.):
    end=time.monotonic()+seconds;world=ctx.base_matrix() if base is None else base;full=ctx.q if q is None else q
    goals=list(candidates(ctx,category,obj,base));indices=list(range(len(goals)))
    if preferred_candidate is not None and preferred_candidate in indices:
        indices.remove(preferred_candidate);indices.insert(0,preferred_candidate)
    if candidate_limit is not None and len(indices)>candidate_limit:
        indices=list(dict.fromkeys(indices[:min(8,candidate_limit)]+[indices[j] for j in np.linspace(0,len(indices)-1,candidate_limit).astype(int)]))[:candidate_limit]
    check=ctx.validator(base=base,max_load=1.05);hand=ctx.gripper[ctx.arm+'_collision']-ctx.ik.tool
    rejected={'geometry':0,'ik':0,'path':0}
    for index in indices:
        if time.monotonic()>end:break
        goal=goals[index];points=hand@goal[:3,:3].T+goal[:3,3]
        if np.min(ctx.scene.contact_clearance(points,np.full(len(points),.0004)))<-.0002:
            rejected['geometry']+=1;continue
        relative=np.linalg.inv(obj)@goal
        geometry_key=(category,tuple(np.round(relative[:3,:3].ravel(),2)),tuple(np.round(relative[:3,3],3)))
        key=(category,ctx.arm,round(float(world[0,3]),1),round(float(world[1,3]),1),*geometry_key[1:])
        if key in ctx.failed_pinches:continue
        tip=np.linalg.inv(world)@goal;point=tip[:3,3];rotation=tip[:3,:3]
        lows=ctx.ik.solve(full,point,rotation,count=count,validator=check,all_solutions=True,
            position_tolerance=.0008,rotation_tolerance=.02,rest_weight=.0005)
        if not lows:rejected['ik']+=1
        for low in lows[:3]:
            start=full.copy();start[ctx.ik.columns]=low
            # Enter and withdraw along the palm axis; never lower a finger through the book.
            direction=rotation[:,2];distance=.16 if abs(goal[2,2])<.7 else .12
            approach=interpolate_cartesian(ctx.ik,start,point,point+direction*distance,rotation,validator=check,step=.01)
            if approach is None:rejected['path']+=1;continue
            # The lifted payload is checked against the shelf as well as the robot.
            old=ctx.held;ctx.held=(ctx.arm,category,np.linalg.inv(tip)@np.linalg.inv(world)@obj)
            try:
                loaded=ctx.validator(base=base,relax_from=start,max_load=1.10)
                cabinet=ctx.scene.meta['bookcases'][ctx.scene.meta['bookcase_assignment'][category]]
                lo=np.asarray(cabinet['aabb_lo']);hi=np.asarray(cabinet['aabb_hi']);axis=int(np.argmin((hi-lo)[:2]))
                front=np.zeros(3);front[axis]=np.sign(world[axis,3]-(lo[axis]+hi[axis])/2)
                delta=world[:3,:3].T@front*.22
                withdrawn=interpolate_cartesian(ctx.ik,start,point,point+delta,rotation,validator=loaded,step=.01)
                if withdrawn is None:continue
                finish=start.copy();finish[ctx.ik.columns]=withdrawn[-1]
                lift=interpolate_cartesian(ctx.ik,finish,point+delta,point+delta+[0.,0.,.10],rotation,validator=loaded,step=.02)
                if lift is None:continue
            finally:ctx.held=old
            return dict(low=low,up=withdrawn+lift,approach=approach,pre=approach[-1],tip=tip,world_tip=goal,
                solutions=len(lows),candidate=index,key=key,geometry_key=geometry_key)
    ctx.event('book_plan_rejected',category=category,candidates=len(goals),rejected=rejected)
    return None
