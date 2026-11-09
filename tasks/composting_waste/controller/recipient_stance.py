"""Reachability-checked recipient approach using its current RGB-D pose."""
import numpy as np
from manip import pose2
from skills import interpolate_cartesian
from recipient_geometry import release_goals
from compost_navigation import route_to

def approach_recipient(ctx, recipient, occupied=()):
    """Fallback after existing release attempts fail; retain all collision guards."""
    obj, angle = recipient
    if ctx.held is None: return False
    category = ctx.held[1]; relation = ctx.held[2]
    goals = list(release_goals(ctx, category, obj, angle, occupied))
    # Entry is east of the recipient. Offset the base opposite the active arm's
    # lateral reach, then compare short anatomical reaching distances.
    lateral = -.15 if ctx.arm == 'right' else .15
    heading=float(ctx.est.base()[2]) if abs(ctx.chain[3])>.5 else np.pi
    candidates = [(obj[0,3]+radius, obj[1,3]+dy, heading)
                  for radius in (.80,.70,.60) for dy in (lateral,0.,-lateral)]
    start = np.asarray(ctx.est.base())
    candidates.sort(key=lambda stance: np.linalg.norm(np.asarray(stance[:2])-start[:2]))
    moved_attempts = 0
    for stance in candidates:
        if np.linalg.norm(np.asarray(stance[:2])-np.asarray(ctx.est.base())[:2]) < .06: continue
        yield ctx.action()
        base = pose2(*stance); full = ctx.q.copy(); check = ctx.validator(base=base,max_load=1.10)
        if not check(full,ctx.body.fk(full)[0]): continue
        feasible = None
        for goal in goals[::max(1,len(goals)//3)][:3]:
            tip = np.linalg.inv(base) @ goal @ np.linalg.inv(relation)
            solutions = ctx.ik.solve(full,tip[:3,3],tip[:3,:3],count=10,validator=check,all_solutions=True)
            for low in solutions[:3]:
                contact = full.copy(); contact[ctx.ik.columns] = low
                up = interpolate_cartesian(ctx.ik,contact,tip[:3,3],tip[:3,3]+[0.,0.,.15],
                                           tip[:3,:3],validator=check,step=.02)
                if up is not None: feasible=goal; break
            if feasible is not None: break
        if feasible is None: continue
        ctx.event('recipient_stance',phase='candidate',stance=list(stance),release_goal=feasible.tolist())
        status = yield from route_to(ctx,stance,max_steps=650)
        ctx.event('recipient_stance',phase='motion',stance=list(stance),status=status)
        moved_attempts += 1
        if status == 'arrived': return True
        if moved_attempts >= 2: break
    return False
