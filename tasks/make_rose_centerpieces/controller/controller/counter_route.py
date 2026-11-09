"""Explicit counter-side and rear viewpoints from observed occupancy geometry."""
import numpy as np


def counter_viewpoints(depth,search,odom,visited):
    if search.counter is None:return []
    center=search.counter[:2]
    normal=center-search.counter_origin
    normal/=max(np.linalg.norm(normal),1e-6)
    tangent=np.array([-normal[1],normal[0]])
    heights=depth.z_min+(np.arange(depth.nz)+.5)*depth.res
    top=(heights>search.counter[2]-.18)&(heights<search.counter[2]+.04)
    cells=np.argwhere((depth.log_odds[:,:,top]>=2).any(axis=2))
    points=depth.xy(cells)
    points=points[np.linalg.norm(points-center,axis=1)<1.4]
    offset=points-center
    front,back=-.35,.35;left,right=-.55,.55
    if len(points)>8:
        front=min(front,float(np.quantile(offset@normal,.03)))
        back=max(back,float(np.quantile(offset@normal,.97)))
        left=min(left,float(np.quantile(offset@tangent,.03)))
        right=max(right,float(np.quantile(offset@tangent,.97)))
    clearance=depth.radius+.18
    proposals=[]
    fridge=getattr(search,'fridge',None)
    if getattr(search,'confirmed',False) and getattr(search,'bin_pose',None) is not None:
        # The confirmed destination is stronger evidence than an appliance cue.
        fridge=search.bin_pose[:3,3]
    sides=(left-clearance,right+clearance)
    if fridge is not None:
        # Approach the counter end nearest the observed appliance cue. Keep
        # a stand-off; the landmark itself is occupied geometry, not a goal.
        sides=tuple(sorted(sides,key=lambda side:np.linalg.norm(center+normal*(back+clearance)+tangent*side-fridge[:2])))
        toward=fridge[:2]-center;toward/=max(np.linalg.norm(toward),1e-6)
        goal=fridge[:2]-toward*max(.65,clearance)
        # The appliance cue defines the kitchen side more directly than the
        # broad worktop extent, which may include nearby cabinet surfaces.
        if (goal-center)@normal>.15:
            proposals.append(('behind_counter',goal))
    # Rear goals have priority. If rear floor is unobserved, use a visible
    # side as a staging point, then replan after acquiring new depth.
    for side in ((0.,)+sides if fridge is None else sides):
        goal=center+normal*(back+clearance)+tangent*side
        if any(np.linalg.norm(goal-p)<.18 for p in visited):continue
        proposals.append(('behind_counter',goal))
    side_goals=[]
    for side in sides:
        for along in ((front+back)/2,front-clearance):
            goal=center+normal*along+tangent*side
            if np.linalg.norm(goal-odom[:2])<.18 or any(np.linalg.norm(goal-p)<.18 for p in visited):continue
            side_goals.append(('counter_side',goal))
    side_goals.sort(key=lambda p:np.linalg.norm(p[1]-odom[:2])+(0. if fridge is None else .25*np.linalg.norm(p[1]-fridge[:2])))
    # Exact corners can lie beyond the current camera rays. Approach them
    # through observed floor alongside the counter, then reveal the next leg.
    known=(depth.log_odds[:,:,:8]<0).any(axis=2)
    occupied=(depth.log_odds>=2).any(axis=2)
    cells=np.argwhere(known&~occupied)
    points=depth.xy(cells);offset=points-center
    along=offset@normal;across=offset@tangent
    distance=np.linalg.norm(points-odom[:2],axis=1)
    exterior=(along<front-.08)|(along>back+.08)|(across<left-.08)|(across>right+.08)
    valid=exterior&(distance>.25)&(distance<1.3)
    for old in visited:valid &= np.linalg.norm(points-old,axis=1)>.22
    edge_goals=[]
    if side_goals:
        target=side_goals[0][1]
        remaining=np.linalg.norm(points-target,axis=1)
        valid &= remaining<np.linalg.norm(odom[:2]-target)-.06
        score=remaining+.15*distance
        for i in np.argsort(np.where(valid,score,np.inf)):
            if not valid[i]:break
            goal=points[i]
            if any(np.linalg.norm(goal-p[1])<.25 for p in edge_goals):continue
            edge_goals.append(('counter_edge',goal))
            if len(edge_goals)>=10:break
    return proposals+side_goals+edge_goals
