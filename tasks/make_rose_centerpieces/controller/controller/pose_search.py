"""Observed-space viewpoint search with an articulated SE(2) footprint."""
import time
import numpy as np
from scipy.ndimage import binary_dilation,uniform_filter
from navigation import rot2,wrap
from pose_navigation import PosePlanner,compress_route
class PoseSearch:
    def __init__(self,policy):
        self.policy=policy;self.visited=[];self.debug={};self.age=0;self.goal=None;self.path=None;self.next_plan=0
        self.counter_views=[];self.counter_backside_reached=False
        self.anchor_revision=-1

    def anchor_changed(self):
        revision=self.policy.bin_search.counter_revision
        if revision==self.anchor_revision:return False
        self.anchor_revision=revision
        self.counter_views=[]
        if not self.policy.bin_search.confirmed:self.counter_backside_reached=False
        self.next_plan=0
        return True

    def rear_area(self,odom):
        search=self.policy.bin_search
        if search.counter is None or search.counter_origin is None or not search.counter_from_food:return False
        normal=search.counter[:2]-search.counter_origin;normal/=max(np.linalg.norm(normal),1e-6)
        rear=(odom[:2]-search.counter[:2])@normal>.15
        destination=search.fridge
        if search.confirmed and search.bin_pose is not None:destination=search.bin_pose[:3,3]
        near=destination is None or np.linalg.norm(odom[:2]-destination[:2])<1.5
        return bool(rear and near)
    def planner(self,depth):
        p=self.policy
        return PosePlanner(depth,p.robot,p.servo,p.current_q,p.level,p.held,None if p.held is None else p.bounds(p.child))
    def plan(self,depth,odom,phase):
        if self.policy.step<self.next_plan:return False
        self.next_plan=self.policy.step+45
        counter_blocked=False;counter_attempts=[]
        routing_to_bin=phase=='find_bin' or (phase=='search_target' and self.policy.held is not None)
        if routing_to_bin and self.policy.bin_search.counter is not None and not self.counter_backside_reached:
            if not self.policy.bin_search.counter_from_food:
                self.debug=dict(kind='waiting_for_fruit_counter');return False
            from counter_route import counter_viewpoints
            proposals=counter_viewpoints(depth,self.policy.bin_search,odom,self.counter_views)
            planner=self.planner(depth);deadline=time.monotonic()+8.;attempts=[]
            for kind,goal in proposals:
                remaining=deadline-time.monotonic()
                if remaining<=0:break
                delta=goal-odom[:2]
                yaw=float(np.arctan2(delta[1],delta[0]))
                route=planner.plan(odom,np.r_[goal,yaw],seconds=min(2.,remaining),forward_only=True)
                attempts.append(dict(kind=kind,goal=goal.tolist(),planning=planner.stats.copy()))
                if route is None:continue
                self.goal=goal;self.route=compress_route(route);self.index=0;self.age=0
                self.motion_budget=max(900,min(1800,int(sum(np.linalg.norm(b[:2]-a[:2]) for a,b in zip(route,route[1:]))/.06*30+240)))
                self.counter_route_kind=kind
                self.debug=dict(kind=kind,goal=goal.tolist(),planner=planner.stats,attempts=attempts)
                return True
            if (phase=='search_target' and self.policy.held is not None) or (phase=='find_bin' and self.policy.bin_search.fridge is not None):
                # Close pickup stances may permit a short translation before
                # there is room to turn onto the full counter-side route.
                guides=[goal for kind,goal in proposals if kind=='counter_side']
                if guides:
                    guide=self.policy.bin_search.fridge[:2] if phase=='find_bin' and self.policy.bin_search.fridge is not None else guides[0]
                    local=[]
                    for radius in (.36,.24,.12):
                        for angle in np.linspace(-np.pi,np.pi,16,endpoint=False):
                            goal=odom[:2]+radius*np.array([np.cos(angle),np.sin(angle)])
                            if phase=='find_bin' and (goal-odom[:2])@np.array([np.cos(odom[2]),np.sin(odom[2])])<.02:continue
                            progress=np.linalg.norm(odom[:2]-guide)-np.linalg.norm(goal-guide)
                            if progress<.03 or any(np.linalg.norm(goal-old)<.08 for old in self.counter_views):continue
                            local.append((np.linalg.norm(goal-guide)-.2*radius,goal))
                    for _,goal in sorted(local,key=lambda x:x[0])[:30]:
                        remaining=deadline-time.monotonic()
                        if remaining<=0:break
                        route=planner.plan(odom,np.r_[goal,odom[2]],seconds=min(.25,remaining),fixed_heading=True,no_reverse=phase=='find_bin')
                        if route is None:continue
                        self.goal=goal;self.route=compress_route(route);self.index=0;self.age=0;self.motion_budget=360
                        self.counter_route_kind='counter_clearance'
                        self.debug=dict(kind='counter_clearance',goal=goal.tolist(),guide=guide.tolist(),planner=planner.stats,attempts=attempts)
                        return True
            # Keep the explicit route objective; turn to reveal more floor
            # before retrying rather than drifting to an unrelated frontier.
            counter_blocked=True;counter_attempts=attempts
            self.debug=dict(kind='counter_route_blocked',attempts=attempts)
        self.counter_route_kind=None
        known=(depth.log_odds[:,:,:8]<0).any(axis=2);occupied=(depth.log_odds>=2).any(axis=2)
        frontier=known&binary_dilation(~known,iterations=2)&~occupied
        information=uniform_filter((~known).astype(float),size=25)
        cells=np.argwhere(frontier);proposals=[]
        if len(cells):
            points=depth.xy(cells);distance=np.linalg.norm(points-odom[:2],axis=1)
            valid=(distance>.5)&(distance<5.)
            for old in self.visited:valid&=np.linalg.norm(points-old,axis=1)>.4
            score=3*information[cells[:,0],cells[:,1]]-.5*distance
            for k in np.argsort(np.where(valid,score,-np.inf))[-40:][::-1]:
                if not valid[k]:continue
                goal=points[k]
                if any(np.linalg.norm(goal-other[1])<.25 for other in proposals):continue
                proposals.append((float(score[k]),goal))
        for radius in (.55,.8,1.2,1.7,2.2):
            for angle in np.linspace(-np.pi,np.pi,24,endpoint=False):
                goal=odom[:2]+radius*np.array([np.cos(angle),np.sin(angle)]);cell=depth.ij(goal)
                if np.any(cell<0) or np.any(cell>=depth.n) or not known[tuple(cell)] or occupied[tuple(cell)]:continue
                novelty=min([np.linalg.norm(goal-p) for p in self.visited] or [2.])
                if novelty<.4:continue
                proposals.append((min(novelty,1.5)+information[tuple(cell)]-.5*radius,goal))
        # Favor viewpoints near the observed food context when looking for a bin.
        # This uses only saved RGB-D estimates, never task-instance locations.
        context=None
        if phase=='search_target' and self.policy.grasp is not None:
            context=self.policy.grasp.get('world_pose')
        if context is not None:
            proposals=[(score-.45*np.linalg.norm(goal-context[:2,3]),goal) for score,goal in proposals]
        if phase=='find_bin':
            proposals=[(score+self.policy.bin_search.viewpoint_bonus(goal),goal) for score,goal in proposals]
            if counter_blocked and self.policy.bin_search.fridge is not None:
                proposals=[(score-.2*np.linalg.norm(goal-self.policy.bin_search.fridge[:2]),goal) for score,goal in proposals]
        planner=self.planner(depth);deadline=time.monotonic()+4.
        for score,goal in sorted(proposals,key=lambda x:-x[0])[:24]:
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            yaw=odom[2]
            if counter_blocked:
                delta=goal-odom[:2];yaw=float(np.arctan2(delta[1],delta[0]))
            elif phase=='find_bin' and self.policy.bin_search.counter is not None:
                delta=self.policy.bin_search.counter[:2]-goal
                yaw=float(np.arctan2(delta[1],delta[0]))
            route=planner.plan(odom,np.r_[goal,yaw],seconds=min(.9,remaining),forward_only=counter_blocked)
            if route is None:continue
            self.goal=goal;self.route=compress_route(route);self.index=0;self.age=0;self.motion_budget=900;self.visited.append(odom[:2].copy())
            self.counter_route_kind='access_search' if counter_blocked else None
            self.debug=dict(kind='access_search' if counter_blocked else 'frontier',goal=goal.tolist(),score=score,planner=planner.stats,candidates=len(proposals),visited=len(self.visited),counter_attempts=counter_attempts)
            return True
        self.debug=dict(no_reachable_viewpoint=True,candidates=len(proposals),planner=planner.stats);return False
    def step(self,depth,odom,action):
        self.age+=1;action[:3]=0
        if self.age>self.motion_budget:
            self.debug['motion_timeout']=True
            self.visited.append(self.goal.copy());return True
        while self.index<len(self.route):
            point=self.route[self.index]
            turn=self.index>0 and np.linalg.norm(point[:2]-self.route[self.index-1][:2])<1e-6
            tolerance=.12 if turn else .045
            if np.linalg.norm(point[:2]-odom[:2])<tolerance and abs(wrap(point[2]-odom[2]))<.10:self.index+=1
            else:break
        if self.index>=len(self.route):
            kind=getattr(self,'counter_route_kind',None)
            if kind is not None:
                self.counter_views.append(self.goal.copy())
                rear=self.rear_area(odom)
                if rear:self.counter_backside_reached=True
                self.debug['physical_rear_arrival']=rear
                self.debug['viewpoint_reached']=True
            return True
        point=self.route[self.index];error=wrap(point[2]-odom[2])
        if getattr(self,'counter_route_kind',None) not in (None,'counter_clearance'):
            turn=self.index>0 and np.linalg.norm(point[:2]-self.route[self.index-1][:2])<1e-6
            if turn:
                if np.linalg.norm(point[:2]-odom[:2])>.15:
                    self.debug['turn_position_drift']=True;return True
                action[2]=np.clip(error*.9,-.30,.30)
                self.debug['turn_in_place']=True
                return False
            delta=point[:2]-odom[:2];distance=np.linalg.norm(delta)
            if distance>=.045:
                travel_heading=float(np.arctan2(delta[1],delta[0]))
                error=wrap(travel_heading-odom[2])
                action[2]=np.clip(error*.9,-.30,.30)
                if abs(error)<.20:
                    action[0]=min(.20,.9*distance)
                self.debug['travel_heading']=travel_heading
                return False
        action[:2]=np.clip(rot2(odom[2]).T@(point[:2]-odom[:2])*.9,-.20,.20)
        action[2]=np.clip(error*.9,-.30,.30)
        if abs(error)>.25:action[:2]=0
        return False
