"""Bounded SE(2) lattice search against frozen observed 3-D occupancy.

Future robot poses transform static robot spheres; recorded pixels are never
reinterpreted with hypothetical future camera/robot calibration.
"""
import heapq,math,time
import numpy as np
from scipy.spatial import cKDTree
from navigation import rot2,wrap

class PosePlanner:
    def __init__(self,depth,robot,servo,q,level,held=None,bounds=None):
        cells=np.argwhere(depth.log_odds>=2)
        points=np.c_[depth.xy(cells[:,:2]),depth.z_min+(cells[:,2]+.5)*depth.res]
        self.tree=cKDTree(points);self.depth=depth
        local=robot.centers(servo.kin.forward(q)[0]);self.centers=local@level[:3,:3].T+level[:3,3]
        self.radii=robot.radii.copy()+.015
        if held is not None:
            pose=servo.pose(q)@held;lo,hi=bounds;center=pose[:3,3]+pose[:3,:3]@((lo+hi)/2)
            center=level[:3,:3]@center+level[:3,3]
            self.centers=np.r_[self.centers,center[None]];self.radii=np.r_[self.radii,np.linalg.norm(hi-lo)/2+.02]
        for extra in getattr(robot,'extra_payloads',[]):
            pose=servo.kin.forward(q)[0][extra['arm']+'_gripper_link'].copy()
            pose[:3,3]+=pose[:3,:3]@np.array([0.,0.,-.072]);pose=pose@extra['local']
            lo,hi=extra['bounds'];center=pose[:3,3]+pose[:3,:3]@((lo+hi)/2)
            center=level[:3,:3]@center+level[:3,3]
            self.centers=np.r_[self.centers,center[None]];self.radii=np.r_[self.radii,np.linalg.norm(hi-lo)/2+.02]
        self.known=(depth.log_odds[:,:,:8]<0).any(axis=2)
        self.angles=np.arange(24)*2*np.pi/24
        self.rotated=[self.centers.copy() for _ in self.angles]
        for centers,angle in zip(self.rotated,self.angles):centers[:,:2]=centers[:,:2]@rot2(angle).T
        self.stats={}

    def margins(self,xy,angle):
        centers=self.centers.copy();centers[:,:2]=centers[:,:2]@rot2(angle).T+xy
        return self.tree.query(centers,k=1)[0]-self.radii

    def plan(self,start,goal,seconds=2.,max_expansions=8000,forward_only=False,fixed_heading=False,no_reverse=False):
        start=np.asarray(start);goal=np.asarray(goal);deadline=time.monotonic()+seconds
        axy=self.depth.ij(start[:2]);offset=start[:2]-self.depth.xy(axy) if fixed_heading else np.zeros(2)
        gxy=self.depth.ij(goal[:2]-offset)
        sa=int(np.rint(start[2]*24/(2*np.pi)))%24;ga=int(np.rint(goal[2]*24/(2*np.pi)))%24
        a=(*axy,sa);b=(*gxy,ga);cache={}
        baseline=self.margins(start[:2],start[2]);threshold=np.minimum(0.,baseline-.004)
        def valid(node):
            if node in cache:return cache[node]
            x,y,angle=node
            if not (0<=x<self.depth.n and 0<=y<self.depth.n):return False
            xy=self.depth.xy([x,y])+offset
            known_cell=self.depth.ij(xy)
            if np.any(known_cell<0) or np.any(known_cell>=self.depth.n):cache[node]=False;return False
            if np.linalg.norm(xy-start[:2])>.48 and not self.known[tuple(known_cell)]:cache[node]=False;return False
            centers=self.rotated[angle].copy();centers[:,:2]+=xy
            if fixed_heading:
                centers=self.centers.copy();centers[:,:2]=centers[:,:2]@rot2(start[2]).T+xy
            value=bool(np.all(self.tree.query(centers,k=1)[0]-self.radii>=threshold))
            cache[node]=value;return value
        if not valid(b) or (fixed_heading and np.any(self.margins(goal[:2],start[2])<threshold)):
            self.stats=dict(reason='goal_pose_collision_or_unknown');return None
        def heuristic(n):
            return np.linalg.norm((np.array(n[:2])-gxy)*self.depth.res)+.12*abs(wrap(self.angles[n[2]]-goal[2]))
        queue=[(heuristic(a),0.,a)];cost={a:0.};parents={};expanded=0
        while queue and expanded<max_expansions and time.monotonic()<deadline:
            _,g,node=heapq.heappop(queue)
            if g>cost[node]+1e-8:continue
            expanded+=1
            if node==b:
                chain=[node]
                while node!=a:node=parents[node];chain.append(node)
                route=[np.r_[self.depth.xy(n[:2])+offset,start[2] if fixed_heading else self.angles[n[2]]] for n in chain[::-1]]
                if forward_only:
                    # Initial turns stay at the actual start, not at a nearby
                    # lattice-cell center that would look like a tiny drive.
                    for point,n in zip(route,chain[::-1]):
                        if tuple(n[:2])!=tuple(axy):break
                        point[:2]=start[:2]
                route[0]=start.copy();route[-1]=goal.copy()
                self.stats=dict(reason='found',expansions=expanded,poses=len(route),cache_entries=len(cache));return route
            x,y,angle=node
            neighbors=[(x+dx,y+dy,angle,self.depth.res*math.hypot(dx,dy)) for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]]
            if not fixed_heading:neighbors += [(x,y,(angle+d)%24,.04) for d in (-1,1)]
            for nx,ny,na,edge_cost in neighbors:
                if no_reverse and (nx!=x or ny!=y) and np.array([nx-x,ny-y])@np.array([np.cos(start[2]),np.sin(start[2])])<-.001:continue
                if forward_only and (nx!=x or ny!=y):
                    direction=np.array([nx-x,ny-y],float)
                    heading=np.array([np.cos(self.angles[angle]),np.sin(self.angles[angle])])
                    if direction@heading/np.linalg.norm(direction)<np.cos(np.pi/8):continue
                candidate=(nx,ny,na)
                if not valid(candidate):continue
                if na!=angle:
                    xy=self.depth.xy([x,y]);delta=wrap(self.angles[na]-self.angles[angle])
                    if any(np.any(self.margins(xy,self.angles[angle]+alpha*delta)<threshold) for alpha in (.25,.5,.75)):continue
                elif nx!=x and ny!=y:
                    if not valid((nx,y,angle)) or not valid((x,ny,angle)):continue
                new_cost=g+edge_cost
                if new_cost>=cost.get(candidate,float('inf')):continue
                cost[candidate]=new_cost;parents[candidate]=node
                heapq.heappush(queue,(new_cost+1.5*heuristic(candidate),new_cost,candidate))
        self.stats=dict(reason='search_limit' if queue else 'no_route',expansions=expanded,cache_entries=len(cache));return None

def compress_route(route):
    """Collapse checked straight runs and monotonic stationary turns below pi.

    Keep angular reversals, translation bends and long turns, so the controller
    follows the same shortest angular sweep already checked by the lattice.
    """
    compressed=[route[0]]
    for point in route[1:]:
        if len(compressed)>=2:
            a,b=compressed[-2:];u=b[:2]-a[:2];v=point[:2]-b[:2]
            stationary=np.linalg.norm(u)<1e-8 and np.linalg.norm(v)<1e-8
            ab=wrap(b[2]-a[2]);bc=wrap(point[2]-b[2])
            if stationary and ab*bc>0 and abs(ab+bc)<np.pi-1e-6:
                compressed[-1]=point;continue
            same=abs(wrap(a[2]-b[2]))<1e-6 and abs(wrap(b[2]-point[2]))<1e-6
            if same and np.linalg.norm(u)>1e-5 and np.linalg.norm(v)>1e-5 and abs(u[0]*v[1]-u[1]*v[0])<1e-6 and u@v>0:
                compressed[-1]=point;continue
        compressed.append(point)
    return compressed
