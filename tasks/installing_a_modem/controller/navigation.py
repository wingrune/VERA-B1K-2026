"""Local depth occupancy map; all coordinates are initialized at episode start."""
import heapq,math
import numpy as np
from scipy.ndimage import binary_dilation
from perception import pointcloud

def rot2(yaw):return np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
def wrap(x):return (x+math.pi)%(2*math.pi)-math.pi
class DepthMap:
 def __init__(self):
  self.res=.06;self.n=160;self.radius=.276;self.z_min=.10;self.nz=24
  self.log_odds=np.zeros((self.n,self.n,self.nz),np.int8);self.last_plan_expansions=0;self.last_update={}
 def ij(self,xy):return np.floor(np.asarray(xy)/self.res+self.n/2).astype(int)
 def xy(self,ij):return (np.asarray(ij)-self.n/2+.5)*self.res
 @staticmethod
 def self_mask(points,capsules):
  mask=np.linalg.norm(points[:,:2],axis=1)<.47
  for a,b,radius in capsules or []:
   axis=b-a;alpha=np.clip(((points-a)@axis)/max(float(axis@axis),1e-10),0,1)
   mask|=np.linalg.norm(points-a-alpha[:,None]*axis,axis=1)<radius
  return mask
 def voxel_indices(self,points):
  xy=self.ij(points[:,:2]);z=np.floor((points[:,2]-self.z_min)/self.res).astype(int)
  cells=np.c_[xy,z];valid=np.all((cells>=0)&(cells<np.array([self.n,self.n,self.nz])),axis=1)
  cells=cells[valid]
  return np.unique((cells[:,0]*self.n+cells[:,1])*self.nz+cells[:,2])
 def integrate(self,points,origin,is_self=None):
  points=np.asarray(points,float);origin=np.asarray(origin,float)
  if not len(points):return
  is_self=np.zeros(len(points),bool) if is_self is None else np.asarray(is_self,bool)
  # Clear only voxels actually crossed by observed 3-D rays. A ray above a low
  # obstacle must never erase the entire ground-plane column.
  sample=points[::max(1,int(np.ceil(len(points)/512)))];rays=sample-origin
  lengths=np.linalg.norm(rays,axis=1);directions=rays/np.maximum(lengths[:,None],1e-9)
  distances=np.arange(self.res/2,5.,self.res)
  visible=distances[None,:]<(lengths[:,None]-self.res*.75)
  free_points=(origin+directions[:,None,:]*distances[None,:,None])[visible]
  free=self.voxel_indices(free_points);hit=self.voxel_indices(points[~is_self]);flat=self.log_odds.ravel()
  flat[free]=np.maximum(-8,flat[free]-1);flat[hit]=np.minimum(8,flat[hit]+3)
  self.last_update={'free_voxels':int(len(free)),'hit_voxels':int(len(hit)),'self_points_rejected':int(is_self.sum()),'occupied_voxels':int(np.count_nonzero(flat>=2))}
 def update(self,cams,odom,robot_capsules=None):
  points,_,_=pointcloud(cams['head'],stride=5)
  is_self=self.self_mask(points,robot_capsules)
  points=points.copy();points[:,:2]=points[:,:2]@rot2(odom[2]).T+odom[:2]
  origin=cams['head']['t'].copy();origin[:2]=rot2(odom[2])@origin[:2]+odom[:2]
  self.integrate(points,origin,is_self)
 def occupancy(self):
  occupied=(self.log_odds>=2).any(axis=2);width=int(np.ceil(self.radius/self.res));k=np.arange(-width,width+1);xx,yy=np.meshgrid(k,k)
  return binary_dilation(occupied,structure=xx*xx+yy*yy<=(self.radius/self.res)**2)
 def path(self,start,goal):
  occ=self.occupancy();a=self.ij(start);b=self.ij(goal)
  if not np.all((b>=0)&(b<self.n)):return None
  if occ[tuple(b)]:return None
  # Starting footprint is already physically occupied by the robot, so allow leaving it.
  xs,ys=np.ogrid[:self.n,:self.n];occ[(xs-a[0])**2+(ys-a[1])**2<(self.radius/self.res)**2]=False
  a=tuple(a);b=tuple(b);queue=[(0.,0.,a)];cost={a:0.};prev={}
  expanded=0
  while queue and expanded<16000:
   expanded+=1
   _,g,u=heapq.heappop(queue)
   if g>cost[u]+1e-6:continue
   if u==b:
    path=[u]
    while u!=a:u=prev[u];path.append(u)
    return [self.xy(p) for p in path[::-1]]
   for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
    v=(u[0]+dx,u[1]+dy)
    if not (0<=v[0]<self.n and 0<=v[1]<self.n) or occ[v]:continue
    if dx and dy and (occ[u[0]+dx,u[1]] or occ[u[0],u[1]+dy]):continue
    ng=g+math.hypot(dx,dy)
    if ng<cost.get(v,float('inf')):cost[v]=ng;prev[v]=u;heapq.heappush(queue,(ng+math.hypot(v[0]-b[0],v[1]-b[1]),ng,v))
  return None
 def line_clear(self,start,goal):
  occ=self.occupancy();a=self.ij(start);xs,ys=np.ogrid[:self.n,:self.n]
  occ[(xs-a[0])**2+(ys-a[1])**2<(self.radius/self.res)**2]=False
  points=np.asarray(start)[None]+np.linspace(0,1,max(2,int(np.linalg.norm(np.asarray(goal)-start)/self.res*2)))[:,None]*(np.asarray(goal)-start)
  cells=self.ij(points)
  return bool(np.all((cells>=0)&(cells<self.n)) and not occ[cells[:,0],cells[:,1]].any())
 def goal(self,button,normal,odom,side_fallback=False,avoid=()):
  choices=[];occ=self.occupancy();normal=np.asarray(normal).copy()
  if np.linalg.norm(normal)<.4:normal=odom[:2]-button
  normal/=max(np.linalg.norm(normal),1e-8)
  for radius in [.70,.78,.86,.94]:
   for turn in (np.linspace(-math.pi,math.pi,33) if side_fallback else np.linspace(-math.pi*.375,math.pi*.375,19)):
    if side_fallback and abs(turn)<=math.pi*.375:continue
    direction=rot2(turn)@normal;heading=math.atan2(-direction[1],-direction[0]);lateral=np.array([-math.sin(heading),math.cos(heading)]);goal=button+direction*radius+lateral*.18;ij=self.ij(goal)
    if not np.all((ij>=0)&(ij<self.n)) or occ[tuple(ij)]:continue
    if any(np.linalg.norm(goal-point)<.24 for point in avoid):continue
    cost=.10*abs(turn)+abs(radius-.75)+3*np.linalg.norm(goal-odom[:2]);choices.append((cost,goal,heading))
  for _,goal,heading in sorted(choices,key=lambda x:x[0])[:12]:
   path=self.path(odom[:2],goal)
   if path:return goal,heading,path
  if not side_fallback:return self.goal(button,normal,odom,side_fallback=True,avoid=avoid)
  return None
