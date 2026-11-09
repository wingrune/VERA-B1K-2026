"""Observed-floor coverage planning, without particles or simulator-state access."""
import numpy as np
from perception import pointcloud
from ground_frame import level_cameras
from navigation import rot2
class FloorCoverage:
 def __init__(self):
  self.samples={};self.completed=[];self.failed=[];self.target=None;self.marks=[];self.mark_attempts=[];self.target_is_mark=False;self.step=0
 def update(self,cams,level,odom):
  for camera in level_cameras(cams,level).values():
   p,_,_=pointcloud(camera,stride=5)
   p=p[(abs(p[:,2])<.012)&(np.linalg.norm(p[:,:2],axis=1)>.4)&(np.linalg.norm(p[:,:2],axis=1)<3.2)]
   p[:,:2]=p[:,:2]@rot2(odom[2]).T+odom[:2]
   for row in p:
    key=tuple(np.rint(row[:2]/.07).astype(int));self.samples[key]=row[:2]
 def observe_marks(self,marks,odom,step):
  self.step=step
  for mark in marks:
   if mark['pixels']<20:continue
   point=np.r_[rot2(odom[2])@np.array(mark['position'][:2])+odom[:2],0.]
   nearest=min(self.marks,key=lambda m:np.linalg.norm(point-m['point']),default=None)
   if nearest is not None and np.linalg.norm(point-nearest['point'])<.12:
    nearest.update(point=point,last=step,hits=nearest['hits']+1,pixels=mark['pixels'])
   else:self.marks.append(dict(point=point,last=step,hits=1,pixels=mark['pixels']))
  self.marks=[m for m in self.marks if step-m['last']<1800]
 def covered(self,xy):
  return any(np.max(abs((np.asarray(xy)-p)@rot2(yaw)))<.18 for p,yaw in self.completed)
 def choose(self,odom,depth_map):
  choices=[]
  for mark in self.marks:
   point=mark['point'];attempts=[t for p,t in self.mark_attempts if np.linalg.norm(p-point[:2])<.15]
   if mark['hits']<2 or len(attempts)>=3 or (attempts and mark['last']<=max(attempts)+45):continue
   if any(np.linalg.norm(point[:2]-p)<.18 for p in self.failed):continue
   distance=np.linalg.norm(point[:2]-odom[:2])
   choices.append((distance,point))
  if choices:
   self.target=min(choices,key=lambda a:a[0])[1].copy();self.target_is_mark=True;return self.target.copy()
  self.target_is_mark=False
  if not self.samples:return None
  points=np.array(list(self.samples.values()));choices=[]
  for point in points[::max(1,len(points)//120)]:
   if self.covered(point):continue
   if any(np.linalg.norm(point-p)<.24 for p in self.failed):continue
   # Require observed floor throughout the proposed patch; avoid furniture edges.
   delta=points-point
   if not all(np.any(np.linalg.norm(delta-[dx,dy],axis=1)<.08) for dx in [-.16,0,.16] for dy in [-.16,0,.16]):continue
   distance=np.linalg.norm(point-odom[:2])
   if distance<.43:continue
   choices.append((distance,point))
  for _,point in sorted(choices,key=lambda a:a[0]):
   cell=depth_map.ij(point)
   if np.all((cell>=0)&(cell<depth_map.n)) and not depth_map.occupancy()[tuple(cell)]:
    self.target=np.r_[point,0.];return self.target.copy()
  return None
 def finish(self,odom):
  if self.target is not None:
   if self.target_is_mark:self.mark_attempts.append((self.target[:2].copy(),self.step))
   self.completed.append((self.target[:2].copy(),float(odom[2])));self.target=None
 def reject(self):
  if self.target is not None:self.failed.append(self.target[:2].copy());self.target=None
