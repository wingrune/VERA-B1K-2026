"""Bounded joint-space planning against observed geometry and robot self-collision."""
import time,numpy as np
from scene_collision import ObservedSceneCollision
def joint_path(servo,robot,q,target,cams,obj,bounds,seconds=4.,locked=(),payload_local=None,scene_override=None):
 scene=scene_override if scene_override is not None else ObservedSceneCollision(cams,robot,servo.kin.forward(q)[0],obj,*bounds)
 ids=servo.indices;start=q[ids].copy();deadline=time.monotonic()+seconds;rng=np.random.default_rng(481)
 def valid(x):
  trial=q.copy();trial[ids]=x
  payload=None if payload_local is None else servo.pose(trial)@payload_local
  return scene.valid(servo.kin.forward(trial)[0],payload)
 def edge(a,b):
  for alpha in np.linspace(0,1,max(2,int(np.max(abs(a-b))/.055)+1)):
   if time.monotonic()>deadline or not valid(a+alpha*(b-a)):return False
  return True
 if not valid(target):return None
 if edge(start,target):return [target.copy()]
 trees=[([start],[-1]),([target.copy()],[-1])]
 def extend(tree,sample):
  nodes,parents=tree;j=int(np.argmin([np.linalg.norm(x-sample) for x in nodes]));delta=sample-nodes[j];d=np.linalg.norm(delta);new=nodes[j]+delta*min(1.,.3/max(d,1e-8))
  if not edge(nodes[j],new):return None,False
  nodes.append(new);parents.append(j);return len(nodes)-1,d<=.3
 def chain(tree,j):
  out=[]
  while j>=0:out.append(tree[0][j]);j=tree[1][j]
  return out[::-1]
 for iteration in range(1000):
  if time.monotonic()>deadline:break
  side=iteration%2
  sample=trees[1-side][0][-1] if rng.random()<.4 else np.clip(start*rng.random()+target*(1-rng.random())+rng.normal(0,.5,len(ids)),servo.limits[0]+1e-4,servo.limits[1]-1e-4)
  sample=sample.copy();sample[list(locked)]=start[list(locked)]
  a,_=extend(trees[side],sample)
  if a is None:continue
  for _ in range(25):
   b,joined=extend(trees[1-side],trees[side][0][a])
   if b is None:break
   if joined:
    left,right=chain(trees[side],a),chain(trees[1-side],b)
    path=left+right[::-1] if side==0 else right+left[::-1]
    # Remove unnecessary random-tree bends while retaining the same swept checks.
    deadline=time.monotonic()+1.5;short=[path[0]];i=0
    while i<len(path)-1:
     j=len(path)-1
     while j>i+1 and not edge(path[i],path[j]):j-=1
     short.append(path[j]);i=j
    return short[1:]
 return None
