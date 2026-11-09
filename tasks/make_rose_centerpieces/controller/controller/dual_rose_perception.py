"""CPU prototype: distinct RGB-D rose tracks and independent per-hand tracking.
No simulator imports, Native state, or post-release observations are required.
The caller supplies accepted RGB-D/world transforms or commanded FK release poses.
"""
import time
import numpy as np
from rose_perception import RosePerception
from scipy.spatial import cKDTree
from rose_geometry import masks
from perception import pointcloud
from rose_geometry import rose_candidates,guided_rose_fit,visible_model_support

class DualRosePerception:
 def __init__(self,folder):
  self.perception=RosePerception(folder);self.tracks={};self.hands={'left':None,'right':None};self.released=[];self.next_id=0;self.diagnostics={}
 def _valid_pose(self,p):
  return isinstance(p,np.ndarray) and p.shape==(4,4) and np.isfinite(p).all() and np.allclose(p[3],[0,0,0,1]) and np.allclose(p[:3,:3].T@p[:3,:3],np.eye(3),atol=1e-4) and np.isclose(np.linalg.det(p[:3,:3]),1.,atol=1e-4)
 def _excluded(self,world):
  # Same conservative point exclusion as existing acquisition. It is used only
  # for table acquisition, never to confirm another placement inside the vase.
  return any(np.linalg.norm(world[:3,3]-p[:3,3])<.14 for p in self.released)
 def acquire(self,cameras,base,level,step):
  started=time.monotonic()
  if not self._valid_pose(base) or not self._valid_pose(level):return []
  self.perception.level=level;self.perception.world_transform=lambda:base
  existing=self.perception.all_roses(cameras)
  # The original fallback runs only when SIFT finds nothing. For two hands,
  # always inspect independent red clusters, then merge duplicate roots.
  extra=[] if any(s.get('method')=='rose_color_canonical_icp' for _,s in existing) else rose_candidates(self.perception.models['rose'],cameras,level)
  raw=existing+extra;accepted=[];rejected=[]
  red_mask,_=masks(self.perception.models['rose']['rgb'][None]);red_model=self.perception.models['rose']['xyz'][red_mask[0]][::3]
  for pose,stats in sorted(raw,key=lambda x:x[1].get('residual',1)):
   if not self._valid_pose(pose):continue
   world=base@pose
   if any(t['state'] in ('assigned','held') and np.linalg.norm(world[:3,3]-t['world_pose'][:3,3])<.065 for t in self.tracks.values()):continue
   if self._excluded(world):rejected.append({'reason':'released_fk_root','root':world[:3,3].tolist()});continue
   if any(np.linalg.norm(world[:3,3]-p['world_pose'][:3,3])<.065 for p in accepted):continue
   support=visible_model_support(self.perception.models['rose'],cameras,pose)
   if support['matched_fraction']<.65 or support['unoccluded_fraction']<=.5:
    rejected.append({'reason':'depth','stats':stats,'support':support});continue
   # Distinct roots alone can alias one flower under different rotations.
   # Require fresh observedred support and deduplicate its actual3D centroid.
   red_tree=cKDTree(red_model@pose[:3,:3].T+pose[:3,3]);anchors=[]
   for role,c in cameras.items():
    rm,_=masks(c['rgb']);points=pointcloud(c,rm)[0];points=points[red_tree.query(points)[0]<.014]
    if len(points)>=20:anchors.append((len(points),np.median(points,axis=0),role))
   if not anchors:rejected.append({'reason':'insufficient_independent_red_support'});continue
   count,anchor,role=max(anchors,key=lambda a:a[0]);anchor_world=base[:3,:3]@anchor+base[:3,3]
   if any(np.linalg.norm(anchor_world-p['observed_flower_world'])<.035 for p in accepted):continue
   accepted.append({'body_pose':pose,'world_pose':world,'observed_flower_world':anchor_world,'stats':dict(stats,depth_support=support,red_support_points=count,red_support_camera=role)})
  available=[i for i,t in self.tracks.items() if t['state']=='available'];used=set()
  for candidate in accepted:
   choices=[(np.linalg.norm(candidate['world_pose'][:3,3]-self.tracks[i]['world_pose'][:3,3]),i) for i in available if i not in used]
   distance,ident=min(choices) if choices else (np.inf,None)
   if distance>.065:ident=f'rose_track_{self.next_id}';self.next_id+=1
   used.add(ident);candidate.update(track_id=ident,seen_step=int(step),state='available');self.tracks[ident]=candidate
  self.diagnostics={'raw_global':len(existing),'raw_color':len(extra),'accepted':len(accepted),'rejected':rejected,'wall_seconds':time.monotonic()-started}
  return accepted
 def assign(self,hand,track_id):
  if hand not in self.hands or self.hands[hand] is not None or track_id not in self.tracks:return False
  if track_id in self.hands.values() or self.tracks[track_id]['state']!='available':return False
  self.hands[hand]=track_id;self.tracks[track_id]['state']='assigned';return True
 def fit_held(self,hand,cameras,predicted_body,base,step,carried_blossom=False):
  if hand not in self.hands or self.hands[hand] is None or not self._valid_pose(predicted_body) or not self._valid_pose(base):return None
  # Independent call: a failed/occluded left fit cannot replace the right track.
  fit=guided_rose_fit(self.perception.models['rose'],cameras,predicted_body,translation_limit=.14 if carried_blossom else .075,red_min=.4,green_min=.5 if carried_blossom else .35)
  if fit is None and carried_blossom:fit=guided_rose_fit(self.perception.models['rose'],cameras,predicted_body,translation_limit=.14,red_min=.4,green_min=.5,multistart=True,visible_only=True)
  if fit is None:return None
  world=base@fit[0];other=self.hands['right' if hand=='left' else 'left']
  if other is not None and np.linalg.norm(world[:3,3]-self.tracks[other]['world_pose'][:3,3])<.065:return None
  ident=self.hands[hand];self.tracks[ident].update(body_pose=fit[0],world_pose=world,stats=dict(fit[1],hand=hand,track_id=ident),seen_step=int(step),state='held')
  return fit
 def mark_released(self,hand,release_world):
  if hand not in self.hands or self.hands[hand] is None or not self._valid_pose(release_world):return False
  ident=self.hands[hand];self.released.append(release_world.copy());self.tracks[ident].update(world_pose=release_world.copy(),state='released');self.hands[hand]=None
  return True

 def register_observed(self,hand,world_pose,base,step,stats=None):
  """Register the caller's prior accepted RGB-D pose, never a Native pose.
  Does not claim a successful grasp; caller confirms holding independently.
  """
  if hand not in self.hands or self.hands[hand] is not None or not self._valid_pose(world_pose) or not self._valid_pose(base):return None
  if self._excluded(world_pose):return None
  if any(t['state'] in ('assigned','held') and np.linalg.norm(world_pose[:3,3]-t['world_pose'][:3,3])<.065 for t in self.tracks.values()):return None
  ident=f'rose_track_{self.next_id}';self.next_id+=1
  self.tracks[ident]={'track_id':ident,'body_pose':np.linalg.inv(base)@world_pose,'world_pose':world_pose.copy(),'stats':{} if stats is None else dict(stats),'seen_step':int(step),'state':'assigned'}
  self.hands[hand]=ident;return ident
 def update_held_fk(self,hand,world_pose,step):
  """Caller-confirmed held payload FK estimate; never refreshes RGB-D seen_step."""
  if hand not in self.hands or self.hands[hand] is None or not self._valid_pose(world_pose):return False
  self.tracks[self.hands[hand]].update(world_pose=world_pose.copy(),fk_step=int(step),state='held');return True
 def normal_search_exclusions(self,except_hand='left'):
  """World roots excluded from the existing normal acquisition, copy-on-read."""
  result=[p[:3,3].copy() for p in self.released]
  for hand,ident in self.hands.items():
   if hand!=except_hand and ident is not None:result.append(self.tracks[ident]['world_pose'][:3,3].copy())
  return result
 def clear_hand(self,hand):
  """Discard failed assignment; a new RGB-D acquisition is required to reuse it."""
  if hand not in self.hands:return False
  ident=self.hands[hand]
  if ident is not None:self.tracks[ident]['state']='unavailable'
  self.hands[hand]=None;return True
