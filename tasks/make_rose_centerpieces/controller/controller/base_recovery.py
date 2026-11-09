"""Recover from observed base-motion stalls without simulator contacts or poses."""
from collections import deque
import numpy as np
from navigation import rot2
class BaseRecovery:
 def __init__(self):self.history=deque(maxlen=45);self.retreat=None;self.age=0;self.count=0;self.debug={}
 def apply(self,policy,action,proprio):
  moving=policy.phase in ['navigate','target_navigate','inspect_parent'] or (policy.phase in ['search','search_target'] and policy.search_moving)
  if not moving:self.history.clear();self.retreat=None;return
  if self.retreat is not None:
   self.age+=1;delta=self.retreat-policy.odom[:2];action[:3]=0.
   if np.linalg.norm(delta)>.025 and self.age<100:action[:2]=np.clip(rot2(policy.odom[2]).T@delta*.8,-.12,.12);return
   self.retreat=None;self.history.clear();policy.search_moving=False;policy.nav=None;policy.change('search_target' if policy.held is not None else 'search');return
  self.history.append((policy.odom[:2].copy(),action[:2].copy(),np.linalg.norm(proprio[:2])))
  if len(self.history)<45:return
  rows=list(self.history)
  if np.mean([np.linalg.norm(x[1]) for x in rows])<.065 or np.mean([x[2] for x in rows])>.025 or np.linalg.norm(rows[-1][0]-rows[0][0])>.03:return
  d=rot2(policy.odom[2])@np.mean([x[1] for x in rows],axis=0);d/=max(np.linalg.norm(d),1e-8);hit=policy.odom[:2]+d*(policy.map.radius+.10);cell=policy.map.ij(hit)
  if np.all((cell>=1)&(cell<policy.map.n-1)):policy.map.log_odds[cell[0]-1:cell[0]+2,cell[1]-1:cell[1]+2,:8]=8
  if policy.searcher.goal is not None:policy.searcher.visited.append(policy.searcher.goal.copy())
  goal=policy.odom[:2]-d*.18;self.count+=1;allowed=policy.map.line_clear(policy.odom[:2],goal);self.debug=dict(count=self.count,inferred_obstacle=hit.tolist(),retreat_allowed=allowed)
  self.history.clear();action[:3]=0.
  if allowed:self.retreat=goal;self.age=0
  else:policy.search_moving=False;policy.nav=None;policy.change('search_target' if policy.held is not None else 'search')

