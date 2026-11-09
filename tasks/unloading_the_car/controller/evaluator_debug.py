"""Development-only simulator observer; no data is returned to the policy."""
import json
from omnigibson.eval.evaluator import BatchedEvaluator
class DiagnosticEvaluator(BatchedEvaluator):
 audit_path=None
 def _apply_actions(self,actions,active_env_indices):
  result=super()._apply_actions(actions,active_env_indices)
  step=getattr(self.policy,'step',0)
  if self.audit_path is None or step%30:return result
  row=dict(step=step,policy_access=False)
  try:
   accessor=self.instance_eval_states[0].env_accessor
   robot=accessor.robot;p,q=robot.get_position_orientation();row['robot_base']=[p.tolist(),q.tolist()]
   row['particles']={name:int(system.n_particles) for name,system in accessor.scene.active_systems.items()}
   row['joint_positions']=robot.get_joint_positions().tolist()
   row['joint_targets']=robot.get_joint_position_targets().tolist()
   row['joint_efforts']=robot.get_joint_efforts().tolist()
   from omnigibson.utils.usd_utils import RigidContactAPI
   row['contacts']=sorted([list(pair) for pair in RigidContactAPI.get_contact_pairs(accessor.env_idx,{robot},None,False)])
   if step==30:
    row['robot_link_diagnostics']={name:dict(contact_reporting=bool(link.contact_reporting_enabled),gravity_disabled=bool(link.prim.GetAttribute('physxRigidBody:disableGravity').Get())) for name,link in robot.links.items()}
   row['particle_poses']={}
   for name,system in accessor.scene.active_systems.items():
    try:
     positions,orientations=system.get_particles_position_orientation();row['particle_poses'][name]=positions.tolist()
    except Exception as exc:row['particle_poses'][name]={'unavailable':type(exc).__name__}
   row['objects']={}
   for name,obj in accessor.object_scope.items():
    if obj is None or not hasattr(obj,'root_link') or not any(x in name for x in ['scrub_brush','pipe_cleaner','keyboard','digital_camera','camera_tripod','cabinet_door','cabinet_base','facsimile']):continue
    p,q=obj.root_link.get_position_orientation();row['objects'][name]=dict(root_position=p.tolist(),root_quaternion=q.tolist())
  except Exception as e:row['error']=str(e)
  with self.audit_path.open('a') as f:f.write(json.dumps(row)+'\n')
  return result
