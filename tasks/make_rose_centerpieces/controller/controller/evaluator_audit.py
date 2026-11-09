"""Separate evaluator-side diagnostic observer. Nothing returns to the policy."""
import json
from omnigibson.eval.evaluator import BatchedEvaluator

class DiagnosticEvaluator(BatchedEvaluator):
    audit_path=None
    def _apply_actions(self,actions,active_env_indices):
        step=getattr(self.policy,'step',0)
        phase=getattr(self.policy,'phase',None)
        detailed=phase in ('descend','close','lift','carry','verify_release') or (phase is not None and phase.startswith('dual_'))
        if self.audit_path is None or (step%30 and not detailed):
            return super()._apply_actions(actions,active_env_indices)
        accessor=self.instance_eval_states[0].env_accessor
        p,q=accessor.robot.get_position_orientation()
        row=dict(step=step,timing='before_native_step; same state as policy input',policy_access=False,robot_base=[p.tolist(),q.tolist()],objects={})
        row['phase']=phase
        held=getattr(accessor.robot,'_ag_obj_in_hand',{}).get('left')
        row['native_assisted_grasp_left']=None if held is None else held.name
        right_held=getattr(accessor.robot,'_ag_obj_in_hand',{}).get('right')
        row['native_assisted_grasp_right']=None if right_held is None else right_held.name
        observed=self.instance_eval_states[0].obs.get('robot_r1::proprio')
        if observed is not None:
            row['observed_left_eef']=observed[17:20].detach().cpu().tolist()
            row['observed_gripper_left_qpos']=observed[24:26].detach().cpu().tolist()
            row['observed_right_eef']=observed[42:45].detach().cpu().tolist()
            row['observed_gripper_right_qpos']=observed[49:51].detach().cpu().tolist()
        for key,obj in accessor.object_scope.items():
            if obj is None or not hasattr(obj,'root_link') or not any(s in key for s in ('rose.n.','vase.n.','coffee_table.n.')):
                continue
            p,q=obj.root_link.get_position_orientation()
            row['objects'][key]=dict(root_position=p.tolist(),root_quaternion=q.tolist())
        from omnigibson.utils.usd_utils import RigidContactAPI
        fingers={link for name,link in accessor.robot.links.items() if 'finger' in name and 'left' in name}
        row['left_finger_links']=[link.prim_path for link in fingers]
        try:
            row['left_finger_contacts']=sorted(list(pair) for pair in RigidContactAPI.get_contact_pairs(accessor.robot.scene.idx,fingers,None,current_only=True)) if fingers else []
        except Exception as exc:
            row['contact_error']=repr(exc)
        from omnigibson.object_states import Inside, OnTop
        bin_obj=next((obj for key,obj in accessor.object_scope.items() if 'vase.n.' in key and obj is not None),None)
        row['native_inside']={}
        table_obj=next((obj for key,obj in accessor.object_scope.items() if 'coffee_table.n.' in key and obj is not None),None)
        row['native_vase_ontop_coffee_table']=None if bin_obj is None or table_obj is None else bool(bin_obj.states[OnTop].get_value(table_obj))
        if bin_obj is not None:
            for key,obj in accessor.object_scope.items():
                if obj is not None and 'rose.n.' in key:
                    row['native_inside'][key]=bool(obj.states[Inside].get_value(bin_obj))
        with self.audit_path.open('a') as f:f.write(json.dumps(row)+'\n')
        return super()._apply_actions(actions,active_env_indices)
