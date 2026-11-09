"""Task-56 native driver. Audit/short probes/full pilot are separately labelled."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import sys
import types
faulthandler.enable()
stub=types.ModuleType('omnigibson.envs.lerobot_data_wrapper')
class UnavailableRecording:
    def __init__(self,*args,**kwargs):raise RuntimeError('Use native evaluator video')
stub.LeRobotDataWrapper=stub.LeRobotPlaybackWrapper=UnavailableRecording
sys.modules[stub.__name__]=stub

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--index',type=int,choices=range(10),required=True)
    p.add_argument('--kind',choices=('scan','survey','rose','view'),default='rose')
    p.add_argument('--steps',type=int)
    p.add_argument('--audit',action='store_true')
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    import torch
    torch.cuda.set_device(0)
    from omegaconf import OmegaConf
    from omnigibson.eval.evaluator import BatchedEvaluator,resolve_instance_ids
    from omnigibson.eval.utils.eval_utils import seed_everything,DEFAULT_EVAL_SEED,TASK_NAMES_TO_INDICES
    from omnigibson.macros import gm
    assert TASK_NAMES_TO_INDICES['make_rose_centerpieces']==56
    instance=resolve_instance_ids('make_rose_centerpieces',[a.index])[0]
    assert instance==301+a.index
    seed_everything(DEFAULT_EVAL_SEED);torch.set_num_threads(4);gm.HEADLESS=True;gm.RENDER_VIEWER_CAMERA=False
    os.environ['POLICY_OUTPUT']=str(a.output/'policy');os.environ['TASK_NAME']='make_rose_centerpieces'
    scored=a.steps is None and not a.audit and not a.pilot and a.kind=='rose'
    os.environ['POLICY_SAVE_OBSERVATIONS']='0' if scored else '1'
    protocol=dict(periodic_stack_dumps=False,observation_archive='reset_only' if scored else 'every180_phase_changes_and_pair_attempts',task='make_rose_centerpieces',task_id=56,public_index=a.index,native_instance=instance,seed=0,
                  kind=a.kind,development=not scored,scored=scored,pilot=a.pilot,native_timeout=a.steps is None,
                  native_horizon=None,debug_step_cap=a.steps,privileged_diagnostics=a.audit,
                  rose_postrelease_rgbd_verification=False,advance_after_open_release_and_safe_withdrawal=True,
                  conditional_dual_pickup=True,dual_targets_require_independent_allowed_observations=True,
                  paired_preclosure_staging=True,paired_shared_trunk_coupled_ik=True,
                  paired_cartesian_first_lift=True,both_pickups_require_independent_rgbd_confirmation=True,unheld_targets_protected_from_other_robot_links=True)
    cfg=OmegaConf.create(dict(env_wrapper={'_target_':'omnigibson.eval.wrappers.RGBDFullResWrapper'},
        policy_name='rose_rgbd',model={'_target_':'scan_policy.ScanPolicy' if a.kind=='survey' else 'rose_scan_policy.RoseScanPolicy' if a.kind=='scan' else 'view_policy.ViewPolicy' if a.kind=='view' else 'rose_policy.Policy'},
        headless=True,partial_scene_load=True,max_steps=a.steps,write_video=True,mode='public_test',seed=0,
        num_envs=1,task={'name':'make_rose_centerpieces'},robot=None))
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (a.output/'config.json').write_text(json.dumps(OmegaConf.to_container(cfg),indent=2))
    for name in ('videos','json'):(a.output/name).mkdir(exist_ok=True)
    cls=BatchedEvaluator
    if a.audit:
        from evaluator_audit import DiagnosticEvaluator
        cls=DiagnosticEvaluator;cls.audit_path=a.output/'diagnostic_only.jsonl'
    with cls(cfg) as evaluator:
        protocol['native_horizon']=int(evaluator.human_stats['length'] * __import__('omnigibson.eval.utils.eval_utils',fromlist=['EVAL_TIMEOUT_MULTIPLIER']).EVAL_TIMEOUT_MULTIPLIER)
        (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
        results=evaluator.run([instance],write_video=True,video_path=str(a.output/'videos'),metrics_dir=str(a.output/'json'),rollout_id=0)
        (a.output/'summary.json').write_text(json.dumps(dict(protocol=protocol,results=list(results.values())),indent=2))

if __name__=='__main__':main()
