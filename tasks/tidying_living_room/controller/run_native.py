"""Native task runner; bounded probes are explicitly excluded from performance scores."""
import os,sys,json,types,argparse,hashlib
from pathlib import Path
stub=types.ModuleType('omnigibson.envs.lerobot_data_wrapper')
class UnavailableRecording:
 def __init__(self,*a,**kw):raise RuntimeError('Use native evaluator videos.')
stub.LeRobotDataWrapper=stub.LeRobotPlaybackWrapper=UnavailableRecording
sys.modules[stub.__name__]=stub
sys.path.insert(0,'/work/campaigns/competition_rgbd_20261003/deps')
import torch
from omegaconf import OmegaConf
from omnigibson.eval.evaluator import BatchedEvaluator,resolve_instance_ids
from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED,seed_everything
from omnigibson.macros import gm
def main():
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--output',required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--probe',action='store_true');p.add_argument('--debug-steps',type=int);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);os.environ['POLICY_OUTPUT']=str(out/'policy');os.environ['TASK_NAME']=a.task
 gm.HEADLESS=True;gm.RENDER_VIEWER_CAMERA=False;seed_everything(DEFAULT_EVAL_SEED);torch.set_num_threads(4)
 cfg=OmegaConf.create(dict(env_wrapper={'_target_':'omnigibson.eval.wrappers.RGBDFullResWrapper'},policy_name='multitask_rgbd',model={'_target_':'probe_policy.ProbePolicy' if a.probe else 'observation_policy.Policy'},headless=True,partial_scene_load=True,max_steps=a.debug_steps,write_video=True,mode='public_test',seed=DEFAULT_EVAL_SEED,num_envs=1,task={'name':a.task},robot=None))
 source=Path(__file__).parent
 (out/'source_sha256.json').write_text(json.dumps({str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*') if f.is_file() and '__pycache__' not in f.parts},indent=2))
 (out/'config.json').write_text(json.dumps(OmegaConf.to_container(cfg),indent=2))
 instance=resolve_instance_ids(a.task,[a.index])[0]
 protocol=dict(task=a.task,public_index=a.index,native_instance=instance,seed=DEFAULT_EVAL_SEED,fresh_process=True,development=bool(a.probe or a.debug_steps is not None),native_timeout=a.debug_steps is None,policy_inputs=['RGB','depth','proprioception','robot-relative camera calibration'],static_priors=True)
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 for n in ['json','videos']:(out/n).mkdir(exist_ok=True)
 evaluator_class=BatchedEvaluator
 if a.debug_steps is not None:
  from evaluator_debug import DiagnosticEvaluator
  evaluator_class=DiagnosticEvaluator;evaluator_class.audit_path=out/'diagnostic_only.jsonl'
 with evaluator_class(cfg) as evaluator:
  if a.probe:
   from export_training_geometry import export_static
   export_static(evaluator,out/'training_geometry')
  result=evaluator.run([instance],write_video=True,video_path=str(out/'videos'),metrics_dir=str(out/'json'),rollout_id=0)
  (out/'summary.json').write_text(json.dumps(dict(task=a.task,development=protocol['development'],results=list(result.values())),indent=2))
  (out/'DONE').write_text('Native evaluation completed; see protocol for diagnostic status.\n')
if __name__=='__main__':main()
