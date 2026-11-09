"""One native public episode per fresh process, preventing held-object reset carryover."""
import os,sys,json,types,argparse,hashlib
from pathlib import Path
stub=types.ModuleType('omnigibson.envs.lerobot_data_wrapper')
class UnavailableRecording:
    def __init__(self,*args,**kwargs):raise RuntimeError('Use native evaluator videos')
stub.LeRobotDataWrapper=stub.LeRobotPlaybackWrapper=UnavailableRecording;sys.modules[stub.__name__]=stub
sys.path.insert(0,'/work/campaigns/competition_rgbd_20261003/deps')
import torch
from omegaconf import OmegaConf
from omnigibson.eval.evaluator import BatchedEvaluator,resolve_instance_ids
from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED,seed_everything
from omnigibson.macros import gm

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--development',action='store_true');a=p.parse_args();assert 0<=a.index<=19
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);os.environ['POLICY_OUTPUT']=str(out/'policy');os.environ['INSTALLATION_TASK']=a.task
    gm.HEADLESS=True;gm.RENDER_VIEWER_CAMERA=False;seed_everything(DEFAULT_EVAL_SEED);torch.set_num_threads(6)
    cfg=OmegaConf.create(dict(env_wrapper={'_target_':'omnigibson.eval.wrappers.RGBDFullResWrapper'},policy_name='installation_rgbd',model={'_target_':'observation_policy.InstallationPolicy'},headless=True,partial_scene_load=True,max_steps=None,write_video=True,mode='public_test',seed=DEFAULT_EVAL_SEED,num_envs=1,task={'name':a.task},robot=None))
    (out/'config.json').write_text(json.dumps(OmegaConf.to_container(cfg),indent=2));source=Path(__file__).parent
    (out/'source_sha256.json').write_text(json.dumps({str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()},indent=2));instance=resolve_instance_ids(a.task,[a.index])[0]
    (out/'protocol.json').write_text(json.dumps(dict(task=a.task,public_index=a.index,native_instance=instance,seed=DEFAULT_EVAL_SEED,fresh_process=True,native_timeout=True,development=a.development),indent=2))
    for d in ['videos','json']:(out/d).mkdir(exist_ok=True)
    with BatchedEvaluator(cfg) as evaluator:
        result=evaluator.run([instance],write_video=True,video_path=str(out/'videos'),metrics_dir=str(out/'json'),rollout_id=0)
        (out/'summary.json').write_text(json.dumps(dict(task=a.task,development=a.development,results=list(result.values()),successes=sum(bool(r['success']) for r in result.values())),indent=2))
        (out/'DONE').write_text('native episode metrics and videos complete')
if __name__=='__main__':main()
