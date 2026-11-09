"""Fresh-process one-step reset validation; never a scored episode."""
import argparse
import json
import os
import sys
import types
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(180, repeat=True)
from pathlib import Path

# Preserve the historical old-LeRobot import workaround, with no foreign paths.
stub = types.ModuleType('omnigibson.envs.lerobot_data_wrapper')
class UnavailableRecording:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('Use native evaluator videos; LeRobot recording is unavailable.')
stub.LeRobotDataWrapper = stub.LeRobotPlaybackWrapper = UnavailableRecording
sys.modules[stub.__name__] = stub

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--index', type=int, choices=range(10), required=True)
    p.add_argument('--export-priors', action='store_true')
    p.add_argument('--export-articulation',action='store_true')
    p.add_argument('--inspect-bin-area',action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    os.environ['POLICY_OUTPUT'] = str(a.output / 'observations')
    os.environ['OMNIGIBSON_GPU_ID'] = '0'
    import torch
    torch.cuda.set_device(0)
    import omnigibson
    import warp
    assert str(Path(omnigibson.__file__).resolve()).startswith('/behavior-src/BEHAVIOR-1K/OmniGibson/')
    from omegaconf import OmegaConf
    from omnigibson.eval.evaluator import BatchedEvaluator, resolve_instance_ids
    from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, TASK_NAMES_TO_INDICES, seed_everything, EVAL_TIMEOUT_MULTIPLIER
    from omnigibson.macros import gm
    assert TASK_NAMES_TO_INDICES['make_rose_centerpieces'] == 56
    assert resolve_instance_ids('make_rose_centerpieces', list(range(10))) == list(range(301, 311))
    seed_everything(DEFAULT_EVAL_SEED)
    torch.set_num_threads(4)
    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False
    cfg = OmegaConf.create(dict(env_wrapper={'_target_': 'omnigibson.eval.wrappers.RGBDFullResWrapper'},
        policy_name='rose_reset_diagnostic', model={'_target_': 'reset_probe.ResetProbe'},
        headless=True, partial_scene_load=True, max_steps=1, write_video=True,
        mode='public_test', seed=0, num_envs=1, task={'name': 'make_rose_centerpieces'}, robot=None))
    (a.output / 'protocol.json').write_text(json.dumps(dict(development=True, scored=False,
        task='make_rose_centerpieces', task_id=56, native_instance=301 + a.index, public_index=a.index, seed=0,
        diagnostic_step_cap=1, native_horizon_unmodified_for_scored_runs=None,
        omnigibson_source=omnigibson.__file__, warp_source=warp.__file__, warp_version=warp.__version__), indent=2))
    (a.output / 'config.json').write_text(json.dumps(OmegaConf.to_container(cfg), indent=2))
    for name in ('videos', 'json'):
        (a.output / name).mkdir(exist_ok=True)
    with BatchedEvaluator(cfg) as evaluator:
        protocol=json.loads((a.output/'protocol.json').read_text())
        protocol['native_horizon_unmodified_for_scored_runs']=int(evaluator.human_stats['length']*EVAL_TIMEOUT_MULTIPLIER)
        (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
        if a.inspect_bin_area:
            evaluator.load_batch({0:301+a.index})
            from inspect_bin_area import inspect_bin_area
            inspect_bin_area(evaluator,a.output/'camera_inspection')
            faulthandler.cancel_dump_traceback_later()
            return
        if a.export_priors:
            from export_priors import export_priors
            export_priors(evaluator, a.output / 'training_priors')
        if a.export_articulation:
            from export_bin_articulation import export_bin_articulation
            export_bin_articulation(evaluator,a.output/'training_articulation')
        result = evaluator.run([301 + a.index], write_video=True, video_path=str(a.output / 'videos'),
                               metrics_dir=str(a.output / 'json'), rollout_id=0)
        (a.output / 'diagnostic_summary.json').write_text(json.dumps(dict(scored=False,
            development=True, results=list(result.values())), indent=2))

if __name__ == '__main__':
    main()
