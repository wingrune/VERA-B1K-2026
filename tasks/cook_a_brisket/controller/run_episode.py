"""One native cook_a_brisket episode, the loop of omnigibson/eval/eval.py with a local policy class.
Evaluator-side diagnostics (true poses, goal predicates) go to diag.jsonl and never reach the policy."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=0)                                          # public index 0..19 -> instance 301..320
parser.add_argument("--mode", default="public_test")
parser.add_argument("--policy", default="brisket_policy.BrisketPolicy")
parser.add_argument("--output", required=True)
parser.add_argument("--max-steps", type=int, default=None)                                   # None: native 1.5 x human length
parser.add_argument("--video", action="store_true")
parser.add_argument("--diag-every", type=int, default=30)
args = parser.parse_args()

from omegaconf import OmegaConf
from omnigibson.eval.evaluator import Evaluator, resolve_instance_ids
from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
from omnigibson.macros import gm
from diagnostics import Diagnostics

TASK = "cook_a_brisket"
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
os.environ["POLICY_OUTPUT"] = str(out / "policy")
gm.HEADLESS = True
seed = seed_everything(DEFAULT_EVAL_SEED)
instance = resolve_instance_ids(TASK, [args.index], mode=args.mode)[0]
cfg = OmegaConf.create({"env_wrapper": {"_target_": "omnigibson.eval.wrappers.RGBDFullResWrapper"}, "policy_name": args.policy,
                        "model": {"_target_": args.policy}, "headless": True, "partial_scene_load": True, "max_steps": args.max_steps,
                        "write_video": args.video, "mode": args.mode, "seed": seed, "task": {"name": TASK}, "robot": None})
(out / "run.json").write_text(json.dumps({"index": args.index, "instance": int(instance), "mode": args.mode, "policy": args.policy,
                                          "max_steps": args.max_steps, "argv": sys.argv}, indent=1))

tic = time.time()
with Evaluator(cfg) as evaluator:
    evaluator.reset()
    evaluator.load_task_instance(int(instance))
    evaluator.reset()
    if args.video:
        (out / "videos").mkdir()
        evaluator.start_recording(str(out / "videos" / f"{TASK}_{instance}.mp4"))
    diag = Diagnostics(evaluator.env, out / "diag.jsonl")
    diag.log(0)
    print(f"LOADED instance {instance} in {time.time() - tic:.0f} s", flush=True)

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        terminated, truncated = evaluator.step()
        steps += 1
        if steps % args.diag_every == 0:
            diag.log(steps)
    diag.log(steps)

    success = bool(evaluator.env.task.success)
    metrics = {}
    for metric in evaluator.metrics:
        metrics.update(metric.aggregate(evaluator.env))
    if args.video:
        evaluator.stop_recording()
    summary = {"task": TASK, "instance": int(instance), "steps": steps, "success": success, "wall_s": time.time() - tic, **metrics}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("RESULT", json.dumps({"instance": int(instance), "steps": steps, "success": success, "q": metrics.get("q_score", {}).get("final")}), flush=True)
