"""One native hiding_Easter_eggs episode, the loop of omnigibson/eval/eval.py with a local policy class.
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
parser.add_argument("--policy", default="easter_policy.EasterPolicy")
parser.add_argument("--output", required=True)
parser.add_argument("--max-steps", type=int, default=None)                                   # None: native 1.5 x human length
parser.add_argument("--video", action="store_true")
parser.add_argument("--diag-every", type=int, default=30)
parser.add_argument("--dev-near-basket", type=float, default=None)                           # DEVELOPMENT ONLY: teleport the robot this far from the basket, facing it
args = parser.parse_args()

from omegaconf import OmegaConf
from omnigibson.eval.evaluator import Evaluator, resolve_instance_ids
from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
from omnigibson.macros import gm
from diagnostics import Diagnostics

TASK = "hiding_Easter_eggs"
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
os.environ["POLICY_OUTPUT"] = str(out / "policy")
gm.HEADLESS = True
seed = seed_everything(DEFAULT_EVAL_SEED)
instance = resolve_instance_ids(TASK, [args.index], mode=args.mode)[0]
cfg = OmegaConf.create({"env_wrapper": {"_target_": "omnigibson.eval.wrappers.RGBDFullResWrapper"}, "policy_name": args.policy,
                        "model": {"_target_": args.policy}, "headless": True, "partial_scene_load": True, "max_steps": args.max_steps,
                        "write_video": args.video, "mode": args.mode, "seed": seed, "task": {"name": TASK}, "robot": None})
(out / "run.json").write_text(json.dumps({"index": args.index, "instance": int(instance), "mode": args.mode, "policy": args.policy, "dev_near_basket": args.dev_near_basket,
                                          "max_steps": args.max_steps, "argv": sys.argv}, indent=1))

tic = time.time()
with Evaluator(cfg) as evaluator:
    evaluator.reset()
    evaluator.load_task_instance(int(instance))
    evaluator.reset()
    if args.dev_near_basket is not None:                                                      # development shortcut past the search; never in a scored run
        import math
        import numpy as np
        import torch as th
        robot = evaluator.robot
        bp = evaluator.env.task.object_scope["wicker_basket.n.01_1"].get_position_orientation()[0]
        d = robot.get_position_orientation()[0][:2] - bp[:2]
        d = d / th.norm(d)
        yaw = math.atan2(-d[1], -d[0])
        robot.reset()
        robot.set_position_orientation(th.tensor([bp[0] + args.dev_near_basket * d[0], bp[1] + args.dev_near_basket * d[1], 0.0]), th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]))
        robot.keep_still()
        s0 = evaluator.obs["robot_r1::proprio"].numpy()
        hold = th.tensor(np.r_[np.zeros(3), s0[53:57], s0[3:10], 1.0, s0[28:35], 1.0], dtype=th.float32)
        for _ in range(10):
            obs, *_ = evaluator.env.step(hold)
        evaluator.obs = evaluator._preprocess_obs(evaluator._sync_lights_and_get_obs(obs))
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
