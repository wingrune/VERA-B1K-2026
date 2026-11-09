"""Entry point for the shared launcher (common/evaluate.py): one native cook_a_brisket episode run by the frozen
run_episode.py, then its native metrics written as {"results": [row]}, the form common/aggregate.py reads.
Added when packaging (2026-10-14); the scored full_v20 runs called run_episode.py directly."""
import argparse
import json
import runpy
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="cook_a_brisket")
parser.add_argument("--index", type=int, required=True)                                      # public index 0..9 -> native instance 301..310
parser.add_argument("--output", required=True)
parser.add_argument("--no-video", action="store_true")
args = parser.parse_args()
assert args.task == "cook_a_brisket", args.task

out = Path(args.output)
native = out / "native"                                                                       # run_episode.py refuses an existing directory
here = Path(__file__).resolve().parent
sys.argv = [str(here / "run_episode.py"), "--index", str(args.index), "--output", str(native)] + ([] if args.no_video else ["--video"])
runpy.run_path(str(here / "run_episode.py"), run_name="__main__")

s = json.loads((native / "summary.json").read_text())
row = {"task": s["task"], "instance_id": s["instance"], "rollout_id": 0, "steps": s["steps"], "success": s["success"],
       **{k: v for k, v in s.items() if k not in ("task", "instance", "steps", "success", "wall_s")}}
(out / "summary.json").write_text(json.dumps({"diagnostic": False, "results": [row]}, indent=1, default=float))
