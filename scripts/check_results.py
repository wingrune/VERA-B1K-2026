#!/usr/bin/env python3
"""Check per-task result folders; print the score the organizers' formula gives.

usage: python3 scripts/check_results.py results/<task> [--metrics-only] [--strict]
       python3 scripts/check_results.py results            # every results/<task> that has json/

Expected layout (what scripts/run_eval.sh writes; one folder per task):
  results/<task>/                                    folder name is the 2026 task
    json/<task>_<instance_id>_<rollout_id>.json      one per rollout, unedited evaluator output
    videos/<task>_<instance_id>_<rollout_id>.mp4     one per json (--write-video), unedited
    logs/<task>_<instance_id>.log                    evaluator log for that rollout
    README.md                                        evaluator command, wrapper, robot config, docker image
    <wrapper>.py                                     evaluation wrapper used
    <robot config>.yaml|.json                        robot config used (eval/r1pro.yaml)

Rules (docs/challenge/submission.md and evaluation.md of BEHAVIOR-1K v3.9.3-post1, omnigibson/eval/utils/score_utils.py):
  file names from this task x public instances 301-320 x rollout 0; reported results use indices 0-9 = instances 301-310
  (311-320 give a warning); each json has the evaluator's keys and types and matches its file name; q_score.final in [0, 1];
  success implies q 1; every json has its video. Errors exit 1; --strict also fails on warnings. --metrics-only skips the
  package files (README, wrapper, robot config).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASK_IDS = json.loads((REPO / 'server' / 'task_ids_2026.json').read_text())
TEST_INSTANCES = list(range(301, 341))
NUM_PUBLIC = 20
REPORTED_PUBLIC = range(301, 311)
EVAL_TIMEOUT_MULTIPLIER = 1.5
NUMBER = (int, float)


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def is_num(x) -> bool:
    return isinstance(x, NUMBER) and not isinstance(x, bool) and not math.isnan(x)


def check_json(path: Path, task: str, instance: int, rollout: int, rep: Report) -> dict | None:
    where = f'json/{path.name}'
    try:
        r = json.loads(path.read_text())
    except (ValueError, UnicodeDecodeError) as exc:
        rep.error(f'{where}: not valid JSON ({exc})')
        return None
    if not isinstance(r, dict):
        rep.error(f'{where}: not a JSON object')
        return None
    expect = {'task': task, 'instance_id': instance, 'rollout_id': rollout}
    for key, value in expect.items():
        if r.get(key) != value:
            rep.error(f'{where}: {key} is {r.get(key)!r}, the file name says {value!r}')
    if not isinstance(r.get('steps'), int) or isinstance(r.get('steps'), bool) or r['steps'] < 0:
        rep.error(f'{where}: steps must be a non-negative int')
    if not isinstance(r.get('success'), bool):
        rep.error(f'{where}: success must be a bool')
    for group in ('agent_distance', 'normalized_agent_distance'):
        g = r.get(group)
        if not isinstance(g, dict) or set(g) != {'base', 'left', 'right'}:
            rep.error(f'{where}: {group} must have exactly base, left, right')
            continue
        for k, v in g.items():
            if not is_num(v) or v < 0:
                rep.error(f'{where}: {group}.{k} must be a non-negative number, got {v!r}')
    q = (r.get('q_score') or {}).get('final') if isinstance(r.get('q_score'), dict) else None
    if not is_num(q) or not 0.0 <= q <= 1.0:
        rep.error(f'{where}: q_score.final must be a number in [0, 1], got {q!r}')
        q = None
    t = r.get('time')
    if not isinstance(t, dict) or not {'simulator_steps', 'simulator_time', 'normalized_time'} <= set(t):
        rep.error(f'{where}: time must have simulator_steps, simulator_time, normalized_time')
    else:
        if t['simulator_steps'] != r.get('steps'):
            rep.error(f'{where}: time.simulator_steps {t["simulator_steps"]} != steps {r.get("steps")}')
        if not is_num(t['simulator_time']) or t['simulator_time'] < 0:
            rep.error(f'{where}: time.simulator_time must be a non-negative number')
        if not is_num(t['normalized_time']) or t['normalized_time'] <= 0:
            rep.error(f'{where}: time.normalized_time must be a positive number (the score divides by it)')
    if r.get('success') is True and q is not None and q != 1.0:
        rep.error(f'{where}: success with q_score.final {q}')
    if r.get('success') is False and q == 1.0:
        rep.warn(f'{where}: q_score.final 1.0 without success')
    extra = set(r) - {'task', 'instance_id', 'rollout_id', 'steps', 'success', 'agent_distance', 'normalized_agent_distance', 'q_score', 'time'}
    if extra:
        rep.warn(f'{where}: keys the evaluator does not write: {sorted(extra)} (outputs must not be edited)')
    return r


def check_video(path: Path, rep: Report, ffprobe: str | None) -> None:
    where = f'videos/{path.name}'
    if not path.is_file():
        rep.error(f'{where}: missing (rollout videos are required, run the evaluator with --write-video)')
        return
    head = path.read_bytes()[:12] if path.stat().st_size >= 12 else b''
    if head[4:8] != b'ftyp':
        rep.error(f'{where}: not an MP4 file (no ftyp box)')
        return
    if ffprobe:
        out = subprocess.run([ffprobe, '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
                             capture_output=True, text=True)
        try:
            ok = out.returncode == 0 and float(out.stdout.strip()) > 0
        except ValueError:
            ok = False
        if not ok:
            rep.error(f'{where}: ffprobe cannot read a positive duration ({out.stderr.strip()[:200]})')


def task_dirs(root: Path) -> list[Path]:
    if not root.is_dir() or root.name in TASK_IDS:
        return [root]
    subs = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name in TASK_IDS and (p / 'json').is_dir()]
    return subs or [root]


def inspect(root: Path, metrics_only: bool, rep: Report) -> tuple[dict[str, dict[int, dict]], int]:
    """Check one results/<task> folder. Returns parsed rollouts and the json file count."""
    where = root.name
    if root.name not in TASK_IDS:
        rep.error(f'folder name {root.name!r} must be a 2026 task (results/<task>/{{json,videos,logs}})')
    instances = TEST_INSTANCES[:NUM_PUBLIC]
    tasks = [root.name] if root.name in TASK_IDS else list(TASK_IDS)
    allowed = {f'{t}_{i}_0.json': (t, i, 0) for t in tasks for i in instances}

    json_dir, video_dir = root / 'json', root / 'videos'
    if not json_dir.is_dir():
        rep.error(f'{where}: json/ is missing')
    files = sorted(json_dir.glob('*.json')) if json_dir.is_dir() else []
    if len(files) > 20:
        rep.error(f'{where}: {len(files)} rollouts, at most 20 (one task x 20 public instances x 1 rollout)')
    for f in sorted(json_dir.iterdir()) if json_dir.is_dir() else []:
        if f.suffix != '.json':
            rep.warn(f'{where}: json/{f.name}: not a .json file (ignored by the scorer)')

    ffprobe = shutil.which('ffprobe')
    results: dict[str, dict[int, dict]] = {}
    for f in files:
        if f.name not in allowed:
            rep.error(f'{where}: json/{f.name}: not <task>_<{instances[0]}..{instances[-1]}>_0.json (one rollout per instance, rollout 0)')
            continue
        task, instance, rollout = allowed[f.name]
        if instance not in REPORTED_PUBLIC:
            rep.warn(f'{where}: json/{f.name}: instance {instance} is index {instance - 301}; reported results use indices 0-9 (301-310)')
        r = check_json(f, task, instance, rollout, rep)
        check_video(video_dir / f'{f.stem}.mp4', rep, ffprobe)
        if r is not None:
            results.setdefault(task, {})[instance] = r
    if video_dir.is_dir():
        for v in sorted(video_dir.glob('*.mp4')):
            if not (json_dir / f'{v.stem}.json').is_file():
                rep.warn(f'{where}: videos/{v.name}: no matching json')

    if not metrics_only:
        if not (root / 'README.md').is_file():
            rep.error(f'{where}: README.md is missing (evaluator command, wrapper path, robot config path, docker image)')
        if not list(root.glob('*.py')):
            rep.error(f'{where}: no wrapper .py in the folder (the evaluation wrapper used)')
        configs = list(root.glob('*.yaml')) + list(root.glob('*.yml')) + [c for c in root.glob('*.json') if c.parent == root]
        if not configs:
            rep.error(f'{where}: no robot config (.yaml or .json) in the folder (e.g. r1pro.yaml)')
    return results, len(files)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder', type=Path)
    p.add_argument('--metrics-only', action='store_true', help='check json and videos only, not README / wrapper / robot config')
    p.add_argument('--strict', action='store_true', help='warnings fail too')
    a = p.parse_args()
    rep = Report()
    root = a.folder.resolve()
    targets = task_dirs(root)

    results: dict[str, dict[int, dict]] = {}
    n_files = 0
    for target in targets:
        got, n = inspect(target, a.metrics_only, rep)
        n_files += n
        for task, insts in got.items():
            results.setdefault(task, {}).update(insts)

    # Score as omnigibson/eval/utils/score_utils.compute_final_q_score: per task, the sum over rollouts divided by the
    # number of public instances (20), missing rollouts count 0; overall = mean over all 100 tasks.
    n_inst = NUM_PUBLIC
    print(f'{"task":42s} {"n":>3s} {"SR":>6s} {"mean Q":>7s}  (over the submitted rollouts)')
    q_total = sr_total = 0.0
    for task in sorted(results, key=TASK_IDS.get):
        rs = results[task].values()
        qs = [r['q_score']['final'] for r in rs if is_num((r.get('q_score') or {}).get('final'))]
        sr = sum(1 for q in qs if q == 1) / max(len(qs), 1)
        print(f'{task:42s} {len(qs):3d} {sr:6.2f} {sum(qs) / max(len(qs), 1):7.3f}')
        q_total += sum(qs) / n_inst
        sr_total += sum(1 for q in qs if q == 1) / n_inst
    print(f'{n_files} rollouts over {len(results)} tasks; official formula over 100 tasks x {n_inst} instances: '
          f'Q {q_total / len(TASK_IDS):.4f}, SR {sr_total / len(TASK_IDS):.4f}')

    for w in rep.warnings:
        print(f'WARNING {w}')
    for e in rep.errors:
        print(f'ERROR {e}')
    failed = bool(rep.errors) or (a.strict and bool(rep.warnings))
    print(f'{"FAIL" if failed else "OK"}: {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
