#!/usr/bin/env python3
"""Compare one task between results_tmp/<task> and results/<task>; move the new results into results/ when they are better.

usage: python3 scripts/promote_task.py <task id or name> [--force] [--dry-run]

The task's rollouts are taken as a whole: every json, video and evaluator log in results/<task> is replaced by the ones in
results_tmp/<task>, never single instances. The challenge forbids assembling the best rollouts across repeated runs, so
promote a task when its code changed and it was evaluated again on all instances, not to keep the luckier of two runs of the
same code. Promotion needs the new set to cover at least the instances of the old one, to pass scripts/check_results.py
--metrics-only, and a higher mean Q (equal Q: higher SR); --force skips the comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get('B1K_RESULTS', REPO / 'results'))
RESULTS_TMP = Path(os.environ.get('B1K_RESULTS_TMP', REPO / 'results_tmp'))
TASK_IDS = json.loads((REPO / 'server' / 'task_ids_2026.json').read_text())


def load(run: Path, task: str) -> dict[int, float]:
    out = {}
    for f in sorted((run / 'json').glob(f'{task}_*_0.json')):
        inst = f.stem[len(task) + 1:].split('_')[0]
        if inst.isdigit():
            out[int(inst)] = float(json.loads(f.read_text())['q_score']['final'])
    return out


def task_files(run: Path, task: str) -> list[Path]:
    pats = [f'json/{task}_*_0.json', f'videos/{task}_*_0.mp4', f'logs/{task}_*.log']
    return [p for pat in pats for p in run.glob(pat)]


def stats(q: dict[int, float]) -> str:
    n = len(q)
    return f'{n} rollouts, mean Q {sum(q.values()) / n:.3f}, SR {sum(v == 1 for v in q.values())}/{n}' if n else 'no rollouts'


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('task')
    p.add_argument('--force', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    task = {str(v): k for k, v in TASK_IDS.items()}.get(a.task, a.task)
    if task not in TASK_IDS:
        sys.exit(f'unknown task {a.task}')
    src, dst = RESULTS_TMP / task, RESULTS / task
    new, old = load(src, task), load(dst, task)
    print(f'{task}\n  results_tmp: {stats(new)}\n  results:     {stats(old)}')
    if not new:
        sys.exit('nothing to promote')

    check = subprocess.run([sys.executable, str(REPO / 'scripts' / 'check_results.py'), str(src), '--metrics-only'],
                           capture_output=True, text=True)
    errors = [line for line in check.stdout.splitlines() if line.startswith('ERROR') and f'/{task}_' in line]
    if errors:
        print('\n'.join(errors))
        sys.exit('the new results of this task do not pass the format check')

    if not a.force:
        if not set(old) <= set(new):
            sys.exit(f'results_tmp lacks instances {sorted(set(old) - set(new))} that results has: evaluate them too or use --force')
        key = lambda q: (sum(q.values()) / len(q), sum(v == 1 for v in q.values()) / len(q)) if q else (-1.0, -1.0)  # noqa: E731
        if key(new) <= key(old):
            print('not better: results/ kept (use --force to replace anyway)')
            return 1
    if a.dry_run:
        print('would replace', len(task_files(dst, task)), 'files with', len(task_files(src, task)))
        return 0

    for f in task_files(dst, task):
        f.unlink()
    for f in task_files(src, task):
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
    for f in ('README.md', 'r1pro.yaml', 'rgbd_full_res_wrapper.py'):
        if (src / f).is_file() and not (dst / f).is_file():
            shutil.copy2(src / f, dst / f)
    print(f'promoted {task} into {dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
