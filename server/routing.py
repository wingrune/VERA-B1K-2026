#!/usr/bin/env python3
"""Routing table: which solution serves which task (config/routing.csv), checked against config/solutions.json.

usage: python3 server/routing.py [config/routing.csv] [--no-files]     prints the table, exit 1 on any error

CSV columns: task,solution,checkpoint[,notes]. task is the 2026 task name (server/task_ids_2026.json), or * for the default
route that serves every task id without a row of its own (a task outside the 100); solution a key of
config/solutions.json, or empty / none for a task that is not served; checkpoint the path of the model weights when the solution
needs one: relative to the repository (checkpoints/...), absolute, or with $VARS from the environment. Lines starting with # are comments. --no-files skips
the existence checks of checkpoints and task folders (inside the image checkpoints sit under /opt/checkpoints).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

REPO = Path(__file__).resolve().parents[1]
TASK_IDS = json.loads((REPO / 'server' / 'task_ids_2026.json').read_text())
DEFAULT = '*'


def task_order(task: str) -> int:
    return TASK_IDS.get(task, len(TASK_IDS))


@dataclass(frozen=True)
class Route:
    task: str
    solution: str
    checkpoint: str


def load_solutions() -> Dict[str, dict]:
    return {k: v for k, v in json.loads((REPO / 'config' / 'solutions.json').read_text()).items() if not k.startswith('_')}


def load_routing(path: Path, check_files: bool = True) -> Dict[str, Route]:
    solutions = load_solutions()
    routes: Dict[str, Route] = {}
    errors = []
    rows = [r for r in csv.DictReader(line for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#'))]
    for n, row in enumerate(rows, 2):
        task = (row.get('task') or '').strip()
        sol = (row.get('solution') or '').strip()
        ckpt = os.path.expandvars((row.get('checkpoint') or '').strip())
        where = f'{path.name} row {n} ({task or "?"})'
        if task not in TASK_IDS and task != DEFAULT:
            errors.append(f'{where}: unknown 2026 task')
            continue
        if task in routes:
            errors.append(f'{where}: task listed twice')
            continue
        if sol in ('', 'none'):
            continue
        spec = solutions.get(sol)
        if spec is None:
            errors.append(f'{where}: unknown solution {sol!r} (config/solutions.json: {", ".join(solutions)})')
            continue
        if spec.get('tasks_from') and task not in json.loads((REPO / spec['tasks_from']).read_text()):
            errors.append(f'{where}: {sol} cannot serve {task} (not in {spec["tasks_from"]}; the {DEFAULT} route needs a solution without tasks_from)')
        if spec.get('needs_checkpoint') and not ckpt:
            errors.append(f'{where}: {sol} needs a checkpoint path')
        if '$' in ckpt:
            errors.append(f'{where}: checkpoint {ckpt}: set that environment variable or write the path')
            continue
        ckpt_dir = REPO / Path(ckpt).expanduser() if ckpt else None      # relative paths: under the repository (checkpoints/)
        if check_files and ckpt_dir and not ckpt_dir.is_dir():
            errors.append(f'{where}: checkpoint {ckpt} is not a directory')
        if check_files and sol in ('planner', 'planner_cpu', 'b1k_classical', 'frozen_rgbd') and not (REPO / 'tasks' / task / 'solution.json').is_file():
            errors.append(f'{where}: no {sol} solution in tasks/{task}/')
        routes[task] = Route(task, sol, str(ckpt_dir) if ckpt_dir else '')
    if errors:
        raise SystemExit('routing errors:\n  ' + '\n  '.join(errors))
    return routes


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = Path(args[0] if args else REPO / 'config' / 'routing.csv')
    routes = load_routing(path, check_files='--no-files' not in sys.argv)
    for r in sorted(routes.values(), key=lambda r: task_order(r.task)):
        print(f'{TASK_IDS.get(r.task, DEFAULT):>3}  {r.task:42s} {r.solution:18s} {r.checkpoint}')
    print(f'{len(set(routes) & set(TASK_IDS))} of {len(TASK_IDS)} tasks routed' + (f'; others: {routes[DEFAULT].solution}' if DEFAULT in routes else ''))


if __name__ == '__main__':
    main()
