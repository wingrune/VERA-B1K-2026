#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def action_chunk_size(task: str, routing: Path, solutions: Path) -> int:
    ids = json.loads((REPO / 'server/task_ids_2026.json').read_text())
    task = {str(v): k for k, v in ids.items()}.get(task, task)
    rows = csv.DictReader(line for line in routing.read_text().splitlines()
                          if line.strip() and not line.lstrip().startswith('#'))
    matches = [row for row in rows if row['task'] == task]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one routing row for {task}')
    spec = json.loads(solutions.read_text()).get(matches[0]['solution'], {})
    count = spec.get('action_chunk_size', 1)
    if type(count) is not int or count < 1:
        raise ValueError(f'Invalid action_chunk_size for {task}: {count!r}')
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    parser.add_argument('--routing', type=Path, default=REPO / 'config/routing.csv')
    args = parser.parse_args()
    print(action_chunk_size(args.task, args.routing, REPO / 'config/solutions.json'))
