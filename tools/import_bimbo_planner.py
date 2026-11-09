#!/usr/bin/env python3
"""Import a BIMBO_EvoX fair planner into tasks/<task>/ of this repository, byte-exact to the evaluated version.

usage:
  python tools/import_bimbo_planner.py --bimbo <BIMBO_EvoX checkout> --task-dir 83_store_honey --package fair_ag --commit c53ee86b67 \
      [--program evo/runs/fair-83-r1/programs/6606a458.json --sha <bundle sha256>] [--env KEY=VALUE ...] [--note "..."]

What lands in tasks/<task name>/:
  root/                       a minimal BIMBO tree (the planners find their IK worker and skills/gpu-ik under the first PYTHONPATH
                              entry, so the paths stay as in BIMBO): core/skills_lib, skills/gpu-ik, tasks/<task-dir>/planners/ik_*worker.py
                              and the planner package, all from --commit (git archive)
  solution.json               provenance (commit, package, evolution program, measured results) and the runtime env of the backend

With --program the package's .py files are replaced by the evolved program's modules (the bundle sha256 is checked as in
skills/evo-status/extract_best.py); the non-.py data of the package (offline models, maps) stays from --commit.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SHARED = ['core/__init__.py', 'core/skills_lib', 'skills/gpu-ik/gpu_ik.py', 'skills/gpu-ik/ik_server.py']


def git(bimbo: Path, *args: str) -> bytes:
    return subprocess.run(['git', '-C', str(bimbo), *args], check=True, capture_output=True).stdout


def archive(bimbo: Path, commit: str, paths: list[str], out: Path) -> None:
    present = [p for p in paths if git(bimbo, 'ls-tree', '--name-only', commit, p).strip()]
    with tarfile.open(fileobj=io.BytesIO(git(bimbo, 'archive', commit, *present))) as tar:
        tar.extractall(out, filter='data')


def bundle_sha(sol: dict) -> str:
    return hashlib.sha256(json.dumps(sol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def readme(s: dict) -> str:
    src, evo = s['source'], s['source']['evolution']
    lines = [f"# {s['task']} (2026 task {s['task_id_2026']})", '',
             'Motion planner on RGB-D + proprioception (BIMBO_EvoX fair planner), served by `solutions/planner/backend.py`.', '',
             f"- Source: BIMBO_EvoX `{src['commit'][:10]}`, `tasks/{src['bimbo_task']}/planners/{src['package']}`",
             f"- Package in this folder: `root/{s['entry']['package']}` (plus the shared `core/skills_lib`, `skills/gpu-ik` and the IK worker"
             ' of the same commit)']
    if s['entry']['env']:
        lines.append('- Backend environment: ' + ', '.join(f'`{k}={v}`' for k, v in s['entry']['env'].items()))
    if evo:
        ok = sum(r['success'] for r in evo['per_seed'])
        lines += [f"- Evolution: run `{evo['run']}`, program `{evo['program']}` (iteration {evo['iteration']}), bundle sha256 "
                  f"`{evo['bundle_sha256'][:16]}...`",
                  f"- Measured (H100 port, seeds {evo['per_seed'][0]['seed']}-{evo['per_seed'][-1]['seed']}): SR {ok}/{len(evo['per_seed'])}, "
                  'Q ' + ' '.join(f"{r['q']:.2f}" for r in evo['per_seed'])]
    if s.get('note'):
        lines.append(f"- Note: {s['note']}")
    lines += ['', 'Regenerate with `tools/import_bimbo_planner.py` (the command is in the repository README).', '']
    return '\n'.join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bimbo', type=Path, required=True)
    p.add_argument('--task-dir', required=True, help='BIMBO task folder, e.g. 83_store_honey')
    p.add_argument('--package', required=True, help='planner package under planners/, e.g. fair_ag')
    p.add_argument('--commit', required=True)
    p.add_argument('--program', help='evolved program json, relative to the BIMBO task folder')
    p.add_argument('--sha', help='expected bundle sha256 of --program')
    p.add_argument('--env', action='append', default=[], help='KEY=VALUE set for the planner backend')
    p.add_argument('--note', default='')
    a = p.parse_args()

    bimbo = a.bimbo.resolve()
    task_yaml = git(bimbo, 'show', f'{a.commit}:tasks/{a.task_dir}/task.yaml').decode()
    name = next(line.split(':', 1)[1].split('#')[0].strip() for line in task_yaml.splitlines() if line.strip().startswith('bddl:'))
    task_ids = json.loads((HERE / 'server' / 'task_ids_2026.json').read_text())
    if name not in task_ids:
        raise SystemExit(f'{name} is not a 2026 challenge task')

    dest = HERE / 'tasks' / name
    root = dest / 'root'
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    planners = f'tasks/{a.task_dir}/planners'
    package = f'{planners}/{a.package}'
    archive(bimbo, a.commit, SHARED + [f'{planners}/ik_worker.py', f'{planners}/ik_arm_worker.py', package], root)
    pkg = root / package
    if not (pkg / '__init__.py').is_file():
        raise SystemExit(f'{package} has no __init__.py at {a.commit}')

    evolution = None
    if a.program:
        program_path = bimbo / 'tasks' / a.task_dir / a.program
        program = json.loads(program_path.read_text())
        sol = program['solution']
        sol = json.loads(sol) if isinstance(sol, str) else sol
        sha = bundle_sha(sol)
        if a.sha and sha != a.sha:
            raise SystemExit(f'bundle sha mismatch: {sha} != {a.sha}')
        for f in pkg.rglob('*.py'):
            f.unlink()
        for rel, src in sol.items():
            target = (pkg / rel).resolve()
            if pkg.resolve() not in target.parents:
                raise SystemExit(f'module path escapes the package: {rel}')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(src)
        back = {f.relative_to(pkg).as_posix(): f.read_text() for f in sorted(pkg.rglob('*.py'))}
        if bundle_sha(back) != sha:
            raise SystemExit('written package does not hash back to the program bundle')
        passport = program_path.parents[1] / 'PASSPORT.json'
        per_seed = json.loads(passport.read_text())['results']['per_seed'] if passport.is_file() else []
        evolution = dict(run=program_path.parents[1].name, program=program['id'], iteration=program.get('iteration'), bundle_sha256=sha,
                         per_seed=[{k: s[k] for k in ('seed', 'q', 'success', 'steps')} for s in per_seed])

    env = dict(kv.split('=', 1) for kv in a.env)
    solution = dict(task=name, task_id_2026=task_ids[name], kind='planner',
                    entry=dict(package=package, env=env),
                    source=dict(repo='BIMBO_EvoX', commit=git(bimbo, 'rev-parse', a.commit).decode().strip(), bimbo_task=a.task_dir,
                                package=a.package, evolution=evolution),
                    note=a.note)
    (dest / 'solution.json').write_text(json.dumps(solution, indent=2) + '\n')
    (dest / 'README.md').write_text(readme(solution))
    print(f'{name}: {package} @ {a.commit}' + (f' + program {evolution["program"]}' if evolution else ''))


if __name__ == '__main__':
    main()
