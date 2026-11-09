#!/usr/bin/env python3
"""Import one frozen classical controller of the b1k_rl_guys repository into tasks/<task>/ (served by solutions/b1k_classical).

usage: python3 tools/import_b1k_classical.py --b1k <b1k_rl_guys checkout> --task <task> --candidate <name> --commit <commit>
                                             [--results <folder with json/<task>_<instance>_0.json>]

The controller folder tasks/<task>/experiments/candidate/<name> (or --source-path) is taken by `git archive` of that commit into
tasks/<task>/controller/. Its source fingerprint (sha256 of each file, then of the sorted list, as b1k_rl_guys'
common/util.py computes it) is recorded in solution.json and checked against the scored runs' fingerprint given with
--fingerprint, if any. --results: the evaluator's per-episode metrics of the scored public episodes (instances 301-310),
recorded per instance in solution.json and README.md.
The rose_m15 adapter imports only the h32 runtime, controller and static priors; it excludes evaluation artifacts.
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

REPO = Path(__file__).resolve().parents[1]
TASK_IDS = json.loads((REPO / 'server' / 'task_ids_2026.json').read_text())

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--b1k', required=True, type=Path); p.add_argument('--task', required=True); p.add_argument('--candidate', required=True)
p.add_argument('--commit', required=True); p.add_argument('--branch', default='fax-manip')
p.add_argument('--source-path', help='controller directory relative to the source repository (overrides the candidate path)')
p.add_argument('--adapter', choices=('observation_policy', 'rose_m15'), default='observation_policy',
               help='Policy API adapter in solutions/b1k_classical')
p.add_argument('--fingerprint', help='expected source fingerprint (from the scored runs)')
p.add_argument('--results', type=Path); p.add_argument('--note', default='')
a = p.parse_args()
if a.task not in TASK_IDS:
    raise SystemExit(f'{a.task}: not a 2026 task (server/task_ids_2026.json)')
src = a.source_path or f'tasks/{a.task}/experiments/candidate/{a.candidate}'
if Path(src).is_absolute() or '..' in Path(src).parts:
    raise SystemExit('--source-path must be a repository-relative directory')
src = str(Path(src))
commit = subprocess.run(['git', '-C', str(a.b1k), 'rev-parse', a.commit], capture_output=True, text=True, check=True).stdout.strip()
tar = subprocess.run(['git', '-C', str(a.b1k), 'archive', '--format=tar', commit, src], capture_output=True, check=True).stdout
dest = REPO / 'tasks' / a.task; code = dest / 'controller'
files = {}
with tarfile.open(fileobj=io.BytesIO(tar)) as t:
    for m in t.getmembers():
        if not m.isfile():
            continue
        rel = Path(m.name).relative_to(src)
        if '__pycache__' in rel.parts or rel.name == '.DS_Store':
            continue
        data = t.extractfile(m).read()
        files[rel] = data
if a.adapter == 'rose_m15':
    # M15 serves only h32. Keep runtime sources byte-for-byte, without the
    # evaluation archive or the unused historical candidate/runtime variants.
    files = {rel: data for rel, data in files.items()
             if str(rel) in ('policy.py', 'configure.py')
             or str(rel).startswith(('controller/', 'candidates/h32/', 'runtime/h32/'))}
    for name in ('controller/EXPERIMENT.md', 'controller/hypothesis.json'):
        files.pop(Path(name), None)
    files = {rel: data for rel, data in files.items() if 'rtx_debug' not in rel.parts}
    manifest_path = Path('runtime/h32/tasks/56_make_rose_centerpieces/CONTROLLER_SOURCE.json')
    original_manifest = json.loads(files[manifest_path])
    controller = {str(rel.relative_to('controller')): data for rel, data in files.items()
                  if rel.parts[0] == 'controller'}
    source_hashes = {rel: hashlib.sha256(data).hexdigest() for rel, data in sorted(controller.items())}
    expected_files = set(original_manifest['sha256']) - {'EXPERIMENT.md', 'hypothesis.json'}
    if set(source_hashes) != expected_files:
        raise SystemExit('M15 controller file set differs from its original source manifest')
    if any(original_manifest['sha256'].get(rel) != digest for rel, digest in source_hashes.items()):
        raise SystemExit('M15 controller files differ from their original source manifest')
    # This manifest verifies the deployed file set. Drop historical report
    # metadata and hashes of the excluded documentation, preserving code hashes.
    manifest = {key: original_manifest[key] for key in ('schema', 'repository', 'commit', 'source_directory')}
    manifest.update(file_count=len(controller), total_bytes=sum(map(len, controller.values())), sha256=source_hashes)
    files[manifest_path] = (json.dumps(manifest, indent=2) + '\n').encode()
hashes = {str(rel): hashlib.sha256(data).hexdigest() for rel, data in files.items()}
hashes = dict(sorted(hashes.items()))
fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
if a.fingerprint and a.fingerprint != fingerprint:
    raise SystemExit(f'fingerprint {fingerprint} of {src}@{commit[:10]} is not the scored {a.fingerprint}')
if code.exists():
    shutil.rmtree(code)
code.mkdir(parents=True)
for rel, data in files.items():
    (code / rel).parent.mkdir(parents=True, exist_ok=True); (code / rel).write_bytes(data)
per = []
if a.results:
    for f in sorted((a.results / 'json').glob(f'{a.task}_*_0.json')):
        d = json.loads(f.read_text())
        if 301 <= int(d['instance_id']) <= 310:
            per.append(dict(instance=int(d['instance_id']), success=bool(d['success']), q=float(d['q_score']['final']), steps=int(d['steps'])))
version = {}
if (code / 'version.json').exists():
    version = json.loads((code / 'version.json').read_text())
spec = dict(task=a.task, task_id_2026=TASK_IDS[a.task], kind='b1k_classical', candidate=a.candidate,
            entry=dict(code='controller', env={}),
            source=dict(repo='github.com/tttonyalpha/b1k_rl_guys', branch=a.branch, commit=commit, path=src, fingerprint=fingerprint, files=len(hashes),
                        measured=dict(protocol='native evaluator, RGBDFullResWrapper, native horizon, one fresh simulator process per episode, rollout 0',
                                      per_instance=per)),
            note=a.note or version.get('summary', '')[:300])
if a.adapter != 'observation_policy':
    spec['entry']['adapter'] = a.adapter
(dest / 'solution.json').write_text(json.dumps(spec, indent=2) + '\n')
n = len(per); s = sum(r['success'] for r in per); q = sum(r['q'] for r in per) / n if n else None
priors = 'controller/controller/priors/' if a.adapter == 'rose_m15' else 'controller/priors/'
lines = [f'# {a.task} (2026 task {TASK_IDS[a.task]})', '',
         'Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by '
         '`solutions/b1k_classical/backend.py`.', '',
         f'- Source: `{src}` at `{commit[:10]}` (branch `{a.branch}`), source fingerprint `{fingerprint[:16]}…` ({len(hashes)} files), in `controller/`',
         f'- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `{priors}` '
         '(robot model, a map of the task\'s scene, object models); no instance pose and no simulator state',
         '- Source documentation, scored runs and reports: b1k_rl_guys `tasks/' + a.task + '/`']
if a.adapter == 'rose_m15':
    lines.append('- Adapter: `solutions/b1k_classical/rose_m15.py`, frozen M15/h32 with CUDA perception. '
                 'Runtime code and static priors retain their source bytes; historical variants and evaluation artifacts '
                 'are excluded. The controller manifest covers the retained files. '
                 'Evaluation reports and videos are stored outside this submission repository.')
if n:
    lines.append(f'- Measured (native evaluator, fresh process per episode, instances 301-310): SR {s}/{n}, mean Q {q:.2f}; per instance '
                 + ', '.join(f"{r['instance']}: {'ok' if r['success'] else 'q %.2f' % r['q']}" for r in per))
lines += ['', 'Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).']
(dest / 'README.md').write_text('\n'.join(lines) + '\n')
print(a.task, a.candidate, commit[:10], fingerprint[:16], f'{len(hashes)} files', f'SR {s}/{n}' if n else 'no results')
