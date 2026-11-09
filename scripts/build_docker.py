#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAYER_BYTES = 5 * 1024**3       # checkpoint files per image layer; GHCR rejects layers over 10 GB
sys.path.insert(0, str(REPO))
from server.routing import load_routing, load_solutions, task_order


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--routing', default=str(REPO / 'config' / 'routing.csv'))
    p.add_argument('--tag', default='b1k-evo-submit:latest')
    p.add_argument('--dry-run', action='store_true', help='write .build/ and print the docker command only')
    p.add_argument('--no-files', action='store_true', help='dry-run only: inspect integration before checkpoints arrive')
    a = p.parse_args()
    if a.no_files and not a.dry_run:
        p.error('--no-files requires --dry-run; real builds always validate checkpoint directories')

    routes = load_routing(Path(a.routing), check_files=not a.no_files)
    solutions = load_solutions()
    if not a.no_files:
        for checkpoint in {r.checkpoint for r in routes.values() if '$COMET2_PYTHON' in solutions[r.solution]['command']}:
            root = Path(checkpoint)
            if not (root / 'params').is_dir() or not (root / 'assets' / 'behavior-1k' / '2026-challenge-rgb224' / 'norm_stats.json').is_file():
                raise SystemExit(f'comet2 checkpoint needs params/ and assets/behavior-1k/2026-challenge-rgb224/norm_stats.json: {root}')
    build = REPO / '.build'
    build.mkdir(exist_ok=True)

    names: dict[str, str] = {}
    for r in routes.values():
        if r.checkpoint and r.checkpoint not in names:
            names[r.checkpoint] = f'{Path(r.checkpoint).name}_{hashlib.sha1(r.checkpoint.encode()).hexdigest()[:8]}'
    with open(build / 'routing.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task', 'solution', 'checkpoint'])
        for r in sorted(routes.values(), key=lambda r: task_order(r.task)):
            w.writerow([r.task, r.solution, f'/opt/checkpoints/{names[r.checkpoint]}' if r.checkpoint else ''])

    with_vla = any('$VLA_PYTHON' in solutions[r.solution]['command'] for r in routes.values())
    with_comet2 = any('$COMET2_PYTHON' in solutions[r.solution]['command'] for r in routes.values())     # comet2 and comet3
    dockerfile = (REPO / 'docker' / 'Dockerfile').read_text().rstrip()
    if with_comet2:
        marker = 'COPY . /opt/submit'
        if dockerfile.count(marker) != 1:
            raise ValueError('Expected one team source COPY instruction')
        runtime = (REPO / 'docker' / 'Dockerfile.comet2').read_text().rstrip()
        check = 'RUN JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/venv-comet2/bin/python /opt/submit/solutions/comet2/tools/check_runtime.py'
        dockerfile = dockerfile.replace(marker, runtime + '\n' + marker + '\n' + check)
    lines = [dockerfile, '']
    contexts = []
    for i, (path, name) in enumerate(names.items()):
        contexts += ['--build-context', f'ckpt{i}={path}']
        if a.no_files:
            lines.append(f'COPY --from=ckpt{i} / /opt/checkpoints/{name}/')
            continue
        # A whole checkpoint in one layer (~12 GB) is over the layer size registries accept (GHCR: 10 GB), and one layer per
        # file is over Docker's 127 layers with five checkpoints: one layer per folder, split into parts of at most LAYER_BYTES.
        root = Path(path)
        folders: dict[str, list[Path]] = {}
        for f in sorted(p for p in root.rglob('*') if p.is_file()):
            folders.setdefault(f.parent.relative_to(root).as_posix(), []).append(f)
        for folder, files in folders.items():
            dest = f'/opt/checkpoints/{name}/' + ('' if folder == '.' else folder + '/')
            part, size = [], 0
            for f in files + [None]:
                if part and (f is None or size + f.stat().st_size > LAYER_BYTES):
                    lines.append(f'COPY --from=ckpt{i} {json.dumps(["/" + p.relative_to(root).as_posix() for p in part] + [dest])}')
                    part, size = [], 0
                if f is not None:
                    part.append(f)
                    size += f.stat().st_size
    lines.append('COPY .build/routing.csv /opt/submit/config/routing.csv')
    layers = sum(l.split(maxsplit=1)[0] in ('COPY', 'ADD', 'RUN') for l in '\n'.join(lines).splitlines() if l.strip())
    if layers > 120:
        raise SystemExit(f'{layers} layers: Docker allows 127; group the checkpoint files')
    (build / 'Dockerfile').write_text('\n'.join(lines) + '\n')

    cmd = ['docker', 'build', '-f', str(build / 'Dockerfile'), '--build-arg', f'WITH_VLA={int(with_vla)}', *contexts, '-t', a.tag, str(REPO)]
    print(f'{len(routes)} tasks routed, {len(names)} checkpoint(s), Larchenko: {with_vla}, comet2: {with_comet2}')
    print(' '.join(shlex.quote(c) for c in cmd))
    if not a.dry_run:
        subprocess.run(cmd, check=True, env={**os.environ, 'DOCKER_BUILDKIT': '1'})
        print(f'built {a.tag}; run: docker run --gpus all --rm -p 8000:8000 {a.tag}')


if __name__ == '__main__':
    main()
