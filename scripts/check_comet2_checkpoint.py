#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify(root, manifest):
    files = manifest['files']
    actual = sorted(str(p.relative_to(root)) for name in ('params', 'assets')
                    for p in (root / name).rglob('*') if p.is_file())
    if actual != sorted(row['path'] for row in files):
        raise ValueError('Checkpoint params/assets file set differs from the comet2 manifest')
    lines = []
    for row in sorted(files, key=lambda item: item['path']):
        path = root / row['path']
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError(f'Checkpoint checksum mismatch: {row["path"]}')
        lines.append(f'{row["sha256"]}  {row["path"]}\n')
    checksum = hashlib.sha256(''.join(lines).encode()).hexdigest()
    if checksum != manifest['checkpoint_sha256']:
        raise ValueError('Canonical checkpoint fingerprint mismatch')
    return {'name': 'comet2', 'passed': True, 'files': len(files), 'checkpoint_sha256': checksum}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    manifest = json.loads((REPO / 'solutions/comet2/checkpoint_provenance.json').read_text())
    print(json.dumps(verify(args.checkpoint, manifest), indent=2))
