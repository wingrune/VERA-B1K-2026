#!/usr/bin/env python3
"""Verify every preserved source file, symlink, and executable permission."""
import hashlib
import json
import os
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / 'release/source-manifest.json').read_text())
    failures = []
    for row in manifest['files']:
        path = ROOT / row['path']
        try:
            if row['mode'] == '120000':
                data = os.readlink(path).encode()
            else:
                assert not path.is_symlink()
                assert bool(path.stat().st_mode & stat.S_IXUSR) == (row['mode'] == '100755')
                data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == row['sha256']
        except (OSError, AssertionError):
            failures.append(row['path'])
    if failures:
        raise SystemExit('Source differs from the imported snapshot:\n' + '\n'.join(failures))
    print(f"Verified {len(manifest['files'])} preserved entries from {manifest['source_commit']}")


if __name__ == '__main__':
    verify()
