"""Verify frozen source identity without importing policies or simulator packages."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    tasks = json.loads((REPO / 'config/solutions.json').read_text())['frozen_rgbd']['tasks']
    total = 0
    for task in tasks:
        folder = REPO / 'tasks' / task
        spec = json.loads((folder / 'solution.json').read_text())
        manifest = json.loads((folder / 'fingerprint.json').read_text())
        root = folder / spec['entry']['code']
        actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
        digest = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if actual != manifest['files'] or digest != manifest['sha256'] or digest != spec['source']['fingerprint']:
            raise SystemExit(f'{task}: frozen source differs')
        if len(actual) != spec['source']['files']:
            raise SystemExit(f'{task}: file count differs')
        total += len(actual)
        print(task, len(actual), digest)
    print(f'PASS: {len(tasks)} tasks, {total} unchanged files')


if __name__ == '__main__':
    main()
