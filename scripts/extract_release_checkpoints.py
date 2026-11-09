#!/usr/bin/env python3
"""Copy the five released checkpoints from the pinned image, without starting it."""
import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[1]
IMAGE = 'ghcr.io/alstar8/b1k-evo-submit@sha256:1ab7d94f5b3ecc347dbb9adccc2d9212cfbd2ed4aad078bda82faad607a240c5'


def extract():
    source = list(csv.DictReader((REPO / 'config/routing.csv').open()))
    targets = {r['solution']: REPO / r['checkpoint'] for r in source if r['checkpoint']}
    for target in targets.values():
        if target.exists() or target.is_symlink():
            raise SystemExit(f'Refusing to overwrite {target}; move it aside first.')
    subprocess.run(['docker', 'pull', IMAGE], check=True)
    container = subprocess.check_output(['docker', 'create', IMAGE], text=True).strip()
    try:
        # Stage on the same filesystem; incomplete copies never become checkpoint roots.
        with tempfile.TemporaryDirectory(prefix='.release-extract-', dir=REPO / 'checkpoints') as temp:
            stage = Path(temp)
            subprocess.run(['docker', 'cp', f'{container}:/opt/submit/config/routing.csv', str(stage / 'routing.csv')], check=True)
            image_rows = list(csv.DictReader((stage / 'routing.csv').open()))
            for solution, target in targets.items():
                paths = {r['checkpoint'] for r in image_rows if r['solution'] == solution}
                if len(paths) != 1:
                    raise RuntimeError(f'Expected one image checkpoint for {solution}: {paths}')
                remote = PurePosixPath(paths.pop())
                if '..' in remote.parts or not remote.is_relative_to('/opt/checkpoints'):
                    raise RuntimeError(f'Unexpected image path: {remote}')
                dest = stage / solution
                dest.mkdir()
                subprocess.run(['docker', 'cp', f'{container}:{remote}/.', str(dest)], check=True)
                if not (dest / 'params').is_dir() or not (dest / 'assets').is_dir():
                    raise RuntimeError(f'Incomplete checkpoint: {solution}')
            from importlib.util import spec_from_file_location, module_from_spec
            spec = spec_from_file_location('checkpoint_check', REPO / 'scripts/check_comet2_checkpoint.py')
            check = module_from_spec(spec)
            spec.loader.exec_module(check)
            manifest = json.loads((REPO / 'solutions/comet2/checkpoint_provenance.json').read_text())
            check.verify(stage / 'comet2', manifest)
            for solution, target in targets.items():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(stage / solution), str(target))
                print(f'{solution}: {target.relative_to(REPO)}')
    finally:
        subprocess.run(['docker', 'rm', container], check=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show-image', action='store_true', help='Print the immutable image URI and exit')
    args = parser.parse_args()
    if args.show_image:
        print(IMAGE)
    else:
        extract()
