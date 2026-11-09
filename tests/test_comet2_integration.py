import csv
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tokenize
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.routing import load_routing, load_solutions
from scripts.eval_policy_args import action_chunk_size
from scripts.check_comet2_checkpoint import verify


class Comet2Integration(unittest.TestCase):
    def test_comment_free_source_and_checkpoint_metadata(self):
        paths = list((ROOT / 'solutions/comet2').rglob('*.py'))
        paths += list((ROOT / 'tests').glob('*comet2*.py'))
        for path in paths:
            source = path.read_text()
            comments = [token for token in tokenize.generate_tokens(io.StringIO(source).readline)
                        if token.type == tokenize.COMMENT and not token.string.startswith('#!')]
            self.assertEqual(comments, [], str(path))
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.assertIsNone(ast.get_docstring(node), str(path))
        metadata = json.loads((ROOT / 'solutions/comet2/checkpoint_provenance.json').read_text())
        self.assertEqual(set(metadata['input_info']) & {'parent_' + 'checkpoint'}, set())

    def test_bundled_runtime_checksums(self):
        root = ROOT / 'solutions/comet2/runtime'
        manifest = json.loads((ROOT / 'solutions/comet2/runtime_manifest.json').read_text())
        for name, row in manifest['files'].items():
            data = (root / name).read_bytes()
            self.assertEqual(len(data), row['bytes'], name)
            self.assertEqual(hashlib.sha256(data).hexdigest(), row['sha256'], name)

    def test_routes_and_isolation(self):
        tasks = json.loads((ROOT / 'solutions/comet2/tasks.json').read_text())
        routes = load_routing(ROOT / 'config/routing.csv', check_files=False)
        routed = {k: v for k, v in routes.items() if v.solution == 'comet2'}
        ids = json.loads((ROOT / 'server/task_ids_2026.json').read_text())
        self.assertEqual(tasks, ids)
        self.assertEqual(set(routes), set(ids) | {'*'})
        self.assertEqual({66, 71, 79, 84, 88, 97, 98}, {ids[k] for k in routed})
        self.assertEqual({v.checkpoint for v in routed.values()}, {str(ROOT / 'checkpoints/comet2')})
        group1 = {k: v for k, v in routes.items() if v.solution == 'comet2_1'}
        self.assertEqual({20, 26, 67, 81, 86, 91}, {ids[k] for k in group1})
        self.assertEqual({v.checkpoint for v in group1.values()}, {str(ROOT / 'checkpoints/comet2_1')})
        fallback = {k: v for k, v in routes.items() if v.solution == 'comet3'}
        self.assertIn('*', fallback)
        self.assertEqual({v.checkpoint for v in fallback.values()}, {str(ROOT / 'checkpoints/comet3')})
        solutions = load_solutions()
        for name in ('comet2', 'comet2_1', 'comet3'):
            spec = solutions[name]
            self.assertEqual(spec['command'], solutions['comet2']['command'])
            self.assertTrue(spec['fresh_process_per_episode'])
            self.assertFalse(spec['per_task'])
            self.assertTrue(spec['needs_checkpoint'])
            self.assertIn('--checkpoint', spec['command'])
            self.assertEqual(spec['action_chunk_size'], 20)
        self.assertNotIn('tasks_from', solutions['comet3'])

    def test_replay_is_task_specific(self):
        routes = ROOT / 'config/routing.csv'
        solutions = ROOT / 'config/solutions.json'
        self.assertEqual(action_chunk_size('66', routes, solutions), 20)
        self.assertEqual(action_chunk_size('polishing_shoes', routes, solutions), 20)
        self.assertEqual(action_chunk_size('make_pizza', routes, solutions), 20)
        for task in ('turning_on_radio', 'organizing_school_stuff', 'picking_up_trash'):
            self.assertEqual(action_chunk_size(task, routes, solutions), 1)
        with self.assertRaises(ValueError):
            action_chunk_size('unknown', routes, solutions)

    def test_build_variants_and_checkpoint_deduplication(self):
        for tasks, expected_contexts, runtime in ((['turning_on_radio'], 0, False),
                                                (['turning_out_all_lights_before_sleep', 'laying_tile_floors'], 1, True)):
            with tempfile.TemporaryDirectory() as directory:
                routing = Path(directory) / 'routing.csv'
                all_rows = list(csv.DictReader((ROOT / 'config/routing.csv').read_text().splitlines()))
                with routing.open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['task', 'solution', 'checkpoint', 'notes'])
                    writer.writeheader()
                    writer.writerows(row for row in all_rows if row['task'] in tasks)
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/build_docker.py'),
                    '--routing', str(routing), '--dry-run', '--no-files'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                dockerfile = (ROOT / '.build/Dockerfile').read_text()
                self.assertEqual(dockerfile.count('COPY --from=ckpt'), expected_contexts)
                self.assertEqual('uv venv /opt/venv-comet2' in dockerfile, runtime)
                self.assertIn('WITH_VLA=0', result.stdout)
                if runtime:
                    self.assertIn('solutions/comet2/requirements.lock', dockerfile)
                    self.assertIn('solutions/comet2/tools/check_runtime.py', dockerfile)
                    self.assertEqual(dockerfile.count('FROM '), 1)
                    fragment = (ROOT / 'docker/Dockerfile.comet2').read_text()
                    self.assertFalse(any(line.startswith(('FROM ', 'COPY --from=')) for line in fragment.splitlines()))

    def test_file_check_bypass_cannot_build(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/build_docker.py'), '--no-files'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires --dry-run', result.stderr)

    def test_checkpoint_required_before_any_model_import(self):
        result = subprocess.run([sys.executable, str(ROOT / 'solutions/comet2/backend.py'), '--port', '8999'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--checkpoint', result.stderr)

    def test_manifest_detects_corruption_and_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for name, value in (('assets/norm.json', b'normalization'), ('params/weights', b'weights')):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
                files.append(dict(path=name, bytes=len(value), sha256=hashlib.sha256(value).hexdigest()))
            canonical = ''.join(f'{row["sha256"]}  {row["path"]}\n' for row in sorted(files, key=lambda x: x['path']))
            manifest = dict(files=files, checkpoint_sha256=hashlib.sha256(canonical.encode()).hexdigest())
            self.assertTrue(verify(root, manifest)['passed'])
            (root / 'params/weights').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify(root, manifest)
            (root / 'params/extra').write_text('extra')
            with self.assertRaisesRegex(ValueError, 'file set'):
                verify(root, manifest)


if __name__ == '__main__':
    unittest.main()
