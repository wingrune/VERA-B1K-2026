"""Exercise checkpoint recovery without downloading the large public image."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_extraction', ROOT / 'scripts/extract_release_checkpoints.py')
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)


class ReleaseExtraction(unittest.TestCase):
    def fixture(self, root):
        (root / 'config').mkdir()
        (root / 'checkpoints').mkdir()
        (root / 'scripts').mkdir()
        (root / 'solutions/comet2').mkdir(parents=True)
        (root / 'scripts/check_comet2_checkpoint.py').write_bytes((ROOT / 'scripts/check_comet2_checkpoint.py').read_bytes())
        content = b'checkpoint bytes'
        digest = hashlib.sha256(content).hexdigest()
        paths = ['params/data', 'assets/stats.json']
        manifest = {'files': [{'path': p, 'bytes': len(content), 'sha256': digest} for p in paths],
                    'checkpoint_sha256': hashlib.sha256(''.join(f'{digest}  {p}\n' for p in sorted(paths)).encode()).hexdigest()}
        (root / 'solutions/comet2/checkpoint_provenance.json').write_text(json.dumps(manifest))
        with (root / 'config/routing.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['task', 'solution', 'checkpoint'])
            writer.writerows([['radio', 'comet2', 'checkpoints/comet2'], ['other', 'comet3', 'checkpoints/comet3']])
        return content, paths

    def test_existing_destination_prevents_pull(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / 'checkpoints/comet2').mkdir()
            with patch.object(extractor, 'REPO', root), patch.object(extractor.subprocess, 'run') as run:
                with self.assertRaisesRegex(SystemExit, 'Refusing to overwrite'):
                    extractor.extract()
                run.assert_not_called()

    def test_success_and_failure_cleanup_without_starting_container(self):
        for fail_copy in (False, True):
            with self.subTest(fail_copy=fail_copy), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                content, paths = self.fixture(root)
                calls = []

                def run(argv, **kwargs):
                    calls.append(argv)
                    if argv[:2] == ['docker', 'cp']:
                        dest = Path(argv[-1])
                        if dest.name == 'routing.csv':
                            dest.write_text('task,solution,checkpoint\nradio,comet2,/opt/checkpoints/group2\nother,comet3,/opt/checkpoints/all63\n')
                        else:
                            if fail_copy:
                                raise RuntimeError('copy failed')
                            for path in paths:
                                target = dest / path
                                target.parent.mkdir(parents=True, exist_ok=True)
                                target.write_bytes(content)

                with patch.object(extractor, 'REPO', root), patch.object(extractor.subprocess, 'run', side_effect=run), \
                        patch.object(extractor.subprocess, 'check_output', return_value='test-container\n'):
                    if fail_copy:
                        with self.assertRaisesRegex(RuntimeError, 'copy failed'):
                            extractor.extract()
                        self.assertFalse((root / 'checkpoints/comet2').exists())
                    else:
                        extractor.extract()
                        self.assertEqual((root / 'checkpoints/comet2/params/data').read_bytes(), content)
                        self.assertEqual((root / 'checkpoints/comet3/params/data').read_bytes(), content)
                self.assertEqual(calls[0], ['docker', 'pull', extractor.IMAGE])
                self.assertEqual(calls[-1], ['docker', 'rm', 'test-container'])
                self.assertFalse(any(c[:2] in (['docker', 'start'], ['docker', 'run']) for c in calls))


if __name__ == '__main__':
    unittest.main()
