"""python3 -m unittest discover tests   (stdlib only)"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_check_results as T  # noqa: E402

PROMOTE = REPO / 'scripts' / 'promote_task.py'


class Promote(unittest.TestCase):
    def setUp(self):
        T.setUpModule()
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.env = os.environ.copy()
        self.env['B1K_RESULTS'] = str(self.root / 'results')
        self.env['B1K_RESULTS_TMP'] = str(self.root / 'results_tmp')

    def write(self, root, qs):
        run = self.root / root / 'store_honey'
        for d in ('json', 'videos', 'logs'):
            (run / d).mkdir(parents=True, exist_ok=True)
        for inst, q in qs.items():
            stem = f'store_honey_{inst}_0'
            (run / 'json' / f'{stem}.json').write_text(json.dumps(T.result(instance=inst, q=q, success=q == 1.0)))
            (run / 'videos' / f'{stem}.mp4').write_bytes(T.MP4)
            (run / 'logs' / f'store_honey_{inst}.log').write_text(root)
        return run

    def promote(self, *args):
        out = subprocess.run([sys.executable, str(PROMOTE), '83', *args], capture_output=True, text=True, env=self.env)
        return out.returncode, out.stdout + out.stderr

    def test_rules(self):
        old = self.write('results', {301: 1.0, 302: 0.0})
        self.write('results_tmp', {301: 1.0, 302: 0.0})
        code, out = self.promote()
        self.assertEqual(code, 1, out)
        self.assertIn('not better', out)

        self.write('results_tmp', {301: 1.0, 302: 1.0})
        code, out = self.promote()
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads((old / 'json' / 'store_honey_302_0.json').read_text())['q_score']['final'], 1.0)
        self.assertEqual((old / 'logs' / 'store_honey_302.log').read_text(), 'results_tmp')

        self.write('results', {303: 0.0})
        code, out = self.promote()
        self.assertEqual(code, 1, out)
        self.assertIn('lacks instances [303]', out)


if __name__ == '__main__':
    unittest.main()
