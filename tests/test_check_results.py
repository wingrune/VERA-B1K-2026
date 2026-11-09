"""python3 -m unittest discover tests   (stdlib only)"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECK = REPO / 'scripts' / 'check_results.py'
MP4 = b'\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2'     # header only: enough when ffprobe is absent


def setUpModule():
    global MP4
    if shutil.which('ffmpeg'):                                       # with ffprobe present the checker reads the duration
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'v.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=16x16:d=0.2', '-pix_fmt', 'yuv420p', str(out)],
                           check=True)
            MP4 = out.read_bytes()


def result(task='store_honey', instance=301, q=1.0, success=True, steps=5177):
    return {'task': task, 'instance_id': instance, 'rollout_id': 0, 'steps': steps, 'success': success,
            'agent_distance': {'base': 3.1, 'left': 1.2, 'right': 2.0},
            'normalized_agent_distance': {'base': 1.4, 'left': 0.9, 'right': float('inf')},
            'q_score': {'final': q},
            'time': {'simulator_steps': steps, 'simulator_time': steps / 30, 'normalized_time': 0.8}}


class CheckResults(unittest.TestCase):
    def folder(self, results, name='store_honey', package=True, videos=True):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / name
        (root / 'json').mkdir(parents=True)
        (root / 'videos').mkdir()
        for r in results:
            stem = f"{r['task']}_{r['instance_id']}_{r['rollout_id']}"
            (root / 'json' / f'{stem}.json').write_text(json.dumps(r))
            if videos:
                (root / 'videos' / f'{stem}.mp4').write_bytes(MP4)
        if package:
            for f in ('README.md', 'rgbd_full_res_wrapper.py', 'r1pro.yaml'):
                (root / f).write_text('x')
        return root

    def run_check(self, root, *args):
        out = subprocess.run([sys.executable, str(CHECK), str(root), *args], capture_output=True, text=True)
        return out.returncode, out.stdout

    def test_valid(self):
        code, out = self.run_check(self.folder([result(), result(instance=302, q=0.5, success=False)]))
        self.assertEqual(code, 0, out)
        self.assertIn('Q 0.0008, SR 0.0005', out)      # (1.0 + 0.5) / 20 / 100 and 1 / 20 / 100

    def test_wrong_folder_name_and_rollout(self):
        r = result()
        r['rollout_id'] = 1
        code, out = self.run_check(self.folder([r], name='results_run1'))
        self.assertEqual(code, 1)
        self.assertIn('2026 task', out)
        self.assertIn('rollout 0', out)

    def test_content_mismatch_and_bad_q(self):
        r = result(q=1.3)
        r['instance_id'] = 305
        root = self.folder([r])
        (root / 'json' / 'store_honey_305_0.json').rename(root / 'json' / 'store_honey_301_0.json')
        code, out = self.run_check(root)
        self.assertEqual(code, 1)
        self.assertIn('instance_id is 305', out)
        self.assertIn('q_score.final must be', out)

    def test_missing_video_and_package(self):
        code, out = self.run_check(self.folder([result()], package=False, videos=False))
        self.assertEqual(code, 1)
        self.assertIn('videos/store_honey_301_0.mp4: missing', out)
        self.assertIn('README.md is missing', out)
        code, out = self.run_check(self.folder([result()], package=False), '--metrics-only')
        self.assertEqual(code, 0, out)

    def test_unknown_task_and_held_out_index(self):
        code, out = self.run_check(self.folder([result(task='make_tea'), result(instance=315)]))
        self.assertEqual(code, 1)
        self.assertIn('make_tea_301_0.json: not', out)
        self.assertIn('index 14', out)
        code, out = self.run_check(self.folder([result(instance=315)]), '--strict')
        self.assertEqual(code, 1)


if __name__ == '__main__':
    unittest.main()
