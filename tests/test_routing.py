"""python3 -m unittest discover tests   (stdlib + numpy)"""
import csv
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'solutions' / 'larchenko_2025')]
from server.routing import DEFAULT, TASK_IDS, load_routing, load_solutions  # noqa: E402


class Routing(unittest.TestCase):
    def table(self, text):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        f = Path(tmp.name) / 'routing.csv'
        f.write_text(text)
        return f

    def test_table_is_plain_csv(self):
        # GitHub renders config/routing.csv as a table: no comment lines, one header, the same width on every row
        text = (REPO / 'config' / 'routing.csv').read_text()
        rows = list(csv.reader(text.splitlines(), strict=True))
        self.assertEqual(rows[0], ['task', 'solution', 'checkpoint', 'notes'])
        self.assertEqual({len(r) for r in rows}, {4})
        self.assertFalse([r for r in rows if r[0].startswith('#')])

    def test_example_lists_every_task_and_validates(self):
        example = REPO / 'config' / 'routing.csv'
        rows = [l for l in example.read_text().splitlines() if l and not l.startswith(('#', 'task,'))]
        self.assertEqual(sorted(r.split(',')[0] for r in rows), sorted([*TASK_IDS, DEFAULT]))
        routes = load_routing(example, check_files=False)          # the checkpoint files are not in git
        larchenko = {2: {0, 1, 7, 8, 9, 12, 16, 17, 18, 20, 21, 22, 26, 30, 43, 45}, 3: {4, 27, 31, 32, 33, 35, 36, 37, 38, 39, 41, 46, 49}}
        for r in routes.values():
            if r.solution.startswith('larchenko_ckpt'):
                self.assertIn(TASK_IDS[r.task], larchenko[int(r.solution[-1])], r.task)   # the checkpoint finetuned on the task
                self.assertEqual(r.checkpoint, str(REPO / 'checkpoints' / 'larchenko' / f'checkpoint_{r.solution[-1]}'))
        # every task is served: comet2 serves its seven group2 tasks, comet2_1 its six group1 tasks, comet3 every other task
        # with no other solution and, through the * row, any task id outside the 100 (an unrouted task scores 0)
        self.assertEqual(sorted(routes), sorted([*TASK_IDS, DEFAULT]))
        self.assertEqual(routes[DEFAULT].solution, 'comet3')
        self.assertEqual(sorted(json.loads((REPO / 'solutions' / 'comet2' / 'tasks.json').read_text())), sorted(TASK_IDS))
        self.assertEqual({TASK_IDS[r.task] for r in routes.values() if r.solution == 'comet2'}, {66, 71, 79, 84, 88, 97, 98})
        self.assertEqual({TASK_IDS[r.task] for r in routes.values() if r.solution == 'comet2_1'}, {20, 26, 67, 81, 86, 91})
        for r in routes.values():
            if r.solution.startswith('comet'):
                self.assertEqual(r.checkpoint, str(REPO / 'checkpoints' / r.solution))
        planners = [r for r in routes.values() if r.solution in ('planner', 'planner_cpu')]
        self.assertEqual(len(planners), 10)
        for r in planners:
            spec = json.loads((REPO / 'tasks' / r.task / 'solution.json').read_text())
            self.assertEqual(spec['task_id_2026'], TASK_IDS[r.task])
            self.assertTrue((REPO / 'tasks' / r.task / 'root' / spec['entry']['package'] / '__init__.py').is_file())

    def test_native_cpu_planners_are_scoped_and_reset_independently(self):
        routes = load_routing(REPO / 'config' / 'routing.csv', check_files=False)
        self.assertEqual({r.task for r in routes.values() if r.solution == 'planner_cpu'},
                         {'make_microwave_popcorn', 'installing_smoke_detectors'})
        spec = load_solutions()['planner_cpu']
        self.assertEqual(spec['command'][0], '$FROZEN_RGBD_PYTHON')
        self.assertIn('{repo}/solutions/planner/backend.py', spec['command'])
        self.assertTrue(spec['per_task'])
        self.assertTrue(spec['fresh_process_per_episode'])
        self.assertFalse(spec['needs_checkpoint'])
        popcorn = json.loads((REPO / 'tasks/make_microwave_popcorn/solution.json').read_text())
        self.assertTrue(popcorn['entry']['reset_task_spec'])

    def test_vla_rows(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ckpt = tmp.name
        routes = load_routing(self.table(f'task,solution,checkpoint\nturning_on_radio,larchenko_ckpt2,{ckpt}\nmake_pizza,larchenko_ckpt3,{ckpt}\n'))
        self.assertEqual(routes['turning_on_radio'].checkpoint, ckpt)
        with self.assertRaises(SystemExit) as e:
            load_routing(self.table(f'task,solution,checkpoint\nstore_honey,larchenko_ckpt2,{ckpt}\ncook_bacon,larchenko_ckpt4,\n'))
        self.assertIn('cannot serve store_honey', str(e.exception))
        self.assertIn('needs a checkpoint path', str(e.exception))
        with self.assertRaises(SystemExit) as e:
            with mock.patch.dict(os.environ):
                os.environ.pop('B1K_NO_SUCH_CKPT', None)
                load_routing(self.table('task,solution,checkpoint\nturning_on_radio,larchenko_ckpt2,$B1K_NO_SUCH_CKPT\n'))
        self.assertIn('set that environment variable', str(e.exception))
        with self.assertRaises(SystemExit) as e:
            load_routing(self.table('task,solution,checkpoint\nturning_on_radio,larchenko_ckpt2,checkpoints/none_here\n'))
        self.assertIn('checkpoints/none_here is not a directory', str(e.exception))

    def test_default_row(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        routes = load_routing(self.table(f'task,solution,checkpoint\nturning_on_radio,planner,\n*,comet3,{tmp.name}\n'), check_files=False)
        self.assertEqual(routes[DEFAULT].solution, 'comet3')
        with self.assertRaises(SystemExit) as e:          # a solution limited to a task list cannot take unknown tasks
            load_routing(self.table(f'task,solution,checkpoint\n*,comet2,{tmp.name}\n'), check_files=False)
        self.assertIn('cannot serve *', str(e.exception))


class Adapter(unittest.TestCase):
    def test_task_remap(self):
        import adapter_2026 as A
        self.assertEqual(len(A.ID_2026_TO_2025), 50)
        self.assertEqual(A.task_2025(TASK_IDS['turning_on_radio']), 0)
        self.assertEqual(A.task_2025(TASK_IDS['make_pizza']), 49)
        with self.assertRaises(ValueError):
            A.task_2025(TASK_IDS['store_honey'])

    def test_act_rewrites_task_id_and_batch(self):
        import numpy as np
        import adapter_2026 as A
        seen = {}

        class Wrapper:
            def act(self, obs):
                seen.update(obs)
                return np.arange(23, dtype=np.float32)

        a = A.Adapter2026(Wrapper())
        out = a.act({'robot_r1::proprio': np.zeros((1, 61)), 'task_id': np.array([[TASK_IDS['make_pizza']]]),
                     'robot_r1::robot_r1:zed_link:Camera:0::depth_linear': np.zeros((1, 2, 2))})
        self.assertEqual(out.shape, (1, 23))
        self.assertEqual(int(seen['task_id'][0]), 49)
        self.assertFalse(any(k.endswith('depth_linear') for k in seen))
        with self.assertRaises(ValueError):
            a.act({'robot_r1::proprio': np.zeros((2, 61)), 'task_id': np.array([[0], [0]])})


if __name__ == '__main__':
    unittest.main()
