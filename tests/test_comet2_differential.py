import ast
import copy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REFERENCE = os.environ.get('COMET2_REFERENCE_BACKEND')


@unittest.skipUnless(REFERENCE, 'Set COMET2_REFERENCE_BACKEND for frozen-source differential tests')
class Comet2Differential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np
        path = Path(REFERENCE)
        spec = importlib.util.spec_from_file_location('comet2_baseline', path / '__init__.py',
                                                     submodule_search_locations=[str(path)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.old = {n: importlib.import_module('comet2_baseline.' + n) for n in ('wrapper', 'preprocessing', 'scheduler', 'recovery', 'config')}
        cls.new = {n: importlib.import_module('solutions.comet2.' + n) for n in cls.old}
        if os.environ.get('COMET2_OBSERVATION'):
            data = np.load(os.environ['COMET2_OBSERVATION'])
            cls.obs = {k: data[k] for k in data.files}
        else:
            rng = np.random.default_rng(8)
            cls.obs = {'robot_r1::proprio': rng.normal(size=(1, 61)).astype(np.float32), 'task_id': np.array([[66]])}
            for i, key in enumerate(cls.new['preprocessing'].CAMERA_KEYS):
                size = 720 if i == 0 else 480
                cls.obs[key] = rng.integers(0, 256, (1, size, size, 4), dtype=np.uint8)

    def equal(self, left, right):
        if isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal(a, b)
        elif isinstance(left, (str, bool)) or left is None:
            self.assertEqual(left, right)
        else:
            a, b = self.np.asarray(left), self.np.asarray(right)
            self.assertEqual(a.dtype, b.dtype)
            self.np.testing.assert_array_equal(a, b)

    def test_execution_code_unchanged(self):
        def semantics(path):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                        node.body.pop(0)
            return ast.dump(tree)
        for name in ('preprocessing', 'policy_config', 'wrapper', 'scheduler', 'recovery', 'serving'):
            self.assertEqual(semantics(Path(REFERENCE) / (name + '.py')),
                             semantics(ROOT / 'solutions/comet2' / (name + '.py')), name)
        self.assertEqual(self.old['config'].MODEL_ENV, self.new['config'].MODEL_ENV)

    def test_real_rgb_state_and_model_packing(self):
        np = self.np
        tasks = json.loads((ROOT / 'solutions/comet2/tasks.json').read_text())
        for task in tasks:
            for batched in (True, False):
                obs = self.obs if batched else {k: np.asarray(v)[0] for k, v in self.obs.items()}
                a = self.old['preprocessing'].ObservationProcessor()(obs, task)
                b = self.new['preprocessing'].ObservationProcessor()(obs, task)
                self.equal(a, b)
                self.equal(self.old['preprocessing'].B1KInputs()(a), self.new['preprocessing'].B1KInputs()(b))
                self.assertEqual(b['prompt'], task)

    def test_all_recovery_rules_and_boundaries(self):
        np = self.np
        old = self.old['recovery'].RecoveryPolicy.from_file(Path(REFERENCE) / 'assets/recovery_rules.json')
        new = self.new['recovery'].RecoveryPolicy.from_file(ROOT / 'solutions/comet2/assets/recovery_rules.json')
        self.assertEqual(old.rules, new.rules)
        for task in [*old.rules, 'unknown']:
            for stage in (None, -1, 0, 1, 15):
                for left in (-1., np.nextafter(-.98, -np.inf), -.98, np.nextafter(-.98, np.inf), 1.):
                    for right in (-1., -.98, 1.):
                        state = np.zeros(23); state[[14, 22]] = [left, right]
                        actions = np.arange(32 * 23, dtype=np.float32).reshape(32, 23) / 1000
                        self.equal(old.correct(task, stage, state, actions.copy()), new.correct(task, stage, state, actions.copy()))
        self.equal(vars(old.stats), vars(new.stats))
        old.reset(); new.reset(); self.equal(vars(old.stats), vars(new.stats))

    def test_actions_chunks_task_switch_and_reset(self):
        np = self.np
        rng = np.random.default_rng(5)
        chunks = rng.normal(size=(5, 32, 23)).astype(np.float32)
        chunks[:, :, [14, 22]] = 0
        chunks[1:, 13:, 14] = .8
        class Model:
            def __init__(self): self.calls = 0
            def infer(self, obs):
                value = chunks[self.calls % len(chunks)].copy(); self.calls += 1
                return {'actions': value}
            def reset(self): pass
        p, q = Model(), Model()
        old = self.old['wrapper'].B1KPolicyWrapper(p)
        new = self.new['wrapper'].B1KPolicyWrapper(q)
        obs = copy.deepcopy(self.obs)
        for task_id in (66, 71, 79, 84, 88, 97, 98):
            obs['task_id'] = np.array([[task_id]])
            for count in [1] * 21 + [20, 20, 7, 3, 16, 4, 20]:
                self.equal(old.act_chunk(obs, count), new.act_chunk(obs, count))
                self.assertEqual(p.calls, q.calls)
            old.reset(); new.reset(); self.equal(old.act(obs), new.act(obs))
        with self.assertRaises(KeyError):
            new.act(dict(obs, task_id=np.array([[9999]])))


if __name__ == '__main__':
    unittest.main()
