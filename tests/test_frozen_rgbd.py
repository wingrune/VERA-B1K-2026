"""Input, output, isolation and frozen-source boundaries of the CPU adapter."""
import json
import unittest
from unittest.mock import patch

import numpy as np
from solutions.frozen_rgbd.backend import FrozenRGBDPolicy, verify_core, REPO
from server.routing import load_routing


class FrozenRGBD(unittest.TestCase):
    def adapter(self, unbatch=False):
        p = FrozenRGBDPolicy.__new__(FrozenRGBDPolicy)
        p.spec = {'task_id_2026': 87, 'entry': {'unbatch_observations': unbatch}}
        p.keys = ('robot_r1::proprio', 'camera')
        self.seen = {}

        class Core:
            def action(_, obs):
                self.seen.update(obs)
                return np.arange(23, dtype=np.float32)

        p.policy = Core()
        return p

    def obs(self):
        return {'robot_r1::proprio': np.zeros((1, 61), np.float32),
                'camera': np.ones((1, 3, 4, 3), np.uint8), 'task_id': np.array([[87]]),
                'forbidden_extra': object()}

    def test_whitelist_preserves_original_arrays(self):
        obs = self.obs(); action = self.adapter().act(obs)
        self.assertEqual(set(self.seen), {'robot_r1::proprio', 'camera'})
        self.assertIs(self.seen['camera'], obs['camera'])
        np.testing.assert_array_equal(action, np.arange(23, dtype=np.float32)[None])

    def test_cans_unbatches_once(self):
        obs = self.obs(); self.adapter(True).act(obs)
        self.assertEqual(self.seen['robot_r1::proprio'].shape, (61,))
        self.assertEqual(self.seen['camera'].shape, (3, 4, 3))
        self.assertTrue(np.shares_memory(self.seen['camera'], obs['camera']))

    def test_wrong_task_and_batch_rejected_before_core(self):
        p = self.adapter()
        for updates in [{'task_id': np.array([[59]])}, {'robot_r1::proprio': np.zeros((2, 61))}]:
            with self.assertRaises(ValueError):
                p.act({**self.obs(), **updates})
        self.assertEqual(self.seen, {})

    def test_nonfinite_and_wrong_action_shape_rejected(self):
        p = self.adapter()
        for value in [np.zeros((1, 23)), np.full(23, np.nan)]:
            p.policy.action = lambda obs: value
            with self.assertRaises(ValueError):
                p.act(self.obs())

    def test_all_fingerprints_and_routes(self):
        tasks = json.loads((REPO / 'config/solutions.json').read_text())['frozen_rgbd']['tasks']
        routes = load_routing(REPO / 'config/routing.csv', check_files=False)
        for task in tasks:
            root, spec = verify_core(task)
            self.assertTrue((root / 'observation_policy.py').is_file())
            self.assertEqual(routes[task].solution, 'frozen_rgbd')
            self.assertEqual(spec['task'], task)

    def test_mutated_manifest_rejected(self):
        task = 'collecting_aluminum_cans'
        original = json.loads((REPO / 'tasks' / task / 'fingerprint.json').read_text())
        with patch('solutions.frozen_rgbd.backend.json.loads') as loads:
            # Supply a manifest with a changed expected digest without touching frozen files.
            import copy
            changed = copy.deepcopy(original); changed['sha256'] = '0' * 64
            loads.side_effect = [{'kind': 'frozen_rgbd', 'entry': {'code': 'controller'},
                                 'source': {'fingerprint': original['sha256']}}, changed]
            with self.assertRaises(RuntimeError):
                verify_core(task)


if __name__ == '__main__':
    unittest.main()
