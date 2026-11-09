import asyncio
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import urllib.request

import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.protocol import health, packb, unpackb
from server.router import Router
from server.routing import TASK_IDS, load_routing


class Comet2Transport(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.router = Router(load_routing(ROOT / 'config/routing.csv', check_files=False))

        command = [sys.executable, str(ROOT / 'tests/comet2_fake_backend.py'), '--port', '{port}']
        for name in ('comet2', 'comet3'):
            self.router.solutions[name] = {
                **self.router.solutions[name], 'command': command, 'startup_timeout_s': 30}
        for task in ('turning_on_radio', 'picking_up_toys'):
            other = self.router.routes[task].solution
            self.router.solutions[other] = {
                **self.router.solutions[other], 'command': [*command, '--plain'], 'startup_timeout_s': 30}
        self.server = await serve(self.router.handle, '127.0.0.1', 0, compression=None,
                                  max_size=None, process_request=health)
        self.port = self.server.sockets[0].getsockname()[1]
        self.obs = {'robot_r1::proprio': np.zeros((1, 61), np.float32)}
        self.obs['robot_r1::proprio'][:, [24, 25, 49, 50]] = .05

        for camera in ('zed', 'left_realsense', 'right_realsense'):
            self.obs[f'robot_r1::robot_r1:{camera}_link:Camera:0::rgb'] = np.zeros((1, 224, 224, 4), np.uint8)

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        if self.router.backend:
            await self.router.backend.stop()

    async def request(self, ws, task, count=20):
        await ws.send(packb(dict(self.obs, task_id=np.array([[task]]), __action_chunk_size__=count)))
        raw = await asyncio.wait_for(ws.recv(), 40)
        self.assertIsInstance(raw, bytes, raw)
        reply = unpackb(raw)
        action = reply['action']
        self.assertEqual(action.shape, (1, 23))
        self.assertEqual(action.dtype, np.float32)
        self.assertTrue(np.isfinite(action).all())
        if count > 1:
            self.assertEqual(reply['action_chunk'].shape, (1, count, 23))
            self.assertEqual(reply['action_chunk'].dtype, np.float32)
            self.assertTrue(np.isfinite(reply['action_chunk']).all())
            np.testing.assert_array_equal(action, reply['action_chunk'][:, 0])
        return action

    async def test_health_chunks_reset_switching_and_connection_isolation(self):
        def get_health():
            with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/healthz', timeout=5) as response:
                return response.status, response.read()
        self.assertEqual(await asyncio.to_thread(get_health), (200, b'OK\n'))
        async with connect(f'ws://127.0.0.1:{self.port}', max_size=None, compression=None) as ws:
            self.assertEqual(unpackb(await ws.recv()), {})
            waiting = await connect(f'ws://127.0.0.1:{self.port}', max_size=None, compression=None)
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(waiting.recv(), 1)
            first = await self.request(ws, 66)
            self.assertEqual(tuple(first[0, 3:5]), (1, 66))
            first_pid = self.router.backend.proc.pid
            second = await self.request(ws, 66, 1)
            cached = await self.request(ws, 66, 7)
            self.assertEqual(second[0, 3], 2)
            np.testing.assert_array_equal(second, cached)
            await ws.send(packb({'reset': True}))
            fresh = await self.request(ws, 66)
            np.testing.assert_array_equal(first, fresh)
            self.assertNotEqual(first_pid, self.router.backend.proc.pid)
            switched = await self.request(ws, 79)
            self.assertEqual(tuple(switched[0, 3:5]), (2, 79))
            await ws.send(packb({'reset': True}))
            for task in ('turning_on_radio', 'picking_up_toys'):
                plain = await self.request(ws, TASK_IDS[task], 1)
                np.testing.assert_array_equal(plain, np.full((1, 23), 7, np.float32))
            await ws.send(packb({'reset': True}))
            back = await self.request(ws, 66)
            np.testing.assert_array_equal(first, back)
            await ws.send(packb({'reset': True}))
            fallback = await self.request(ws, TASK_IDS['can_meat'])
            self.assertEqual(self.router.backend.route.solution, 'comet3')
            self.assertEqual(tuple(fallback[0, 3:5]), (1, TASK_IDS['can_meat']))
            await ws.send(packb({'reset': True}))
            unknown = await self.request(ws, 9999)
            self.assertEqual(self.router.backend.route.solution, 'comet3')
            self.assertEqual(tuple(unknown[0, 3:5]), (1, -1))
            await self.assert_episode_fails_then_recovers(ws, 9999, lambda: self.router.routes.pop('*'))
            plain_route = self.router.routes['turning_on_radio']
            for mode in ('raise', 'nan', 'hang'):
                failing = dict(self.router.solutions[plain_route.solution], request_timeout_s=3)
                failing['command'] = [*failing['command'][:-1], '--fail', mode]
                self.router.solutions['failing'] = failing
                await self.assert_episode_fails_then_recovers(
                    ws, TASK_IDS['turning_on_radio'],
                    lambda: self.router.routes.__setitem__('turning_on_radio', replace(plain_route, solution='failing')))
        self.assertEqual(unpackb(await asyncio.wait_for(waiting.recv(), 10)), {}, 'the waiting evaluator is served next')
        await waiting.close()

    async def assert_episode_fails_then_recovers(self, ws, task, break_route):
        restore = dict(self.router.routes)
        break_route()
        await ws.send(packb({'reset': True}))
        held = await self.request(ws, task)
        expected = np.zeros((1, 23), np.float32)
        expected[:, [14, 22]] = 1
        np.testing.assert_array_equal(held, expected)
        self.assertIsNone(self.router.backend)
        self.router.routes = restore
        np.testing.assert_array_equal(await self.request(ws, task, 1), expected)
        self.assertIsNone(self.router.backend, 'a failed episode holds until the next reset')
        await ws.send(packb({'reset': True}))
        recovered = await self.request(ws, task, 1)
        self.assertIsNotNone(self.router.backend)
        self.assertFalse(np.array_equal(recovered, expected))


if __name__ == '__main__':
    unittest.main()
