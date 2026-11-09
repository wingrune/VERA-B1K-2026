#!/usr/bin/env python3
"""Submission policy server: one websocket endpoint for the 2026 evaluator, each task routed to its own solution.

usage: python server/router.py [--port 8000] [--routing config/routing.csv] [--preload <task>]

Every observation carries obs["task_id"] (2026 index). The router looks the task up in the routing table, makes sure the
solution's backend process runs (config/solutions.json: command, one process per task or per checkpoint, optionally a fresh
process per episode), and passes the evaluator's messages through unchanged. One backend runs at a time, so a 24 GB GPU holds
whichever model the current task needs; switching tasks stops the previous backend. A task id without a row of its own (a task
outside the 100) goes to the * route when the table has one. A backend error, a backend reply the evaluator would reject (no
finite 23-d action, or an action chunk of the wrong shape), a backend that does not answer within request_timeout_s, or an
unrouted task fails only the current episode: the router stops the backend and answers every request until the next reset
with an action that holds the robot's current pose, so the evaluator goes on to its next episode instead of stopping. A second
evaluator connection waits until the first one closes.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import websockets
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve as ws_serve

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from server.protocol import health, packb, unpackb  # noqa: E402
from server.routing import DEFAULT, TASK_IDS, Route, load_routing, load_solutions, task_order  # noqa: E402

NAMES = {i: n for n, i in TASK_IDS.items()}
BACKEND_PORT = 8100
REQUEST_TIMEOUT_S = 1800    # per backend reply; a planner's first reply includes the cuRobo warm-up (~380 s on an RTX 4090)
log = logging.getLogger('router')

# R1Pro action (23): base velocity 0:3, trunk 3:7, left arm 7:14, left gripper 14, right arm 15:22, right gripper 22.
# Compact 2026 proprio (61): left arm qpos 3:10, left fingers 24:26, right arm qpos 28:35, right fingers 49:51, trunk qpos 53:57.
ACTION_DIM, PROPRIO_DIM, FINGER_OPEN = 23, 61, 0.05


def hold_action(msg: dict) -> bytes:
    """Reply that keeps the robot where it is: base stopped, trunk and arms at their positions, grippers as open as they are."""
    proprio = np.asarray(msg.get('robot_r1::proprio', np.zeros((1, PROPRIO_DIM))), dtype=np.float32)
    if proprio.ndim != 2 or proprio.shape[1] != PROPRIO_DIM or not np.isfinite(proprio).all():
        proprio = np.zeros((1, PROPRIO_DIM), np.float32)
    action = np.zeros((proprio.shape[0], ACTION_DIM), np.float32)
    action[:, 3:7] = proprio[:, 53:57]
    action[:, 7:14] = proprio[:, 3:10]
    action[:, 15:22] = proprio[:, 28:35]
    action[:, 14] = proprio[:, 24:26].mean(axis=1) / FINGER_OPEN * 2 - 1
    action[:, 22] = proprio[:, 49:51].mean(axis=1) / FINGER_OPEN * 2 - 1
    action[:, [14, 22]] = np.clip(action[:, [14, 22]], -1, 1)
    reply = {'action': action}
    chunk = int(np.asarray(msg.get('__action_chunk_size__', 0)).reshape(-1)[0])
    if chunk > 1:
        reply['action_chunk'] = np.repeat(action[:, None], chunk, axis=1)
    return packb(reply)


def check_reply(raw, msg: dict) -> None:
    """Raise unless the evaluator's client accepts the reply: a finite (..., 23) action and, when it asked for a chunk and one is
    sent, a finite chunk of the asked length whose first step equals the action."""
    if isinstance(raw, str):
        raise RuntimeError(f'backend error: {raw}')
    reply = unpackb(raw)
    action = np.asarray(reply['action'], dtype=np.float32)
    if action.shape[-1:] != (ACTION_DIM,) or not np.isfinite(action).all():
        raise RuntimeError(f'backend action has shape {action.shape} or is not finite')
    chunk_size = int(np.asarray(msg.get('__action_chunk_size__', 0)).reshape(-1)[0])
    if chunk_size > 1 and 'action_chunk' in reply:
        chunk = np.asarray(reply['action_chunk'], dtype=np.float32)
        if chunk.shape != (*action.shape[:-1], chunk_size, ACTION_DIM) or not np.isfinite(chunk).all() \
                or not np.array_equal(chunk[..., 0, :], action):
            raise RuntimeError(f'backend action_chunk has shape {chunk.shape}, is not finite or does not start with the action')


class Backend:
    def __init__(self, key: tuple, route: Route, spec: dict, port: int) -> None:
        self.key, self.route, self.spec, self.port = key, route, spec, port
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.ws = None
        self.used = False

    async def start(self) -> None:
        fill = dict(repo=REPO, task=self.route.task, port=self.port, checkpoint=self.route.checkpoint)
        argv = [os.path.expandvars(str(x).format(**fill)) for x in self.spec['command']]
        log.info('start %s for %s: %s', self.route.solution, self.route.task, ' '.join(argv))
        self.proc = await asyncio.create_subprocess_exec(*argv, start_new_session=True)
        deadline = time.monotonic() + float(self.spec.get('startup_timeout_s', 600))
        while time.monotonic() < deadline:
            if self.proc.returncode is not None:
                raise RuntimeError(f'{self.route.solution} backend exited with {self.proc.returncode} during start')
            try:
                self.ws = await connect(f'ws://127.0.0.1:{self.port}', compression=None, max_size=None, ping_interval=None, open_timeout=5)
                await self.ws.recv()                              # metadata
                log.info('%s ready on port %d', self.route.solution, self.port)
                return
            except (OSError, asyncio.TimeoutError, websockets.InvalidHandshake):
                await asyncio.sleep(2)
        raise RuntimeError(f'{self.route.solution} backend not ready after {self.spec.get("startup_timeout_s")} s')

    async def stop(self) -> None:
        if self.ws is not None:
            await self.ws.close()
        if self.proc is not None and self.proc.returncode is None:
            for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
                try:
                    os.killpg(self.proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    await asyncio.wait_for(self.proc.wait(), wait)
                    break
                except asyncio.TimeoutError:
                    continue
        log.info('stopped %s (%s)', self.route.solution, self.route.task)


class Router:
    def __init__(self, routes: dict) -> None:
        self.routes, self.solutions = routes, load_solutions()
        self.backend: Optional[Backend] = None
        self.pending_reset = False
        self.episode_failed = False
        self.port = BACKEND_PORT
        self.busy = asyncio.Lock()

    async def ensure(self, route: Route) -> Backend:
        spec = self.solutions[route.solution]
        key = (route.solution, route.task if spec.get('per_task') else route.checkpoint)
        b = self.backend
        if b is not None and b.key == key and not (self.pending_reset and spec.get('fresh_process_per_episode') and b.used):
            return b
        if b is not None:
            await b.stop()
            self.backend = None
        self.port = BACKEND_PORT + (self.port - BACKEND_PORT + 1) % 50     # a new port: the old one may linger in TIME_WAIT
        b = Backend(key, route, spec, self.port)
        try:
            await b.start()
        except Exception:
            await b.stop()
            raise
        self.backend = b
        return b

    async def handle(self, ws) -> None:
        async with self.busy:                       # one evaluator at a time: a second connection waits for the first to close
            self.episode_failed = False
            await ws.send(packb({}))
            while True:
                try:
                    raw = await ws.recv()
                except websockets.ConnectionClosed:
                    return
                msg = unpackb(raw)
                if 'reset' in msg:
                    self.pending_reset = True
                    self.episode_failed = False
                    continue
                if self.episode_failed:
                    await ws.send(hold_action(msg))
                    continue
                try:
                    task_id = int(np.asarray(msg['task_id']).reshape(-1)[0])
                    task = NAMES.get(task_id, str(task_id))
                    route = self.routes.get(task) or self.routes.get(DEFAULT)
                    if route is None:
                        raise RuntimeError(f'task {task} (id {task_id}) is not routed in the routing table')
                    b = await self.ensure(route)
                    if self.pending_reset:
                        await b.ws.send(packb({'reset': True}))
                        self.pending_reset = False
                    await b.ws.send(raw)
                    reply = await asyncio.wait_for(b.ws.recv(), float(b.spec.get('request_timeout_s', REQUEST_TIMEOUT_S)))
                    check_reply(reply, msg)
                    b.used = True
                except Exception:  # noqa: BLE001 - fail this episode only, the evaluator moves on after its reset
                    log.exception('request failed, holding the pose until the next episode')
                    self.episode_failed = True
                    reply = hold_action(msg)
                    if self.backend is not None:
                        await self.backend.stop()
                        self.backend = None
                await ws.send(reply)


async def amain(a: argparse.Namespace) -> None:
    routes = load_routing(Path(a.routing), check_files=not a.no_file_checks)
    for r in sorted(routes.values(), key=lambda r: task_order(r.task)):
        log.info('route %3s %-40s -> %s %s', TASK_IDS.get(r.task, DEFAULT), r.task, r.solution, r.checkpoint)
    router = Router(routes)
    if a.preload:
        router.pending_reset = True
        await router.ensure(routes[a.preload])
    async with ws_serve(router.handle, '0.0.0.0', a.port, compression=None, max_size=None, process_request=health,
                        ping_interval=60, ping_timeout=300) as srv:
        log.info('router on port %d, %d tasks routed', a.port, len(routes))
        await srv.serve_forever()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8000)))
    p.add_argument('--routing', default=os.environ.get('ROUTING', str(REPO / 'config' / 'routing.csv')))
    p.add_argument('--preload', default=os.environ.get('PRELOAD_TASK') or None, help='start this task\'s backend before the first request')
    p.add_argument('--no-file-checks', action='store_true')
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s router %(message)s')
    asyncio.run(amain(a))


if __name__ == '__main__':
    main()
