#!/usr/bin/env python3
"""Smoke test of a running policy server without the simulator: reset + N synthetic observations of one task.

usage: python3 tools/smoke_client.py <task> [--host 127.0.0.1] [--port 8000] [--steps 3]     (needs numpy, msgpack, websockets)

The observation has the keys and shapes the 2026 evaluator sends with RGBDFullResWrapper and --num-envs 1 (head 720x720,
wrists 480x480 RGBA + depth, 61-d proprio, 21-d camera poses, task id), filled with a flat grey scene 1 m away. It checks that
the router starts the routed backend and that each reply is a finite 23-d action; what the policy does with a blank scene is
not tested.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from websockets.sync.client import connect

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from server.protocol import packb, unpackb  # noqa: E402

CAMS = {'zed_link': 720, 'left_realsense_link': 480, 'right_realsense_link': 480}


def observation(task_id: int) -> dict:
    obs = {}
    for link, px in CAMS.items():
        key = f'robot_r1::robot_r1:{link}:Camera:0'
        obs[f'{key}::rgb'] = np.full((1, px, px, 4), 128, np.uint8)
        obs[f'{key}::depth_linear'] = np.ones((1, px, px), np.float32)
    proprio = np.zeros((1, 61), np.float32)
    proprio[0, 20:24] = proprio[0, 45:49] = [0, 0, 0, 1]             # eef quaternions
    obs['robot_r1::proprio'] = proprio
    obs['robot_r1::cam_rel_poses'] = np.tile(np.array([0, 0, 1, 0, 0, 0, 1], np.float32), 3)[None]
    obs['task_id'] = np.array([[task_id]], np.int64)
    return obs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('task')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--steps', type=int, default=3)
    a = p.parse_args()
    task_id = json.loads((REPO / 'server' / 'task_ids_2026.json').read_text())[a.task]
    with connect(f'ws://{a.host}:{a.port}', compression=None, max_size=None, ping_interval=60, ping_timeout=600, open_timeout=30) as ws:
        print('metadata', unpackb(ws.recv()))
        ws.send(packb({'reset': True}))
        for step in range(a.steps):
            t = time.monotonic()
            ws.send(packb(observation(task_id)))
            reply = ws.recv()
            if isinstance(reply, str):
                print(f'step {step}: server error\n{reply}')
                return 1
            action = np.asarray(unpackb(reply)['action'])
            ok = action.shape[-1] == 23 and np.isfinite(action).all()
            print(f'step {step}: action {action.shape} {np.round(action.reshape(-1)[:6], 3)}... {time.monotonic() - t:.1f} s {"OK" if ok else "BAD"}')
            if not ok:
                return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
