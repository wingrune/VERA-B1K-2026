"""GPU-IK worker of the task 69 fair planner: runs in the cuRobo venv (the simulator venv has no cuRobo), one per episode process, started by
fair/ik.py. Builds both arm solvers once (skills/gpu-ik warmup) and the arm-only solvers of ik_arm_worker.py, prints READY, then answers one
JSON request per stdin line with one JSON line on stdout:

  {"op": "scan", "side", "T": K x 16 (world gripper-link goals, row major), "stands": N x 3, "base_z", "point": 3, "tol": [m, rad]}
      -> {"mask": K x N, "q": K x N x 11}   reachable_stands for every goal, float32 hits re-checked in float64 (skills/gpu-ik _verify64)
  {"op": "near", "side", "T": 16 (robot-frame goal), "q_seed": 11, "point": 3, "tol": [m, rad]}
      -> {"mask": bool, "q": 11}            solve_near: branch-continuous solve seeded with the current trunk + arm joints
  {"op": "arm_near" | "arm_near_loose", "side", "T": 16 (gripper-link goal in the arm mount frame), "q_seed": 7}
      -> {"mask": bool, "q": 7}             the 7 arm joints with the trunk frozen (ik_arm_worker.py); the client re-checks it in float64

Any other stdout line (warp prints its banner there) is not JSON and is skipped by the client. Errors answer {"error": text}.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'skills' / 'gpu-ik'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_ik  # noqa: E402
import importlib.util  # noqa: E402
import ik_arm_worker as arm  # noqa: E402

_spec = importlib.util.spec_from_file_location('ik_arm_worker_loose', Path(__file__).resolve().parent / 'ik_arm_worker.py')
arm_loose = importlib.util.module_from_spec(_spec)      # the same solver module with a looser tolerance: the pushes of the press (orientation is not critical)
_spec.loader.exec_module(arm_loose)

TOL = (0.005, 0.05)   # metres, radians (the near solves; the oracle's scans)
SCAN_TOL = (0.010, 0.10)   # the fair planner's stand scans (requests carry their own tol, both solvers are built at warm-up)

arm.TOL = (0.008, 0.10)   # the arm-only press solves of the fair planner (the oracle's own worker keeps 0.005 / 0.05)
arm_loose.TOL = (0.015, 0.30)
t0 = time.time()
gpu_ik.warmup(tol=TOL)
gpu_ik.warmup(tol=SCAN_TOL)
_T = np.eye(4)
_T[:3, 3] = (0.3, 0.0, -0.3)
for _side in ('left', 'right'):
    arm.near(_side, _T, arm.K.rest(_side)[4:])
    arm_loose.near(_side, _T, arm.K.rest(_side)[4:])
print(json.dumps(dict(ready=True, seconds=round(time.time() - t0, 1))), flush=True)
for line in sys.stdin:
    try:
        req = json.loads(line)
        t = time.perf_counter()
        side = req['side']
        if req['op'] == 'quit':
            break
        if req['op'] in ('arm_near', 'arm_near_loose'):
            mask, q = (arm_loose if req['op'] == 'arm_near_loose' else arm).near(side, np.asarray(req['T'], float).reshape(4, 4), np.asarray(req['q_seed'], float))
            out = dict(mask=bool(mask), q=np.asarray(q, float).tolist())
        else:
            point = np.asarray(req['point'], float)
            tol = tuple(req.get('tol', TOL))
            if req['op'] == 'scan':
                stands = np.asarray(req['stands'], float).reshape(-1, 3)
                base_z = float(req['base_z'])
                masks, qs = [], []
                for T in np.asarray(req['T'], float).reshape(-1, 4, 4):
                    mask, q = gpu_ik.reachable_stands_gpu(T, stands, side, tol, point=point, base_z=base_z)
                    mask = gpu_ik._verify64(T, stands, side, mask, q, tol, point, base_z)
                    masks.append(mask.astype(bool).tolist())
                    qs.append(np.round(q, 5).tolist())
                out = dict(mask=masks, q=qs)
            elif req['op'] == 'near':
                T = np.asarray(req['T'], float).reshape(4, 4)
                mask, q = gpu_ik.solve_near(T, np.zeros((1, 3)), side, np.asarray(req['q_seed'], float), tol=tol, point=point, backend='gpu')
                out = dict(mask=bool(mask[0]), q=np.asarray(q[0], float).tolist())
            else:
                raise ValueError(f"unknown op {req['op']!r}")
        out['seconds'] = round(time.perf_counter() - t, 4)
        print(json.dumps(out), flush=True)
    except Exception:  # noqa: BLE001 - the worker keeps serving, the client decides
        print(json.dumps(dict(error=traceback.format_exc()[-600:])), flush=True)
