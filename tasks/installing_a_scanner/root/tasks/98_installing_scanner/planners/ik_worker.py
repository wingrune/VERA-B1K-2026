"""GPU-IK worker of the task 98 planners (oracle_ag, fair_ag): runs in the cuRobo venv (the simulator venv has no cuRobo), one per episode
process, started by <planner>/ik.py. Builds both arm solvers once (skills/gpu-ik warmup), prints READY, then answers one JSON request per
stdin line with one JSON line on stdout:

  {"op": "scan", "side", "T": K x 16 (world gripper-link goals, row major), "stands": N x 3, "base_z", "point": 3}
      -> {"mask": K x N, "q": K x N x 11}   reachable_stands for every goal, float32 hits re-checked in float64 (skills/gpu-ik _verify64)
  {"op": "near", "side", "T": 16 (robot-frame goal), "q_seed": 11, "point": 3}
      -> {"mask": bool, "q": 11}            solve_near: branch-continuous solve seeded with the current trunk + arm joints

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
import gpu_ik  # noqa: E402

TOL = (0.010, 0.10)   # metres, radians; the servo of r1pro_kinematics accepts 0.015 / 0.12

t0 = time.time()
gpu_ik.warmup(tol=TOL)
print(json.dumps(dict(ready=True, seconds=round(time.time() - t0, 1))), flush=True)
for line in sys.stdin:
    try:
        req = json.loads(line)
        t = time.perf_counter()
        side, point = req['side'], np.asarray(req['point'], float)
        if req['op'] == 'scan':
            stands = np.asarray(req['stands'], float).reshape(-1, 3)
            base_z = float(req['base_z'])
            masks, qs = [], []
            for T in np.asarray(req['T'], float).reshape(-1, 4, 4):
                mask, q = gpu_ik.reachable_stands_gpu(T, stands, side, TOL, point=point, base_z=base_z)
                mask = gpu_ik._verify64(T, stands, side, mask, q, TOL, point, base_z)
                masks.append(mask.astype(bool).tolist())
                qs.append(np.round(q, 5).tolist())
            out = dict(mask=masks, q=qs)
        elif req['op'] == 'near':
            T = np.asarray(req['T'], float).reshape(4, 4)
            mask, q = gpu_ik.solve_near(T, np.zeros((1, 3)), side, np.asarray(req['q_seed'], float), tol=TOL, point=point, backend='gpu')
            out = dict(mask=bool(mask[0]), q=np.asarray(q[0], float).tolist())
        else:
            raise ValueError(f"unknown op {req['op']!r}")
        out['seconds'] = round(time.perf_counter() - t, 4)
        print(json.dumps(out), flush=True)
    except Exception:  # noqa: BLE001 - the worker keeps serving, the client decides
        print(json.dumps(dict(error=traceback.format_exc()[-600:])), flush=True)
