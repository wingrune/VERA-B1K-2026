"""GPU-IK worker of the task 55 planners (oracle_ag): runs in the cuRobo venv (the simulator venv has no cuRobo), one per episode process,
started by <planner>/ik.py. Builds both arm solvers once (skills/gpu-ik warmup), prints READY, then answers one JSON request per stdin line
with one JSON line on stdout:

  {"op": "scan", "side", "T": N x 16 (world gripper-link goal per stand, row major), "stands": N x 3, "base_z", "point": 3}
      -> {"mask": N, "q": N x 11}   reachable_stands, float32 hits re-checked in float64 (skills/gpu-ik _verify64)
  {"op": "near", "side", "T": N x 16, "stands": N x 3, "base_z", "q_seed": N x 11, "point": 3}
      -> {"mask": N, "q": N x 11}   solve_near: branch-continuous solve seeded with q_seed (trunk + arm), nearest solution to it

The servo sends one robot-frame goal with the stand (0, 0, 0) and base_z 0. Any other stdout line (warp prints its banner there) is not JSON
and is skipped by the client. Errors answer {"error": text}.
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

TOL = dict(scan=(0.015, 0.25), near=(0.015, 0.12))   # metres, radians: stand scans as the r49 oracle, solve_near as the CPU servo (0.015 / 0.12)

t0 = time.time()
for tol in TOL.values():
    gpu_ik.warmup(tol=tol)
print(json.dumps(dict(ready=True, seconds=round(time.time() - t0, 1))), flush=True)
for line in sys.stdin:
    try:
        req = json.loads(line)
        t = time.perf_counter()
        side, point, base_z = req['side'], np.asarray(req['point'], float), float(req.get('base_z', 0.))
        stands = np.asarray(req['stands'], float).reshape(-1, 3)
        T = np.asarray(req['T'], float).reshape(-1, 4, 4)
        if req['op'] not in TOL:
            raise ValueError(f"unknown op {req['op']!r}")
        tol = TOL[req['op']]
        seed = np.asarray(req['q_seed'], float).reshape(-1, 11) if req['op'] == 'near' else None
        mask, q = gpu_ik.reachable_stands_gpu(T, stands, side, tol, point=point, base_z=base_z, q_seed=seed)
        mask = gpu_ik._verify64(T, stands, side, mask, q, tol, point, base_z)
        out = dict(mask=mask.astype(bool).tolist(), q=np.round(q, 6).tolist(), seconds=round(time.perf_counter() - t, 4))
        print(json.dumps(out), flush=True)
    except Exception:  # noqa: BLE001 - the worker keeps serving, the client decides
        print(json.dumps(dict(error=traceback.format_exc()[-600:])), flush=True)
