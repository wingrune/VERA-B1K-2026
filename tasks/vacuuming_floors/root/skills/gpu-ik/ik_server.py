"""cuRobo process for gpu_ik backend='server': run with the cuRobo venv python, one solver per process, requests over stdin/stdout.

    <cuRobo python> ik_server.py          (started by gpu_ik itself when BIMBO_CUROBO_PY is set; not by hand)

One JSON per line. Replies start with `IKR ` (library noise on stdout is ignored by the client). Request {"id", "op", "args"} with op
'warmup' | 'reachable_stands' | 'solve_near' and the keyword arguments of that gpu_ik function (arrays as nested lists) -> {"id", "ms", ...}:
warmup -> {}, the others -> {"mask": [...], "q": [[11]...]}; any exception -> {"id", "error"}. {"op": "quit"} ends the process.
JSON floats round-trip exactly, so the results equal an in-process call.
BIMBO_GPU_IK_SERVER_BACKEND=cpu (tests only) solves with the CPU solver, so the protocol is testable without a GPU.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.pop('BIMBO_CUROBO_PY', None)   # this process solves in-process; never spawn a server of its own
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import gpu_ik  # noqa: E402

MARK = 'IKR '
BACKEND = os.environ.get('BIMBO_GPU_IK_SERVER_BACKEND', 'gpu')


def _send(obj) -> None:
    sys.stdout.write(MARK + json.dumps(obj) + '\n')
    sys.stdout.flush()


def _call(op: str, a: dict) -> dict:
    if op == 'warmup':
        if BACKEND == 'gpu':
            gpu_ik.warmup(tuple(a.get('arms', ('left', 'right'))), tuple(a.get('tol', gpu_ik.DEFAULT_TOL)))
        return {}
    fn = dict(reachable_stands=gpu_ik.reachable_stands, solve_near=gpu_ik.solve_near)[op]
    for k in ('target_pose_world', 'target_world', 'stands', 'q_seed', 'point'):
        if a.get(k) is not None:
            a[k] = np.asarray(a[k], float)
    if 'tol' in a:
        a['tol'] = tuple(a['tol'])
    mask, q = fn(**a, backend=BACKEND)
    return dict(mask=np.asarray(mask, bool).tolist(), q=np.asarray(q, float).tolist())


def main() -> int:
    if BACKEND not in ('gpu', 'cpu'):
        _send(dict(id=None, error=f'BIMBO_GPU_IK_SERVER_BACKEND={BACKEND}'))
        return 1
    if BACKEND == 'gpu' and not gpu_ik.gpu_available():
        _send(dict(id=None, error='no CUDA/cuRobo in the server python ' + sys.executable))
        return 1
    for line in sys.stdin:
        req = json.loads(line)
        if req.get('op') == 'quit':
            break
        t = time.perf_counter()
        try:
            out = _call(req['op'], dict(req.get('args') or {}))
        except Exception:  # noqa: BLE001
            out = dict(error=traceback.format_exc()[-600:])
        out.update(id=req.get('id'), ms=round(1e3 * (time.perf_counter() - t), 2))
        _send(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
