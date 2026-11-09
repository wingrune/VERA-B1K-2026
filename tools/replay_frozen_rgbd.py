"""Replay permitted recorded RGB-D only; compare router actions to frozen core, then reset and error recovery.
Not an evaluation rollout or Q measurement. Reference uses an independently supplied frozen core.
"""
import argparse
import hashlib
import tempfile
import json
import os
import resource
import sys
import time
from pathlib import Path
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from websockets.sync.client import connect

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from server.protocol import packb, unpackb


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=('collecting_aluminum_cans', 'organizing_school_stuff', 'organizing_art_supplies', 'composting_waste', 'boxing_books_up_for_storage'), required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--observation', type=Path, required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--observation-origin', required=True, help='Task and episode that produced this frame; disclose cross-task fixtures')
    a = p.parse_args()
    if a.steps < 1:
        p.error('--steps must be positive')
    spec = json.loads((REPO / 'tasks' / a.task / 'solution.json').read_text())
    a.task_id = spec['task_id_2026']
    reference_hashes = {str(f.relative_to(a.reference_root)): hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in sorted(a.reference_root.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
    reference_sha = hashlib.sha256(json.dumps(reference_hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert reference_sha == spec['source']['fingerprint'], 'independent reference fingerprint mismatch'
    os.environ['POLICY_OUTPUT'] = tempfile.mkdtemp(prefix='frozen_reference_')
    import urllib.request
    assert urllib.request.urlopen(f'http://{a.host}:{a.port}/healthz').status == 200
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(a.reference_root))
    from observation_policy import Policy, KEYS
    reference = Policy()
    with np.load(a.observation, allow_pickle=False) as data:
        assert set(KEYS).issubset(data.files), data.files
        obs = {}
        for k in KEYS:
            ndim = 1 if k.endswith(('proprio','cam_rel_poses')) else 2 if k.endswith('depth_linear') else 3
            value = data[k]
            assert value.ndim in (ndim, ndim+1), (k, value.shape)
            obs[k] = value[None] if value.ndim == ndim else value
    obs['task_id'] = np.array([[a.task_id]], np.int64)
    reference_obs = {k: obs[k][0] if spec['entry'].get('unbatch_observations', False) else obs[k] for k in KEYS}
    durations, episodes = [], []
    for episode in range(2):
        reference.reset()
        actions = []
        with connect(f'ws://{a.host}:{a.port}', compression=None, max_size=None, ping_timeout=600) as ws:
            unpackb(ws.recv())
            ws.send(packb({'reset': True}))
            for step in range(a.steps):
                start = time.monotonic()
                ws.send(packb(obs)); reply = ws.recv(timeout=600)
                elapsed = time.monotonic() - start
                assert not isinstance(reply, str), reply
                actual = unpackb(reply)['action']
                expected = np.asarray(reference.action(reference_obs), dtype=np.float32).reshape(1,23)
                assert actual.shape == (1, 23) and actual.dtype == np.float32 and np.isfinite(actual).all()
                np.testing.assert_array_equal(actual, expected)
                durations.append(elapsed);actions.append(actual.tolist())
        episodes.append(actions)
    assert episodes[0] == episodes[1], 'reset leaked episode state'
    # Invalid batch must produce an explicit protocol error; next reset must recover.
    with connect(f'ws://{a.host}:{a.port}', compression=None, max_size=None) as ws:
        unpackb(ws.recv());ws.send(packb({'reset': True}))
        invalid = dict(obs);invalid['robot_r1::proprio'] = np.zeros((2,61), np.float32)
        ws.send(packb(invalid));error = ws.recv(timeout=600)
        assert isinstance(error,str) and 'num-envs 1' in error, error
    with connect(f'ws://{a.host}:{a.port}', compression=None, max_size=None) as ws:
        unpackb(ws.recv());ws.send(packb({'reset': True}));ws.send(packb(obs))
        reply=ws.recv(timeout=600);assert not isinstance(reply,str), reply
        reference.reset();np.testing.assert_array_equal(unpackb(reply)['action'],np.asarray(reference.action(reference_obs), dtype=np.float32).reshape(1,23))
    report = dict(passed=True, task=a.task, task_id=a.task_id, reference_root=str(a.reference_root), reference_sha256=reference_sha, observation=str(a.observation), observation_origin=a.observation_origin, observation_sha256=hashlib.sha256(a.observation.read_bytes()).hexdigest(), observations='8 whitelisted RGB-D/proprio/calibration arrays only',
                  simulator_started=False, episodes=2, requests=len(durations), bitwise_equal=True, reset_equal=True,
                  error_recovery=True, mean_round_trip_s=float(np.mean(durations)), max_round_trip_s=max(durations),
                  fps=len(durations)/sum(durations), client_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Repeated recorded observation; not a live rollout or a complete worst-case latency proof')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
