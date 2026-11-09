"""Policy server of one BIMBO fair planner (tasks/<task>/): RGB-D + proprio in, 23-d R1Pro action out.

usage: python solutions/planner/backend.py --task store_honey --port 8101

The planner runs exactly as in the BIMBO native runner: obs is the evaluator's flattened dict as torch tensors with the env batch
dim (only --num-envs 1 is supported), act(obs) -> (23,) action. Its GPU IK runs in a cuRobo worker subprocess of this same Python
(BIMBO_IK_PYTHON); without a GPU the episode fails rather than falling back to CPU (BIMBO_IK_REQUIRE_GPU=1). The planner reads
the first PYTHONPATH entry as its repository root, so it is set to tasks/<task>/root before the package is imported.

Debugging only: PLANNER_TRACE_DIR=<dir> appends the planner's log() to <dir>/<task>_<pid>.jsonl on every phase change and every
PLANNER_TRACE_EVERY controls (default 25). Unset (the submission), nothing is written.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from server.protocol import serve  # noqa: E402


class PlannerPolicy:
    def __init__(self, task: str) -> None:
        folder = REPO / 'tasks' / task
        spec = json.loads((folder / 'solution.json').read_text())
        root = folder / 'root'
        os.environ['PYTHONPATH'] = os.pathsep.join([str(root), os.environ.get('PYTHONPATH', '')]).rstrip(os.pathsep)
        os.environ.setdefault('BIMBO_IK_PYTHON', sys.executable)
        os.environ.setdefault('BIMBO_IK_REQUIRE_GPU', '1')
        os.environ.update(spec['entry']['env'])
        sys.path.insert(0, str(root))
        import torch
        self.torch = torch
        package = root / spec['entry']['package']
        name = f'b1k_submit_planner_{task}'
        mod_spec = importlib.util.spec_from_file_location(name, package / '__init__.py', submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[name] = module
        mod_spec.loader.exec_module(module)
        self.planner = module.entrypoint()
        self.ready = False
        # BIMBO planners take the env index; planners whose reset() reads a task spec dict set entry.reset_task_spec.
        self.reset_arg = {'task_name': task} if spec['entry'].get('reset_task_spec') else 0
        trace_dir = os.environ.get('PLANNER_TRACE_DIR')
        self.trace = open(Path(trace_dir) / f'{task}_{os.getpid()}.jsonl', 'a') if trace_dir and hasattr(self.planner, 'log') else None
        self.trace_every = int(os.environ.get('PLANNER_TRACE_EVERY', 25))
        self.trace_phase = None
        logging.info('planner %s loaded from %s', task, package)

    def reset(self) -> None:
        self.planner.reset(self.reset_arg)
        self.ready = True

    def act(self, obs: dict) -> np.ndarray:
        batch = np.shape(obs['robot_r1::proprio'])[0]
        if batch != 1:
            raise ValueError(f'planner backends serve one environment, got a batch of {batch}: run the evaluator with --num-envs 1')
        if not self.ready:
            self.reset()
        tensors = {k: self.torch.as_tensor(np.array(v)) for k, v in obs.items()}
        action = np.asarray(self.planner.act(tensors), dtype=np.float32).reshape(1, -1)
        if self.trace is not None:
            self.write_trace()
        return action

    def write_trace(self) -> None:
        phase, step = getattr(self.planner, 'phase', None), getattr(self.planner, 'step', 0)
        if phase == self.trace_phase and step % self.trace_every:
            return
        self.trace_phase = phase
        self.trace.write(json.dumps(self.planner.log(), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
        self.trace.flush()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--port', type=int, required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s planner[%(process)d] %(message)s')
    serve(PlannerPolicy(a.task), a.port)


if __name__ == '__main__':
    main()
