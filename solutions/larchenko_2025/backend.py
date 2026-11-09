"""Policy server of one Larchenko 2025 checkpoint behind the 2026 adapter.

usage: python solutions/larchenko_2025/backend.py --checkpoint /opt/checkpoints/checkpoint_2 --port 8102

Runs in the VLA venv (JAX, upstream openpi + b1k). One checkpoint per process; the router starts the process the routing table
names for the task and stops it when another solution is needed.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.7')
# b1k.training.config reads <root>/2026-challenge-demos/meta/tasks.jsonl at import (training task lists); the dataset is not
# in the image, so serve the bundled copy of that one file.
os.environ.setdefault('OPENPI_BEHAVIOR_DATA_ROOT', str(HERE / 'data'))

from server.protocol import serve  # noqa: E402
from adapter_2026 import build  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--num-steps', type=int, default=20, help='flow-matching denoising steps (2025 submission: 20)')
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s vla[%(process)d] %(message)s')
    if not (Path(a.checkpoint) / 'params').is_dir():
        raise SystemExit(f'{a.checkpoint} is not an openpi checkpoint (no params/)')
    serve(build(a.checkpoint, a.num_steps), a.port)


if __name__ == '__main__':
    main()
