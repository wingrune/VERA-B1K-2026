"""Isolate the frozen controller's legacy absolute imports and runtime settings.

Source files remain byte-identical; a task-local import overlay disables their
search wall-clock cutoffs as explicitly requested by the user. Calls are serialized because Python's import
aliases and the legacy output environment are process-global.
"""
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading

from .search_time import SearchTimeFinder

_LOCK = threading.RLock()
MANIFEST = Path(__file__).resolve().parents[2] / "CONTROLLER_SOURCE.json"


def verify_source(root):
    root = Path(root).resolve()
    manifest = json.loads(MANIFEST.read_text())
    expected = manifest["sha256"]
    actual = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])
        raise ValueError(f"Frozen source differs: missing={missing}, extra={extra}, changed={changed}")
    return root


class SourceSession:
    def __init__(self, root, output):
        self.root = verify_source(root)
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.names = {p.stem for p in self.root.glob("*.py")}
        self.modules = {}
        self.policy = None
        self.search_time_finder = SearchTimeFinder(self.root)
        backend = os.environ.get('BIMBO_ROSE_PERCEPTION_BACKEND', 'cpu')
        if backend not in ('cpu', 'cuda'):
            raise ValueError('Unknown FAIR perception backend: ' + backend)
        self.compute_backend = backend
        self.cuda_finder = None
        if backend == 'cuda':
            from .cuda_overlay import CudaFinder
            self.cuda_finder = CudaFinder(self.root)

    @contextmanager
    def scope(self):
        with _LOCK:
            previous = {name: sys.modules.get(name) for name in self.names}
            old_path = list(sys.path)
            old_meta_path = list(sys.meta_path)
            settings = {"TASK_NAME": "make_rose_centerpieces", "POLICY_OUTPUT": str(self.output)}
            old_env = {name: os.environ.get(name) for name in settings}
            try:
                for name in self.names:
                    sys.modules.pop(name, None)
                sys.modules.update(self.modules)
                sys.path.insert(0, str(self.root))
                sys.meta_path.insert(0, self.search_time_finder)
                if self.cuda_finder is not None:
                    sys.meta_path.insert(0, self.cuda_finder)
                os.environ.update(settings)
                yield
            finally:
                # Includes modules imported lazily during forward(), not just initialization.
                self.modules.update({name: sys.modules[name] for name in self.names if name in sys.modules})
                for name, module in previous.items():
                    sys.modules.pop(name, None)
                    if module is not None:
                        sys.modules[name] = module
                sys.path[:] = old_path
                sys.meta_path[:] = old_meta_path
                for name, value in old_env.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

    def create(self):
        with self.scope():
            self.policy = importlib.import_module("rose_policy").Policy()
            self.policy.set_action_dim(23)
        return self.policy

