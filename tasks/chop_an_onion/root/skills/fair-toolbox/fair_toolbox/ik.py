"""GPU IK for fair planners. Servo is solve_near seeded with the current joints; stands are a mask only.

There is no CPU backend and no silent fallback. ``skills/gpu-ik`` runs with ``backend='gpu'`` and raises
when cuRobo (or ``BIMBO_CUROBO_PY``) is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_GPU = None


def _gpu():
    """Import skills/gpu-ik (directory name is not a package)."""
    global _GPU
    if _GPU is None:
        path = str(Path(__file__).resolve().parents[2] / "gpu-ik")
        if path not in sys.path:
            sys.path.insert(0, path)
        import gpu_ik
        _GPU = gpu_ik
    return _GPU


def warmup() -> None:
    """Load cuRobo once per process. Raises when the GPU helper is unavailable."""
    _gpu().warmup(wait=True)


def stand_mask(target_pose_world, stands, arm: str, tol=(0.005, 0.05), point=None, base_z: float = 0.0) -> np.ndarray:
    """Boolean mask of stands that reach the target. Not a servo command (the joint branch may differ)."""
    if arm not in ("left", "right"):
        raise ValueError(arm)
    mask, _q = _gpu().reachable_stands(target_pose_world, stands, arm, tol=tol, point=point, base_z=base_z, backend="gpu")
    return np.asarray(mask, bool)


def servo(target_world, stands, arm: str, q_seed, tol=(0.005, 0.05), point=None, base_z: float = 0.0):
    """solve_near from q_seed (current trunk 4 + arm 7). Returns (mask, q). GPU only; raises without cuRobo."""
    if arm not in ("left", "right"):
        raise ValueError(arm)
    if q_seed is None:
        raise ValueError("servo requires q_seed equal to the current joints")
    seed = np.asarray(q_seed, float)
    if seed.shape[-1] != 11:
        raise ValueError("q_seed must be 11 joints (trunk 4 + arm 7)")
    return _gpu().solve_near(target_world, stands, arm, seed, tol=tol, point=point, base_z=base_z, backend="gpu")
