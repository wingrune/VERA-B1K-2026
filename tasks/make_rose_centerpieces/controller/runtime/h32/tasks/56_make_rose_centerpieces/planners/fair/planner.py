"""Observation/action adapter; no changes to frozen control decisions."""
import os
from pathlib import Path

import numpy as np

from core.env.native_iface import spec
from .source import SourceSession
from .search_time import VARIANT


def controller_observation(obs):
    """Batch NumPy inputs if needed; retain tensor values, pixels and depth exactly."""
    import torch
    shapes = {spec.PROPRIO_KEY: (61,), spec.CAM_POSES_KEY: (21,)}
    for camera in spec.CAMERAS:
        shapes[camera.rgb_key] = (camera.size, camera.size, 4)
        shapes[camera.depth_key] = (camera.size, camera.size)
    extra = set(obs) - set(shapes) - {spec.TASK_ID_KEY}
    if extra or set(shapes) - set(obs):
        raise ValueError(f"Observation keys differ: extra={sorted(extra)}, missing={sorted(set(shapes)-set(obs))}")
    out = {}
    for key, shape in shapes.items():
        value = obs[key]
        tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(value))
        if tuple(tensor.shape) == shape:
            tensor = tensor.unsqueeze(0)
        if tuple(tensor.shape) != (1,) + shape:
            raise ValueError(f"{key}: shape {tuple(tensor.shape)}, expected {(1,)+shape}")
        wanted = torch.uint8 if key.endswith("::rgb") else torch.float32
        if tensor.dtype != wanted:
            raise ValueError(f"{key}: dtype {tensor.dtype}, expected {wanted}")
        if key.endswith("::depth_linear"):
            if torch.isnan(tensor).any() or (tensor <= 0).any():
                raise ValueError(f"{key}: expected positive planar depth or +inf")
        elif wanted == torch.float32 and not torch.isfinite(tensor).all():
            raise ValueError(f"{key}: non-finite values")
        out[key] = tensor
    # Task IDs are harness metadata. The controller receives only onboard inputs.
    return out


class Planner:
    def __init__(self, source_dir=None, output_dir=None):
        source_dir = source_dir or os.environ.get("BIMBO_ROSE_CONTROLLER")
        output_dir = output_dir or os.environ.get("BIMBO_ROSE_OUTPUT")
        if not source_dir or not output_dir:
            raise ValueError("Set BIMBO_ROSE_CONTROLLER and BIMBO_ROSE_OUTPUT in the task runner")
        self.session = SourceSession(Path(source_dir), Path(output_dir))
        self.controller = self.session.create()
        self.last_phase = None
        self.reason = "initialized frozen v36"

    def reset(self, seed=0):
        # Legacy reset is seedless; native evaluator owns RNG initialization.
        # In particular, harness instance IDs must not become controller inputs.
        with self.session.scope():
            self.controller.reset()
        self.last_phase = None
        self.reason = "reset frozen v36"

    def act(self, obs):
        allowed = controller_observation(obs)
        with self.session.scope():
            action = self.controller.forward(allowed)
        result = action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
        if result.shape != (1, 23) or result.dtype != np.float32 or not np.isfinite(result).all():
            raise ValueError(f"Controller returned invalid action {result.shape} {result.dtype}")
        phase = str(self.controller.phase)
        if phase != self.last_phase:
            self.reason = f"frozen v36 transition {self.last_phase or 'reset'} -> {phase}"
            self.last_phase = phase
        return result

    def log(self):
        c = self.controller
        return {"phase": str(c.phase), "reason": self.reason, "age": int(c.age),
                "controller_variant": VARIANT,
                "perception_backend": self.session.compute_backend,
                "step": int(c.step), "stage": str(getattr(c, "stage", "")),
                "active_child": str(c.child), "active_parent": str(c.parent),
                "issued_rose_releases": len(getattr(c, "rose_releases", [])),
                "note": "issued releases are controller estimates, not native goal satisfaction"}


def entrypoint():
    return Planner()
