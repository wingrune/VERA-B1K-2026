"""Assisted grasp is applied by the environment (skills/assisted-grasp). The planner only commands the gripper.

Close (command toward -1) while the object sits between the fingers: both fingers touch and the rays between
the grasp points hit the body for 0.3 s. The hold is a PhysX joint. Open (command +1) releases it at once.
Fair observations do not include that joint.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

_AG = None


def _ag():
    global _AG
    if _AG is None:
        path = Path(__file__).resolve().parents[2] / "assisted-grasp" / "assisted_grasp.py"
        spec = importlib.util.spec_from_file_location("bimbo_skill_assisted_grasp", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # dataclass looks the module up while the file executes
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(spec.name, None)
            raise
        _AG = mod
    return _AG


def facts() -> Dict[str, Any]:
    """Window 0.3 s, two fingers, rays between the fingers, PhysX joint; the environment applies it."""
    ag = _ag()
    return {
        "window_s": ag.GRASP_WINDOW_S,
        "min_fingers": ag.MIN_FINGERS,
        "release_s": ag.RELEASE_WINDOW_S,
        "mass_kg": ag.MASS_THRESHOLD_KG,
        "applying": "gripper command < +1",
        "hold": "PhysX joint created by the environment",
    }


def close_command() -> float:
    """Gripper command that keeps the grasp applying (below +1)."""
    return -1.0


def open_command() -> float:
    """Gripper command that releases the hold immediately."""
    return 1.0


def hold_state(*_args, **_kwargs):
    """Not implemented: fair has no grasp-hold query; assisted grasp is automatic in the environment."""
    raise NotImplementedError(
        "Fair sees RGB-D, proprio and camera calibration only. Assisted grasp (skills/assisted-grasp) "
        "creates the PhysX hold inside the environment. Missing: a hold signal from proprio or RGB-D. "
        "Command the gripper with close_command() and open_command(); do not reimplement the grasp."
    )
