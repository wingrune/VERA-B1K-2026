"""Frozen helpers for a fair planner: GPU IK, assisted-grasp facts, object templates, an offline map.

Import as ``fair_toolbox`` with ``skills/fair-toolbox`` on ``PYTHONPATH``. The package is not mutable
during evolution. Servo IK is ``ik.servo`` (cuRobo ``solve_near``, current joints, GPU only).
"""
from . import grasp, ik, offline_map, templates

__all__ = ["grasp", "ik", "offline_map", "templates"]
