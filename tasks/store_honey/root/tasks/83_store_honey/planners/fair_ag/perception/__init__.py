"""Perception of the fair planner of task 83 as modules by meaning (the class Perception is assembled from their mixins in state.py)."""
from .cloud import CAMS, rot2, self_keep
from .state import Perception

__all__ = ['CAMS', 'Perception', 'rot2', 'self_keep']
