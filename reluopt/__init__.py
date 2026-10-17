"""Convex two-layer ReLU regression by fully corrective atom selection."""

from .model import ReLUModel
from .solver import solve_rcg

__all__ = ["ReLUModel", "solve_rcg"]
