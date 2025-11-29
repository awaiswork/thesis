"""Utility functions for object detection evaluation."""

from .device import get_device
from .metrics import extract_metrics, format_results
from .visualization import plot_comparison

__all__ = ["get_device", "extract_metrics", "format_results", "plot_comparison"]


