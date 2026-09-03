"""Evaluation: depth metrics, obstacle distance, navigation, latency, ablation."""

from .metrics import DepthEvaluator, compute_depth_metrics

__all__ = ["DepthEvaluator", "compute_depth_metrics"]
