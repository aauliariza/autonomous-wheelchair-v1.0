"""Obstacle-aware free-path navigation (spec sections V-AE)."""

from .distance import DistanceEstimator, ObstacleDistance, robust_depth_statistics
from .free_path import FreePathSelector, NavigationCommand
from .hysteresis import MajorityVoteHysteresis
from .obstacle_fusion import Obstacle, ObstacleFusion
from .roi import ROI, compute_bbox_inner_roi, compute_global_roi
from .safety import SafetyMonitor, SafetyState
from .sectors import Sector, SectorMap

__all__ = [
    "ROI",
    "DistanceEstimator",
    "FreePathSelector",
    "MajorityVoteHysteresis",
    "NavigationCommand",
    "Obstacle",
    "ObstacleDistance",
    "ObstacleFusion",
    "SafetyMonitor",
    "SafetyState",
    "Sector",
    "SectorMap",
    "compute_bbox_inner_roi",
    "compute_global_roi",
    "robust_depth_statistics",
]
