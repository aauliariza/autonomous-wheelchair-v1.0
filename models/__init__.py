"""Model wrappers, feature hooks and projections for YOLO26-Depth distillation."""

from .feature_hooks import FeatureExtractor, resolve_depth_feature_layers
from .model_utils import (
    ModelLoadError,
    denormalize_depth_space,
    get_depth_head,
    load_depth_model,
    measure_latency,
    model_complexity,
    select_device,
)
from .projection import ProjectionBank
from .student import StudentDepthModel
from .teacher import TeacherDepthModel

__all__ = [
    "FeatureExtractor",
    "ModelLoadError",
    "ProjectionBank",
    "StudentDepthModel",
    "TeacherDepthModel",
    "denormalize_depth_space",
    "get_depth_head",
    "load_depth_model",
    "measure_latency",
    "model_complexity",
    "resolve_depth_feature_layers",
    "select_device",
]
