"""Detection-depth fusion (spec section BC).

Combines YOLO26n bounding boxes with the YOLO26n-Depth metric depth map into a
list of ``Obstacle`` records, assigns them to sectors, and marks sectors blocked.

EVERY obstacle carries the label ``obstacle`` (spec section A). No object class
is recognised, reported, or acted upon differently: a chair, a person and a
wardrobe are all simply things not to hit.

Conservative-by-default rules
-----------------------------
- An obstacle whose distance is INVALID blocks its sector when
  ``safety.invalid_depth_blocks`` is true. Unknown distance is treated as
  "possibly too close", never as "clear".
- An obstacle whose fused confidence is below ``min_confidence_score`` also
  blocks, when ``low_confidence_blocks`` is true — a low-confidence obstacle is
  still evidence of something there.
- Obstacles outside the navigation ROI are retained with ``in_roi=False`` for
  visualization but never influence the decision (spec section W).

Tracking (spec section BD) is intentionally NOT required. Association is
frame-wise; ``Obstacle.id`` is a per-frame index. The dataclass carries an
optional ``track_id`` so a tracker such as ByteTrack can be added later without
changing this interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .distance import DistanceEstimator, ObstacleDistance
from .sectors import SectorMap

OBSTACLE_LABEL = "obstacle"


@dataclass
class Obstacle:
    """One detected obstacle fused with depth (spec section BC).

    Attributes:
        id (int): Per-frame index.
        label (str): Always ``"obstacle"``.
        bbox (tuple): ``(x1, y1, x2, y2)`` in image pixels.
        confidence (float): Raw detector confidence.
        depth_m (float | None): Median axial depth in the inner ROI.
        euclidean_distance_m (float | None): Slant range, if intrinsics exist.
        distance_m (float | None): Distance under the configured mode.
        valid_depth_ratio (float): Usable fraction of inner-ROI pixels.
        sector (str | None): Primary (nearest-to-centre) sector.
        sectors (list[str]): Every sector this obstacle occupies.
        blocked (bool): Whether it violates the safety distance.
        confidence_score (float): Fused detection/depth/temporal score.
        in_roi (bool): Whether it intersects the navigation ROI.
        depth_stats (dict): Full statistics from the estimator.
        track_id (int | None): Reserved for a future tracker.
    """

    id: int
    label: str = OBSTACLE_LABEL
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    confidence: float = 0.0
    depth_m: float | None = None
    euclidean_distance_m: float | None = None
    distance_m: float | None = None
    valid_depth_ratio: float = 0.0
    sector: str | None = None
    sectors: list[str] = field(default_factory=list)
    blocked: bool = False
    confidence_score: float = 0.0
    in_roi: bool = True
    depth_stats: dict[str, Any] = field(default_factory=dict)
    track_id: int | None = None
    reason: str = ""

    @property
    def has_valid_distance(self) -> bool:
        """True when a usable distance was measured."""
        return self.distance_m is not None

    @property
    def center_x(self) -> float:
        """Horizontal centre of the bounding box."""
        return (self.bbox[0] + self.bbox[2]) / 2.0

    def to_dict(self) -> dict[str, Any]:
        """Serializable record matching the spec section BC schema."""
        return {
            "id": self.id,
            "label": self.label,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "depth_m": self.depth_m,
            "euclidean_distance_m": self.euclidean_distance_m,
            "distance_m": self.distance_m,
            "valid_depth_ratio": self.valid_depth_ratio,
            "sector": self.sector,
            "sectors": list(self.sectors),
            "blocked": self.blocked,
            "confidence_score": self.confidence_score,
            "in_roi": self.in_roi,
            "track_id": self.track_id,
            "reason": self.reason,
        }


class ObstacleFusion:
    """Fuses detections with depth and computes sector occupancy.

    Args:
        config (dict): Full ``navigation.yaml`` mapping.
        intrinsics (CameraIntrinsics, optional): Enables Euclidean distance.
    """

    def __init__(self, config: dict[str, Any] | None = None, intrinsics: Any | None = None):
        self.config = config or {}
        safety = self.config.get("safety", {}) or {}

        self.safety_distance_m = float(safety.get("safety_distance_m", 1.0))
        self.invalid_depth_blocks = bool(safety.get("invalid_depth_blocks", True))
        self.min_detection_confidence = float(safety.get("min_detection_confidence", 0.25))
        self.low_confidence_blocks = bool(safety.get("low_confidence_blocks", True))
        self.min_confidence_score = float(safety.get("min_confidence_score", 0.20))

        self.estimator = DistanceEstimator(self.config, intrinsics=intrinsics)
        self.intrinsics = intrinsics

    def fuse(
        self,
        boxes: np.ndarray | list,
        confidences: np.ndarray | list,
        depth_map: np.ndarray,
        sector_map: SectorMap,
        image_size: tuple[int, int] | None = None,
        temporal_consistency: float = 1.0,
    ) -> list[Obstacle]:
        """Fuse one frame's detections with its depth map.

        Args:
            boxes: ``(N, 4)`` xyxy boxes in image pixels.
            confidences: ``(N,)`` detector confidences.
            depth_map (np.ndarray): ``(H, W)`` metric depth. May be smaller than
                the image (the head predicts at input/4); boxes are rescaled.
            sector_map (SectorMap): Reset and populated in place.
            image_size (tuple, optional): ``(H, W)`` the boxes were measured in.
            temporal_consistency (float): Multiplier for the confidence score.

        Returns:
            (list[Obstacle]): All obstacles, in-ROI and out.
        """
        sector_map.reset()

        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4) if len(boxes) else np.zeros((0, 4))
        confidences = np.asarray(confidences, dtype=np.float64).ravel() if len(confidences) else np.zeros((0,))

        if len(confidences) != len(boxes):
            raise ValueError(f"Got {len(boxes)} boxes but {len(confidences)} confidences; they must match.")

        depth = np.squeeze(np.asarray(depth_map))
        obstacles: list[Obstacle] = []

        for i, (box, conf) in enumerate(zip(boxes, confidences, strict=True)):
            if conf < self.min_detection_confidence:
                continue

            x1, y1, x2, y2 = (float(v) for v in box)
            ob = Obstacle(id=i, bbox=(x1, y1, x2, y2), confidence=float(conf), label=OBSTACLE_LABEL)

            ob.in_roi = sector_map.is_in_roi(x1, x2)

            dist: ObstacleDistance = self.estimator.estimate(depth, (x1, y1, x2, y2), image_size=image_size)
            ob.depth_stats = dist.to_dict()
            ob.valid_depth_ratio = dist.valid_ratio
            ob.depth_m = dist.depth_median_m
            ob.euclidean_distance_m = dist.euclidean_distance_m
            ob.distance_m = dist.distance_m if dist.valid else None
            if not dist.valid:
                ob.reason = dist.reason

            ob.confidence_score = self.estimator.confidence_score(
                detection_confidence=ob.confidence,
                distance=dist,
                temporal_consistency=temporal_consistency,
            )

            occupied = sector_map.assign_bbox(x1, x2) if ob.in_roi else []
            ob.sectors = [s.name for s in occupied]
            ob.sector = self._primary_sector(ob, occupied, sector_map)

            ob.blocked = self._is_blocking(ob)

            if ob.in_roi:
                for s in occupied:
                    s.obstacle_ids.append(ob.id)
                    if ob.blocked:
                        s.blocked = True
                    if ob.distance_m is not None:
                        s.min_distance_m = (
                            ob.distance_m if s.min_distance_m is None else min(s.min_distance_m, ob.distance_m)
                        )

            obstacles.append(ob)

        return obstacles

    def _primary_sector(self, ob: Obstacle, occupied: list, sector_map: SectorMap) -> str | None:
        """Pick the single representative sector for reporting.

        The sector containing the box centre is preferred; otherwise the first
        occupied sector. Only ``sector`` is a simplification — the full
        ``sectors`` list is what drives blocking.
        """
        if not occupied:
            return None
        centre = sector_map.sector_at_x(ob.center_x)
        if centre is not None and centre in occupied:
            return centre.name
        return occupied[0].name

    def _is_blocking(self, ob: Obstacle) -> bool:
        """Decide whether an obstacle blocks its sectors (spec section AA)."""
        if not ob.in_roi:
            return False

        if ob.distance_m is None:
            # Unknown distance: conservative by default.
            if self.invalid_depth_blocks:
                ob.reason = ob.reason or "distance invalid; treated as blocking (invalid_depth_blocks)"
                return True
            return False

        if ob.distance_m < self.safety_distance_m:
            ob.reason = f"distance {ob.distance_m:.3f}m < safety_distance_m {self.safety_distance_m:.3f}m"
            return True

        if self.low_confidence_blocks and ob.confidence_score < self.min_confidence_score:
            ob.reason = (
                f"confidence score {ob.confidence_score:.3f} < min_confidence_score "
                f"{self.min_confidence_score:.3f}; treated as blocking"
            )
            return True

        return False
