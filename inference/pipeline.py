"""End-to-end perception -> planning pipeline (spec sections BE, BF, AE).

.. code-block:: text

    Frame -> preprocessing -> [ YOLO26n detection | YOLO26n-Depth student ]
          -> obstacle fusion -> distance estimation -> 60% global ROI
          -> sector mapping -> safety threshold -> free-path selection
          -> majority vote (N=3) -> navigation command

ARCHITECTURAL BOUNDARY (spec section AE): this class ends at a COMMAND. It never
touches a motor. The motor controller is a separate component with its own
independent safety layer.

PERFORMANCE (spec section BF): models are loaded once in ``__init__`` and reused;
inference runs under ``torch.inference_mode()``; the sector map and estimators are
allocated once and reset per frame rather than rebuilt; and tensor->NumPy
conversions happen once per frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from navigation.free_path import FreePathSelector, NavigationCommand
from navigation.hysteresis import MajorityVoteHysteresis
from navigation.obstacle_fusion import Obstacle, ObstacleFusion
from navigation.roi import compute_global_roi
from navigation.safety import SafetyMonitor
from navigation.sectors import SectorMap
from utils.logger import get_logger

LOG = get_logger("pipeline")


@dataclass
class PipelineResult:
    """Everything one frame produced.

    Attributes:
        frame_id (int): Monotonic counter.
        command (NavigationCommand): Final command after hysteresis and safety.
        raw_command (NavigationCommand): Pre-hysteresis planner output.
        obstacles (list[Obstacle]): Detected obstacles with fused depth.
        depth_map (np.ndarray | None): Metric depth at the frame's resolution.
        occupancy (dict): ``{sector: blocked}``.
        latency_ms (dict): Per-stage timings.
        safety (dict): Safety monitor report.
        decision_reason (str): Why this command was chosen.
    """

    frame_id: int
    command: NavigationCommand
    raw_command: NavigationCommand
    obstacles: list[Obstacle] = field(default_factory=list)
    depth_map: np.ndarray | None = None
    occupancy: dict[str, bool] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    decision_reason: str = ""

    @property
    def total_latency_ms(self) -> float:
        """Total pipeline latency for this frame."""
        return self.latency_ms.get("total", 0.0)

    @property
    def fps(self) -> float:
        """Instantaneous frame rate implied by this frame's latency."""
        t = self.total_latency_ms
        return 1000.0 / t if t > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serializable summary (excludes the depth map)."""
        return {
            "frame_id": self.frame_id,
            "command": str(self.command),
            "raw_command": str(self.raw_command),
            "num_obstacles": len(self.obstacles),
            "occupancy": self.occupancy,
            "latency_ms": self.latency_ms,
            "safety": self.safety,
            "decision_reason": self.decision_reason,
            "obstacles": [o.to_dict() for o in self.obstacles],
        }


class NavigationPipeline:
    """Loads the models once and processes frames.

    Args:
        nav_config (dict): ``configs/navigation.yaml`` contents.
        depth_weights (str | Path, optional): Overrides ``depth.weights``.
        detector_weights (str | Path, optional): Overrides ``detection.weights``.
        camera_config (str | Path, optional): Intrinsics for Euclidean distance.
        device (str, optional): ``0`` | ``cpu`` | ``cuda:0``.
    """

    def __init__(
        self,
        nav_config: dict[str, Any],
        depth_weights: str | Path | None = None,
        detector_weights: str | Path | None = None,
        camera_config: str | Path | None = None,
        device: str | None = None,
    ):
        from ultralytics import YOLO

        from models.model_utils import select_device

        self.config = nav_config
        self.device = select_device(device if device is not None else nav_config.get("device"))

        det_cfg = nav_config.get("detection", {}) or {}
        dep_cfg = nav_config.get("depth", {}) or {}

        self.imgsz = int(det_cfg.get("imgsz", 640))
        self.conf = float(det_cfg.get("conf", 0.25))
        self.iou = float(det_cfg.get("iou", 0.45))
        self.max_det = int(det_cfg.get("max_det", 100))
        self.class_agnostic = bool(det_cfg.get("class_agnostic", True))
        self.detection_label = str(det_cfg.get("label", "obstacle"))

        # --- intrinsics (optional; Euclidean distance requires them) ---
        self.intrinsics = None
        if camera_config is not None:
            from calibration.intrinsics import CameraIntrinsics, IntrinsicsError

            need_metric = nav_config.get("safety", {}).get("distance_mode") == "euclidean"
            try:
                self.intrinsics = CameraIntrinsics.from_yaml(camera_config, require_calibrated=need_metric)
                LOG.info("Camera intrinsics: %s", self.intrinsics)
            except IntrinsicsError as e:
                if need_metric:
                    raise
                LOG.warning("Camera intrinsics unavailable (%s); Euclidean distance will not be reported.", e)

        # --- models, loaded ONCE (spec section BF) ---
        det_w = str(detector_weights or det_cfg.get("weights", "yolo26n.pt"))
        dep_w = str(depth_weights or dep_cfg.get("weights", "yolo26n-depth.pt"))

        LOG.info("Loading detector: %s", det_w)
        self.detector = YOLO(det_w)

        # Spec section A: every detection becomes a single 'obstacle' class. With
        # class_agnostic=True (the default) a COCO-80 checkpoint is usable with no
        # retraining, because ObstacleFusion discards the predicted class entirely.
        # Warn when a multi-class model is used WITHOUT that flag, since the
        # configuration then implies an nc=1 obstacle model that this is not.
        num_classes = len(getattr(self.detector.model, "names", {}) or {})
        if num_classes > 1 and not self.class_agnostic:
            LOG.warning(
                "Detector '%s' has %d classes but detection.class_agnostic is false. "
                "Class identities are still discarded (every detection is labelled '%s'); "
                "set class_agnostic: true, or fine-tune an nc=1 model with "
                "training/train_detection.py.",
                det_w,
                num_classes,
                self.detection_label,
            )
        elif num_classes > 1:
            LOG.info("Detector has %d classes; all are collapsed to '%s'.", num_classes, self.detection_label)

        dep_path = Path(dep_w)
        if not dep_path.exists() and not dep_path.name.startswith("yolo"):
            fallback = dep_cfg.get("fallback_weights", "yolo26n-depth.pt")
            LOG.warning("Depth weights '%s' not found; falling back to '%s'.", dep_w, fallback)
            dep_w = str(fallback)
        LOG.info("Loading depth model: %s", dep_w)
        self.depth_model = YOLO(dep_w)

        # --- navigation components, allocated once ---
        self.fusion = ObstacleFusion(nav_config, intrinsics=self.intrinsics)
        self.selector = FreePathSelector(nav_config)
        self.hysteresis = MajorityVoteHysteresis.from_config(nav_config)
        self.safety_monitor = SafetyMonitor(nav_config)
        self.safety_distance_m = float(nav_config.get("safety", {}).get("safety_distance_m", 1.0))

        self._sector_map: SectorMap | None = None
        self._frame_shape: tuple[int, int] | None = None
        self.frame_id = 0

    def _sectors_for(self, height: int, width: int) -> SectorMap:
        """Return the sector map, rebuilding it only when the frame size changes."""
        if self._sector_map is None or self._frame_shape != (height, width):
            roi = compute_global_roi(
                width,
                height,
                width_ratio=float(self.config.get("roi", {}).get("width_ratio", 0.60)),
                x_center=float(self.config.get("roi", {}).get("x_center", 0.50)),
                height_ratio=float(self.config.get("roi", {}).get("height_ratio", 1.0)),
                y_center=float(self.config.get("roi", {}).get("y_center", 0.50)),
            )
            self._sector_map = SectorMap.from_config(self.config, roi)
            self._frame_shape = (height, width)
            LOG.info("Navigation ROI %s with sectors %s", roi.as_tuple(), [s.name for s in self._sector_map])
        return self._sector_map

    def process_frame(self, frame: np.ndarray, timestamp: float | None = None) -> PipelineResult:
        """Run the full pipeline on one BGR frame.

        Never raises for a perception failure: an exception in detection or depth
        is converted into a STOP through the safety monitor, because a crashed
        pipeline on a moving wheelchair is the worst possible outcome
        (spec section AD).
        """
        import torch

        t_start = time.perf_counter()
        ts = time.time() if timestamp is None else timestamp
        self.frame_id += 1
        latency: dict[str, float] = {}

        h, w = frame.shape[:2]
        sector_map = self._sectors_for(h, w)
        sector_map.reset()

        boxes = np.zeros((0, 4))
        confs = np.zeros((0,))
        depth_map: np.ndarray | None = None
        model_exception: Exception | None = None
        detection_ok = True
        depth_valid = True

        try:
            with torch.inference_mode():
                t = time.perf_counter()
                det = self.detector.predict(
                    frame,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    max_det=self.max_det,
                    device=str(self.device),
                    verbose=False,
                )[0]
                latency["detection"] = (time.perf_counter() - t) * 1000

                if det.boxes is not None and len(det.boxes):
                    boxes = det.boxes.xyxy.cpu().numpy()
                    confs = det.boxes.conf.cpu().numpy()
                else:
                    detection_ok = True  # zero detections is valid, not a failure

                t = time.perf_counter()
                dep = self.depth_model.predict(frame, imgsz=self.imgsz, device=str(self.device), verbose=False)[0]
                latency["depth"] = (time.perf_counter() - t) * 1000

                raw = dep.depth.data
                depth_map = np.squeeze(raw.detach().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw))

            if depth_map is None or not np.isfinite(depth_map).any() or (depth_map > 0).sum() == 0:
                depth_valid = False
        except (RuntimeError, ValueError, AttributeError, IndexError) as e:
            # Perception failed: record it and let the safety layer force STOP.
            LOG.error("Inference failure on frame %d (%s): %s", self.frame_id, type(e).__name__, e)
            model_exception = e
            detection_ok = False
            depth_valid = False

        # --- fusion + distance ---
        obstacles: list[Obstacle] = []
        if depth_valid and depth_map is not None:
            t = time.perf_counter()
            obstacles = self.fusion.fuse(boxes, confs, depth_map, sector_map, image_size=(h, w))
            latency["fusion"] = (time.perf_counter() - t) * 1000

        # --- free path ---
        t = time.perf_counter()
        decision = self.selector.select(sector_map)
        latency["free_path"] = (time.perf_counter() - t) * 1000

        # If depth is unusable the planner's view is meaningless; force STOP
        # before it can produce a confident FORWARD from an empty sector map.
        raw_command = decision.command if depth_valid else NavigationCommand.STOP

        elapsed_s = time.perf_counter() - t_start
        report = self.safety_monitor.check(
            frame_age_s=max(0.0, time.time() - ts),
            inference_latency_s=elapsed_s,
            depth_valid=depth_valid,
            detection_ok=detection_ok,
            sector_distances={s.name: s.min_distance_m for s in sector_map if s.min_distance_m is not None},
            model_exception=model_exception,
            camera_ok=frame is not None and frame.size > 0,
            frame_timestamp=ts,
        )

        # Hysteresis smooths the planner; the safety monitor then has the final
        # word, so a fault can never be voted away by two good frames.
        smoothed = self.hysteresis.update(raw_command)
        command = self.safety_monitor.apply(smoothed, report)

        latency["total"] = (time.perf_counter() - t_start) * 1000

        return PipelineResult(
            frame_id=self.frame_id,
            command=command,
            raw_command=raw_command,
            obstacles=obstacles,
            depth_map=depth_map,
            occupancy=sector_map.occupancy(),
            latency_ms=latency,
            safety=report.to_dict(),
            decision_reason=decision.reason,
        )

    @property
    def sector_map(self) -> SectorMap | None:
        """The current sector map, for overlay rendering."""
        return self._sector_map

    def reset(self) -> None:
        """Clear temporal state between independent sequences."""
        self.hysteresis.reset()
        self.safety_monitor.reset()
        self.frame_id = 0
