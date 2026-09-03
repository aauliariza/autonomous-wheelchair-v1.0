"""Robust obstacle distance estimation (spec sections V, X, Y).

Reading a single pixel from a monocular depth map is not a distance measurement:
depth maps are noisiest exactly at object boundaries, and one bad pixel can move
a reported distance by metres. This module therefore reduces the inner-60%
bounding-box ROI to a robust statistic.

Pipeline per obstacle
---------------------
1. Crop the inner 60% of the box (``compute_bbox_inner_roi``).
2. Drop invalid depths: ``0``, ``NaN``, ``Inf``, ``<= 0``, and values outside the
   sensor's trustworthy range.
3. Optionally clip the 5th/95th percentiles to shed residual outliers.
4. Reduce with the MEDIAN (default), which tolerates up to 50% contamination —
   important because a box always contains some background.
5. Report dispersion via the MAD, rescaled by 1.4826 so it is comparable to a
   standard deviation for normally distributed data.
6. If the valid fraction falls below ``min_valid_ratio``, mark the distance
   INVALID rather than returning a number computed from a handful of pixels.

An INVALID distance is not "no obstacle". The safety layer treats it as blocking
(spec section X: "obstacle harus diperlakukan konservatif").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .roi import ROI, compute_bbox_inner_roi

# Scale factor making MAD a consistent estimator of sigma under normality.
MAD_TO_SIGMA = 1.4826


@dataclass
class ObstacleDistance:
    """Result of estimating one obstacle's distance.

    Attributes:
        valid (bool): False when too few pixels were usable.
        distance_m (float | None): Distance under the configured mode.
        depth_median_m (float | None): Median axial depth.
        depth_min_m, depth_max_m (float | None): Range of valid depths.
        depth_std_m (float | None): Robust spread (MAD * 1.4826).
        euclidean_distance_m (float | None): Slant range, if intrinsics given.
        valid_ratio (float): Fraction of ROI pixels that were usable.
        num_valid (int): Count of usable pixels.
        num_total (int): ROI pixel count.
        roi (tuple | None): The inner ROI actually sampled.
        reason (str): Why the estimate is invalid, when it is.
    """

    valid: bool
    distance_m: float | None = None
    depth_median_m: float | None = None
    depth_min_m: float | None = None
    depth_max_m: float | None = None
    depth_std_m: float | None = None
    euclidean_distance_m: float | None = None
    valid_ratio: float = 0.0
    num_valid: int = 0
    num_total: int = 0
    roi: tuple[int, int, int, int] | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serializable summary."""
        return {
            "valid": self.valid,
            "distance_m": self.distance_m,
            "depth_median_m": self.depth_median_m,
            "depth_min_m": self.depth_min_m,
            "depth_max_m": self.depth_max_m,
            "depth_std_m": self.depth_std_m,
            "euclidean_distance_m": self.euclidean_distance_m,
            "valid_ratio": self.valid_ratio,
            "num_valid": self.num_valid,
            "num_total": self.num_total,
            "roi": self.roi,
            "reason": self.reason,
        }


def robust_depth_statistics(
    depth_patch: np.ndarray,
    min_depth_m: float = 0.1,
    max_depth_m: float = 10.0,
    clip_lower_percentile: float = 5.0,
    clip_upper_percentile: float = 95.0,
    statistic: str = "median",
    percentile: float = 25.0,
) -> dict[str, Any]:
    """Reduce a depth patch to robust statistics (spec section X).

    Args:
        depth_patch (np.ndarray): Depth values in metres; any shape.
        min_depth_m, max_depth_m (float): Trustworthy sensor range.
        clip_lower_percentile, clip_upper_percentile (float): Percentile band
            kept before reduction. Set both to 0/100 to disable clipping.
        statistic (str): ``median`` | ``mean`` | ``percentile``.
        percentile (float): Used when ``statistic == "percentile"``. A low value
            (e.g. 25) biases toward the NEAR surface, which is the conservative
            choice for obstacle avoidance.

    Returns:
        (dict): ``value``, ``median``, ``min``, ``max``, ``std``, ``num_valid``,
            ``num_total``, ``valid_ratio``.
    """
    flat = np.asarray(depth_patch, dtype=np.float64).ravel()
    total = int(flat.size)

    if total == 0:
        return {
            "value": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
            "num_valid": 0,
            "num_total": 0,
            "valid_ratio": 0.0,
        }

    # Reject every invalid case named in the spec in one pass.
    valid = np.isfinite(flat) & (flat > 0) & (flat >= min_depth_m) & (flat <= max_depth_m)
    vals = flat[valid]
    num_valid = int(vals.size)

    if num_valid == 0:
        return {
            "value": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
            "num_valid": 0,
            "num_total": total,
            "valid_ratio": 0.0,
        }

    # Percentile clipping needs enough samples for the percentiles to mean
    # anything; below 10 values it would just discard scarce real data.
    if num_valid >= 10 and (clip_lower_percentile > 0 or clip_upper_percentile < 100):
        lo = np.percentile(vals, clip_lower_percentile)
        hi = np.percentile(vals, clip_upper_percentile)
        clipped = vals[(vals >= lo) & (vals <= hi)]
        if clipped.size > 0:
            vals = clipped

    median = float(np.median(vals))
    if statistic == "mean":
        value = float(np.mean(vals))
    elif statistic == "percentile":
        value = float(np.percentile(vals, percentile))
    elif statistic == "median":
        value = median
    else:
        raise ValueError(f"statistic must be 'median', 'mean' or 'percentile', got '{statistic}'.")

    # MAD rather than std: a bimodal patch (obstacle + background) would inflate
    # the standard deviation and understate confidence in the median.
    mad = float(np.median(np.abs(vals - median)))

    return {
        "value": value,
        "median": median,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "std": mad * MAD_TO_SIGMA,
        "num_valid": num_valid,
        "num_total": total,
        "valid_ratio": num_valid / float(total),
    }


class DistanceEstimator:
    """Estimates obstacle distances from a dense depth map.

    Args:
        config (dict): The ``navigation.yaml`` mapping (or its relevant subset).
        intrinsics (CameraIntrinsics, optional): Enables Euclidean output. Without
            it only axial depth is reported — never a fabricated slant range.

    Examples:
        >>> import numpy as np
        >>> est = DistanceEstimator({})
        >>> d = np.full((100, 100), 2.0)
        >>> r = est.estimate(d, (10, 10, 90, 90))
        >>> r.valid, round(r.distance_m, 3)
        (True, 2.0)
    """

    def __init__(self, config: dict[str, Any] | None = None, intrinsics: Any | None = None):
        cfg = config or {}
        stats = cfg.get("depth_stats", {}) or {}
        bbox = cfg.get("bbox_roi", {}) or {}
        safety = cfg.get("safety", {}) or {}

        self.statistic = stats.get("statistic", "median")
        self.percentile = float(stats.get("percentile", 25.0))
        self.min_depth_m = float(stats.get("min_depth_m", 0.1))
        self.max_depth_m = float(stats.get("max_depth_m", 10.0))
        self.clip_lower = float(stats.get("clip_lower_percentile", 5.0))
        self.clip_upper = float(stats.get("clip_upper_percentile", 95.0))
        self.min_valid_ratio = float(stats.get("min_valid_ratio", 0.30))

        self.inner_ratio = float(bbox.get("inner_ratio", 0.60))
        self.min_size_px = int(bbox.get("min_size_px", 4))

        self.distance_mode = safety.get("distance_mode", "axial")
        if self.distance_mode not in ("axial", "euclidean"):
            raise ValueError(f"safety.distance_mode must be 'axial' or 'euclidean', got '{self.distance_mode}'.")

        self.intrinsics = intrinsics

    def estimate(
        self,
        depth_map: np.ndarray,
        bbox: tuple[float, float, float, float],
        image_size: tuple[int, int] | None = None,
    ) -> ObstacleDistance:
        """Estimate one obstacle's distance from its bounding box.

        Args:
            depth_map (np.ndarray): ``(H, W)`` metric depth in metres.
            bbox (tuple): ``(x1, y1, x2, y2)`` in the depth map's pixel frame,
                unless ``image_size`` says otherwise.
            image_size (tuple, optional): ``(H, W)`` the box was measured in.
                Boxes are rescaled when this differs from the depth map, which is
                required because the depth head predicts at input/4.

        Returns:
            (ObstacleDistance): Always returned; check ``.valid`` before use.
        """
        if depth_map.ndim != 2:
            depth_map = np.squeeze(depth_map)
        if depth_map.ndim != 2:
            raise ValueError(f"depth_map must be 2D (H, W); got shape {depth_map.shape}.")

        dh, dw = depth_map.shape
        x1, y1, x2, y2 = (float(v) for v in bbox)

        if image_size is not None and tuple(image_size) != (dh, dw):
            sy = dh / float(image_size[0])
            sx = dw / float(image_size[1])
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy

        roi: ROI = compute_bbox_inner_roi(
            x1,
            y1,
            x2,
            y2,
            inner_ratio=self.inner_ratio,
            image_width=dw,
            image_height=dh,
            min_size_px=self.min_size_px,
        )

        if roi.is_empty:
            return ObstacleDistance(
                valid=False, roi=roi.as_tuple(), reason="inner ROI is empty after clipping to the depth map"
            )

        patch = depth_map[roi.y1 : roi.y2, roi.x1 : roi.x2]
        stats = robust_depth_statistics(
            patch,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            clip_lower_percentile=self.clip_lower,
            clip_upper_percentile=self.clip_upper,
            statistic=self.statistic,
            percentile=self.percentile,
        )

        result = ObstacleDistance(
            valid=False,
            depth_median_m=stats["median"],
            depth_min_m=stats["min"],
            depth_max_m=stats["max"],
            depth_std_m=stats["std"],
            valid_ratio=stats["valid_ratio"],
            num_valid=stats["num_valid"],
            num_total=stats["num_total"],
            roi=roi.as_tuple(),
        )

        if stats["value"] is None:
            result.reason = "no valid depth pixels in the inner ROI"
            return result

        if stats["valid_ratio"] < self.min_valid_ratio:
            result.reason = (
                f"valid depth ratio {stats['valid_ratio']:.3f} below min_valid_ratio {self.min_valid_ratio:.3f}"
            )
            return result

        axial = float(stats["value"])

        # Euclidean range is only reported when intrinsics exist. Substituting
        # axial depth for it would be a silent geometric error (spec section T).
        euclid: float | None = None
        if self.intrinsics is not None:
            ucx, ucy = roi.center
            if image_size is not None and tuple(image_size) != (dh, dw):
                ucx *= image_size[1] / float(dw)
                ucy *= image_size[0] / float(dh)
            euclid = float(self.intrinsics.euclidean_from_axial(axial, ucx, ucy))

        result.valid = True
        result.euclidean_distance_m = euclid
        result.depth_median_m = stats["median"]

        if self.distance_mode == "euclidean":
            if euclid is None:
                result.valid = False
                result.reason = (
                    "distance_mode='euclidean' requires camera intrinsics; none were supplied. "
                    "Recovery: pass CameraIntrinsics, or set safety.distance_mode: axial."
                )
                return result
            result.distance_m = euclid
        else:
            result.distance_m = axial

        return result

    def confidence_score(
        self,
        detection_confidence: float,
        distance: ObstacleDistance,
        temporal_consistency: float = 1.0,
    ) -> float:
        """Fuse detection, depth-validity and temporal evidence (spec section Y).

        .. math::
            c = c_{det} \\cdot r_{valid} \\cdot c_{temporal} \\cdot s_{dispersion}

        The dispersion factor penalizes obstacles whose ROI depth is internally
        inconsistent (a high MAD relative to the distance usually means the box
        straddles an object and the wall behind it).

        This is a heuristic CONFIDENCE SCORE, deliberately not called a
        probability: it is not calibrated, and no claim is made that a score of
        0.8 corresponds to 80% correctness (spec section Y).

        Returns:
            (float): Score in ``[0, 1]``.
        """
        det = float(np.clip(detection_confidence, 0.0, 1.0))
        ratio = float(np.clip(distance.valid_ratio, 0.0, 1.0))
        temporal = float(np.clip(temporal_consistency, 0.0, 1.0))

        dispersion = 1.0
        if distance.depth_std_m is not None and distance.distance_m:
            rel = distance.depth_std_m / max(distance.distance_m, 1e-6)
            # Half-weight at rel == 1.0; approaches 0 as spread exceeds distance.
            dispersion = float(1.0 / (1.0 + rel))

        return float(np.clip(det * ratio * temporal * dispersion, 0.0, 1.0))
