"""Depth metrics with an explicit metric-vs-aligned distinction (spec sections R, S).

THE MOST IMPORTANT DECISION IN THIS FILE
----------------------------------------
Monocular depth benchmarks conventionally rescale each prediction by
``median(gt)/median(pred)`` before scoring. That protocol exists because most
monocular models are scale-AMBIGUOUS: they recover structure, not metres.

This research is about ABSOLUTE metric distance for obstacle avoidance, so
median alignment would hide the exact error that matters. Measured in this
project's own audit, on one model and one dataset:

.. code-block:: text

    align="median"  ->  delta1 = 0.7977      (benchmark protocol)
    align="none"    ->  delta1 = 0.2310      (true metric accuracy)

The same model looks 3.5x better under alignment. A wheelchair braking at 1.0 m
cares only about the second number.

Two modes are therefore provided and always reported separately:

* ``MODE 1 — metric``   ``align="none"``   the deployment-relevant number.
* ``MODE 2 — aligned``  ``align="median"`` for comparison against published
  monocular benchmarks.

Never quote an aligned figure as evidence of metric accuracy.

METRICS (spec section R — "metrics evaluation", not "matrix evaluation")
------------------------------------------------------------------------
AbsRel, SqRel, RMSE, RMSE_log, delta1/2/3, SILog.

Following the standard Eigen protocol, only pixels with ground truth inside
``(min_depth, max_depth)`` are scored, and predictions are clamped into that
range. Metrics are finalized PER IMAGE and then averaged, so every image counts
equally regardless of how many valid pixels it has — matching Depth Anything V2
and Monodepth2, and matching Ultralytics' own DepthMetrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

# Images with fewer valid pixels than this are skipped: a median computed from a
# handful of pixels is meaningless (the Depth Anything V2 floor).
MIN_VALID_PIXELS = 10

METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3", "silog")


def compute_depth_metrics(
    pred: torch.Tensor | np.ndarray,
    gt: torch.Tensor | np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float = 10.0,
    align: str = "none",
) -> dict[str, float] | None:
    """Compute all eight depth metrics for ONE image.

    Args:
        pred: Predicted depth ``(H, W)`` in metres.
        gt: Ground-truth depth ``(H, W)`` in metres; invalid pixels are 0/NaN/Inf.
        min_depth, max_depth (float): Eigen-protocol valid range.
        align (str): ``"none"`` for metric evaluation (MODE 1) or ``"median"``
            for the scale-aligned benchmark protocol (MODE 2).

    Returns:
        (dict | None): Metric name -> value, or None when the image has too few
            valid pixels to score.
    """
    if align not in ("none", "median"):
        raise ValueError(f"align must be 'none' or 'median', got '{align}'.")

    p = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
    g = gt.detach().cpu().numpy() if isinstance(gt, torch.Tensor) else np.asarray(gt)
    p, g = np.squeeze(p).astype(np.float64), np.squeeze(g).astype(np.float64)

    if p.shape != g.shape:
        raise ValueError(f"Prediction shape {p.shape} does not match ground truth {g.shape}.")

    mask = np.isfinite(g) & (g > min_depth) & (g < max_depth)
    if int(mask.sum()) < MIN_VALID_PIXELS:
        return None

    pv, gv = p[mask], g[mask]

    if align == "median":
        finite = np.isfinite(pv)
        if finite.any():
            denom = np.median(np.clip(pv[finite], min_depth, None))
            if denom > 0:
                pv = pv * (np.median(gv[finite]) / denom)

    # Non-finite predictions are scored at a bound rather than dropped, so a
    # model that emits NaN cannot improve its score by hiding pixels.
    pv = np.nan_to_num(pv, nan=max_depth, posinf=max_depth, neginf=min_depth)
    pv = np.clip(pv, min_depth, max_depth)

    thresh = np.maximum(pv / gv, gv / pv)
    diff = pv - gv
    log_diff = np.log(pv) - np.log(gv)

    return {
        "abs_rel": float(np.mean(np.abs(diff) / gv)),
        "sq_rel": float(np.mean(diff**2 / gv)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "rmse_log": float(np.sqrt(np.mean(log_diff**2))),
        "delta1": float(np.mean(thresh < 1.25)),
        "delta2": float(np.mean(thresh < 1.25**2)),
        "delta3": float(np.mean(thresh < 1.25**3)),
        # lambda=1 variance form (ZoeDepth/KITTI convention), x100.
        "silog": float(np.sqrt(max(np.mean(log_diff**2) - np.mean(log_diff) ** 2, 0.0)) * 100),
        "valid_pixels": int(mask.sum()),
    }


class DepthEvaluator:
    """Accumulates per-image depth metrics under BOTH evaluation modes.

    Both modes are computed from the same predictions in one pass, so a report
    can present them side by side and the gap between them is itself a result:
    it quantifies how much of the model's apparent accuracy depends on scale
    alignment.

    Examples:
        >>> ev = DepthEvaluator(max_depth=10.0)
        >>> import numpy as np
        >>> _ = ev.update(np.full((32, 32), 2.0), np.full((32, 32), 2.0))
        >>> ev.compute()["metric"]["delta1"]
        1.0
    """

    def __init__(self, min_depth: float = 1e-3, max_depth: float = 10.0):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self._acc: dict[str, list[dict[str, float]]] = {"metric": [], "aligned": []}
        self.skipped = 0

    def update(self, pred, gt) -> bool:
        """Score one image under both modes. Returns False when it was skipped."""
        m = compute_depth_metrics(pred, gt, self.min_depth, self.max_depth, align="none")
        a = compute_depth_metrics(pred, gt, self.min_depth, self.max_depth, align="median")
        if m is None or a is None:
            self.skipped += 1
            return False
        self._acc["metric"].append(m)
        self._acc["aligned"].append(a)
        return True

    def update_batch(self, preds, gts) -> int:
        """Score a batch, returning how many images were counted."""
        preds = preds.squeeze(1) if getattr(preds, "ndim", 0) == 4 else preds
        gts = gts.squeeze(1) if getattr(gts, "ndim", 0) == 4 else gts
        return sum(1 for p, g in zip(preds, gts) if self.update(p, g))

    def compute(self) -> dict[str, Any]:
        """Average per-image metrics for both modes.

        Returns:
            (dict): ``{"metric": {...}, "aligned": {...}, "num_images": int,
                "skipped": int, "alignment_gap": {...}}``. The gap is
                ``aligned - metric`` per key, making the cost of absolute-scale
                error explicit rather than something a reader must infer.
        """
        out: dict[str, Any] = {"num_images": len(self._acc["metric"]), "skipped": self.skipped}

        for mode, rows in self._acc.items():
            out[mode] = (
                {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}
                if rows
                else dict.fromkeys(METRIC_KEYS, float("nan"))
            )

        if self._acc["metric"]:
            out["alignment_gap"] = {k: out["aligned"][k] - out["metric"][k] for k in METRIC_KEYS}
        return out

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._acc = {"metric": [], "aligned": []}
        self.skipped = 0

    @staticmethod
    def format_table(results: dict[str, Any]) -> str:
        """Render both modes as an aligned text table."""
        lines = [
            f"{'metric':<12} {'MODE 1 (metric)':>18} {'MODE 2 (aligned)':>18} {'gap':>12}",
            "-" * 64,
        ]
        for k in METRIC_KEYS:
            m = results.get("metric", {}).get(k, float("nan"))
            a = results.get("aligned", {}).get(k, float("nan"))
            lines.append(f"{k:<12} {m:>18.4f} {a:>18.4f} {a - m:>12.4f}")
        lines.append("-" * 64)
        lines.append(f"images scored: {results.get('num_images', 0)}  skipped: {results.get('skipped', 0)}")
        lines.append("MODE 1 (align=none) is the deployment-relevant metric accuracy.")
        lines.append("MODE 2 (align=median) is the benchmark protocol and HIDES absolute-scale error.")
        return "\n".join(lines)
