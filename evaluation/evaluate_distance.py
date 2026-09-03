#!/usr/bin/env python3
"""Evaluate obstacle DISTANCE accuracy (spec section AK).

Depth metrics score every pixel equally. This script scores the quantity the
wheelchair actually acts on: one distance per obstacle.

GROUND-TRUTH DEFINITION — stated explicitly, as spec section AK requires
------------------------------------------------------------------------
Ground-truth obstacle distance is the MEDIAN of the valid ground-truth depth
inside the same inner-60% bounding-box ROI used for the prediction, reduced by
the same robust statistics. Concretely:

* the SAME ``compute_bbox_inner_roi`` region,
* the SAME invalid-pixel rejection (0 / NaN / Inf / out-of-range),
* the SAME percentile clipping and median reduction.

Prediction and ground truth therefore differ ONLY in their depth source (student
network vs sensor). Any other choice would confound distance error with a
difference in the reduction itself.

By default both are AXIAL depth (camera-axis Z), matching
``navigation.safety.distance_mode``. Passing ``--distance-mode euclidean``
converts BOTH sides with the same intrinsics, so the comparison stays consistent
(spec section T).

METRICS
-------
MAE, RMSE, MAPE, median absolute error, and the fraction of obstacles predicted
within +/-10%, +/-20% and +/-30% of ground truth.

Usage:
    python evaluation/evaluate_distance.py --model outputs/checkpoints/student_distilled_best.pt \
        --data configs/data/sunrgbd.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_yaml, save_json  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("evaluate_distance")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def distance_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    """Compute obstacle-distance error metrics (spec section AK)."""
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    ok = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    if not ok.any():
        return {"num_obstacles": 0, "note": "NOT MEASURED - no valid obstacle pairs"}

    p, g = pred[ok], gt[ok]
    err = p - g
    rel = np.abs(err) / g

    return {
        "num_obstacles": int(ok.sum()),
        "obstacle_distance_mae": float(np.mean(np.abs(err))),
        "obstacle_distance_rmse": float(np.sqrt(np.mean(err**2))),
        "obstacle_distance_mape": float(np.mean(rel) * 100.0),
        "median_absolute_error": float(np.median(np.abs(err))),
        "percentage_within_10_percent": float(np.mean(rel <= 0.10) * 100.0),
        "percentage_within_20_percent": float(np.mean(rel <= 0.20) * 100.0),
        "percentage_within_30_percent": float(np.mean(rel <= 0.30) * 100.0),
        # A signed bias reveals systematic over- or under-estimation, which for a
        # wheelchair is far more dangerous in one direction: predicting FARTHER
        # than reality is what causes a collision.
        "mean_signed_error": float(np.mean(err)),
        "overestimated_fraction": float(np.mean(err > 0)),
    }


def synthetic_boxes(
    gt_depth: np.ndarray, grid: int = 3, margin: float = 0.15
) -> list[tuple[float, float, float, float]]:
    """Generate a deterministic grid of evaluation regions.

    Used when no obstacle annotations exist. These are EVALUATION REGIONS, not
    fabricated obstacle labels: no claim is made that they contain objects. They
    let distance-estimation accuracy be measured on real depth without inventing
    detections. Regions whose GT is mostly invalid are dropped by the estimator.
    """
    h, w = gt_depth.shape[:2]
    boxes = []
    bw, bh = w * (1 - 2 * margin) / grid, h * (1 - 2 * margin) / grid
    for r in range(grid):
        for c in range(grid):
            x1 = w * margin + c * bw
            y1 = h * margin + r * bh
            boxes.append((x1, y1, x1 + bw, y1 + bh))
    return boxes


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Evaluate obstacle distance accuracy.")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--nav-config", type=Path, default=Path("configs/navigation.yaml"))
    ap.add_argument("--camera-config", type=Path, default=Path("configs/camera.yaml"))
    ap.add_argument("--distance-mode", choices=["axial", "euclidean"], default=None)
    ap.add_argument(
        "--detector",
        type=Path,
        default=None,
        help="Obstacle detector. Without it, a deterministic grid of evaluation regions is used.",
    )
    ap.add_argument("--output", type=Path, default=Path("outputs/evaluation/distance_metrics.json"))
    args = ap.parse_args(argv)

    try:
        import cv2
        from ultralytics import YOLO

        from evaluation.evaluate_depth import load_pairs
        from models.model_utils import select_device
        from navigation.distance import DistanceEstimator

        nav_cfg = load_yaml(args.nav_config)
        if args.distance_mode:
            nav_cfg.setdefault("safety", {})["distance_mode"] = args.distance_mode
        mode = nav_cfg.get("safety", {}).get("distance_mode", "axial")

        intrinsics = None
        if mode == "euclidean":
            from calibration.intrinsics import CameraIntrinsics

            intrinsics = CameraIntrinsics.from_yaml(args.camera_config, require_calibrated=True)
            LOG.info("Euclidean mode using %s", intrinsics)

        data_cfg = load_yaml(args.data)
        depth_scale = float(data_cfg.get("depth_scale", 1000))

        pairs = load_pairs(args.data, args.split)
        if args.limit:
            pairs = pairs[: args.limit]

        device = select_device(args.device)
        model = YOLO(str(args.model))
        detector = YOLO(str(args.detector)) if args.detector else None

        # One estimator instance drives BOTH sides, so prediction and ground
        # truth are reduced identically (see module docstring).
        estimator = DistanceEstimator(nav_cfg, intrinsics=intrinsics)

        preds: list[float] = []
        gts: list[float] = []
        LOG.info(
            "Evaluating %d image(s) | distance_mode=%s | source=%s",
            len(pairs),
            mode,
            "detector" if detector else "grid regions",
        )

        for img_path, dep_path in pairs:
            raw = cv2.imread(str(dep_path), cv2.IMREAD_ANYDEPTH)
            if raw is None:
                continue
            gt_depth = raw.astype(np.float32) / depth_scale

            result = model.predict(str(img_path), imgsz=args.imgsz, device=str(device), verbose=False)[0]
            pd = result.depth.data
            pd = pd.detach().cpu().numpy() if hasattr(pd, "detach") else np.asarray(pd)
            pred_depth = np.squeeze(pd)
            if pred_depth.shape != gt_depth.shape:
                pred_depth = cv2.resize(
                    pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR
                )

            if detector is not None:
                det = detector.predict(str(img_path), imgsz=args.imgsz, device=str(device), verbose=False)[0]
                boxes = det.boxes.xyxy.cpu().numpy().tolist() if det.boxes is not None else []
            else:
                boxes = synthetic_boxes(gt_depth)

            for box in boxes:
                p = estimator.estimate(pred_depth, tuple(box))
                g = estimator.estimate(gt_depth, tuple(box))
                if p.valid and g.valid:
                    preds.append(p.distance_m)
                    gts.append(g.distance_m)

        metrics = distance_metrics(np.array(preds), np.array(gts))

        print()
        print(f"{'obstacle distance metric':<34} {'value':>12}")
        print("-" * 48)
        for k, v in metrics.items():
            print(f"{k:<34} {v:>12.4f}" if isinstance(v, float) else f"{k:<34} {v:>12}")
        print("-" * 48)
        print(f"ground truth definition: median of valid GT depth in the inner-60% ROI ({mode} depth)")

        save_json(
            {
                "model": str(args.model),
                "data": str(args.data),
                "split": args.split,
                "distance_mode": mode,
                "ground_truth_definition": (
                    f"median of valid ground-truth depth within the inner-60% bbox ROI, {mode} depth, "
                    "reduced with the same robust statistics as the prediction"
                ),
                "region_source": "detector" if detector else "deterministic grid (no obstacle annotations)",
                **metrics,
            },
            args.output,
        )
        LOG.info("Results written to %s", args.output)
    except FileNotFoundError as e:
        LOG.error("%s", e)
        return 1
    except (RuntimeError, ValueError, ImportError) as e:
        LOG.error("Distance evaluation failed (%s): %s", type(e).__name__, e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
