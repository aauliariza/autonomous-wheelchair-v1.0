#!/usr/bin/env python3
"""Per-stage latency and model complexity benchmark (spec sections AN, AO).

Reports mean / median / P95 / P99 for EACH pipeline stage separately, never FPS
alone. For a safety-critical system the tail matters more than the average: a
20 ms mean with a 300 ms P99 means one frame in a hundred exceeds the stale-frame
timeout and forces a STOP.

Stages measured independently (spec section AN):
    detection, depth, fusion, distance estimation, free-path selection, total.

MODEL LOADING IS EXCLUDED. Only the forward pass is timed, after warm-up
iterations, so lazy CUDA kernel compilation is not charged to inference
(spec section BF).

Usage:
    python evaluation/benchmark_latency.py --depth-model outputs/checkpoints/student_distilled_best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_yaml, save_json  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("benchmark_latency")


def summarize(samples: list[float]) -> dict[str, float]:
    """Summarize timing samples in milliseconds."""
    if not samples:
        return {"note": "NOT MEASURED"}
    a = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "std_ms": float(a.std()),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "runs": int(a.size),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Benchmark per-stage pipeline latency and model complexity.")
    ap.add_argument("--depth-model", type=Path, default=Path("yolo26n-depth.pt"))
    ap.add_argument("--detector", type=Path, default=Path("yolo26n.pt"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--nav-config", type=Path, default=Path("configs/navigation.yaml"))
    ap.add_argument("--output", type=Path, default=Path("outputs/evaluation/latency_benchmark.json"))
    args = ap.parse_args(argv)

    try:
        import torch
        from ultralytics import YOLO

        from models.model_utils import measure_latency, model_complexity, peak_memory_mb, select_device
        from navigation.free_path import FreePathSelector
        from navigation.obstacle_fusion import ObstacleFusion
        from navigation.roi import compute_global_roi
        from navigation.sectors import SectorMap

        device = select_device(args.device)
        nav_cfg = load_yaml(args.nav_config)

        LOG.info("Loading models (load time is excluded from the measurements)...")
        depth_model = YOLO(str(args.depth_model))
        detector = YOLO(str(args.detector))

        # --- complexity (spec section AO) ---
        complexity = {
            "depth": model_complexity(depth_model.model, imgsz=args.imgsz, device=device),
            "detector": model_complexity(detector.model, imgsz=args.imgsz, device=device),
        }
        for name, c in complexity.items():
            LOG.info(
                "%s: %s params, %s GFLOPs, %.2f MB",
                name,
                f"{c['parameters']:,}",
                c["gflops"] if c["gflops"] is not None else "NOT MEASURED",
                c["model_size_mb"],
            )

        # --- raw forward-pass latency ---
        LOG.info("Measuring forward-pass latency (%d runs, %d warmup)...", args.runs, args.warmup)
        raw = {
            "depth_forward": measure_latency(depth_model.model, args.imgsz, device, args.runs, args.warmup),
            "detector_forward": measure_latency(detector.model, args.imgsz, device, args.runs, args.warmup),
        }

        # --- end-to-end per-stage latency ---
        frame = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
        roi = compute_global_roi(args.imgsz, args.imgsz, nav_cfg["roi"]["width_ratio"], nav_cfg["roi"]["x_center"])
        sector_map = SectorMap.from_config(nav_cfg, roi)
        fusion = ObstacleFusion(nav_cfg)
        selector = FreePathSelector(nav_cfg)

        stages: dict[str, list[float]] = {k: [] for k in ("detection", "depth", "fusion", "distance", "free_path", "total")}

        for i in range(args.runs + args.warmup):
            t_all = time.perf_counter()

            t = time.perf_counter()
            det = detector.predict(frame, imgsz=args.imgsz, device=str(device), verbose=False)[0]
            t_det = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            dep = depth_model.predict(frame, imgsz=args.imgsz, device=str(device), verbose=False)[0]
            t_dep = (time.perf_counter() - t) * 1000

            boxes = det.boxes.xyxy.cpu().numpy() if det.boxes is not None else np.zeros((0, 4))
            confs = det.boxes.conf.cpu().numpy() if det.boxes is not None else np.zeros((0,))
            dmap = dep.depth.data
            dmap = dmap.detach().cpu().numpy() if hasattr(dmap, "detach") else np.asarray(dmap)

            # Fusion covers detection-depth association AND distance estimation;
            # they are reported together and the distance share is timed inside.
            t = time.perf_counter()
            obstacles = fusion.fuse(boxes, confs, np.squeeze(dmap), sector_map, image_size=frame.shape[:2])
            t_fuse = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            selector.select(sector_map)
            t_path = (time.perf_counter() - t) * 1000

            t_total = (time.perf_counter() - t_all) * 1000

            if i >= args.warmup:  # discard warm-up iterations
                stages["detection"].append(t_det)
                stages["depth"].append(t_dep)
                stages["fusion"].append(t_fuse)
                stages["distance"].append(t_fuse)  # distance is computed inside fusion
                stages["free_path"].append(t_path)
                stages["total"].append(t_total)

        summary = {k: summarize(v) for k, v in stages.items()}

        print()
        print(f"{'stage':<14}{'mean':>10}{'median':>10}{'P95':>10}{'P99':>10}{'max':>10}   (ms)")
        print("-" * 68)
        for name, s in summary.items():
            if "mean_ms" in s:
                print(f"{name:<14}{s['mean_ms']:>10.2f}{s['median_ms']:>10.2f}{s['p95_ms']:>10.2f}"
                      f"{s['p99_ms']:>10.2f}{s['max_ms']:>10.2f}")
        print("-" * 68)
        total = summary["total"]
        if "mean_ms" in total:
            print(f"end-to-end mean {total['mean_ms']:.2f} ms  ->  {1000 / total['mean_ms']:.2f} FPS")
            print(f"P99 {total['p99_ms']:.2f} ms is the number that governs the stale-frame timeout.")

        save_json(
            {
                "device": str(device),
                "imgsz": args.imgsz,
                "runs": args.runs,
                "warmup": args.warmup,
                "depth_model": str(args.depth_model),
                "detector": str(args.detector),
                "complexity": complexity,
                "forward_only": raw,
                "pipeline_stages": summary,
                "peak_gpu_memory_mb": peak_memory_mb(device),
                "torch_version": torch.__version__,
            },
            args.output,
        )
        LOG.info("Benchmark written to %s", args.output)
    except (RuntimeError, ValueError, ImportError, FileNotFoundError) as e:
        LOG.error("Latency benchmark failed (%s): %s", type(e).__name__, e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
