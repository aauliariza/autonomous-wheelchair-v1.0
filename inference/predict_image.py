#!/usr/bin/env python3
"""Single-image inference with full navigation output (spec section AF).

Outputs: obstacle boxes labelled ``obstacle``, confidence, distance in metres,
sector assignment, blocked/free state, navigation command, depth visualization
and ROI overlay.

NOTE ON HYSTERESIS: a single image cannot fill the N=3 vote window, so the
smoothed command is the warm-up default (STOP). The RAW per-frame command is
reported alongside it and is the meaningful one for a still image
(spec section AC).

Usage:
    python inference/predict_image.py --source data/test.jpg
    python inference/predict_image.py --source img.jpg --model outputs/checkpoints/student_distilled_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_yaml, save_json  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("predict_image")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Run the navigation pipeline on one image.")
    ap.add_argument("--source", type=Path, required=True, help="Input image")
    ap.add_argument("--model", type=Path, default=None, help="Depth checkpoint")
    ap.add_argument("--detector", type=Path, default=None, help="Obstacle detector checkpoint")
    ap.add_argument("--config", type=Path, default=Path("configs/navigation.yaml"))
    ap.add_argument("--camera-config", type=Path, default=Path("configs/camera.yaml"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=None, help="Override detection confidence")
    ap.add_argument("--output", type=Path, default=Path("outputs/predictions"))
    ap.add_argument("--save-depth", action="store_true", help="Also write the colorized depth map")
    ap.add_argument("--no-show-depth", action="store_true", help="Do not append the depth panel")
    args = ap.parse_args(argv)

    try:
        import cv2

        from inference.pipeline import NavigationPipeline
        from visualization.depth_visualizer import colorize_depth
        from visualization.navigation_overlay import draw_navigation_overlay
        from visualization.obstacle_overlay import draw_obstacles

        if not args.source.exists():
            LOG.error("Image not found: %s", args.source.resolve())
            return 1

        frame = cv2.imread(str(args.source))
        if frame is None:
            LOG.error("Could not decode image: %s (unsupported format or corrupt file)", args.source)
            return 1

        cfg = load_yaml(args.config)
        if args.conf is not None:
            cfg.setdefault("detection", {})["conf"] = args.conf

        pipeline = NavigationPipeline(
            cfg,
            depth_weights=args.model,
            detector_weights=args.detector,
            camera_config=args.camera_config if args.camera_config.exists() else None,
            device=args.device,
        )

        # Warm-up pass: the first inference pays lazy CUDA/kernel initialisation,
        # which on this path measured ~2.1 s and tripped the latency fail-safe into
        # DEGRADED. A still image is an offline analysis, so the reported latency
        # should reflect steady-state inference, not one-off setup cost.
        pipeline.process_frame(frame)
        pipeline.reset()

        result = pipeline.process_frame(frame)

        # --- console report ---
        print()
        print(f"image            : {args.source}")
        print(f"obstacles        : {len(result.obstacles)}")
        print(f"raw command      : {result.raw_command}   <- meaningful for a single image")
        print(f"smoothed command : {result.command}   (N=3 hysteresis is in warm-up)")
        print(f"safety state     : {result.safety.get('state')}")
        print(f"latency          : {result.total_latency_ms:.1f} ms")
        print()
        print(f"{'id':<4}{'label':<10}{'conf':>7}{'distance':>11}{'euclid':>10}{'valid':>8}{'sector':>8}{'state':>9}")
        print("-" * 68)
        for ob in result.obstacles:
            dist = f"{ob.distance_m:.3f}m" if ob.distance_m is not None else "--"
            euc = f"{ob.euclidean_distance_m:.3f}m" if ob.euclidean_distance_m is not None else "--"
            state = "BLOCKED" if ob.blocked else ("free" if ob.in_roi else "out-ROI")
            print(
                f"{ob.id:<4}{ob.label:<10}{ob.confidence:>7.2f}{dist:>11}{euc:>10}"
                f"{ob.valid_depth_ratio:>8.2f}{str(ob.sector or '-'):>8}{state:>9}"
            )
        print("-" * 68)
        print("sector occupancy :", {k: ("blocked" if v else "free") for k, v in result.occupancy.items()})

        # --- visualization ---
        vis = draw_obstacles(frame, result.obstacles, show_inner_roi=True)
        vis = draw_navigation_overlay(
            vis,
            pipeline.sector_map,
            command=str(result.raw_command),
            safety_distance_m=pipeline.safety_distance_m,
            latency_ms=result.total_latency_ms,
            fps=result.fps,
            num_obstacles=len(result.obstacles),
            safety_state=result.safety.get("state"),
        )

        if not args.no_show_depth and result.depth_map is not None:
            panel = colorize_depth(result.depth_map)
            panel = cv2.resize(panel, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.putText(panel, "metric depth (m)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            import numpy as np

            vis = np.hstack([vis, panel])

        args.output.mkdir(parents=True, exist_ok=True)
        out_img = args.output / f"{args.source.stem}_navigation.jpg"
        cv2.imwrite(str(out_img), vis)
        LOG.info("Annotated image written to %s", out_img)

        if args.save_depth and result.depth_map is not None:
            depth_png = args.output / f"{args.source.stem}_depth.png"
            cv2.imwrite(str(depth_png), colorize_depth(result.depth_map))
            LOG.info("Depth visualization written to %s", depth_png)

        save_json(result.to_dict(), args.output / f"{args.source.stem}_result.json")
    except (RuntimeError, ValueError, ImportError, OSError) as e:
        LOG.error("Image inference failed (%s): %s", type(e).__name__, e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
