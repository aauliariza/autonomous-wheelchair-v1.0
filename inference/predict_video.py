#!/usr/bin/env python3
"""Video / webcam inference with the full navigation HUD (spec sections AG, AI).

Overlays FPS, inference latency, obstacle count, per-obstacle distances, the five
sectors and their state, the current command, the safety threshold and the ROI.

Unlike a single image, a video exercises the N=3 majority-vote hysteresis and the
temporal safety monitor, so the SMOOTHED command is the meaningful output here.

Per-frame telemetry is written to CSV in the spec section AI schema.

Usage:
    python inference/predict_video.py --source data/test.mp4
    python inference/predict_video.py --source 0 --show          # webcam
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_yaml  # noqa: E402
from utils.logger import FrameLogger, get_logger  # noqa: E402

LOG = get_logger("predict_video")


def open_source(source: str):
    """Open a video file or camera index, with an actionable error on failure."""
    import cv2

    # A bare integer means a camera index; anything else is a path or URL.
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise OSError(
            f"Could not open video source '{source}'.\n"
            f"  If it is a file: check the path and that the codec is supported by your OpenCV build.\n"
            f"  If it is a camera index: check the device exists and is not in use by another process."
        )
    return cap


def run(args) -> int:
    """Process every frame of the source."""
    import cv2

    from inference.pipeline import NavigationPipeline
    from visualization.depth_visualizer import colorize_depth
    from visualization.navigation_overlay import draw_navigation_overlay
    from visualization.obstacle_overlay import draw_obstacles

    cfg = load_yaml(args.config)
    if args.conf is not None:
        cfg.setdefault("detection", {})["conf"] = args.conf

    pipeline = NavigationPipeline(
        cfg,
        depth_weights=args.model,
        detector_weights=args.detector,
        camera_config=args.camera_config if Path(args.camera_config).exists() else None,
        device=args.device,
    )

    cap = open_source(args.source)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_width = width * 2 if args.show_depth else width
    args.output.mkdir(parents=True, exist_ok=True)
    stem = Path(str(args.source)).stem if not str(args.source).isdigit() else f"camera{args.source}"
    out_path = args.output / f"{stem}_navigation.mp4"

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (out_width, height))
    if not writer.isOpened():
        raise OSError(
            f"Could not open the video writer for {out_path}. "
            f"Your OpenCV build may lack the mp4v encoder; try --output with a different location."
        )

    csv_path = args.output / f"{stem}_telemetry.csv"
    counts: dict[str, int] = {}
    processed = 0
    t_start = time.perf_counter()

    LOG.info("Processing %s (%dx%d @ %.1f fps, %s frames)", args.source, width, height, fps_in,
             total if total > 0 else "unknown")

    with FrameLogger(csv_path) as flog:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and processed >= args.max_frames:
                break

            result = pipeline.process_frame(frame)
            processed += 1
            counts[str(result.command)] = counts.get(str(result.command), 0) + 1

            flog.log_frame(
                frame_id=result.frame_id,
                obstacles=result.obstacles,
                command=str(result.command),
                latency_ms=result.total_latency_ms,
            )

            vis = draw_obstacles(frame, result.obstacles)
            vis = draw_navigation_overlay(
                vis,
                pipeline.sector_map,
                command=str(result.command),
                safety_distance_m=pipeline.safety_distance_m,
                latency_ms=result.total_latency_ms,
                fps=result.fps,
                num_obstacles=len(result.obstacles),
                safety_state=result.safety.get("state"),
                raw_command=str(result.raw_command),
                hysteresis_state=pipeline.hysteresis.state(),
            )

            if args.show_depth and result.depth_map is not None:
                import numpy as np

                panel = cv2.resize(colorize_depth(result.depth_map), (width, height), interpolation=cv2.INTER_NEAREST)
                vis = np.hstack([vis, panel])
            elif args.show_depth:
                import numpy as np

                vis = np.hstack([vis, np.zeros_like(vis)])

            writer.write(vis)

            if args.show:
                cv2.imshow("autonomous wheelchair", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    LOG.info("Interrupted by user.")
                    break

            if processed % 30 == 0:
                LOG.info("frame %d | %s | %.1f ms", processed, result.command, result.total_latency_ms)

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    LOG.info("Processed %d frames in %.1fs (%.2f FPS end-to-end)", processed, elapsed,
             processed / elapsed if elapsed > 0 else 0.0)
    LOG.info("Command distribution: %s", counts)
    LOG.info("Annotated video: %s", out_path)
    LOG.info("Telemetry CSV  : %s", csv_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Run the navigation pipeline over a video or camera.")
    ap.add_argument("--source", required=True, help="Video file path, URL, or camera index")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--detector", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=Path("configs/navigation.yaml"))
    ap.add_argument("--camera-config", type=Path, default=Path("configs/camera.yaml"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--output", type=Path, default=Path("outputs/predictions"))
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--show", action="store_true", help="Display a live window (needs a GUI)")
    ap.add_argument("--show-depth", action="store_true", help="Append the depth panel to each frame")
    args = ap.parse_args(argv)

    try:
        return run(args)
    except OSError as e:
        LOG.error("%s", e)
        return 1
    except (RuntimeError, ValueError, ImportError) as e:
        LOG.error("Video inference failed (%s): %s", type(e).__name__, e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
