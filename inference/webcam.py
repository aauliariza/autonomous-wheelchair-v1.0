#!/usr/bin/env python3
"""Live webcam navigation (spec section AG).

A thin wrapper over ``predict_video`` that defaults to a live camera and enables
the display window.

SAFETY NOTE: this is a PERCEPTION demo. It emits navigation commands and never
drives a motor. Wiring these commands to a real wheelchair requires the
independent motor-controller safety layer described in spec section AE.

Usage:
    python inference/webcam.py --camera 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.predict_video import main as video_main  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("webcam")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Live webcam navigation demo.")
    ap.add_argument("--camera", default="0", help="Camera index")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--detector", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=Path("configs/navigation.yaml"))
    ap.add_argument("--camera-config", type=Path, default=Path("configs/camera.yaml"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--output", type=Path, default=Path("outputs/predictions"))
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-show", action="store_true", help="Headless: record without a window")
    ap.add_argument("--show-depth", action="store_true")
    args = ap.parse_args(argv)

    LOG.warning("PERCEPTION DEMO ONLY: commands are displayed, never sent to a motor controller.")

    forwarded = [
        "--source",
        str(args.camera),
        "--config",
        str(args.config),
        "--camera-config",
        str(args.camera_config),
        "--output",
        str(args.output),
    ]
    if args.model:
        forwarded += ["--model", str(args.model)]
    if args.detector:
        forwarded += ["--detector", str(args.detector)]
    if args.device:
        forwarded += ["--device", str(args.device)]
    if args.conf is not None:
        forwarded += ["--conf", str(args.conf)]
    if args.max_frames:
        forwarded += ["--max-frames", str(args.max_frames)]
    if not args.no_show:
        forwarded.append("--show")
    if args.show_depth:
        forwarded.append("--show-depth")

    return video_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
