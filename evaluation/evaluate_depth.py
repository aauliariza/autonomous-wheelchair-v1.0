#!/usr/bin/env python3
"""Evaluate depth accuracy under both evaluation modes (spec sections R, S).

Reports MODE 1 (metric, ``align=none``) and MODE 2 (aligned, ``align=median``)
side by side. The metric mode is the one that matters for obstacle distance; the
aligned mode exists only for comparison with published monocular benchmarks.

Usage:
    python evaluation/evaluate_depth.py --model outputs/checkpoints/student_distilled_best.pt \
        --data configs/data/sunrgbd.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import DepthEvaluator  # noqa: E402
from utils.io import load_yaml, save_json  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("evaluate_depth")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def load_pairs(data_yaml: Path, split: str) -> list[tuple[Path, Path]]:
    """Resolve (image, depth) pairs for a split, using the images/->depth/ convention."""
    cfg = load_yaml(data_yaml)
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    rel = cfg.get(split, f"images/{split}")
    img_dir = root / rel
    dep_dir = Path(str(img_dir).replace("/images/", "/depth/"))

    if not img_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {img_dir}\n  Recovery: check '{split}' in {data_yaml}.")
    if not dep_dir.exists():
        raise FileNotFoundError(
            f"Depth directory not found: {dep_dir}\n"
            f"  Expected the images/ -> depth/ layout produced by prepare_sunrgbd.py."
        )

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    depth_by_stem = {p.stem: p for p in dep_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    pairs = [(im, depth_by_stem[im.stem]) for im in images if im.stem in depth_by_stem]

    if not pairs:
        raise FileNotFoundError(f"No matched image/depth pairs between {img_dir} and {dep_dir}.")
    return pairs


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Evaluate depth accuracy (metric and aligned modes).")
    ap.add_argument("--model", type=Path, required=True, help="Depth checkpoint to evaluate")
    ap.add_argument("--data", type=Path, required=True, help="Dataset YAML")
    ap.add_argument("--split", default="val", help="Split to evaluate")
    ap.add_argument("--device", default=None, help="0 | cpu | cuda:0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=None, help="Evaluate at most N images")
    ap.add_argument("--output", type=Path, default=Path("outputs/evaluation/depth_metrics.json"))
    args = ap.parse_args(argv)

    try:
        import cv2
        from ultralytics import YOLO

        from models.model_utils import select_device

        cfg = load_yaml(args.data)
        depth_scale = float(cfg.get("depth_scale", 1000))
        max_depth = float(cfg.get("max_depth", 10.0))

        pairs = load_pairs(args.data, args.split)
        if args.limit:
            pairs = pairs[: args.limit]
        LOG.info("Evaluating %d image(s) from split '%s'", len(pairs), args.split)

        device = select_device(args.device)
        model = YOLO(str(args.model))
        LOG.info("Model: %s on %s", args.model, device)

        evaluator = DepthEvaluator(max_depth=max_depth)

        for img_path, dep_path in pairs:
            raw = cv2.imread(str(dep_path), cv2.IMREAD_ANYDEPTH)
            if raw is None:
                LOG.warning("Unreadable depth map, skipping: %s", dep_path)
                continue
            gt = raw.astype(np.float32) / depth_scale

            result = model.predict(str(img_path), imgsz=args.imgsz, device=str(device), verbose=False)[0]
            pred = result.depth.data
            pred = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
            pred = np.squeeze(pred)

            # Compare at the GROUND TRUTH's resolution: upsampling GT would
            # invent depth in regions the sensor never measured.
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

            evaluator.update(pred, gt)

        results = evaluator.compute()
        print()
        print(DepthEvaluator.format_table(results))

        save_json(
            {
                "model": str(args.model),
                "data": str(args.data),
                "split": args.split,
                "imgsz": args.imgsz,
                "max_depth": max_depth,
                **results,
            },
            args.output,
        )
        LOG.info("Results written to %s", args.output)
    except FileNotFoundError as e:
        LOG.error("%s", e)
        return 1
    except (RuntimeError, ValueError, ImportError) as e:
        LOG.error("Depth evaluation failed (%s): %s", type(e).__name__, e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
