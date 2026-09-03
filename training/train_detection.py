#!/usr/bin/env python3
"""Fine-tune YOLO26n as a single-class obstacle detector (spec sections A, BB).

The dataset must have ``nc = 1`` with class 0 named ``obstacle``. ``single_cls``
is forced on as a second guarantee, so even a dataset that slipped through with
extra classes collapses to one.

If you have no obstacle annotations yet, you do NOT need this script to run the
navigation pipeline: ``configs/navigation.yaml`` sets
``detection.class_agnostic: true``, which relabels every COCO detection from the
stock ``yolo26n.pt`` as ``obstacle``. Fine-tuning here is the higher-accuracy
path once real indoor annotations exist.

Usage:
    python training/train_detection.py --config configs/detection.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.common import (  # noqa: E402
    build_ultralytics_args,
    export_best_checkpoint,
    resolve_data_yaml,
    setup_experiment,
)
from utils.io import load_config, load_yaml, merge_overrides, save_json  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("train_detection")


def validate_single_class(data_yaml: Path) -> None:
    """Refuse to train unless the dataset really is single-class ``obstacle``."""
    cfg = load_yaml(data_yaml)
    nc = int(cfg.get("nc", 0))
    names = cfg.get("names", {}) or {}

    if nc != 1:
        raise ValueError(
            f"Detection dataset must have nc=1, but {data_yaml} declares nc={nc} with names={names}.\n"
            f"  The system performs obstacle DETECTION, not object recognition (spec section A).\n"
            f"  Recovery: python datasets/scripts/convert_to_obstacle_dataset.py "
            f"--format yolo --source <your dataset> --output datasets/obstacle"
        )

    label = names.get(0) if isinstance(names, dict) else (names[0] if names else None)
    if str(label).lower() != "obstacle":
        LOG.warning("Class 0 is named '%s', not 'obstacle'. Renaming it in the run config for consistency.", label)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Fine-tune YOLO26n as a single-class obstacle detector.")
    ap.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    ap.add_argument("--set", nargs="*", dest="overrides", default=None)
    ap.add_argument("--export-to", type=Path, default=Path("outputs/checkpoints/detector_obstacle_best.pt"))
    args = ap.parse_args(argv)

    try:
        config = merge_overrides(load_config(args.config), args.overrides)
        data_yaml = resolve_data_yaml(config)
        validate_single_class(data_yaml)
        run_dir = setup_experiment(config, args.config, extra={"role": "detector"})

        from ultralytics import YOLO

        model_cfg = config.get("model", {}) or {}
        model = YOLO(model_cfg.get("weights", "yolo26n.pt"), task="detect")

        train_args = build_ultralytics_args(config, data_yaml)
        train_args["single_cls"] = True  # hard guarantee (spec section BB)
        LOG.info("Training obstacle detector with: %s", {k: train_args[k] for k in sorted(train_args)})

        results = model.train(**train_args)
        actual_dir = Path(getattr(results, "save_dir", run_dir))
        metrics = dict(getattr(results, "results_dict", {}) or {})
        save_json(
            {"role": "detector", "nc": 1, "metrics": metrics, "save_dir": str(actual_dir)}, run_dir / "metrics.json"
        )
        LOG.info("Detection metrics: %s", metrics)

        export_best_checkpoint(actual_dir, args.export_to)
    except (FileNotFoundError, ValueError) as e:
        LOG.error("%s", e)
        return 1
    except (RuntimeError, KeyError) as e:
        LOG.error("Detector training failed (%s): %s", type(e).__name__, e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
