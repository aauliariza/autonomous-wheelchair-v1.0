#!/usr/bin/env python3
"""Train / fine-tune the YOLO26x-Depth teacher on SUN RGB-D (spec section F).

The teacher is the distillation target, so it is trained FIRST and to
convergence; every KD experiment then reuses the same frozen checkpoint, which
keeps the ablation rows comparable.

Post-training, Ultralytics automatically fits the log-affine calibration
(``cal_a``/``cal_b``) on the validation split and writes it into the checkpoint,
under a "calibrate only if it helps" policy. That is what makes the saved model
emit metric-scaled depth, since the SILog training loss is scale-invariant.

Usage:
    python training/train_teacher.py --config configs/teacher.yaml
    python training/train_teacher.py --config configs/teacher.yaml --set train.epochs=1 train.device=cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.common import build_ultralytics_args, export_best_checkpoint, resolve_data_yaml, setup_experiment  # noqa: E402
from utils.io import load_config, merge_overrides, save_json  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("train_teacher")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Train the YOLO26x-Depth teacher.")
    ap.add_argument("--config", type=Path, default=Path("configs/teacher.yaml"))
    ap.add_argument("--set", nargs="*", dest="overrides", default=None, help="Config overrides, e.g. train.epochs=50")
    ap.add_argument("--export-to", type=Path, default=Path("outputs/checkpoints/teacher_best.pt"))
    args = ap.parse_args(argv)

    try:
        config = merge_overrides(load_config(args.config), args.overrides)
        data_yaml = resolve_data_yaml(config)
        run_dir = setup_experiment(config, args.config, extra={"role": "teacher"})

        from ultralytics import YOLO

        model_cfg = config.get("model", {}) or {}
        weights = model_cfg.get("weights") or model_cfg.get("cfg")
        LOG.info("Loading teacher from %s", weights)
        model = YOLO(weights, task=model_cfg.get("task", "depth"))

        n_params = sum(p.numel() for p in model.model.parameters())
        LOG.info("Teacher parameters: %s", f"{n_params:,}")

        train_args = build_ultralytics_args(config, data_yaml)
        LOG.info("Training with: %s", {k: train_args[k] for k in sorted(train_args)})
        results = model.train(**train_args)

        actual_dir = Path(getattr(results, "save_dir", run_dir))
        metrics = dict(getattr(results, "results_dict", {}) or {})
        save_json(
            {"role": "teacher", "parameters": n_params, "metrics": metrics, "save_dir": str(actual_dir)},
            run_dir / "metrics.json",
        )
        LOG.info("Validation metrics: %s", metrics)

        export_best_checkpoint(actual_dir, args.export_to)
        LOG.info("Teacher training complete. Next: python evaluation/evaluate_depth.py --model %s", args.export_to)
    except FileNotFoundError as e:
        LOG.error("%s", e)
        return 1
    except (RuntimeError, ValueError, KeyError) as e:
        LOG.error("Teacher training failed (%s): %s", type(e).__name__, e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
