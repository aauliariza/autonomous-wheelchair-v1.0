#!/usr/bin/env python3
"""Train the distilled YOLO26n-Depth student (spec sections H-Q).

Runs EXPERIMENTS B-E from the ablation table. Which experiment you get is
determined entirely by which ``kd.<term>.enabled`` flags are set in the config,
so no source change is needed between rows:

.. code-block:: text

    B  --set kd.feature.enabled=false kd.boundary.enabled=false \\
             kd.relative.enabled=false kd.roi.enabled=false
    C  --set kd.boundary.enabled=false kd.relative.enabled=false kd.roi.enabled=false
    D  --set kd.relative.enabled=false kd.roi.enabled=false
    E  (defaults in configs/distillation.yaml)

Usage:
    python training/train_distillation.py --config configs/distillation.yaml
    python training/train_distillation.py --config configs/distillation.yaml --experiment B
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
from utils.io import load_config, merge_overrides, save_json  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("train_distillation")

# Ablation presets (spec section Q). Each disables the terms a row excludes.
EXPERIMENT_PRESETS = {
    "B": {"feature": False, "boundary": False, "relative": False, "roi": False},
    "C": {"boundary": False, "relative": False, "roi": False},
    "D": {"relative": False, "roi": False},
    "E": {},
}


def apply_experiment_preset(config: dict, tag: str) -> dict:
    """Disable the KD terms a given ablation row excludes."""
    if tag not in EXPERIMENT_PRESETS:
        raise ValueError(f"Unknown experiment '{tag}'. Available: {sorted(EXPERIMENT_PRESETS)}.")
    kd = config.setdefault("kd", {})
    for term, enabled in EXPERIMENT_PRESETS[tag].items():
        kd.setdefault(term, {})["enabled"] = enabled
    config.setdefault("experiment", {})["tag"] = tag
    name = config["experiment"].get("name", "student_distilled")
    config["experiment"]["name"] = f"{name}_{tag}" if not name.endswith(f"_{tag}") else name
    return config


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Train the distilled YOLO26n-Depth student.")
    ap.add_argument("--config", type=Path, default=Path("configs/distillation.yaml"))
    ap.add_argument("--experiment", choices=sorted(EXPERIMENT_PRESETS), default=None, help="Ablation preset B-E")
    ap.add_argument("--set", nargs="*", dest="overrides", default=None, help="Config overrides")
    ap.add_argument("--export-to", type=Path, default=Path("outputs/checkpoints/student_distilled_best.pt"))
    args = ap.parse_args(argv)

    try:
        config = merge_overrides(load_config(args.config), args.overrides)
        if args.experiment:
            config = apply_experiment_preset(config, args.experiment)

        data_yaml = resolve_data_yaml(config)
        run_dir = setup_experiment(config, args.config, extra={"role": "student_distilled"})

        from ultralytics import YOLO
        from ultralytics.models.yolo.depth import DepthTrainer

        from models.model_utils import select_device
        from models.projection import ProjectionBank
        from models.student import StudentDepthModel
        from models.teacher import TeacherDepthModel
        from training.kd_trainer import KDDepthModel, clear_kd_context, set_kd_context, strip_kd_wrapper

        train_cfg = config.get("train", {}) or {}
        device = select_device(train_cfg.get("device"))
        imgsz = int(train_cfg.get("imgsz", 640))

        # --- teacher: frozen, eval, no grad ---
        t_cfg = config.get("teacher", {}) or {}
        teacher = TeacherDepthModel(
            weights=t_cfg.get("weights", "yolo26x-depth.pt"),
            fallback=t_cfg.get("fallback_weights", "yolo26x-depth.pt"),
            device=device,
            space=config.get("teacher_space", "calibrated"),
        )
        LOG.info("Teacher: %s", teacher)
        if not teacher.is_frozen:
            raise RuntimeError("Teacher is not frozen; refusing to train (it would drift during distillation).")

        # --- projections sized from MEASURED feature widths ---
        kd_cfg = config.get("kd", {}) or {}
        feat_cfg = kd_cfg.get("feature", {}) or {}
        projections = None
        if feat_cfg.get("enabled"):
            s_cfg = config.get("student", {}) or {}
            probe = StudentDepthModel(
                weights=s_cfg.get("weights", "yolo26n-depth.pt"),
                fallback=s_cfg.get("fallback_weights", "yolo26n-depth.pt"),
                layers=feat_cfg.get("student_layers"),
                device=device,
            )
            t_ch = teacher.feature_channels(imgsz)
            s_ch = probe.feature_channels(imgsz)
            probe.close()
            LOG.info("Measured feature channels — teacher %s, student %s", t_ch, s_ch)
            projections = ProjectionBank(
                t_ch, s_ch, direction=feat_cfg.get("projection_direction", "teacher_to_student")
            ).to(device)
            LOG.info(
                "Projection bank: %s auxiliary parameters (excluded from the deployed student)",
                f"{projections.num_parameters:,}",
            )

        # --- obstacle detector for the ROI term ---
        detector = None
        roi_cfg = kd_cfg.get("roi", {}) or {}
        if roi_cfg.get("enabled"):
            detector = YOLO(roi_cfg.get("detector_weights", "yolo26n.pt"))
            LOG.info("ROI term will use detector %s (all classes -> obstacle)", roi_cfg.get("detector_weights"))

        # --- inject the KD criterion via the module-level DepthModel subclass ---
        set_kd_context(config, teacher, projections, detector)
        kd_model_cls = KDDepthModel

        class KDDepthTrainer(DepthTrainer):
            """DepthTrainer that builds the KD-aware model and trains the projections too."""

            def get_model(self, cfg=None, weights=None, verbose=True):
                """Return a KD-aware DepthModel with the student's weights loaded."""
                model = kd_model_cls(cfg, ch=self.data.get("channels", 3), nc=self.data["nc"], verbose=verbose)
                if weights:
                    model.load(weights)
                # Guard the audit finding: checkpoints load with requires_grad=False.
                for p in model.parameters():
                    p.requires_grad_(True)
                return model

            def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
                """Include the projection bank's parameters in the optimizer.

                The projections are auxiliary and never deployed, but they MUST be
                optimized: a frozen random projection would make the feature term
                measure distance to noise.
                """
                optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
                if projections is not None:
                    optimizer.add_param_group({"params": list(projections.parameters()), "weight_decay": 0.0})
                    LOG.info(
                        "Added %d projection parameter tensors to the optimizer.", len(list(projections.parameters()))
                    )
                return optimizer

        s_cfg = config.get("student", {}) or {}
        overrides = build_ultralytics_args(config, data_yaml)
        overrides["model"] = s_cfg.get("weights", "yolo26n-depth.pt")
        overrides["task"] = "depth"

        LOG.info("Starting KD training: %s", {k: overrides[k] for k in sorted(overrides)})
        trainer = KDDepthTrainer(overrides=overrides)
        trainer.train()

        actual_dir = Path(trainer.save_dir)
        metrics = dict(getattr(trainer, "metrics", {}) or {})
        save_json(
            {
                "role": "student_distilled",
                "experiment_tag": config.get("experiment", {}).get("tag"),
                "kd_terms": sorted(k for k, v in kd_cfg.items() if isinstance(v, dict) and v.get("enabled")),
                "teacher": teacher.info(),
                "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                "save_dir": str(actual_dir),
            },
            run_dir / "metrics.json",
        )

        # Strip the KD wrapper so the shipped student is a stock DepthModel.
        for ckpt in (actual_dir / "weights" / "best.pt", actual_dir / "weights" / "last.pt"):
            strip_kd_wrapper(ckpt)

        clear_kd_context()
        teacher.close()
        export_best_checkpoint(actual_dir, args.export_to)
        LOG.info("Distillation complete. Next: python evaluation/evaluate_depth.py --model %s", args.export_to)
    except FileNotFoundError as e:
        LOG.error("%s", e)
        return 1
    except (RuntimeError, ValueError, KeyError, ImportError) as e:
        LOG.error("Distillation training failed (%s): %s", type(e).__name__, e)
        import traceback

        LOG.debug("%s", traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
