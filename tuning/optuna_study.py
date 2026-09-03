#!/usr/bin/env python3
"""Optuna hyperparameter search over the KD configuration (spec section P).

The optimizer library is OPTUNA (a TPE-based hyperparameter search framework).

Trials are stored in SQLite, so an interrupted study RESUMES from its completed
trials rather than restarting -- essential when a search spans days of GPU time.

Each trial trains a short proxy run (``trial.epochs``, optionally on a fraction of
the data), evaluates it in MODE 1 (metric, ``align=none``), measures latency, and
scores it with the composite objective. The winning configuration is written to
YAML and should then be retrained at full length.

Usage:
    python tuning/optuna_study.py --config configs/optuna.yaml
    python tuning/optuna_study.py --config configs/optuna.yaml --n-trials 100   # resumes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tuning.objective import CompositeObjective, build_trial_config, suggest  # noqa: E402
from utils.io import load_config, save_json, save_yaml  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.seed import seed_everything  # noqa: E402

LOG = get_logger("optuna_study")


def build_sampler(cfg: dict[str, Any], seed: int):
    """Construct the configured sampler."""
    import optuna

    name = (cfg.get("name") or "TPESampler").lower()
    if name.startswith("random"):
        return optuna.samplers.RandomSampler(seed=seed)
    if name.startswith("cma"):
        return optuna.samplers.CmaEsSampler(seed=seed)
    return optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=int(cfg.get("n_startup_trials", 10)),
        multivariate=bool(cfg.get("multivariate", True)),
    )


def build_pruner(cfg: dict[str, Any]):
    """Construct the configured pruner."""
    import optuna

    name = (cfg.get("name") or "MedianPruner").lower()
    if name.startswith("nop"):
        return optuna.pruners.NopPruner()
    if name.startswith("hyperband"):
        return optuna.pruners.HyperbandPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=int(cfg.get("n_startup_trials", 5)),
        n_warmup_steps=int(cfg.get("n_warmup_steps", 5)),
        interval_steps=int(cfg.get("interval_steps", 1)),
    )


def run_trial(trial, config: dict[str, Any], objective: CompositeObjective, work_dir: Path) -> float:
    """Train, evaluate and score one trial."""
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.models.yolo.depth import DepthTrainer

    from evaluation.metrics import DepthEvaluator
    from models.model_utils import measure_latency, select_device
    from models.projection import ProjectionBank
    from models.student import StudentDepthModel
    from models.teacher import TeacherDepthModel
    from training.common import build_ultralytics_args, resolve_data_yaml
    from training.kd_trainer import KDDepthModel, clear_kd_context, set_kd_context

    space = config.get("search_space", {}) or {}
    params = {name: suggest(trial, name, spec) for name, spec in space.items()}
    LOG.info("Trial %d parameters: %s", trial.number, params)

    trial_cfg = config.get("trial", {}) or {}
    base = load_config(trial_cfg.get("base_config", "configs/distillation.yaml"))
    cfg = build_trial_config(base, params, trial_cfg)

    run_name = f"trial_{trial.number:04d}"
    cfg.setdefault("experiment", {})["name"] = run_name
    cfg["experiment"]["output_dir"] = str(work_dir)

    data_yaml = resolve_data_yaml(cfg)
    device = select_device(cfg.get("train", {}).get("device"))
    imgsz = int(cfg.get("train", {}).get("imgsz", 640))

    teacher = TeacherDepthModel(
        weights=cfg.get("teacher", {}).get("weights", "yolo26x-depth.pt"),
        fallback=cfg.get("teacher", {}).get("fallback_weights", "yolo26x-depth.pt"),
        device=device,
        space=cfg.get("teacher_space", "calibrated"),
    )

    projections = None
    feat = (cfg.get("kd", {}) or {}).get("feature", {}) or {}
    if feat.get("enabled"):
        probe = StudentDepthModel(weights=cfg.get("student", {}).get("weights", "yolo26n-depth.pt"), device=device)
        projections = ProjectionBank(
            teacher.feature_channels(imgsz),
            probe.feature_channels(imgsz),
            direction=feat.get("projection_direction", "teacher_to_student"),
        ).to(device)
        probe.close()

    detector = None
    roi = (cfg.get("kd", {}) or {}).get("roi", {}) or {}
    if roi.get("enabled"):
        detector = YOLO(roi.get("detector_weights", "yolo26n.pt"))

    set_kd_context(cfg, teacher, projections, detector)

    class TrialTrainer(DepthTrainer):
        """DepthTrainer that builds the KD-aware model for this trial."""

        def get_model(self, cfg_=None, weights=None, verbose=True):
            """Return the KD model with gradients explicitly enabled."""
            model = KDDepthModel(cfg_, ch=self.data.get("channels", 3), nc=self.data["nc"], verbose=verbose)
            if weights:
                model.load(weights)
            for p in model.parameters():
                p.requires_grad_(True)
            return model

        def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
            """Include the trial's projection parameters in the optimizer."""
            opt = super().build_optimizer(model, name, lr, momentum, decay, iterations)
            if projections is not None:
                opt.add_param_group({"params": list(projections.parameters()), "weight_decay": 0.0})
            return opt

    try:
        overrides = build_ultralytics_args(cfg, data_yaml)
        overrides["model"] = cfg.get("student", {}).get("weights", "yolo26n-depth.pt")
        overrides["task"] = "depth"
        trainer = TrialTrainer(overrides=overrides)
        trainer.train()

        best = Path(trainer.save_dir) / "weights" / "best.pt"
        model = YOLO(str(best))

        # MODE 1 (metric) evaluation — an aligned score would ignore absolute scale.
        import cv2

        from evaluation.evaluate_depth import load_pairs
        from utils.io import load_yaml

        data_meta = load_yaml(data_yaml)
        scale = float(data_meta.get("depth_scale", 1000))
        evaluator = DepthEvaluator(max_depth=float(data_meta.get("max_depth", 10.0)))

        for img_path, dep_path in load_pairs(Path(data_yaml), "val"):
            raw = cv2.imread(str(dep_path), cv2.IMREAD_ANYDEPTH)
            if raw is None:
                continue
            gt = raw.astype(np.float32) / scale
            pd = model.predict(str(img_path), imgsz=imgsz, device=str(device), verbose=False)[0].depth.data
            pred = np.squeeze(pd.detach().cpu().numpy() if hasattr(pd, "detach") else np.asarray(pd))
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            evaluator.update(pred, gt)

        results = evaluator.compute()
        metrics = results["metric"]

        obj_cfg = config.get("objective", {}) or {}
        latency = measure_latency(
            model.model,
            imgsz,
            device,
            runs=int(obj_cfg.get("latency_runs", 50)),
            warmup=int(obj_cfg.get("latency_warmup", 10)),
        )

        score = objective.score(metrics, latency["mean_ms"])

        for k, v in metrics.items():
            trial.set_user_attr(f"metric_{k}", float(v))
        trial.set_user_attr("latency_mean_ms", latency["mean_ms"])
        trial.set_user_attr("aligned_delta1", results["aligned"]["delta1"])
        trial.set_user_attr("checkpoint", str(best))

        LOG.info(
            "Trial %d: score=%.4f | delta1=%.4f abs_rel=%.4f rmse=%.4f | latency=%.1f ms",
            trial.number,
            score,
            metrics["delta1"],
            metrics["abs_rel"],
            metrics["rmse"],
            latency["mean_ms"],
        )
        return score
    finally:
        clear_kd_context()
        teacher.close()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Run the Optuna hyperparameter search.")
    ap.add_argument("--config", type=Path, default=Path("configs/optuna.yaml"))
    ap.add_argument("--n-trials", type=int, default=None, help="Overrides study.n_trials")
    ap.add_argument("--work-dir", type=Path, default=Path("outputs/optuna"))
    args = ap.parse_args(argv)

    try:
        import optuna
    except ImportError:
        LOG.error("Optuna is not installed. Recovery: pip install optuna  (it is in requirements.txt)")
        return 1

    try:
        config = load_config(args.config)
        study_cfg = config.get("study", {}) or {}
        seed = int(study_cfg.get("seed", 42))
        seed_everything(seed)

        storage = study_cfg.get("storage")
        if storage and storage.startswith("sqlite:///"):
            Path(storage.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)

        objective = CompositeObjective(config.get("objective", {}))
        LOG.info("Composite objective: %s", objective.describe())

        study = optuna.create_study(
            study_name=study_cfg.get("name", "kd_depth_search"),
            storage=storage,
            direction=study_cfg.get("direction", "maximize"),
            sampler=build_sampler(config.get("sampler", {}), seed),
            pruner=build_pruner(config.get("pruner", {})),
            load_if_exists=bool(study_cfg.get("load_if_exists", True)),
        )

        done = len([t for t in study.trials if t.state.is_finished()])
        if done:
            LOG.info("Resuming study '%s' with %d completed trial(s).", study.study_name, done)

        n_trials = args.n_trials or int(study_cfg.get("n_trials", 50))
        args.work_dir.mkdir(parents=True, exist_ok=True)

        study.optimize(
            lambda t: run_trial(t, config, objective, args.work_dir),
            n_trials=n_trials,
            timeout=study_cfg.get("timeout_s"),
            # A crashed trial must not abort a multi-day study.
            catch=(RuntimeError, ValueError, OSError),
        )

        if study.best_trial is None:
            LOG.warning("No trial completed successfully; nothing to report.")
            return 1

        best = study.best_trial
        LOG.info("Best trial %d scored %.4f", best.number, best.value)
        LOG.info("Best parameters: %s", best.params)

        base = load_config((config.get("trial", {}) or {}).get("base_config", "configs/distillation.yaml"))
        best_cfg = build_trial_config(base, best.params, {})
        best_cfg.setdefault("experiment", {})["name"] = "student_distilled_tuned"
        save_yaml(best_cfg, args.work_dir / "best_config.yaml")
        save_json(
            {
                "study": study.study_name,
                "best_trial": best.number,
                "best_score": best.value,
                "best_params": best.params,
                "user_attrs": best.user_attrs,
                "objective": objective.describe(),
                "n_trials": len(study.trials),
            },
            args.work_dir / "best_trial.json",
        )
        LOG.info("Best config written to %s", args.work_dir / "best_config.yaml")
        LOG.info(
            "Retrain at full length: python training/train_distillation.py --config %s",
            args.work_dir / "best_config.yaml",
        )
    except FileNotFoundError as e:
        LOG.error("%s", e)
        return 1
    except (RuntimeError, ValueError) as e:
        LOG.error("Study failed (%s): %s", type(e).__name__, e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
