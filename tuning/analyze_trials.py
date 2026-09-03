#!/usr/bin/env python3
"""Analyze a completed Optuna study (spec section P).

Reports the best trial, parameter importances, and the accuracy/latency
trade-off, so the chosen configuration can be justified in a paper rather than
merely asserted.

Usage:
    python tuning/analyze_trials.py --storage sqlite:///outputs/optuna/kd_depth_search.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import save_json  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOG = get_logger("analyze_trials")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Analyze an Optuna study.")
    ap.add_argument("--storage", default="sqlite:///outputs/optuna/kd_depth_search.db")
    ap.add_argument("--study-name", default="kd_depth_search")
    ap.add_argument("--output", type=Path, default=Path("outputs/optuna/analysis.json"))
    ap.add_argument("--top", type=int, default=10, help="Trials to list")
    ap.add_argument("--plots", action="store_true", help="Write optimization-history and importance plots")
    args = ap.parse_args(argv)

    try:
        import optuna
    except ImportError:
        LOG.error("Optuna is not installed. Recovery: pip install optuna")
        return 1

    try:
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    except (KeyError, RuntimeError, ValueError) as e:
        LOG.error("Could not load study '%s' from %s (%s: %s)", args.study_name, args.storage, type(e).__name__, e)
        LOG.error("Recovery: run tuning/optuna_study.py first, or check --study-name and --storage.")
        return 1

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        LOG.warning("Study '%s' has no completed trials; nothing to analyze.", args.study_name)
        return 1

    LOG.info("Study '%s': %d trials (%d complete)", args.study_name, len(study.trials), len(complete))

    ranked = sorted(complete, key=lambda t: t.value, reverse=True)[: args.top]
    print()
    print(f"{'rank':<6}{'trial':<8}{'score':>9}{'delta1':>9}{'abs_rel':>10}{'rmse':>9}{'latency ms':>12}")
    print("-" * 63)
    for i, t in enumerate(ranked, 1):
        a = t.user_attrs
        print(
            f"{i:<6}{t.number:<8}{t.value:>9.4f}"
            f"{a.get('metric_delta1', float('nan')):>9.4f}"
            f"{a.get('metric_abs_rel', float('nan')):>10.4f}"
            f"{a.get('metric_rmse', float('nan')):>9.4f}"
            f"{a.get('latency_mean_ms', float('nan')):>12.2f}"
        )
    print("-" * 63)

    importances: dict[str, float] = {}
    method = "none"
    # Importance needs enough completed trials to mean anything.
    if len(complete) < 4:
        LOG.info("Only %d completed trial(s); parameter importance needs at least 4.", len(complete))
    else:
        # fANOVA is the default but requires scikit-learn, which is optional here.
        # PED-ANOVA is pure-Optuna, so importance still works on a minimal install
        # instead of aborting the whole analysis on a missing optional dependency.
        for name, factory in (
            ("fANOVA", lambda: optuna.importance.FanovaImportanceEvaluator()),
            ("PED-ANOVA", lambda: optuna.importance.PedAnovaImportanceEvaluator()),
        ):
            try:
                raw = optuna.importance.get_param_importances(study, evaluator=factory())
                importances = {k: float(v) for k, v in raw.items()}
                method = name
                break
            except (ImportError, ValueError, RuntimeError) as e:
                LOG.debug("Importance via %s unavailable (%s: %s)", name, type(e).__name__, e)

        if importances:
            print()
            print(f"parameter importance ({method}):")
            for k, v in sorted(importances.items(), key=lambda x: -x[1]):
                print(f"  {k:<24}{v:>8.4f}  {'#' * int(v * 50)}")
        else:
            LOG.warning(
                "Parameter importance could not be computed. Recovery: pip install scikit-learn "
                "to enable the fANOVA evaluator."
            )

    best = study.best_trial
    print()
    print(f"best trial : {best.number}  score {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:<24}{v}")

    if args.plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            out_dir = args.output.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, fn in (
                ("optimization_history", optuna.visualization.matplotlib.plot_optimization_history),
                ("param_importances", optuna.visualization.matplotlib.plot_param_importances),
                ("parallel_coordinate", optuna.visualization.matplotlib.plot_parallel_coordinate),
            ):
                try:
                    ax = fn(study)
                    ax.figure.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
                    LOG.info("Plot written: %s", out_dir / f"{name}.png")
                except (ValueError, RuntimeError, ImportError) as e:
                    LOG.warning("Plot '%s' skipped (%s)", name, e)
        except ImportError:
            LOG.warning("matplotlib is unavailable; plots skipped.")

    save_json(
        {
            "study": args.study_name,
            "n_trials": len(study.trials),
            "n_complete": len(complete),
            "best_trial": best.number,
            "best_score": best.value,
            "best_params": best.params,
            "best_user_attrs": best.user_attrs,
            "param_importances": importances,
            "importance_method": method,
            "top_trials": [{"trial": t.number, "score": t.value, "params": t.params, **t.user_attrs} for t in ranked],
        },
        args.output,
    )
    LOG.info("Analysis written to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
