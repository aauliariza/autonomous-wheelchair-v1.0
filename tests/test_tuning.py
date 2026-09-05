"""Optuna tuning contract tests (spec section P)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


class TestTrialTrainerContract:
    """The trial trainer must match Ultralytics' own call convention."""

    def test_get_model_accepts_cfg_by_keyword(self) -> None:
        """Ultralytics calls ``get_model(cfg=..., weights=..., verbose=...)``.

        The parameter had once been renamed ``cfg_`` to avoid shadowing the
        enclosing trial config, which made every trial die with
        ``TypeError: unexpected keyword argument 'cfg'`` before training a single
        batch — the tuning step was unusable while still importing cleanly.
        """
        from tuning.optuna_study import build_trial_trainer

        params = list(inspect.signature(build_trial_trainer(None).get_model).parameters)
        assert params[:4] == ["self", "cfg", "weights", "verbose"], (
            f"get_model signature is {params}; Ultralytics passes cfg, weights and verbose by keyword."
        )

    def test_signature_matches_ultralytics_base(self) -> None:
        """Guard against drift if Ultralytics renames its own parameters."""
        pytest.importorskip("ultralytics")
        from ultralytics.engine.trainer import BaseTrainer

        from tuning.optuna_study import build_trial_trainer

        base = list(inspect.signature(BaseTrainer.get_model).parameters)
        ours = list(inspect.signature(build_trial_trainer(None).get_model).parameters)
        assert ours[:3] == base[:3], f"override {ours} diverges from BaseTrainer {base}"


class TestAnalyzeTraining:
    """The results.csv diagnosis used to decide how to retrain the teacher."""

    @staticmethod
    def _csv(tmp_path, rows: str):
        p = tmp_path / "results.csv"
        p.write_text("epoch,train/dlog_loss,val/dlog_loss,metrics/delta1,metrics/rmse\n" + rows)
        return p

    def test_finds_best_epoch_by_delta1(self, tmp_path) -> None:
        """best.pt corresponds to the highest delta1, not the last epoch."""
        from scripts.analyze_training import analyze, load

        p = self._csv(tmp_path, "1,0.3,0.4,0.50,0.9\n2,0.2,0.3,0.70,0.7\n3,0.1,0.4,0.60,0.8\n")
        a = analyze(load(p))
        assert a["best"]["epoch"] == 2
        assert a["wasted"] == 1

    def test_detects_widening_train_val_gap(self, tmp_path) -> None:
        """A val/train ratio that grows while train loss falls is overfitting."""
        from scripts.analyze_training import analyze, load

        p = self._csv(tmp_path, "1,0.30,0.39,0.55,0.9\n2,0.20,0.35,0.70,0.7\n3,0.10,0.36,0.68,0.8\n")
        a = analyze(load(p))
        first, last = a["ratios"][0][1], a["ratios"][-1][1]
        assert last > first * 1.25, f"gap {first:.2f} -> {last:.2f} should read as overfitting"

    def test_missing_file_reports_recovery(self, tmp_path) -> None:
        """A wrong path fails with the command to fix it, not a traceback."""
        from scripts.analyze_training import load

        with pytest.raises(FileNotFoundError, match="results.csv not found"):
            load(tmp_path / "absent.csv")

    def test_non_depth_csv_is_rejected(self, tmp_path) -> None:
        """A detection results.csv has no delta1 and must not be scored silently."""
        from scripts.analyze_training import analyze, load

        p = tmp_path / "results.csv"
        p.write_text("epoch,train/box_loss\n1,0.5\n2,0.4\n")
        with pytest.raises(ValueError, match="metrics/delta1"):
            analyze(load(p))
