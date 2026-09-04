"""Optuna tuning contract tests (spec section P)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
