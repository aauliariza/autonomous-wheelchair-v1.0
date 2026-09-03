"""Shared pytest fixtures.

Tests run on SYNTHETIC tensors and images (spec section AU): the full SUN RGB-D
dataset and the model checkpoints are never required, so the suite runs in
seconds on any machine and in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def nav_config(repo_root: Path) -> dict:
    """The real navigation config, so tests exercise shipped defaults."""
    with open(repo_root / "configs" / "navigation.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def flat_depth() -> np.ndarray:
    """A 480x640 depth map at a uniform 3.0 m."""
    return np.full((480, 640), 3.0, dtype=np.float32)


@pytest.fixture
def depth_with_invalids() -> np.ndarray:
    """A depth map containing every invalid case: 0, NaN, Inf and negative."""
    d = np.full((100, 100), 2.0, dtype=np.float32)
    d[0:10, :] = 0.0
    d[10:20, :] = np.nan
    d[20:30, :] = np.inf
    d[30:40, :] = -1.0
    return d


@pytest.fixture
def student_pred() -> torch.Tensor:
    """A (2,1,40,40) positive depth prediction that requires grad."""
    torch.manual_seed(0)
    return (torch.rand(2, 1, 40, 40) * 3.0 + 0.5).requires_grad_(True)


@pytest.fixture
def teacher_pred() -> torch.Tensor:
    """A (2,1,40,40) positive depth target."""
    torch.manual_seed(1)
    return torch.rand(2, 1, 40, 40) * 3.0 + 0.5


@pytest.fixture
def gt_depth() -> torch.Tensor:
    """A (2,1,160,160) ground-truth map with an invalid band and NaN/Inf pixels."""
    torch.manual_seed(2)
    g = torch.rand(2, 1, 160, 160) * 3.0 + 0.5
    g[0, 0, :30, :] = 0.0
    g[0, 0, 35, 0] = float("nan")
    g[1, 0, 40, 5] = float("inf")
    return g
