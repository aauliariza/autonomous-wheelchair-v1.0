"""Dataset preparation and verification tests (spec section AU)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str, relpath: str):
    """Import a CLI script by path (they are scripts, not package modules)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load("prepare_sunrgbd", "datasets/scripts/prepare_sunrgbd.py")
convert = _load("convert_obstacle", "datasets/scripts/convert_to_obstacle_dataset.py")


class TestSunRGBDDepthDecode:
    """The SUN RGB-D bit-rotation decode must be exact."""

    @pytest.mark.parametrize("mm", [1, 500, 2345, 9999, 10000])
    def test_round_trip(self, mm: int) -> None:
        """Encoding then decoding a millimetre value returns the same metres."""
        encoded = np.array([[((mm << 3) & 0xFFFF) | (mm >> 13)]], dtype=np.uint16)
        decoded = prepare.decode_sunrgbd_depth(encoded)[0, 0]
        assert decoded == pytest.approx(mm / 1000.0, abs=1e-6)

    def test_zero_stays_invalid(self) -> None:
        """Code 0 means 'no return' and must decode to exactly 0."""
        assert prepare.decode_sunrgbd_depth(np.zeros((4, 4), dtype=np.uint16)).max() == 0.0

    def test_clipped_at_sensor_limit(self) -> None:
        """Depth is clipped at 10 m, beyond which SUN RGB-D sensors are unreliable."""
        out = prepare.decode_sunrgbd_depth(np.full((4, 4), 0xFFFF, dtype=np.uint16))
        assert out.max() <= prepare.MAX_DEPTH_M

    def test_output_is_finite(self) -> None:
        """No NaN or Inf may survive the decode."""
        rng = np.random.default_rng(0)
        raw = rng.integers(0, 65535, (32, 32)).astype(np.uint16)
        assert np.isfinite(prepare.decode_sunrgbd_depth(raw)).all()


class TestObstacleConversion:
    """Every annotation must collapse to class 0 with degenerate boxes dropped."""

    @pytest.mark.parametrize(
        "box,valid",
        [
            ((0.5, 0.5, 0.4, 0.4), True),
            ((0.5, 0.5, 0.0, 0.4), False),  # zero width
            ((0.5, 0.5, 0.4, 0.0), False),  # zero height
            ((1.5, 0.5, 0.4, 0.4), False),  # centre outside the frame
            ((0.5, 0.5, 1.4, 0.4), False),  # wider than the frame
            ((0.5, 0.5, 0.001, 0.001), False),  # sub-pixel
        ],
    )
    def test_box_validity(self, box, valid) -> None:
        """Invalid geometry is rejected, valid geometry is kept."""
        assert convert.valid_yolo_box(*box, 200, 200, 4.0) is valid

    def test_clamp01(self) -> None:
        """Normalized coordinates are clamped into [0, 1]."""
        assert convert.clamp01(-0.5) == 0.0
        assert convert.clamp01(1.5) == 1.0
        assert convert.clamp01(0.25) == 0.25

    def test_single_class_constants(self) -> None:
        """The class id and name are fixed by spec section A."""
        assert convert.OBSTACLE_CLASS_ID == 0
        assert convert.OBSTACLE_CLASS_NAME == "obstacle"

    def test_scaffold_creates_empty_structure(self, tmp_path: Path) -> None:
        """Scaffolding creates directories but fabricates no labels."""
        counts = convert.scaffold(tmp_path / "obs", ["train", "val"])
        assert counts == {"train": 0, "val": 0}
        for split in ("train", "val"):
            assert (tmp_path / "obs" / "images" / split).is_dir()
            assert (tmp_path / "obs" / "labels" / split).is_dir()
            assert not list((tmp_path / "obs" / "labels" / split).glob("*.txt"))
