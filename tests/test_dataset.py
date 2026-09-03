"""Dataset preparation and verification tests (spec section AU)."""

from __future__ import annotations

import importlib.util
import json
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
sunrgbd_obstacle = _load("convert_sunrgbd_obstacle", "datasets/scripts/convert_sunrgbd_obstacle.py")


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


class TestSunRGBDObstacleSceneKey:
    """Scene-key derivation must be immune to an unverified sequenceName prefix.

    The exact leading-path convention of SUN RGB-D's ``sequenceName`` field
    could not be confirmed without the source .mat file (rgbd.cs.princeton.edu
    is unreachable from this environment). Anchoring on the four KNOWN sensor
    root directory names sidesteps that ambiguity entirely.
    """

    @pytest.mark.parametrize(
        "sequence_name,expected",
        [
            ("/n/fs/sun3d/data/SUNRGBD/kv1/NYUdata/NYU0001", "kv1/NYUdata/NYU0001"),
            ("/n/fs/sun3d/data/SUNRGBD/kv2/kinect2data/000123", "kv2/kinect2data/000123"),
            ("/SUNRGBD/realsense/lg/2014_10_21", "realsense/lg/2014_10_21"),
            ("/xtion/sun3ddata/scene01/", "xtion/sun3ddata/scene01"),
            ("kv1/NYUdata/NYU0001", "kv1/NYUdata/NYU0001"),  # no leading slash at all
        ],
    )
    def test_recognized_sensor_roots(self, sequence_name: str, expected: str) -> None:
        """Any machine-specific prefix before a known sensor root is discarded."""
        assert sunrgbd_obstacle.scene_key_from_sequence_name(sequence_name) == expected

    def test_unrecognized_root_returns_none(self) -> None:
        """A sequenceName with none of the four known sensor roots is not guessed at."""
        assert sunrgbd_obstacle.scene_key_from_sequence_name("/some/other/dataset/scene") is None

    def test_windows_style_backslashes_normalized(self) -> None:
        """Backslash path separators are normalized before matching."""
        assert sunrgbd_obstacle.scene_key_from_sequence_name("C:\\data\\kv1\\NYUdata\\NYU0001") == "kv1/NYUdata/NYU0001"


class TestSunRGBDObstacleBoxValidity:
    """valid_yolo_box must reject degenerate/out-of-frame boxes (spec section C)."""

    @pytest.mark.parametrize(
        "box,valid",
        [
            ((0.5, 0.5, 0.4, 0.4), True),
            ((0.5, 0.5, 0.0, 0.4), False),
            ((0.5, 0.5, 0.005, 0.005), False),  # sub-pixel at 200x200
        ],
    )
    def test_box_validity(self, box, valid) -> None:
        """Matches the same policy as convert_to_obstacle_dataset.valid_yolo_box."""
        assert sunrgbd_obstacle.valid_yolo_box(*box, 200, 200, 4.0) is valid


class TestSunRGBDObstacleImageResolution:
    """Image resolution must try the direct path, then a documented fallback."""

    def test_direct_path_preferred(self, tmp_path: Path) -> None:
        """When rgbname matches exactly, no fallback is used."""
        img_dir = tmp_path / "kv1" / "sceneA" / "image"
        img_dir.mkdir(parents=True)
        (img_dir / "0001.jpg").write_bytes(b"fake")
        path, how = sunrgbd_obstacle.resolve_image_path(tmp_path, "kv1/sceneA", "0001.jpg")
        assert path == img_dir / "0001.jpg"
        assert how == "direct"

    def test_falls_back_to_single_file(self, tmp_path: Path) -> None:
        """A renamed/mismatched file is still found when it is the only one present."""
        img_dir = tmp_path / "kv1" / "sceneB" / "image"
        img_dir.mkdir(parents=True)
        (img_dir / "actual_name.jpg").write_bytes(b"fake")
        path, how = sunrgbd_obstacle.resolve_image_path(tmp_path, "kv1/sceneB", "expected_name.jpg")
        assert path == img_dir / "actual_name.jpg"
        assert "fallback" in how

    def test_missing_scene_returns_none(self, tmp_path: Path) -> None:
        """A scene absent from disk is reported, never guessed at."""
        path, how = sunrgbd_obstacle.resolve_image_path(tmp_path, "kv1/nonexistent", "x.jpg")
        assert path is None and how == "not found"


class TestSunRGBDObstacleSplitAssignment:
    """Split lookup must match exactly first, then fall back to a suffix search."""

    def test_exact_match(self) -> None:
        """An exact scene key resolves directly."""
        lookup = {"kv1/NYUdata/NYU0001": "train"}
        assert sunrgbd_obstacle.assign_split("kv1/NYUdata/NYU0001", lookup) == "train"

    def test_suffix_match(self) -> None:
        """A longer or shorter but consistent path still resolves."""
        lookup = {"NYUdata/NYU0001": "val"}
        assert sunrgbd_obstacle.assign_split("kv1/NYUdata/NYU0001", lookup) == "val"

    def test_no_match_returns_none(self) -> None:
        """An unrelated scene is not silently assigned to a split."""
        lookup = {"kv1/NYUdata/NYU0001": "train"}
        assert sunrgbd_obstacle.assign_split("kv2/kinect2data/000999", lookup) is None


class TestSunRGBDObstacleEndToEnd:
    """Full conversion against a synthetic SUNRGBDMeta2DBB_v2.mat fixture.

    The real file could not be downloaded (rgbd.cs.princeton.edu is unreachable
    from this environment), so this builds a fixture with the SAME nested
    struct-array layout, verified against two independent reference
    implementations (facebookresearch/votenet, charlesq34/frustum-pointnets)
    that consume the real file. See the module docstring in
    convert_sunrgbd_obstacle.py for the source citations.
    """

    @pytest.fixture
    def fixture_mat(self, tmp_path: Path):
        """Build a synthetic SUNRGBDMeta2DBB_v2.mat with known ground truth."""
        scipy_io = pytest.importorskip("scipy.io")

        def box_struct(boxes):
            dt = np.dtype([("gtBb2D", "O"), ("classname", "O")])
            arr = np.zeros(len(boxes), dtype=dt)
            for i, (x, y, w, h, cname) in enumerate(boxes):
                arr[i]["gtBb2D"] = np.array([x, y, w, h], dtype=np.float64)
                arr[i]["classname"] = cname
            return arr

        entries = [
            ("/n/fs/sun3d/data/SUNRGBD/kv1/NYUdata/NYU0001", "0001.jpg", [(10, 20, 100, 150, "chair")]),
            ("/n/fs/sun3d/data/SUNRGBD/kv2/kinect2data/000123", "0123.jpg", []),
        ]
        dt_top = np.dtype([("sequenceName", "O"), ("rgbname", "O"), ("groundtruth2DBB", "O")])
        top = np.zeros(len(entries), dtype=dt_top)
        for i, (seq, rgb, boxes) in enumerate(entries):
            top[i]["sequenceName"] = seq
            top[i]["rgbname"] = rgb
            top[i]["groundtruth2DBB"] = box_struct(boxes)

        meta_path = tmp_path / "SUNRGBDMeta2DBB_v2.mat"
        scipy_io.savemat(str(meta_path), {"SUNRGBDMeta2DBB": top})
        return meta_path

    @pytest.fixture
    def fixture_source(self, tmp_path: Path):
        """Build a matching --source tree with real, readable images."""
        cv2 = pytest.importorskip("cv2")
        rng = np.random.default_rng(0)

        source = tmp_path / "src"
        for scene, name in (("kv1/NYUdata/NYU0001", "0001.jpg"), ("kv2/kinect2data/000123", "0123.jpg")):
            img_dir = source / scene / "image"
            img_dir.mkdir(parents=True)
            cv2.imwrite(str(img_dir / name), rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
        return source

    @pytest.fixture
    def fixture_split(self, tmp_path: Path):
        """Both scenes assigned to train, per the official split JSON convention."""
        split_path = tmp_path / "split.json"
        split_path.write_text(json.dumps({"train": ["kv1/NYUdata/NYU0001", "kv2/kinect2data/000123"], "test": []}))
        return split_path

    def test_full_conversion(self, tmp_path, fixture_mat, fixture_source, fixture_split) -> None:
        """One real box and one zero-object scene both convert correctly."""
        pytest.importorskip("scipy.io")
        pytest.importorskip("cv2")

        output = tmp_path / "obstacle"
        split_lookup = sunrgbd_obstacle.load_split_file(fixture_split)
        stats = sunrgbd_obstacle.convert(fixture_source, fixture_mat, split_lookup, output, min_box_size=4.0)

        assert stats["train"]["images"] == 2
        assert stats["train"]["boxes"] == 1  # only the chair box; the second scene has none

        label = (output / "labels" / "train" / "kv1_NYUdata_NYU0001.txt").read_text().strip()
        cls_id, xc, yc, w, h = label.split()
        assert cls_id == "0"  # spec section A: every object collapses to class 0
        # x=10, y=20, w=100, h=150 on a 640x480 image
        assert float(xc) == pytest.approx((10 + 100 / 2) / 640, abs=1e-4)
        assert float(yc) == pytest.approx((20 + 150 / 2) / 480, abs=1e-4)
        assert float(w) == pytest.approx(100 / 640, abs=1e-4)
        assert float(h) == pytest.approx(150 / 480, abs=1e-4)

        # A scene with zero annotated objects gets an EMPTY label file, not a
        # missing one -- the image was still converted, just has no boxes.
        empty_label = output / "labels" / "train" / "kv2_kinect2data_000123.txt"
        assert empty_label.exists()
        assert empty_label.read_text().strip() == ""

    def test_degenerate_box_dropped(self, tmp_path, fixture_source, fixture_split) -> None:
        """A near-zero-size box never reaches the output label file."""
        scipy_io = pytest.importorskip("scipy.io")

        dt = np.dtype([("gtBb2D", "O"), ("classname", "O")])
        boxes = np.zeros(1, dtype=dt)
        boxes[0]["gtBb2D"] = np.array([5.0, 5.0, 1.0, 1.0])
        boxes[0]["classname"] = "sliver"
        dt_top = np.dtype([("sequenceName", "O"), ("rgbname", "O"), ("groundtruth2DBB", "O")])
        top = np.zeros(1, dtype=dt_top)
        top[0]["sequenceName"] = "/n/fs/sun3d/data/SUNRGBD/kv1/NYUdata/NYU0001"
        top[0]["rgbname"] = "0001.jpg"
        top[0]["groundtruth2DBB"] = boxes
        meta_path = tmp_path / "degenerate.mat"
        scipy_io.savemat(str(meta_path), {"SUNRGBDMeta2DBB": top})

        split_lookup = sunrgbd_obstacle.load_split_file(fixture_split)
        output = tmp_path / "obstacle2"
        stats = sunrgbd_obstacle.convert(fixture_source, meta_path, split_lookup, output, min_box_size=4.0)
        assert stats["train"]["boxes"] == 0
        assert stats["train"]["dropped_boxes"] == 1

    def test_missing_split_file_raises(self, tmp_path: Path) -> None:
        """A missing split file fails with a recovery command, not a stack trace."""
        with pytest.raises(sunrgbd_obstacle.ConversionError, match="Split file not found"):
            sunrgbd_obstacle.load_split_file(tmp_path / "nope.json")
