"""Obstacle distance estimation tests (spec section AU items 4, 9)."""

from __future__ import annotations

import numpy as np
import pytest

from calibration.intrinsics import CameraIntrinsics, IntrinsicsError
from navigation.distance import MAD_TO_SIGMA, DistanceEstimator, robust_depth_statistics


@pytest.fixture
def estimator(nav_config) -> DistanceEstimator:
    """An estimator built from the shipped navigation config."""
    return DistanceEstimator(nav_config)


@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    """Nominal 640x480 pinhole intrinsics (60 deg HFOV)."""
    return CameraIntrinsics(554.26, 554.26, 320.0, 240.0, 640, 480)


class TestRobustStatistics:
    """Robust reduction must reject every invalid case and resist outliers."""

    def test_uniform_patch(self) -> None:
        """A uniform patch reduces to its value."""
        s = robust_depth_statistics(np.full((20, 20), 2.5))
        assert s["value"] == pytest.approx(2.5)
        assert s["valid_ratio"] == pytest.approx(1.0)

    def test_rejects_all_invalid_kinds(self) -> None:
        """0, NaN, Inf and negative values are excluded from the statistic."""
        patch = np.array([2.0, 0.0, np.nan, np.inf, -1.0, 2.0, 2.0, 2.0])
        s = robust_depth_statistics(patch)
        assert s["value"] == pytest.approx(2.0)
        assert s["num_valid"] == 4

    def test_median_resists_outliers(self) -> None:
        """A minority of extreme values must not move the median."""
        patch = np.full((20, 20), 2.0)
        patch[:6, :] = 9.5  # 30% far outliers
        assert robust_depth_statistics(patch)["value"] == pytest.approx(2.0)

    def test_empty_and_all_invalid(self) -> None:
        """Empty and fully-invalid patches return None, not a number."""
        assert robust_depth_statistics(np.array([]))["value"] is None
        assert robust_depth_statistics(np.zeros((8, 8)))["value"] is None
        assert robust_depth_statistics(np.full((8, 8), np.nan))["value"] is None

    def test_out_of_range_excluded(self) -> None:
        """Values outside the sensor range are not counted."""
        patch = np.array([0.05, 2.0, 2.0, 50.0])
        s = robust_depth_statistics(patch, min_depth_m=0.1, max_depth_m=10.0)
        assert s["num_valid"] == 2

    def test_mad_scaling(self) -> None:
        """Dispersion is MAD * 1.4826 so it is comparable to a sigma."""
        patch = np.array([1.0, 2.0, 3.0, 2.0, 2.0])
        assert robust_depth_statistics(patch, clip_lower_percentile=0, clip_upper_percentile=100)[
            "std"
        ] == pytest.approx(0.0 * MAD_TO_SIGMA, abs=1e-6)

    @pytest.mark.parametrize("stat", ["median", "mean", "percentile"])
    def test_statistic_choices(self, stat) -> None:
        """Every configurable statistic works."""
        assert robust_depth_statistics(np.full((10, 10), 2.0), statistic=stat)["value"] == pytest.approx(2.0)

    def test_unknown_statistic_rejected(self) -> None:
        """An unknown statistic is a config error."""
        with pytest.raises(ValueError, match="statistic must be"):
            robust_depth_statistics(np.ones((4, 4)), statistic="mode")

    def test_percentile_is_conservative(self) -> None:
        """A low percentile biases toward the nearer surface."""
        patch = np.concatenate([np.full(50, 1.0), np.full(50, 5.0)])
        near = robust_depth_statistics(patch, statistic="percentile", percentile=25.0)["value"]
        assert near < robust_depth_statistics(patch, statistic="median")["value"] + 1e-9


class TestDistanceEstimator:
    """End-to-end distance estimation from a depth map and a box."""

    def test_uniform_scene(self, estimator, flat_depth) -> None:
        """A uniform 3 m scene yields a 3 m obstacle distance."""
        r = estimator.estimate(flat_depth, (100, 100, 300, 300))
        assert r.valid and r.distance_m == pytest.approx(3.0)
        assert r.valid_ratio == pytest.approx(1.0)

    def test_uses_inner_roi_not_the_full_box(self, estimator) -> None:
        """Background at the box edge must not bias the distance."""
        depth = np.full((200, 200), 10.0)  # far background everywhere
        depth[60:140, 60:140] = 1.5  # near object in the box centre
        r = estimator.estimate(depth, (50, 50, 150, 150))
        assert r.valid and r.distance_m == pytest.approx(1.5, abs=0.01)

    def test_invalid_when_too_few_valid_pixels(self, estimator) -> None:
        """Below min_valid_ratio the result is INVALID, not a guess."""
        depth = np.zeros((200, 200))
        depth[95:105, 95:105] = 2.0
        r = estimator.estimate(depth, (0, 0, 200, 200))
        assert not r.valid and "valid depth ratio" in r.reason

    def test_all_invalid_depth(self, estimator) -> None:
        """A fully invalid patch is reported with a reason."""
        r = estimator.estimate(np.zeros((100, 100)), (10, 10, 90, 90))
        assert not r.valid and r.distance_m is None and r.reason

    def test_depth_map_smaller_than_image(self, estimator) -> None:
        """Boxes rescale when depth is predicted at input/4."""
        depth = np.full((120, 160), 2.0)
        r = estimator.estimate(depth, (200, 200, 400, 400), image_size=(480, 640))
        assert r.valid and r.distance_m == pytest.approx(2.0)

    def test_box_outside_the_depth_map(self, estimator, flat_depth) -> None:
        """A box entirely off-frame yields an invalid result, not a crash."""
        r = estimator.estimate(flat_depth, (5000, 5000, 6000, 6000))
        assert not r.valid

    def test_euclidean_requires_intrinsics(self, nav_config, flat_depth) -> None:
        """Euclidean mode without intrinsics fails loudly rather than substituting Z."""
        cfg = {**nav_config, "safety": {**nav_config["safety"], "distance_mode": "euclidean"}}
        r = DistanceEstimator(cfg, intrinsics=None).estimate(flat_depth, (100, 100, 300, 300))
        assert not r.valid and "intrinsics" in r.reason

    def test_euclidean_exceeds_axial_off_axis(self, nav_config, flat_depth, intrinsics) -> None:
        """Off-axis slant range must exceed axial depth (spec section T)."""
        est = DistanceEstimator(nav_config, intrinsics=intrinsics)
        r = est.estimate(flat_depth, (0, 0, 100, 100))  # top-left corner
        assert r.valid
        assert r.euclidean_distance_m > r.distance_m

    def test_euclidean_equals_axial_at_principal_point(self, nav_config, flat_depth, intrinsics) -> None:
        """At the principal point the two measures coincide."""
        est = DistanceEstimator(nav_config, intrinsics=intrinsics)
        r = est.estimate(flat_depth, (300, 220, 340, 260))
        assert r.euclidean_distance_m == pytest.approx(r.distance_m, rel=0.01)

    def test_invalid_distance_mode_rejected(self, nav_config) -> None:
        """Only 'axial' and 'euclidean' are valid."""
        cfg = {**nav_config, "safety": {**nav_config["safety"], "distance_mode": "manhattan"}}
        with pytest.raises(ValueError, match="distance_mode"):
            DistanceEstimator(cfg)


class TestConfidenceScore:
    """The fused confidence score (spec section Y)."""

    def test_bounded_and_monotonic(self, estimator, flat_depth) -> None:
        """The score stays in [0,1] and rises with detection confidence."""
        d = estimator.estimate(flat_depth, (100, 100, 300, 300))
        low = estimator.confidence_score(0.3, d)
        high = estimator.confidence_score(0.9, d)
        assert 0.0 <= low <= high <= 1.0

    def test_penalises_low_valid_ratio(self, estimator) -> None:
        """Sparse valid depth reduces confidence."""
        good = estimator.estimate(np.full((100, 100), 2.0), (10, 10, 90, 90))
        patchy = np.full((100, 100), 2.0)
        patchy[::2, :] = 0.0
        poor = estimator.estimate(patchy, (10, 10, 90, 90))
        assert estimator.confidence_score(0.9, poor) < estimator.confidence_score(0.9, good)

    def test_temporal_factor_applies(self, estimator, flat_depth) -> None:
        """Temporal inconsistency lowers the score."""
        d = estimator.estimate(flat_depth, (100, 100, 300, 300))
        assert estimator.confidence_score(0.9, d, 0.2) < estimator.confidence_score(0.9, d, 1.0)


class TestIntrinsics:
    """Camera geometry (spec sections T, U)."""

    def test_pixel_to_3d_round_trip(self, intrinsics) -> None:
        """Back-projection matches the closed-form Euclidean distance."""
        x, y, z = intrinsics.pixel_to_3d(0, 0, 1.0)
        assert intrinsics.euclidean_from_axial(1.0, 0, 0) == pytest.approx(np.sqrt(x * x + y * y + z * z))

    def test_corner_error_magnitude(self, intrinsics) -> None:
        """The documented 23.3% corner discrepancy is reproduced."""
        assert intrinsics.euclidean_from_axial(1.0, 0, 0) == pytest.approx(1.2332, abs=1e-3)

    def test_fov_recovered(self, intrinsics) -> None:
        """The 60 deg HFOV used to build the intrinsics is recovered."""
        assert intrinsics.horizontal_fov_deg() == pytest.approx(60.0, abs=0.1)

    def test_scaling_is_linear(self, intrinsics) -> None:
        """Intrinsics scale linearly with resolution."""
        scaled = intrinsics.scaled(1280, 960)
        assert scaled.fx == pytest.approx(intrinsics.fx * 2)
        assert scaled.cx == pytest.approx(intrinsics.cx * 2)

    def test_rejects_non_positive_focal_length(self) -> None:
        """A non-positive focal length is physically impossible."""
        with pytest.raises(IntrinsicsError):
            CameraIntrinsics(0.0, 554.0, 320.0, 240.0, 640, 480)

    def test_uncalibrated_flag_defaults_false(self, intrinsics) -> None:
        """Hand-constructed intrinsics are never marked calibrated."""
        assert intrinsics.calibrated is False
