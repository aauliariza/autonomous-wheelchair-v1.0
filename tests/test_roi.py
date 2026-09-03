"""ROI computation tests (spec section AU items 5, 6)."""

from __future__ import annotations

import pytest

from navigation.roi import ROI, compute_bbox_inner_roi, compute_global_roi


class TestGlobalROI:
    """The 60% navigation ROI (spec section W)."""

    def test_default_is_central_60_percent(self) -> None:
        """Defaults give x in [0.20W, 0.80W] and full height."""
        roi = compute_global_roi(1000, 500)
        assert roi.as_tuple() == (200, 0, 800, 500)
        assert roi.width == 600

    @pytest.mark.parametrize("ratio,expected_width", [(0.5, 500), (0.6, 600), (1.0, 1000)])
    def test_configurable_width(self, ratio, expected_width) -> None:
        """The ROI width follows width_ratio."""
        assert compute_global_roi(1000, 500, width_ratio=ratio).width == expected_width

    def test_off_centre_roi(self) -> None:
        """x_center shifts the ROI without changing its width."""
        roi = compute_global_roi(1000, 500, width_ratio=0.6, x_center=0.3)
        assert roi.width == 600
        assert roi.x1 == 0  # clipped at the frame edge

    def test_partial_height(self) -> None:
        """height_ratio restricts the vertical extent."""
        roi = compute_global_roi(1000, 500, height_ratio=0.5)
        assert roi.height == 250

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_ratio_rejected(self, bad) -> None:
        """Out-of-range ratios are configuration errors."""
        with pytest.raises(ValueError):
            compute_global_roi(640, 480, width_ratio=bad)

    def test_always_within_frame(self) -> None:
        """The ROI never extends past the frame."""
        roi = compute_global_roi(640, 480, width_ratio=1.0, x_center=0.9)
        assert 0 <= roi.x1 <= roi.x2 <= 640


class TestBBoxInnerROI:
    """The inner-60% bounding-box ROI (spec section V)."""

    def test_default_inset(self) -> None:
        """0.6 keeps the central 60%, insetting 20% per side."""
        assert compute_bbox_inner_roi(0, 0, 100, 100, 0.6).as_tuple() == (20, 20, 80, 80)

    def test_full_box_when_ratio_is_one(self) -> None:
        """A ratio of 1.0 keeps the entire box."""
        assert compute_bbox_inner_roi(10, 20, 110, 120, 1.0).as_tuple() == (10, 20, 110, 120)

    def test_flipped_corners_normalised(self) -> None:
        """Reversed corners produce the same region, never a negative one."""
        a = compute_bbox_inner_roi(0, 0, 100, 100, 0.6)
        b = compute_bbox_inner_roi(100, 100, 0, 0, 0.6)
        assert a.as_tuple() == b.as_tuple()
        assert b.width > 0 and b.height > 0

    def test_tiny_box_keeps_minimum_size(self) -> None:
        """A distant obstacle must not inset to nothing and vanish."""
        roi = compute_bbox_inner_roi(50, 50, 53, 53, 0.6, min_size_px=4)
        assert roi.width >= 4 and roi.height >= 4

    def test_clipped_to_image(self) -> None:
        """A partially off-frame box is clipped to the image."""
        roi = compute_bbox_inner_roi(-50, -50, 20, 20, 0.6, image_width=640, image_height=480)
        assert roi.x1 >= 0 and roi.y1 >= 0 and roi.x2 <= 640 and roi.y2 <= 480

    def test_invalid_ratio_rejected(self) -> None:
        """inner_ratio must lie in (0, 1]."""
        with pytest.raises(ValueError):
            compute_bbox_inner_roi(0, 0, 10, 10, 0.0)

    def test_independent_of_global_roi(self) -> None:
        """The two ROIs are separate concepts and must not interact (spec V/W)."""
        g = compute_global_roi(1000, 500, width_ratio=0.5)
        b = compute_bbox_inner_roi(0, 0, 100, 100, 0.6)
        assert g.as_tuple() != b.as_tuple()
        assert b.as_tuple() == (20, 20, 80, 80)  # unaffected by the global ratio


class TestROIGeometry:
    """ROI helper geometry."""

    def test_area_and_emptiness(self) -> None:
        """Area and emptiness are consistent."""
        assert ROI(0, 0, 10, 10).area == 100
        assert ROI(5, 5, 5, 5).is_empty
        assert ROI(10, 10, 0, 0).is_empty  # inverted region has no area

    def test_contains_x_is_half_open(self) -> None:
        """The horizontal span is [x1, x2)."""
        roi = ROI(10, 0, 20, 10)
        assert roi.contains_x(10) and roi.contains_x(19)
        assert not roi.contains_x(20) and not roi.contains_x(9)

    def test_overlap_ratio(self) -> None:
        """Overlap is measured as a fraction of this region's width."""
        roi = ROI(0, 0, 100, 10)
        assert roi.overlap_ratio_x(0, 50) == pytest.approx(0.5)
        assert roi.overlap_ratio_x(-100, 200) == pytest.approx(1.0)
        assert roi.overlap_ratio_x(200, 300) == pytest.approx(0.0)
