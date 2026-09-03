"""Sector decomposition and assignment tests (spec section AU item 6)."""

from __future__ import annotations

import pytest

from navigation.roi import ROI
from navigation.sectors import DEFAULT_SECTOR_NAMES, SectorMap


@pytest.fixture
def sector_map() -> SectorMap:
    """A 5-sector map over a 600px-wide ROI."""
    return SectorMap(ROI(200, 0, 800, 500))


class TestSectorLayout:
    """Sector geometry must tile the ROI exactly."""

    def test_default_names_and_order(self, sector_map) -> None:
        """Sectors are FL | L | CTR | R | FR, left to right."""
        assert [s.name for s in sector_map] == list(DEFAULT_SECTOR_NAMES)
        assert [s.index for s in sector_map] == [0, 1, 2, 3, 4]

    def test_sectors_tile_the_roi_without_gaps(self, sector_map) -> None:
        """Adjacent sectors touch, and the span covers the ROI exactly."""
        regions = [s.region for s in sector_map]
        assert regions[0].x1 == sector_map.roi.x1
        assert regions[-1].x2 == sector_map.roi.x2
        for a, b in zip(regions, regions[1:], strict=False):
            assert a.x2 == b.x1, "sectors must not overlap or leave a gap"

    def test_uneven_widths_still_tile(self) -> None:
        """Unequal widths are supported and still tile exactly."""
        sm = SectorMap(ROI(0, 0, 100, 10), widths=[0.1, 0.2, 0.4, 0.2, 0.1])
        assert sm.sectors[0].region.width == 10
        assert sm.sectors[2].region.width == 40
        assert sm.sectors[-1].region.x2 == 100

    def test_widths_must_sum_to_one(self) -> None:
        """A width vector that does not sum to 1.0 is rejected."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            SectorMap(ROI(0, 0, 100, 10), widths=[0.5, 0.5, 0.5, 0.5, 0.5])

    def test_name_width_length_mismatch(self) -> None:
        """Names and widths must have equal length."""
        with pytest.raises(ValueError, match="names but"):
            SectorMap(ROI(0, 0, 100, 10), names=["A", "B"], widths=[1.0])

    def test_invalid_assignment_mode(self) -> None:
        """Only 'overlap' and 'center' are valid assignment policies."""
        with pytest.raises(ValueError, match="assignment must be"):
            SectorMap(ROI(0, 0, 100, 10), assignment="nearest")


class TestSectorAssignment:
    """Obstacle-to-sector assignment (spec section Z)."""

    def test_narrow_box_hits_one_sector(self, sector_map) -> None:
        """A box inside one lane occupies only that lane."""
        ctr = sector_map.sector_by_name("CTR").region
        assert [s.name for s in sector_map.assign_bbox(ctr.x1 + 5, ctr.x2 - 5)] == ["CTR"]

    def test_wide_box_blocks_every_lane_it_touches(self, sector_map) -> None:
        """Overlap assignment is the safety-correct policy."""
        names = [s.name for s in sector_map.assign_bbox(200, 800)]
        assert names == list(DEFAULT_SECTOR_NAMES)

    def test_centre_mode_assigns_one_sector(self) -> None:
        """Centre mode assigns by midpoint only."""
        sm = SectorMap(ROI(200, 0, 800, 500), assignment="center")
        assert [s.name for s in sm.assign_bbox(200, 800)] == ["CTR"]

    def test_box_outside_roi_is_ignored(self, sector_map) -> None:
        """Obstacles outside the navigation ROI take no part in the decision."""
        assert sector_map.assign_bbox(0, 100) == []
        assert sector_map.assign_bbox(900, 1000) == []
        assert not sector_map.is_in_roi(0, 100)

    def test_box_straddling_the_roi_edge_counts(self, sector_map) -> None:
        """A box partially inside the ROI is still relevant."""
        assert sector_map.is_in_roi(100, 250)
        assert [s.name for s in sector_map.assign_bbox(100, 250)] == ["FL"]

    def test_sub_threshold_box_still_assigned(self, sector_map) -> None:
        """A box narrower than min_overlap_ratio is not silently dropped."""
        ctr = sector_map.sector_by_name("CTR").region
        mid = (ctr.x1 + ctr.x2) // 2
        assert [s.name for s in sector_map.assign_bbox(mid, mid + 1)] == ["CTR"]

    def test_sector_at_x_bounds(self, sector_map) -> None:
        """Lookup covers the ROI and returns None outside it."""
        assert sector_map.sector_at_x(sector_map.roi.x1).name == "FL"
        assert sector_map.sector_at_x(sector_map.roi.x2 - 1).name == "FR"
        assert sector_map.sector_at_x(0) is None


class TestSectorState:
    """Per-frame sector state handling."""

    def test_reset_clears_state_but_keeps_geometry(self, sector_map) -> None:
        """Reset clears occupancy without rebuilding the layout."""
        s = sector_map.sector_by_name("CTR")
        before = s.region.as_tuple()
        s.blocked, s.min_distance_m, s.obstacle_ids = True, 0.5, [1, 2]
        sector_map.reset()
        assert not s.blocked and s.min_distance_m is None and s.obstacle_ids == []
        assert s.region.as_tuple() == before

    def test_occupancy_reporting(self, sector_map) -> None:
        """Blocked and free lists are consistent with occupancy."""
        sector_map.sector_by_name("CTR").blocked = True
        assert sector_map.blocked_sectors() == ["CTR"]
        assert "CTR" not in sector_map.free_sectors()
        assert sector_map.occupancy()["CTR"] is True

    def test_unknown_sector_name_raises(self, sector_map) -> None:
        """Looking up a missing sector is an error, not a silent None."""
        with pytest.raises(KeyError, match="No sector named"):
            sector_map.sector_by_name("MIDDLE")
