"""Free-path policy and obstacle-fusion tests (spec section AU items 7, 8)."""

from __future__ import annotations

import numpy as np
import pytest

from navigation.free_path import FreePathSelector, NavigationCommand
from navigation.obstacle_fusion import OBSTACLE_LABEL, ObstacleFusion
from navigation.roi import compute_global_roi
from navigation.sectors import SectorMap

WIDTH, HEIGHT = 640, 480


@pytest.fixture
def sector_map(nav_config) -> SectorMap:
    """A sector map over the configured 60% ROI."""
    roi = compute_global_roi(WIDTH, HEIGHT, nav_config["roi"]["width_ratio"], nav_config["roi"]["x_center"])
    return SectorMap.from_config(nav_config, roi)


@pytest.fixture
def fusion(nav_config) -> ObstacleFusion:
    """Fusion built from the shipped config."""
    return ObstacleFusion(nav_config)


@pytest.fixture
def selector(nav_config) -> FreePathSelector:
    """Free-path selector built from the shipped config."""
    return FreePathSelector(nav_config)


def block(sector_map: SectorMap, *names: str) -> None:
    """Mark the named sectors blocked."""
    for n in names:
        sector_map.sector_by_name(n).blocked = True


class TestFreePathPolicy:
    """The spec section AB decision table, exhaustively."""

    def test_all_free_is_forward(self, sector_map, selector) -> None:
        """A clear path goes straight ahead."""
        assert selector.select(sector_map).command == NavigationCommand.FORWARD

    def test_ctr_blocked_turns_left(self, sector_map, selector) -> None:
        """CTR blocked with L free turns left."""
        block(sector_map, "CTR")
        assert selector.select(sector_map).command == NavigationCommand.TURN_LEFT

    def test_ctr_l_blocked_turns_right(self, sector_map, selector) -> None:
        """CTR and L blocked with R free turns right."""
        block(sector_map, "CTR", "L")
        assert selector.select(sector_map).command == NavigationCommand.TURN_RIGHT

    def test_ctr_l_r_blocked_uses_fl(self, sector_map, selector) -> None:
        """Falls back to the wider left lane."""
        block(sector_map, "CTR", "L", "R")
        d = selector.select(sector_map)
        assert d.command == NavigationCommand.TURN_LEFT and d.chosen_sector == "FL"

    def test_only_fr_free_turns_right(self, sector_map, selector) -> None:
        """The last free lane is used before stopping."""
        block(sector_map, "CTR", "L", "R", "FL")
        d = selector.select(sector_map)
        assert d.command == NavigationCommand.TURN_RIGHT and d.chosen_sector == "FR"

    def test_all_blocked_stops(self, sector_map, selector) -> None:
        """With no free lane the only safe action is STOP."""
        block(sector_map, "FL", "L", "CTR", "R", "FR")
        d = selector.select(sector_map)
        assert d.command == NavigationCommand.STOP and d.chosen_sector is None

    def test_priority_order_respected(self, sector_map, selector) -> None:
        """CTR wins over L even when both are free."""
        assert selector.select(sector_map).chosen_sector == "CTR"

    def test_decision_carries_a_reason(self, sector_map, selector) -> None:
        """Every decision explains itself for the audit log."""
        assert selector.select(sector_map).reason

    def test_missing_command_mapping_rejected(self) -> None:
        """A prioritised sector with no command mapping is a config error."""
        with pytest.raises(ValueError, match="no entry for prioritised"):
            FreePathSelector({"free_path": {"priority": ["CTR", "GHOST"], "commands": {"CTR": "FORWARD"}}})


class TestObstacleFusion:
    """Detection-depth fusion and blocking rules."""

    def _scene(self, boxes, depth_value=0.5):
        """A far-background depth map with near boxes painted in."""
        d = np.full((HEIGHT, WIDTH), 5.0, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            d[int(y1) : int(y2), int(x1) : int(x2)] = depth_value
        return d

    def test_every_detection_is_labelled_obstacle(self, fusion, sector_map) -> None:
        """Spec section A: no object class is ever reported."""
        boxes = [(282, 100, 358, 400), (150, 100, 200, 300)]
        obs = fusion.fuse(np.array(boxes), [0.9, 0.8], self._scene(boxes), sector_map, image_size=(HEIGHT, WIDTH))
        assert obs and all(o.label == OBSTACLE_LABEL for o in obs)

    def test_near_obstacle_blocks(self, fusion, sector_map) -> None:
        """An obstacle inside the safety distance blocks its sector."""
        boxes = [(282, 100, 358, 400)]
        obs = fusion.fuse(np.array(boxes), [0.9], self._scene(boxes, 0.5), sector_map, image_size=(HEIGHT, WIDTH))
        assert obs[0].blocked and obs[0].distance_m < 1.0

    def test_far_obstacle_does_not_block(self, fusion, sector_map) -> None:
        """An obstacle beyond the safety distance leaves the sector free."""
        boxes = [(282, 100, 358, 400)]
        obs = fusion.fuse(np.array(boxes), [0.9], self._scene(boxes, 2.5), sector_map, image_size=(HEIGHT, WIDTH))
        assert not obs[0].blocked and obs[0].distance_m == pytest.approx(2.5, abs=0.05)

    def test_invalid_depth_blocks_conservatively(self, fusion, sector_map) -> None:
        """An unmeasurable obstacle is treated as blocking (spec section X)."""
        obs = fusion.fuse(
            np.array([[282, 100, 358, 400]]), [0.9], np.zeros((HEIGHT, WIDTH)), sector_map, image_size=(HEIGHT, WIDTH)
        )
        assert obs[0].distance_m is None and obs[0].blocked

    def test_out_of_roi_obstacle_ignored_but_reported(self, fusion, sector_map) -> None:
        """Outside the ROI: still returned for display, never blocking."""
        boxes = [(0, 100, 100, 400)]
        obs = fusion.fuse(np.array(boxes), [0.9], self._scene(boxes), sector_map, image_size=(HEIGHT, WIDTH))
        assert obs and not obs[0].in_roi and not obs[0].blocked
        assert not sector_map.blocked_sectors()

    def test_empty_detection_list(self, fusion, sector_map) -> None:
        """Zero detections is a valid frame with no blocked sectors."""
        obs = fusion.fuse(np.zeros((0, 4)), [], np.full((HEIGHT, WIDTH), 3.0), sector_map, image_size=(HEIGHT, WIDTH))
        assert obs == [] and not sector_map.blocked_sectors()

    def test_low_confidence_detection_filtered(self, fusion, sector_map) -> None:
        """Detections below the confidence floor are discarded."""
        boxes = [(282, 100, 358, 400)]
        obs = fusion.fuse(np.array(boxes), [0.01], self._scene(boxes), sector_map, image_size=(HEIGHT, WIDTH))
        assert obs == []

    def test_wide_obstacle_blocks_multiple_sectors(self, fusion, sector_map) -> None:
        """A wide obstacle blocks every lane it overlaps."""
        boxes = [(205, 100, 435, 400)]
        fusion.fuse(np.array(boxes), [0.9], self._scene(boxes), sector_map, image_size=(HEIGHT, WIDTH))
        assert set(sector_map.blocked_sectors()) >= {"L", "CTR", "R"}

    def test_sector_records_nearest_distance(self, fusion, sector_map) -> None:
        """A sector reports the nearest obstacle within it."""
        boxes = [(282, 100, 358, 200), (290, 250, 350, 400)]
        depth = np.full((HEIGHT, WIDTH), 5.0, dtype=np.float32)
        depth[100:200, 282:358] = 3.0
        depth[250:400, 290:350] = 1.5
        fusion.fuse(np.array(boxes), [0.9, 0.9], depth, sector_map, image_size=(HEIGHT, WIDTH))
        assert sector_map.sector_by_name("CTR").min_distance_m == pytest.approx(1.5, abs=0.05)

    def test_box_confidence_length_mismatch(self, fusion, sector_map) -> None:
        """Mismatched box/confidence lengths are a caller error."""
        with pytest.raises(ValueError, match="must match"):
            fusion.fuse(np.zeros((2, 4)), [0.9], np.full((HEIGHT, WIDTH), 3.0), sector_map)


class TestEndToEndDecisions:
    """Fusion + policy together, on synthetic scenes."""

    @pytest.mark.parametrize(
        "box,expected",
        [
            (None, NavigationCommand.FORWARD),
            ((282, 100, 358, 400), NavigationCommand.TURN_LEFT),
            ((205, 100, 358, 400), NavigationCommand.TURN_RIGHT),
            ((128, 100, 512, 400), NavigationCommand.STOP),
        ],
    )
    def test_scene_to_command(self, fusion, selector, sector_map, box, expected) -> None:
        """Each synthetic scene produces the command the spec table requires."""
        depth = np.full((HEIGHT, WIDTH), 5.0, dtype=np.float32)
        boxes = np.zeros((0, 4))
        confs: list[float] = []
        if box is not None:
            x1, y1, x2, y2 = box
            depth[y1:y2, x1:x2] = 0.5
            boxes, confs = np.array([box]), [0.9]

        fusion.fuse(boxes, confs, depth, sector_map, image_size=(HEIGHT, WIDTH))
        assert selector.select(sector_map).command == expected
