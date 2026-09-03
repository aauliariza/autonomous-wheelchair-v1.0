"""Region-of-interest computation (spec sections V and W).

TWO INDEPENDENT ROIs — deliberately kept separate
-------------------------------------------------
1. ``GLOBAL_NAVIGATION_ROI`` (section W): the central 60% of the horizontal FOV.
   Defines which obstacles are RELEVANT to the wheelchair's path. Obstacles
   outside it are still detected and drawn, but take no part in the free-path
   decision — a chair two metres to the left of a corridor is not in the way.

2. ``BBOX_DEPTH_ROI`` (section V): the central 60% of each obstacle bounding box.
   Defines which pixels are read to estimate THAT obstacle's distance. Box edges
   straddle the background, so depth sampled there mixes the obstacle with the
   wall behind it and biases the distance estimate far.

Both default to 60% but they answer different questions and are configured
independently. Conflating them is a correctness bug, not a style choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ROI:
    """An axis-aligned pixel region ``[x1, x2) x [y1, y2)``."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Width in pixels (never negative)."""
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        """Height in pixels (never negative)."""
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        """Area in pixels."""
        return self.width * self.height

    @property
    def is_empty(self) -> bool:
        """True when the region encloses no pixels."""
        return self.area == 0

    @property
    def center(self) -> tuple[float, float]:
        """Region centre ``(x, y)`` in pixels."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_tuple(self) -> tuple[int, int, int, int]:
        """``(x1, y1, x2, y2)``."""
        return (self.x1, self.y1, self.x2, self.y2)

    def contains_x(self, x: float) -> bool:
        """True when ``x`` falls inside the horizontal span."""
        return self.x1 <= x < self.x2

    def clip_to(self, width: int, height: int) -> ROI:
        """Clamp the region to an image of the given size."""
        return ROI(
            x1=max(0, min(int(self.x1), width)),
            y1=max(0, min(int(self.y1), height)),
            x2=max(0, min(int(self.x2), width)),
            y2=max(0, min(int(self.y2), height)),
        )

    def overlap_ratio_x(self, x1: float, x2: float) -> float:
        """Fraction of THIS region's width covered by the span ``[x1, x2]``.

        Used for sector occupancy: the denominator is the sector's width, so the
        result answers "how much of this sector does the obstacle cover?".
        """
        if self.width <= 0:
            return 0.0
        lo = max(self.x1, min(x1, x2))
        hi = min(self.x2, max(x1, x2))
        return max(0.0, hi - lo) / float(self.width)


def compute_global_roi(
    image_width: int,
    image_height: int,
    width_ratio: float = 0.60,
    x_center: float = 0.50,
    height_ratio: float = 1.0,
    y_center: float = 0.50,
) -> ROI:
    """Central navigation ROI (spec section W).

    With the defaults (``width_ratio=0.60``, ``x_center=0.50``) this yields
    ``x in [0.20W, 0.80W]`` and the full frame height.

    Args:
        image_width (int): Frame width in pixels.
        image_height (int): Frame height in pixels.
        width_ratio (float): Fraction of frame width to keep, in ``(0, 1]``.
        x_center (float): Centre of the ROI as a fraction of width.
        height_ratio (float): Fraction of frame height to keep.
        y_center (float): Centre of the ROI as a fraction of height.

    Returns:
        (ROI): The clipped navigation region.

    Examples:
        >>> compute_global_roi(1000, 500).as_tuple()
        (200, 0, 800, 500)
    """
    if not 0.0 < width_ratio <= 1.0:
        raise ValueError(f"width_ratio must be in (0, 1], got {width_ratio}.")
    if not 0.0 < height_ratio <= 1.0:
        raise ValueError(f"height_ratio must be in (0, 1], got {height_ratio}.")

    half_w = image_width * width_ratio / 2.0
    half_h = image_height * height_ratio / 2.0
    cx = image_width * x_center
    cy = image_height * y_center

    return ROI(
        x1=int(round(cx - half_w)),
        y1=int(round(cy - half_h)),
        x2=int(round(cx + half_w)),
        y2=int(round(cy + half_h)),
    ).clip_to(image_width, image_height)


def compute_bbox_inner_roi(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    inner_ratio: float = 0.60,
    image_width: int | None = None,
    image_height: int | None = None,
    min_size_px: int = 4,
) -> ROI:
    """Inner ROI of one bounding box (spec section V).

    With ``inner_ratio=0.60`` this insets each side by 20%, keeping the central
    60% of the box in both axes:

    .. code-block:: text

        x1_inner = x1 + 0.20 * width       x2_inner = x2 - 0.20 * width
        y1_inner = y1 + 0.20 * height      y2_inner = y2 - 0.20 * height

    Small boxes are protected by ``min_size_px``: a distant obstacle a few pixels
    wide would otherwise inset to nothing and be reported as having no valid
    depth — precisely the far-obstacle case that must not silently disappear.

    Args:
        x1, y1, x2, y2 (float): Box corners, in any order.
        inner_ratio (float): Central fraction to keep, in ``(0, 1]``.
        image_width, image_height (int, optional): Clip bounds.
        min_size_px (int): Minimum width/height of the returned ROI.

    Returns:
        (ROI): The inner region, clipped when image bounds are supplied.

    Examples:
        >>> compute_bbox_inner_roi(0, 0, 100, 100, 0.6).as_tuple()
        (20, 20, 80, 80)
    """
    if not 0.0 < inner_ratio <= 1.0:
        raise ValueError(f"inner_ratio must be in (0, 1], got {inner_ratio}.")

    # Normalize corner order so a flipped box does not produce a negative region.
    bx1, bx2 = (x1, x2) if x1 <= x2 else (x2, x1)
    by1, by2 = (y1, y2) if y1 <= y2 else (y2, y1)

    bw, bh = bx2 - bx1, by2 - by1
    inset = (1.0 - inner_ratio) / 2.0

    ix1 = bx1 + inset * bw
    ix2 = bx2 - inset * bw
    iy1 = by1 + inset * bh
    iy2 = by2 - inset * bh

    # Re-expand about the centre if the inset collapsed the region below the floor.
    if ix2 - ix1 < min_size_px:
        cx = (bx1 + bx2) / 2.0
        ix1, ix2 = cx - min_size_px / 2.0, cx + min_size_px / 2.0
    if iy2 - iy1 < min_size_px:
        cy = (by1 + by2) / 2.0
        iy1, iy2 = cy - min_size_px / 2.0, cy + min_size_px / 2.0

    roi = ROI(int(round(ix1)), int(round(iy1)), int(round(ix2)), int(round(iy2)))
    if image_width is not None and image_height is not None:
        roi = roi.clip_to(image_width, image_height)
    return roi


def roi_from_config(config: dict[str, Any], image_width: int, image_height: int) -> ROI:
    """Build the global navigation ROI from a ``navigation.yaml`` ``roi`` block."""
    roi_cfg = config.get("roi", config) or {}
    return compute_global_roi(
        image_width=image_width,
        image_height=image_height,
        width_ratio=float(roi_cfg.get("width_ratio", 0.60)),
        x_center=float(roi_cfg.get("x_center", 0.50)),
        height_ratio=float(roi_cfg.get("height_ratio", 1.0)),
        y_center=float(roi_cfg.get("y_center", 0.50)),
    )
