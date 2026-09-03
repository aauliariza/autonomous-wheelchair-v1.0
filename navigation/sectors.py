"""Five-sector decomposition of the navigation ROI (spec section Z).

The global ROI is divided left-to-right into:

.. code-block:: text

    |  FL  |  L  |  CTR  |  R  |  FR  |
     <---------- 60% ROI ---------->

``FL``/``FR`` mean Forward-Left / Forward-Right: the outermost lanes still inside
the navigable corridor, reachable by a wider turn than ``L``/``R``. They are not
"far left/right" — anything genuinely off-path is outside the ROI entirely.

Sector widths are configurable and need not be equal; the defaults are five 20%
lanes of the ROI.

Assignment policy
-----------------
``overlap`` (default) marks EVERY sector a bounding box touches. ``center`` marks
only the sector containing the box's horizontal midpoint. Overlap is the
safety-correct choice: a wide obstacle straddling CTR and L blocks both, whereas
centre-assignment would leave L nominally free and could steer the wheelchair
into the part of the obstacle its own midpoint did not fall in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .roi import ROI

DEFAULT_SECTOR_NAMES = ("FL", "L", "CTR", "R", "FR")


@dataclass
class Sector:
    """One vertical lane of the navigation ROI.

    Attributes:
        name (str): Sector label, e.g. ``"CTR"``.
        index (int): Position, 0 = leftmost.
        region (ROI): Pixel extent.
        blocked (bool): Set by the fusion stage when an obstacle violates the
            safety distance inside this sector.
        min_distance_m (float | None): Nearest valid obstacle distance, or None.
        obstacle_ids (list[int]): Obstacles assigned to this sector.
    """

    name: str
    index: int
    region: ROI
    blocked: bool = False
    min_distance_m: float | None = None
    obstacle_ids: list[int] = field(default_factory=list)

    @property
    def is_free(self) -> bool:
        """True when the sector is traversable."""
        return not self.blocked

    def reset(self) -> None:
        """Clear per-frame state, keeping the geometry."""
        self.blocked = False
        self.min_distance_m = None
        self.obstacle_ids = []

    def to_dict(self) -> dict[str, Any]:
        """Serializable summary for logging and overlays."""
        return {
            "name": self.name,
            "index": self.index,
            "region": self.region.as_tuple(),
            "blocked": self.blocked,
            "state": "blocked" if self.blocked else "free",
            "min_distance_m": self.min_distance_m,
            "obstacle_ids": list(self.obstacle_ids),
        }


class SectorMap:
    """Partitions a navigation ROI into named sectors and assigns obstacles.

    Args:
        roi (ROI): The global navigation ROI to subdivide.
        names (list[str], optional): Sector names, left to right.
        widths (list[float], optional): Fractional widths summing to 1.0.
        assignment (str): ``"overlap"`` or ``"center"``.
        min_overlap_ratio (float): Under ``overlap``, the minimum fraction of a
            sector's width a box must cover to occupy it. Filters boxes that
            merely graze a sector boundary by a pixel.

    Examples:
        >>> from navigation.roi import ROI
        >>> sm = SectorMap(ROI(200, 0, 800, 500))
        >>> [s.name for s in sm.sectors]
        ['FL', 'L', 'CTR', 'R', 'FR']
        >>> sm.sector_by_name("CTR").region.as_tuple()
        (440, 0, 560, 500)
    """

    def __init__(
        self,
        roi: ROI,
        names: list[str] | None = None,
        widths: list[float] | None = None,
        assignment: str = "overlap",
        min_overlap_ratio: float = 0.05,
    ):
        self.roi = roi
        self.names = list(names) if names else list(DEFAULT_SECTOR_NAMES)
        self.widths = list(widths) if widths else [1.0 / len(self.names)] * len(self.names)

        if len(self.widths) != len(self.names):
            raise ValueError(f"Got {len(self.names)} sector names but {len(self.widths)} widths; they must match.")
        total = sum(self.widths)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Sector widths must sum to 1.0, got {total:.6f} ({self.widths}).")
        if assignment not in ("overlap", "center"):
            raise ValueError(f"assignment must be 'overlap' or 'center', got '{assignment}'.")

        self.assignment = assignment
        self.min_overlap_ratio = min_overlap_ratio
        self.sectors: list[Sector] = self._build()

    def _build(self) -> list[Sector]:
        """Lay out sectors across the ROI, absorbing rounding into the last one."""
        out: list[Sector] = []
        span = self.roi.width
        cursor = float(self.roi.x1)

        for i, (name, frac) in enumerate(zip(self.names, self.widths)):
            x1 = int(round(cursor))
            cursor += span * frac
            # The final sector is pinned to the ROI edge so cumulative rounding
            # can never leave a one-pixel gap or overhang at the boundary.
            x2 = self.roi.x2 if i == len(self.names) - 1 else int(round(cursor))
            out.append(Sector(name=name, index=i, region=ROI(x1, self.roi.y1, x2, self.roi.y2)))
        return out

    # ---------------- lookup ----------------

    def sector_by_name(self, name: str) -> Sector:
        """Fetch a sector by name."""
        for s in self.sectors:
            if s.name == name:
                return s
        raise KeyError(f"No sector named '{name}'. Available: {[s.name for s in self.sectors]}.")

    def sector_at_x(self, x: float) -> Sector | None:
        """Sector containing pixel column ``x``, or None if outside the ROI."""
        for s in self.sectors:
            if s.region.contains_x(x):
                return s
        # The ROI's right edge is exclusive everywhere else; treat an exact hit
        # on it as the last sector rather than "outside".
        if abs(x - self.roi.x2) < 1e-9 and self.sectors:
            return self.sectors[-1]
        return None

    def reset(self) -> None:
        """Clear per-frame state on every sector."""
        for s in self.sectors:
            s.reset()

    # ---------------- assignment ----------------

    def assign_bbox(self, x1: float, x2: float) -> list[Sector]:
        """Sectors occupied by a bounding box spanning ``[x1, x2]``.

        Returns an empty list when the box lies wholly outside the navigation
        ROI, which is how out-of-ROI obstacles are excluded from the decision
        while remaining available for visualization (spec section W).
        """
        lo, hi = (x1, x2) if x1 <= x2 else (x2, x1)

        # Reject boxes with no horizontal intersection with the ROI at all.
        if hi <= self.roi.x1 or lo >= self.roi.x2:
            return []

        if self.assignment == "center":
            s = self.sector_at_x((lo + hi) / 2.0)
            return [s] if s is not None else []

        hits = [s for s in self.sectors if s.region.overlap_ratio_x(lo, hi) >= self.min_overlap_ratio]
        if hits:
            return hits
        # A box narrower than min_overlap_ratio still occupies wherever its
        # centre lands; dropping it entirely would be unsafe.
        s = self.sector_at_x((lo + hi) / 2.0)
        return [s] if s is not None else []

    def is_in_roi(self, x1: float, x2: float) -> bool:
        """True when a box horizontally intersects the navigation ROI."""
        lo, hi = (x1, x2) if x1 <= x2 else (x2, x1)
        return hi > self.roi.x1 and lo < self.roi.x2

    # ---------------- reporting ----------------

    def blocked_sectors(self) -> list[str]:
        """Names of currently blocked sectors."""
        return [s.name for s in self.sectors if s.blocked]

    def free_sectors(self) -> list[str]:
        """Names of currently free sectors."""
        return [s.name for s in self.sectors if s.is_free]

    def occupancy(self) -> dict[str, bool]:
        """``{sector name: blocked}`` for the current frame."""
        return {s.name: s.blocked for s in self.sectors}

    def to_dict(self) -> dict[str, Any]:
        """Serializable snapshot of the sector layout and state."""
        return {
            "roi": self.roi.as_tuple(),
            "assignment": self.assignment,
            "sectors": [s.to_dict() for s in self.sectors],
        }

    @classmethod
    def from_config(cls, config: dict[str, Any], roi: ROI) -> SectorMap:
        """Build from a ``navigation.yaml`` ``sectors`` block."""
        sec = config.get("sectors", config) or {}
        return cls(
            roi=roi,
            names=sec.get("names"),
            widths=sec.get("widths"),
            assignment=sec.get("assignment", "overlap"),
            min_overlap_ratio=float(sec.get("min_overlap_ratio", 0.05)),
        )

    def __len__(self) -> int:
        return len(self.sectors)

    def __iter__(self):
        return iter(self.sectors)
