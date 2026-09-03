"""Free-path selection policy (spec sections AB and AA).

Priority order, evaluated left to right; the first FREE sector wins:

.. code-block:: text

    CTR -> L -> R -> FL -> FR -> STOP

Mapping to commands:

.. code-block:: text

    CTR free ................................. FORWARD
    CTR blocked, L free ...................... TURN_LEFT
    CTR, L blocked, R free ................... TURN_RIGHT
    CTR, L, R blocked, FL free ............... TURN_LEFT
    CTR, L, R, FL blocked, FR free ........... TURN_RIGHT
    every sector blocked ..................... STOP

Straight ahead is preferred because turning a wheelchair costs time and space and
disturbs the occupant; the inner lanes (L/R) are preferred over the outer ones
(FL/FR) because they need a smaller turn radius.

STOP is a genuine decision here, not an error path. Both the "all blocked" case
and any unresolvable state resolve to STOP, per the fail-safe requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .sectors import SectorMap


class NavigationCommand(str, Enum):
    """Commands emitted by the planner.

    Inherits from ``str`` so a command serializes directly into CSV/JSON while
    still comparing equal to its literal name.
    """

    FORWARD = "FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"
    EMERGENCY_STOP = "EMERGENCY_STOP"

    def is_safe_halt(self) -> bool:
        """True for commands that bring the chair to a halt."""
        return self in (NavigationCommand.STOP, NavigationCommand.EMERGENCY_STOP)

    def __str__(self) -> str:
        return self.value


DEFAULT_PRIORITY = ["CTR", "L", "R", "FL", "FR"]
DEFAULT_COMMAND_MAP = {
    "CTR": NavigationCommand.FORWARD,
    "L": NavigationCommand.TURN_LEFT,
    "R": NavigationCommand.TURN_RIGHT,
    "FL": NavigationCommand.TURN_LEFT,
    "FR": NavigationCommand.TURN_RIGHT,
}


@dataclass
class FreePathDecision:
    """One frame's raw (pre-hysteresis) planning result.

    Attributes:
        command (NavigationCommand): Selected command.
        chosen_sector (str | None): Sector that produced it, None when stopping.
        occupancy (dict): ``{sector: blocked}`` snapshot.
        blocked_sectors (list[str]): Blocked sector names.
        free_sectors (list[str]): Free sector names.
        reason (str): Human-readable justification for the decision.
        sector_distances (dict): Nearest valid distance per sector.
    """

    command: NavigationCommand
    chosen_sector: str | None
    occupancy: dict[str, bool] = field(default_factory=dict)
    blocked_sectors: list[str] = field(default_factory=list)
    free_sectors: list[str] = field(default_factory=list)
    reason: str = ""
    sector_distances: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializable summary."""
        return {
            "command": str(self.command),
            "chosen_sector": self.chosen_sector,
            "occupancy": self.occupancy,
            "blocked_sectors": self.blocked_sectors,
            "free_sectors": self.free_sectors,
            "reason": self.reason,
            "sector_distances": self.sector_distances,
        }


class FreePathSelector:
    """Chooses a navigation command from sector occupancy.

    Args:
        config (dict): ``navigation.yaml`` mapping, or its ``free_path`` block.

    Examples:
        >>> from navigation.roi import ROI
        >>> from navigation.sectors import SectorMap
        >>> sm = SectorMap(ROI(0, 0, 500, 500))
        >>> FreePathSelector({}).select(sm).command
        <NavigationCommand.FORWARD: 'FORWARD'>
        >>> sm.sector_by_name("CTR").blocked = True
        >>> FreePathSelector({}).select(sm).command
        <NavigationCommand.TURN_LEFT: 'TURN_LEFT'>
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = (config or {}).get("free_path", config or {}) or {}
        self.priority: list[str] = list(cfg.get("priority", DEFAULT_PRIORITY))

        raw_map = cfg.get("commands")
        if raw_map:
            self.command_map = {k: NavigationCommand(str(v)) for k, v in raw_map.items()}
        else:
            self.command_map = dict(DEFAULT_COMMAND_MAP)

        self.all_blocked_command = NavigationCommand(str(cfg.get("all_blocked_command", "STOP")))

        missing = [s for s in self.priority if s not in self.command_map]
        if missing:
            raise ValueError(
                f"free_path.commands has no entry for prioritised sector(s) {missing}. "
                f"Every sector in free_path.priority needs a command mapping."
            )

    def select(self, sector_map: SectorMap) -> FreePathDecision:
        """Apply the priority policy to the current sector occupancy."""
        occupancy = sector_map.occupancy()
        blocked = sector_map.blocked_sectors()
        free = sector_map.free_sectors()
        distances = {s.name: s.min_distance_m for s in sector_map}

        for name in self.priority:
            try:
                sector = sector_map.sector_by_name(name)
            except KeyError:
                # A configured priority entry that this sector layout does not
                # define is skipped rather than crashing mid-drive.
                continue
            if sector.is_free:
                return FreePathDecision(
                    command=self.command_map[name],
                    chosen_sector=name,
                    occupancy=occupancy,
                    blocked_sectors=blocked,
                    free_sectors=free,
                    reason=f"sector {name} is free (priority {self.priority.index(name) + 1}/{len(self.priority)})",
                    sector_distances=distances,
                )

        return FreePathDecision(
            command=self.all_blocked_command,
            chosen_sector=None,
            occupancy=occupancy,
            blocked_sectors=blocked,
            free_sectors=free,
            reason="all prioritised sectors are blocked",
            sector_distances=distances,
        )
