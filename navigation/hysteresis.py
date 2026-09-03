"""Majority-vote temporal smoothing (spec section AC).

Per-frame decisions are noisy: a detection flickering on and off, or a depth
estimate crossing the safety threshold by a centimetre, would otherwise make the
chair judder between FORWARD and TURN_LEFT. A majority vote over the last
``N = 3`` commands suppresses single-frame noise while adding only two frames of
latency.

Tie policy
----------
With ``N = 3`` a tie means three different commands, i.e. the scene is changing
faster than the filter can track. The default response is STOP — for an
autonomous wheelchair, indecision must resolve to the safe action, never to a
guess. ``hold_previous_on_tie`` can hold the last command instead, but that is
explicitly NOT recommended for deployment.

Warm-up
-------
Before the window fills, there is not enough evidence for a majority. The filter
emits ``warmup_command`` (STOP by default) rather than trusting the first frame.

STOP override
-------------
A STOP entering the filter is never voted away. If any frame in the window says
STOP, the output is STOP: it is always safe to stop when motion was warranted,
and never safe to move when a stop was warranted. This deliberately breaks the
symmetry of a plain majority vote.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

from .free_path import NavigationCommand


class MajorityVoteHysteresis:
    """Sliding-window majority filter over navigation commands.

    Args:
        window (int): Frames to consider (``N``). Must be >= 1.
        hold_previous_on_tie (bool): Hold the last stable command on a tie
            instead of emitting ``tie_command``. NOT recommended.
        tie_command (str): Command emitted on an unresolved tie.
        warmup_command (str): Command emitted until the window fills.
        stop_override (bool): Any STOP in the window forces STOP out.
        enabled (bool): When False the filter is a pass-through, for ablation.

    Examples:
        >>> h = MajorityVoteHysteresis(window=3)
        >>> h.update(NavigationCommand.FORWARD)     # warm-up
        <NavigationCommand.STOP: 'STOP'>
        >>> h.update(NavigationCommand.FORWARD)     # warm-up
        <NavigationCommand.STOP: 'STOP'>
        >>> h.update(NavigationCommand.FORWARD)     # window full
        <NavigationCommand.FORWARD: 'FORWARD'>
    """

    def __init__(
        self,
        window: int = 3,
        hold_previous_on_tie: bool = False,
        tie_command: str = "STOP",
        warmup_command: str = "STOP",
        stop_override: bool = True,
        enabled: bool = True,
    ):
        if window < 1:
            raise ValueError(f"hysteresis window must be >= 1, got {window}.")
        self.window = window
        self.hold_previous_on_tie = hold_previous_on_tie
        self.tie_command = NavigationCommand(str(tie_command))
        self.warmup_command = NavigationCommand(str(warmup_command))
        self.stop_override = stop_override
        self.enabled = enabled

        self.history: deque[NavigationCommand] = deque(maxlen=window)
        self.previous_command: NavigationCommand = self.warmup_command
        self.last_output: NavigationCommand = self.warmup_command

    def update(self, command: NavigationCommand | str) -> NavigationCommand:
        """Push a raw command and return the filtered one."""
        cmd = command if isinstance(command, NavigationCommand) else NavigationCommand(str(command))

        if not self.enabled:
            self.last_output = cmd
            self.previous_command = cmd
            return cmd

        # EMERGENCY_STOP bypasses the filter entirely: it must take effect on the
        # frame it is raised, not two frames later.
        if cmd == NavigationCommand.EMERGENCY_STOP:
            self.history.append(cmd)
            self.last_output = cmd
            return cmd

        self.history.append(cmd)

        if len(self.history) < self.window:
            self.last_output = self.warmup_command
            return self.warmup_command

        if self.stop_override and NavigationCommand.STOP in self.history:
            self.last_output = NavigationCommand.STOP
            self.previous_command = NavigationCommand.STOP
            return NavigationCommand.STOP

        counts = Counter(self.history)
        top = counts.most_common()
        best_count = top[0][1]
        winners = [c for c, n in top if n == best_count]

        if len(winners) == 1:
            out = winners[0]
            self.previous_command = out
        elif self.hold_previous_on_tie:
            out = self.previous_command
        else:
            out = self.tie_command

        self.last_output = out
        return out

    def reset(self) -> None:
        """Clear history and return to the warm-up state."""
        self.history.clear()
        self.previous_command = self.warmup_command
        self.last_output = self.warmup_command

    @property
    def is_warmed_up(self) -> bool:
        """True once the window holds a full set of frames."""
        return len(self.history) >= self.window

    def state(self) -> dict[str, Any]:
        """Snapshot for logging and overlays."""
        return {
            "window": self.window,
            "history": [str(c) for c in self.history],
            "warmed_up": self.is_warmed_up,
            "last_output": str(self.last_output),
            "previous_stable": str(self.previous_command),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MajorityVoteHysteresis:
        """Build from a ``navigation.yaml`` ``hysteresis`` block."""
        cfg = config.get("hysteresis", config) or {}
        return cls(
            window=int(cfg.get("window", 3)),
            hold_previous_on_tie=bool(cfg.get("hold_previous_on_tie", False)),
            tie_command=cfg.get("tie_command", "STOP"),
            warmup_command=cfg.get("warmup_command", "STOP"),
            stop_override=bool(cfg.get("stop_override", True)),
            enabled=bool(cfg.get("enabled", True)),
        )
