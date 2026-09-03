"""Temporal safety monitoring and emergency stop (spec sections AD and AE).

ARCHITECTURAL BOUNDARY
----------------------
This module produces a COMMAND. It does not drive motors. The separation
required by spec section AE is:

.. code-block:: text

    perception -> planning -> command generation -> [ motor controller ]
                                                     ^
                                        independent safety layer lives here,
                                        outside this software

A neural network must never have direct actuator authority. The motor controller
is expected to enforce its own limits (current, speed, watchdog) and to halt on
loss of command stream, independently of anything decided here.

FAIL-SAFE CONDITIONS (all resolve to STOP)
------------------------------------------
1. stale frame            5. low confidence
2. inference timeout      6. sudden depth change
3. missing detection      7. camera failure
4. invalid depth          8. model exception
                          9. invalid frame timestamp

After ``max_consecutive_failures`` the monitor LATCHES ``EMERGENCY_STOP``, which
persists until ``reset()`` is called explicitly. Latching is deliberate: a system
oscillating in and out of a fault state is more dangerous than one that stops and
demands attention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .free_path import NavigationCommand


class SafetyState(str, Enum):
    """Overall health of the perception/planning chain."""

    OK = "OK"
    DEGRADED = "DEGRADED"          # a recoverable fault; commands forced to STOP
    EMERGENCY_STOP = "EMERGENCY_STOP"  # latched; requires reset()

    def __str__(self) -> str:
        return self.value


@dataclass
class SafetyReport:
    """Outcome of one frame's safety checks.

    Attributes:
        state (SafetyState): Current monitor state.
        safe (bool): True when no violation was detected this frame.
        command (NavigationCommand | None): Overriding command, if any.
        violations (list[str]): Machine-readable violation codes.
        messages (list[str]): Human-readable explanations.
        consecutive_failures (int): Current failure streak.
    """

    state: SafetyState
    safe: bool
    command: NavigationCommand | None = None
    violations: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    consecutive_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serializable summary."""
        return {
            "state": str(self.state),
            "safe": self.safe,
            "command": None if self.command is None else str(self.command),
            "violations": list(self.violations),
            "messages": list(self.messages),
            "consecutive_failures": self.consecutive_failures,
        }


class SafetyMonitor:
    """Validates each frame and overrides unsafe commands.

    Args:
        config (dict): ``navigation.yaml`` mapping or its ``temporal_safety`` block.

    Examples:
        >>> m = SafetyMonitor({})
        >>> m.check(frame_age_s=0.0, inference_latency_s=0.01).safe
        True
        >>> m.check(frame_age_s=10.0).safe          # stale frame
        False
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = (config or {}).get("temporal_safety", config or {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.max_frame_age_s = float(cfg.get("max_frame_age_s", 0.5))
        self.max_inference_latency_s = float(cfg.get("max_inference_latency_s", 0.5))
        self.max_depth_jump_m = float(cfg.get("max_depth_jump_m", 2.0))
        self.max_consecutive_failures = int(cfg.get("max_consecutive_failures", 3))
        self.latch_emergency_stop = bool(cfg.get("latch_emergency_stop", True))

        self.state = SafetyState.OK
        self.consecutive_failures = 0
        self._previous_distances: dict[str, float] = {}
        self._last_frame_time: float | None = None

    # ---------------- checks ----------------

    def check(
        self,
        frame_age_s: float | None = None,
        inference_latency_s: float | None = None,
        depth_valid: bool = True,
        detection_ok: bool = True,
        confidence: float | None = None,
        min_confidence: float = 0.0,
        sector_distances: dict[str, float] | None = None,
        model_exception: Exception | None = None,
        camera_ok: bool = True,
        frame_timestamp: float | None = None,
    ) -> SafetyReport:
        """Run every fail-safe check for one frame.

        Each argument corresponds to one numbered condition in the module
        docstring. ``None`` means "not applicable this frame" and is not a fault.

        Returns:
            (SafetyReport): ``command`` is set whenever an override is required.
        """
        violations: list[str] = []
        messages: list[str] = []

        if not self.enabled:
            return SafetyReport(state=SafetyState.OK, safe=True, consecutive_failures=0)

        # Already latched: stay latched regardless of this frame's health.
        if self.state == SafetyState.EMERGENCY_STOP and self.latch_emergency_stop:
            return SafetyReport(
                state=SafetyState.EMERGENCY_STOP,
                safe=False,
                command=NavigationCommand.EMERGENCY_STOP,
                violations=["latched_emergency_stop"],
                messages=["EMERGENCY_STOP is latched; call SafetyMonitor.reset() to clear it."],
                consecutive_failures=self.consecutive_failures,
            )

        # 8. Model exception
        if model_exception is not None:
            violations.append("model_exception")
            messages.append(f"Model inference raised {type(model_exception).__name__}: {model_exception}")

        # 7. Camera failure
        if not camera_ok:
            violations.append("camera_failure")
            messages.append("Camera reported a failure or returned no frame.")

        # 1. Stale frame
        if frame_age_s is not None and frame_age_s > self.max_frame_age_s:
            violations.append("stale_frame")
            messages.append(f"Frame age {frame_age_s:.3f}s exceeds max_frame_age_s {self.max_frame_age_s:.3f}s.")

        # 2. Inference timeout
        if inference_latency_s is not None and inference_latency_s > self.max_inference_latency_s:
            violations.append("inference_timeout")
            messages.append(
                f"Inference latency {inference_latency_s:.3f}s exceeds "
                f"max_inference_latency_s {self.max_inference_latency_s:.3f}s."
            )

        # 9. Frame timestamp validity — must exist and move forward.
        if frame_timestamp is not None:
            if not (frame_timestamp > 0) or frame_timestamp != frame_timestamp:  # NaN-safe
                violations.append("invalid_timestamp")
                messages.append(f"Frame timestamp {frame_timestamp} is not a valid positive value.")
            elif self._last_frame_time is not None and frame_timestamp < self._last_frame_time:
                violations.append("timestamp_regression")
                messages.append(
                    f"Frame timestamp {frame_timestamp:.6f} precedes the previous frame "
                    f"{self._last_frame_time:.6f}; frames are out of order."
                )
            self._last_frame_time = frame_timestamp

        # 4. Invalid depth
        if not depth_valid:
            violations.append("invalid_depth")
            messages.append("Depth map was missing or contained no usable values.")

        # 3. Missing detection stage
        if not detection_ok:
            violations.append("detection_failure")
            messages.append("Object detection stage failed to produce a result.")

        # 5. Low confidence
        if confidence is not None and confidence < min_confidence:
            violations.append("low_confidence")
            messages.append(f"Fused confidence {confidence:.3f} is below the minimum {min_confidence:.3f}.")

        # 6. Sudden depth change
        if sector_distances:
            for name, dist in sector_distances.items():
                prev = self._previous_distances.get(name)
                if prev is not None and dist is not None and abs(dist - prev) > self.max_depth_jump_m:
                    violations.append("depth_discontinuity")
                    messages.append(
                        f"Sector {name} distance jumped {abs(dist - prev):.2f}m "
                        f"({prev:.2f} -> {dist:.2f}), exceeding max_depth_jump_m {self.max_depth_jump_m:.2f}m."
                    )
            self._previous_distances = {k: v for k, v in sector_distances.items() if v is not None}

        if not violations:
            self.consecutive_failures = 0
            self.state = SafetyState.OK
            return SafetyReport(state=SafetyState.OK, safe=True, consecutive_failures=0)

        self.consecutive_failures += 1

        if self.consecutive_failures >= self.max_consecutive_failures:
            self.state = SafetyState.EMERGENCY_STOP
            command = NavigationCommand.EMERGENCY_STOP
            messages.append(
                f"{self.consecutive_failures} consecutive unsafe frames reached "
                f"max_consecutive_failures ({self.max_consecutive_failures}); latching EMERGENCY_STOP."
            )
        else:
            self.state = SafetyState.DEGRADED
            command = NavigationCommand.STOP

        return SafetyReport(
            state=self.state,
            safe=False,
            command=command,
            violations=violations,
            messages=messages,
            consecutive_failures=self.consecutive_failures,
        )

    def apply(self, command: NavigationCommand, report: SafetyReport) -> NavigationCommand:
        """Override a planned command when the safety report demands it."""
        if report.command is not None:
            return report.command
        return command

    def trigger_emergency_stop(self, reason: str = "manual trigger") -> SafetyReport:
        """Raise EMERGENCY_STOP immediately (software e-stop, spec section AE)."""
        self.state = SafetyState.EMERGENCY_STOP
        self.consecutive_failures = max(self.consecutive_failures, self.max_consecutive_failures)
        return SafetyReport(
            state=self.state,
            safe=False,
            command=NavigationCommand.EMERGENCY_STOP,
            violations=["manual_emergency_stop"],
            messages=[f"EMERGENCY_STOP triggered: {reason}"],
            consecutive_failures=self.consecutive_failures,
        )

    def reset(self) -> None:
        """Clear the latch and all history. Requires a deliberate call."""
        self.state = SafetyState.OK
        self.consecutive_failures = 0
        self._previous_distances = {}
        self._last_frame_time = None

    @property
    def is_emergency(self) -> bool:
        """True while EMERGENCY_STOP is latched."""
        return self.state == SafetyState.EMERGENCY_STOP

    def state_dict(self) -> dict[str, Any]:
        """Snapshot for logging."""
        return {
            "state": str(self.state),
            "consecutive_failures": self.consecutive_failures,
            "is_emergency": self.is_emergency,
            "tracked_sectors": sorted(self._previous_distances),
        }


class FrameTimer:
    """Measures frame age and per-stage latency (spec sections AD, AN)."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._marks: dict[str, float] = {}
        self._t0 = time.perf_counter()

    def start(self, stage: str) -> None:
        """Begin timing a stage."""
        self._marks[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        """End a stage and record its duration in milliseconds."""
        if stage not in self._marks:
            raise KeyError(f"Stage '{stage}' was never started; call start('{stage}') first.")
        dt = (time.perf_counter() - self._marks.pop(stage)) * 1000.0
        self.stages[stage] = dt
        return dt

    def total_ms(self) -> float:
        """Total elapsed time since the timer was created."""
        return (time.perf_counter() - self._t0) * 1000.0

    def reset(self) -> None:
        """Restart the timer for a new frame."""
        self.stages = {}
        self._marks = {}
        self._t0 = time.perf_counter()
