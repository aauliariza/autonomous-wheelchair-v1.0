"""Temporal safety and emergency-stop tests (spec section AU item 11)."""

from __future__ import annotations

import pytest

from navigation.free_path import NavigationCommand
from navigation.safety import FrameTimer, SafetyMonitor, SafetyState


@pytest.fixture
def monitor(nav_config) -> SafetyMonitor:
    """A monitor built from the shipped navigation config."""
    return SafetyMonitor(nav_config)


class TestHealthyOperation:
    """A healthy frame must not be flagged."""

    def test_healthy_frame_passes(self, monitor) -> None:
        """Fresh frame, fast inference, valid depth: no violation."""
        r = monitor.check(frame_age_s=0.01, inference_latency_s=0.02)
        assert r.safe and r.state == SafetyState.OK and r.command is None

    def test_failure_streak_resets_on_recovery(self, monitor) -> None:
        """A good frame clears the streak before it can latch."""
        monitor.check(frame_age_s=10.0)
        assert monitor.consecutive_failures == 1
        monitor.check(frame_age_s=0.01)
        assert monitor.consecutive_failures == 0


class TestFailSafeConditions:
    """Each of the nine conditions in spec section AD must force STOP."""

    @pytest.mark.parametrize(
        "kwargs,violation",
        [
            ({"frame_age_s": 10.0}, "stale_frame"),
            ({"inference_latency_s": 5.0}, "inference_timeout"),
            ({"depth_valid": False}, "invalid_depth"),
            ({"detection_ok": False}, "detection_failure"),
            ({"camera_ok": False}, "camera_failure"),
            ({"model_exception": RuntimeError("CUDA OOM")}, "model_exception"),
            ({"confidence": 0.05, "min_confidence": 0.5}, "low_confidence"),
            ({"frame_timestamp": -1.0}, "invalid_timestamp"),
        ],
    )
    def test_condition_forces_stop(self, monitor, kwargs, violation) -> None:
        """Every fail-safe condition is detected and forces a halt."""
        r = monitor.check(**kwargs)
        assert not r.safe
        assert violation in r.violations
        assert r.command in (NavigationCommand.STOP, NavigationCommand.EMERGENCY_STOP)
        assert r.messages, "every violation must carry an explanation"

    def test_sudden_depth_change(self, monitor) -> None:
        """A large jump between frames is flagged as a discontinuity."""
        monitor.check(sector_distances={"CTR": 5.0})
        r = monitor.check(sector_distances={"CTR": 0.5})
        assert "depth_discontinuity" in r.violations

    def test_timestamp_regression(self, monitor) -> None:
        """Out-of-order frames are detected."""
        monitor.check(frame_timestamp=1000.0)
        r = monitor.check(frame_timestamp=999.0)
        assert "timestamp_regression" in r.violations


class TestEmergencyStop:
    """Latching behaviour (spec section AE)."""

    def test_latches_after_consecutive_failures(self, monitor) -> None:
        """Three consecutive unsafe frames latch EMERGENCY_STOP."""
        for _ in range(2):
            r = monitor.check(frame_age_s=10.0)
            assert r.state == SafetyState.DEGRADED
        r = monitor.check(frame_age_s=10.0)
        assert r.state == SafetyState.EMERGENCY_STOP
        assert r.command == NavigationCommand.EMERGENCY_STOP

    def test_latch_survives_recovery(self, monitor) -> None:
        """A latched stop is NOT cleared by healthy frames."""
        for _ in range(3):
            monitor.check(frame_age_s=10.0)
        r = monitor.check(frame_age_s=0.0, inference_latency_s=0.01)
        assert r.command == NavigationCommand.EMERGENCY_STOP
        assert monitor.is_emergency

    def test_reset_clears_the_latch(self, monitor) -> None:
        """Only an explicit reset clears the latch."""
        for _ in range(3):
            monitor.check(frame_age_s=10.0)
        monitor.reset()
        assert not monitor.is_emergency
        assert monitor.check(frame_age_s=0.01, inference_latency_s=0.01).safe

    def test_manual_trigger(self, monitor) -> None:
        """A software e-stop can be raised directly."""
        r = monitor.trigger_emergency_stop("user pressed stop")
        assert r.command == NavigationCommand.EMERGENCY_STOP
        assert monitor.is_emergency
        assert "user pressed stop" in r.messages[0]


class TestCommandOverride:
    """The safety layer must have the final word."""

    def test_override_replaces_a_moving_command(self, monitor) -> None:
        """An unsafe frame overrides FORWARD with STOP."""
        r = monitor.check(depth_valid=False)
        assert monitor.apply(NavigationCommand.FORWARD, r) == NavigationCommand.STOP

    def test_healthy_frame_passes_command_through(self, monitor) -> None:
        """A healthy frame leaves the planner's decision intact."""
        r = monitor.check(frame_age_s=0.01, inference_latency_s=0.01)
        assert monitor.apply(NavigationCommand.FORWARD, r) == NavigationCommand.FORWARD

    def test_disabled_monitor_never_blocks(self) -> None:
        """Disabling the monitor is possible but reports OK unconditionally."""
        m = SafetyMonitor({"temporal_safety": {"enabled": False}})
        assert m.check(frame_age_s=999.0, depth_valid=False).safe


class TestFrameTimer:
    """Per-stage timing used by the latency fail-safe."""

    def test_records_stage_durations(self) -> None:
        """Stages are timed and recorded in milliseconds."""
        t = FrameTimer()
        t.start("detection")
        assert t.stop("detection") >= 0.0
        assert "detection" in t.stages

    def test_stopping_an_unstarted_stage_raises(self) -> None:
        """Stopping a stage that was never started is a programming error."""
        with pytest.raises(KeyError, match="never started"):
            FrameTimer().stop("ghost")

    def test_reset_clears_stages(self) -> None:
        """Reset prepares the timer for a new frame."""
        t = FrameTimer()
        t.start("a")
        t.stop("a")
        t.reset()
        assert t.stages == {}
