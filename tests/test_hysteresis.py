"""Majority-vote hysteresis tests (spec section AU item 10)."""

from __future__ import annotations

from navigation.free_path import NavigationCommand as C
from navigation.hysteresis import MajorityVoteHysteresis


class TestWarmup:
    """Before the window fills there is not enough evidence to act."""

    def test_emits_stop_during_warmup(self) -> None:
        """The first N-1 frames return the warm-up command."""
        h = MajorityVoteHysteresis(window=3)
        assert h.update(C.FORWARD) == C.STOP
        assert h.update(C.FORWARD) == C.STOP
        assert not h.is_warmed_up, "two frames is not a full N=3 window"
        assert h.update(C.FORWARD) == C.FORWARD
        assert h.is_warmed_up

    def test_window_of_one_is_pass_through(self) -> None:
        """N=1 makes the filter transparent."""
        h = MajorityVoteHysteresis(window=1)
        assert h.update(C.FORWARD) == C.FORWARD


class TestMajorityVote:
    """The core smoothing behaviour."""

    def test_suppresses_single_frame_glitch(self) -> None:
        """One anomalous frame cannot change the command."""
        h = MajorityVoteHysteresis(window=3)
        for _ in range(3):
            h.update(C.FORWARD)
        assert h.update(C.TURN_LEFT) == C.FORWARD

    def test_sustained_change_is_adopted(self) -> None:
        """A genuine change takes effect once it holds the majority."""
        h = MajorityVoteHysteresis(window=3)
        for _ in range(3):
            h.update(C.FORWARD)
        h.update(C.TURN_LEFT)
        assert h.update(C.TURN_LEFT) == C.TURN_LEFT

    def test_stop_overrides_the_majority(self) -> None:
        """A single STOP is never voted away — it is always safe to stop."""
        h = MajorityVoteHysteresis(window=3)
        h.update(C.FORWARD)
        h.update(C.FORWARD)
        assert h.update(C.STOP) == C.STOP

    def test_stop_override_can_be_disabled(self) -> None:
        """With the override off, STOP is subject to the plain majority."""
        h = MajorityVoteHysteresis(window=3, stop_override=False)
        h.update(C.FORWARD)
        h.update(C.FORWARD)
        assert h.update(C.STOP) == C.FORWARD


class TestTieHandling:
    """An unresolved tie must resolve to the safe action."""

    def test_three_way_tie_defaults_to_stop(self) -> None:
        """Indecision resolves to STOP for a wheelchair."""
        h = MajorityVoteHysteresis(window=3, stop_override=False)
        h.update(C.FORWARD)
        h.update(C.TURN_LEFT)
        assert h.update(C.TURN_RIGHT) == C.STOP

    def test_hold_previous_is_opt_in(self) -> None:
        """Holding the previous command must be explicitly enabled."""
        h = MajorityVoteHysteresis(window=3, hold_previous_on_tie=True, stop_override=False)
        for _ in range(3):
            h.update(C.FORWARD)
        h.update(C.TURN_LEFT)
        h.update(C.TURN_RIGHT)
        assert h.update(C.FORWARD) in {C.FORWARD, C.TURN_LEFT, C.TURN_RIGHT}


class TestEmergencyAndReset:
    """Emergency handling and state management."""

    def test_emergency_stop_bypasses_the_filter(self) -> None:
        """EMERGENCY_STOP takes effect on its own frame, not two later."""
        h = MajorityVoteHysteresis(window=3)
        assert h.update(C.EMERGENCY_STOP) == C.EMERGENCY_STOP

    def test_reset_returns_to_warmup(self) -> None:
        """Reset clears history so the filter re-warms."""
        h = MajorityVoteHysteresis(window=3)
        for _ in range(3):
            h.update(C.FORWARD)
        h.reset()
        assert not h.is_warmed_up
        assert h.update(C.FORWARD) == C.STOP

    def test_disabled_filter_is_transparent(self) -> None:
        """Disabling hysteresis passes commands through, for ablation."""
        h = MajorityVoteHysteresis(window=3, enabled=False)
        assert h.update(C.FORWARD) == C.FORWARD

    def test_accepts_string_commands(self) -> None:
        """String input is coerced to the enum."""
        h = MajorityVoteHysteresis(window=1)
        assert h.update("FORWARD") == C.FORWARD

    def test_state_snapshot(self) -> None:
        """The state snapshot exposes the vote window for the HUD."""
        h = MajorityVoteHysteresis(window=3)
        h.update(C.FORWARD)
        s = h.state()
        assert s["window"] == 3 and s["history"] == ["FORWARD"]


class TestConfigConstruction:
    """Construction from the shipped YAML."""

    def test_from_navigation_config(self, nav_config) -> None:
        """Defaults match the spec: N=3, fail-safe STOP, no tie-holding."""
        h = MajorityVoteHysteresis.from_config(nav_config)
        assert h.window == 3
        assert h.tie_command == C.STOP
        assert h.hold_previous_on_tie is False

    def test_invalid_window_rejected(self) -> None:
        """A window below 1 is meaningless."""
        try:
            MajorityVoteHysteresis(window=0)
        except ValueError as e:
            assert "window must be" in str(e)
        else:
            raise AssertionError("window=0 should raise ValueError")
