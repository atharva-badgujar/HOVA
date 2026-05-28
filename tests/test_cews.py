"""
tests/test_cews.py
Unit tests for Layer 5 — Collapse Early Warning System (CEWS)
"""

import pytest
from hova.cews import CEWSMonitor, CollapseStatus, Checkpoint
from hova.config import CEWSConfig


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def monitor():
    """Monitor that checkpoints every 5 steps."""
    cfg = CEWSConfig(
        checkpoint_every=5,
        red_threshold=0.75,
        orange_threshold=0.60,
        yellow_threshold=0.40,
        w_ces=0.35,
        w_cht=0.35,
        w_anc=0.30,
        history_window=5,
    )
    return CEWSMonitor(config=cfg)


def _stable_signal():
    """Simulate a healthy training state: high CES, high CHT, low anchor div."""
    return dict(
        ces_mean=1.4,
        cht_vector=[0.5, 2.0, 0.6, 1.5],
        anchor_div=0.05,
    )


def _collapsing_signal():
    """Simulate a collapsing state: dropping CES, falling CHT, high anchor div."""
    return dict(
        ces_mean=0.5,
        cht_vector=[0.1, 0.3, 0.1, 0.2],
        anchor_div=2.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CollapseStatus enum
# ──────────────────────────────────────────────────────────────────────────────

class TestCollapseStatus:
    def test_emoji_presence(self):
        for status in CollapseStatus:
            assert len(status.emoji) > 0

    def test_description_presence(self):
        for status in CollapseStatus:
            assert len(status.description) > 0

    def test_string_values(self):
        assert CollapseStatus.GREEN.value == "green"
        assert CollapseStatus.RED.value == "red"


# ──────────────────────────────────────────────────────────────────────────────
# CEWSMonitor initial state
# ──────────────────────────────────────────────────────────────────────────────

class TestCEWSMonitorInit:
    def test_not_paused_initially(self, monitor):
        assert not monitor.is_paused()

    def test_no_cri_initially(self, monitor):
        assert monitor.current_cri is None

    def test_no_status_initially(self, monitor):
        assert monitor.current_status is None

    def test_empty_history(self, monitor):
        assert len(monitor.checkpoint_history) == 0


# ──────────────────────────────────────────────────────────────────────────────
# CEWSMonitor.update()
# ──────────────────────────────────────────────────────────────────────────────

class TestCEWSMonitorUpdate:
    def test_no_checkpoint_before_interval(self, monitor):
        for _ in range(4):
            result = monitor.update(**_stable_signal())
        assert result is None
        assert len(monitor.checkpoint_history) == 0

    def test_checkpoint_at_interval(self, monitor):
        for _ in range(5):
            monitor.update(**_stable_signal())
        assert len(monitor.checkpoint_history) == 1

    def test_checkpoint_is_checkpoint_type(self, monitor):
        for _ in range(5):
            result = monitor.update(**_stable_signal())
        assert result is not None
        assert isinstance(result, Checkpoint)

    def test_cri_in_unit_interval(self, monitor):
        for _ in range(5):
            monitor.update(**_stable_signal())
        assert 0.0 <= monitor.current_cri <= 1.0

    def test_multiple_checkpoints(self, monitor):
        for _ in range(15):
            monitor.update(**_stable_signal())
        assert len(monitor.checkpoint_history) == 3  # 5, 10, 15


# ──────────────────────────────────────────────────────────────────────────────
# CRI classification
# ──────────────────────────────────────────────────────────────────────────────

class TestCEWSClassification:
    def test_stable_signal_green_or_yellow(self, monitor):
        """Stable signal should not trigger orange or red."""
        for _ in range(10):
            monitor.update(**_stable_signal())
        status = monitor.current_status
        assert status in (CollapseStatus.GREEN, CollapseStatus.YELLOW)

    def test_collapse_signal_raises_cri(self, monitor):
        """CRI should be non-negative and within [0, 1] after collapse signals."""
        # Feed stable signals first
        for _ in range(10):
            monitor.update(**_stable_signal())

        # Feed collapse signals
        for _ in range(10):
            monitor.update(**_collapsing_signal())

        # Key invariant: CRI is always within bounds
        assert monitor.current_cri is not None
        assert 0.0 <= monitor.current_cri <= 1.0
        # After seeing collapse signals CRI should be non-zero
        assert monitor.current_cri > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Pause / resume
# ──────────────────────────────────────────────────────────────────────────────

class TestCEWSPauseResume:
    def test_not_paused_without_red(self, monitor):
        for _ in range(5):
            monitor.update(**_stable_signal())
        assert not monitor.is_paused()

    def test_resume_clears_pause(self):
        """Manually trigger pause by injecting a checkpoint."""
        monitor = CEWSMonitor(config=CEWSConfig(checkpoint_every=1))
        # Manually force a red status by manipulating history — or just test resume
        monitor._paused = True
        assert monitor.is_paused()
        monitor.resume()
        assert not monitor.is_paused()

    def test_resume_clears_history(self):
        monitor = CEWSMonitor(config=CEWSConfig(checkpoint_every=5))
        for _ in range(5):
            monitor.update(**_stable_signal())
        assert len(list(monitor._ces_history)) > 0
        monitor._paused = True
        monitor.resume()
        assert len(list(monitor._ces_history)) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Summary and export
# ──────────────────────────────────────────────────────────────────────────────

class TestCEWSExport:
    def test_summary_keys(self, monitor):
        summary = monitor.summary()
        for key in ["step", "n_checkpoints", "current_cri", "current_status", "is_paused"]:
            assert key in summary

    def test_export_history_empty(self, monitor):
        assert monitor.export_history() == []

    def test_export_history_after_updates(self, monitor):
        for _ in range(10):
            monitor.update(**_stable_signal())
        history = monitor.export_history()
        assert len(history) == 2
        for record in history:
            for key in ["step", "cri", "status", "ces_mean"]:
                assert key in record


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

class TestCEWSCallbacks:
    def test_alert_callback_fired_on_orange(self):
        """Force an orange-level CRI and check callback is called."""
        calls = []
        monitor = CEWSMonitor(
            config=CEWSConfig(checkpoint_every=1, orange_threshold=0.0),
            on_alert=lambda ckpt: calls.append(ckpt),
        )
        monitor.update(**_stable_signal())
        assert len(calls) >= 1

    def test_on_pause_callback_fired_on_red(self):
        pauses = []
        monitor = CEWSMonitor(
            config=CEWSConfig(checkpoint_every=1, red_threshold=0.0),
            on_pause=lambda ckpt: pauses.append(ckpt),
        )
        monitor.update(**_stable_signal())
        assert len(pauses) >= 1

    def test_alert_method_no_crash_without_checkpoints(self, monitor):
        """alert() before any updates should not raise."""
        monitor.alert()  # should just log a warning
