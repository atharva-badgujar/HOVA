"""
hova/cews.py
Layer 5 — Collapse Early Warning System (CEWS)

All upstream layers reduce collapse risk.  If contamination is severe enough
to slip through, CEWS detects the onset of collapse BEFORE it becomes
irreversible by monitoring three independent signals unified into a single
Collapse Risk Index (CRI).

    CRI(c) = w1·σ(CES_drift) + w2·σ(CHT_grad) + w3·σ(AnchorDiv)
    Thresholds: Green<0.40 | Yellow<0.60 | Orange<0.75 | Red≥0.75
"""

from __future__ import annotations

import logging
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from hova.config import CEWSConfig

logger = logging.getLogger("hova.cews")


# ──────────────────────────────────────────────────────────────────────────────
# Status enum
# ──────────────────────────────────────────────────────────────────────────────

class CollapseStatus(str, Enum):
    """Health status levels for the Collapse Early Warning System."""

    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"

    @property
    def emoji(self) -> str:
        return {
            CollapseStatus.GREEN: "🟢",
            CollapseStatus.YELLOW: "🟡",
            CollapseStatus.ORANGE: "🟠",
            CollapseStatus.RED: "🔴",
        }[self]

    @property
    def description(self) -> str:
        return {
            CollapseStatus.GREEN: "Normal training — no action required",
            CollapseStatus.YELLOW: "Log warning — increase monitoring frequency",
            CollapseStatus.ORANGE: "Alert human operator — flag data batch for audit",
            CollapseStatus.RED: "AUTOMATIC TRAINING PAUSE — mandatory data pipeline audit",
        }[self]


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint data structure
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Checkpoint:
    """Single CEWS checkpoint observation.

    Attributes
    ----------
    step:
        Training step number.
    ces_mean:
        Mean CES score across the last batch.
    cht_vector:
        CHT values [Ω_op, Ω_emo, Ω_lex, Ω_cult] at this checkpoint.
    anchor_div:
        KL divergence from the anchor model on the probe set.
    cri:
        Computed Collapse Risk Index.
    status:
        Colour-coded collapse status.
    timestamp:
        Wall-clock time of this checkpoint.
    """

    step: int
    ces_mean: float
    cht_vector: List[float]
    anchor_div: float
    cri: float
    status: CollapseStatus
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "ces_mean": round(self.ces_mean, 4),
            "cht_vector": [round(x, 4) for x in self.cht_vector],
            "anchor_div": round(self.anchor_div, 4),
            "cri": round(self.cri, 4),
            "status": self.status.value,
            "timestamp": self.timestamp,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Min-max normalisation helper
# ──────────────────────────────────────────────────────────────────────────────

def _minmax_normalise(values: List[float]) -> List[float]:
    """Min-max normalise a list of floats to [0, 1]."""
    if not values:
        return values
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class CEWSMonitor:
    """Monitor training collapse risk and emit alerts or pause signals.

    Maintains a rolling history of three independent signals
    (CES drift, CHT gradient, Anchor divergence) and combines them into
    a single Collapse Risk Index (CRI ∈ [0, 1]).

    Parameters
    ----------
    config:
        ``CEWSConfig`` hyperparameters.  Uses defaults if ``None``.
    on_alert:
        Optional callback invoked when CRI reaches Orange or Red.
        Signature: ``callback(checkpoint: Checkpoint)``.
    on_pause:
        Optional callback invoked when training should auto-pause (Red).
        Signature: ``callback(checkpoint: Checkpoint)``.

    Examples
    --------
    >>> monitor = CEWSMonitor()
    >>> monitor.update(ces_mean=1.2, cht_vector=[0.4, 1.6, 0.45, 1.3], anchor_div=0.1)
    >>> print(monitor.current_status)
    >>> if monitor.is_paused():
    ...     monitor.alert()
    """

    def __init__(
        self,
        config: Optional[CEWSConfig] = None,
        on_alert: Optional[Callable[[Checkpoint], None]] = None,
        on_pause: Optional[Callable[[Checkpoint], None]] = None,
    ) -> None:
        self._cfg = config or CEWSConfig()
        self._on_alert = on_alert
        self._on_pause = on_pause
        self._step: int = 0

        # Rolling histories for normalisation
        self._ces_history: Deque[float] = deque(maxlen=self._cfg.history_window + 1)
        self._cht_history: Deque[List[float]] = deque(maxlen=self._cfg.history_window + 1)
        self._anc_history: Deque[float] = deque(maxlen=self._cfg.history_window + 1)

        # Full checkpoint log
        self.checkpoint_history: List[Checkpoint] = []

        self._paused: bool = False
        self._latest_checkpoint: Optional[Checkpoint] = None

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        ces_mean: float,
        cht_vector: List[float],
        anchor_div: float,
    ) -> Optional[Checkpoint]:
        """Register a new observation and optionally emit a CRI checkpoint.

        This method should be called every training step.  A full CRI
        computation is only performed every ``config.checkpoint_every`` steps.

        Parameters
        ----------
        ces_mean:
            Mean CES score across the most recent batch of documents.
        cht_vector:
            Current CHT values as a list [Ω_op, Ω_emo, Ω_lex, Ω_cult].
        anchor_div:
            Divergence from the anchor model on the probe set.

        Returns
        -------
        Checkpoint or None
            A ``Checkpoint`` object if a CRI computation was performed,
            otherwise ``None``.
        """
        self._step += 1
        self._ces_history.append(ces_mean)
        self._cht_history.append(list(cht_vector))
        self._anc_history.append(anchor_div)

        if self._step % self._cfg.checkpoint_every != 0:
            return None

        cri, signals = self._compute_cri()
        status = self._classify(cri)
        ckpt = Checkpoint(
            step=self._step,
            ces_mean=ces_mean,
            cht_vector=list(cht_vector),
            anchor_div=anchor_div,
            cri=cri,
            status=status,
        )
        self.checkpoint_history.append(ckpt)
        self._latest_checkpoint = ckpt

        if status == CollapseStatus.RED:
            self._paused = True

        self._log_checkpoint(ckpt, signals)
        self._fire_callbacks(ckpt)

        return ckpt

    # ── Status queries ────────────────────────────────────────────────────────

    def is_paused(self) -> bool:
        """Return True if a RED CRI has triggered an automatic training pause."""
        return self._paused

    def resume(self) -> None:
        """Reset the paused state after a manual audit clears the pipeline.

        Call this after:
        1. Identifying the contaminated data batches.
        2. Re-enriching with verified human data.
        3. Resetting the CRI history.
        """
        self._paused = False
        self._ces_history.clear()
        self._cht_history.clear()
        self._anc_history.clear()
        logger.info("CEWS: Training resumed after audit. CRI history reset.")

    @property
    def current_cri(self) -> Optional[float]:
        """Most recently computed CRI value, or None if no checkpoint yet."""
        return self._latest_checkpoint.cri if self._latest_checkpoint else None

    @property
    def current_status(self) -> Optional[CollapseStatus]:
        """Most recent collapse status colour."""
        return self._latest_checkpoint.status if self._latest_checkpoint else None

    def alert(self) -> None:
        """Log a structured alert message with the latest checkpoint details.

        Also fires the ``on_alert`` callback if registered.
        """
        if self._latest_checkpoint is None:
            logger.warning("CEWS: alert() called but no checkpoints have been recorded.")
            return
        ckpt = self._latest_checkpoint
        msg = (
            f"\n{'='*60}\n"
            f"  HOVA CEWS ALERT — Step {ckpt.step}\n"
            f"{'='*60}\n"
            f"  Status  : {ckpt.status.emoji} {ckpt.status.value.upper()}\n"
            f"  CRI     : {ckpt.cri:.4f}\n"
            f"  CES mean: {ckpt.ces_mean:.4f}\n"
            f"  CHT     : {[round(x,3) for x in ckpt.cht_vector]}\n"
            f"  AncDiv  : {ckpt.anchor_div:.4f}\n"
            f"  Action  : {ckpt.status.description}\n"
            f"{'='*60}"
        )
        if ckpt.status == CollapseStatus.RED:
            logger.critical(msg)
        elif ckpt.status == CollapseStatus.ORANGE:
            logger.warning(msg)
        else:
            logger.info(msg)

    def summary(self) -> dict:
        """Return a summary of the monitor's current state.

        Returns
        -------
        dict
            Keys: ``step``, ``n_checkpoints``, ``current_cri``, ``current_status``,
            ``is_paused``, ``latest_checkpoint``.
        """
        return {
            "step": self._step,
            "n_checkpoints": len(self.checkpoint_history),
            "current_cri": round(self.current_cri, 4) if self.current_cri is not None else None,
            "current_status": self.current_status.value if self.current_status else None,
            "is_paused": self._paused,
            "latest_checkpoint": (
                self._latest_checkpoint.to_dict()
                if self._latest_checkpoint else None
            ),
        }

    def export_history(self) -> List[dict]:
        """Export the full checkpoint history as a list of dicts.

        Returns
        -------
        list of dict
            Each dict corresponds to one ``Checkpoint``.
        """
        return [ckpt.to_dict() for ckpt in self.checkpoint_history]

    # ── CRI computation ───────────────────────────────────────────────────────

    def _compute_cri(self) -> Tuple[float, dict]:
        """Compute CRI from rolling signal histories.

        Returns
        -------
        (cri, signals)
            CRI ∈ [0, 1] and a dict of intermediate values.
        """
        ces_vals = list(self._ces_history)
        cht_vals = list(self._cht_history)
        anc_vals = list(self._anc_history)

        # CES_drift: drop in mean CES (positive = declining quality)
        if len(ces_vals) >= 2:
            ces_drift = max(ces_vals[-2] - ces_vals[-1], 0.0)
        else:
            ces_drift = 0.0

        # CHT_grad: L2 decline in health tensor
        if len(cht_vals) >= 2:
            delta = np.array(cht_vals[-2]) - np.array(cht_vals[-1])
            cht_grad = float(np.linalg.norm(np.clip(delta, 0, None)))
        else:
            cht_grad = 0.0

        # AnchorDiv: latest anchor divergence
        anchor_div = anc_vals[-1] if anc_vals else 0.0

        # Min-max normalise over recent history
        all_drifts = [max(ces_vals[i - 1] - ces_vals[i], 0.0) for i in range(1, len(ces_vals))]
        all_grads = [
            float(np.linalg.norm(np.clip(np.array(cht_vals[i - 1]) - np.array(cht_vals[i]), 0, None)))
            for i in range(1, len(cht_vals))
        ]

        def _norm(val, history):
            history_with_val = history + [val]
            mn, mx = min(history_with_val), max(history_with_val)
            if mx == mn:
                return 0.5
            return (val - mn) / (mx - mn)

        ces_norm = _norm(ces_drift, all_drifts[:-1] if all_drifts else [])
        cht_norm = _norm(cht_grad, all_grads[:-1] if all_grads else [])
        anc_norm = _norm(anchor_div, list(anc_vals)[:-1] if len(anc_vals) > 1 else [])

        cri = (
            self._cfg.w_ces * ces_norm
            + self._cfg.w_cht * cht_norm
            + self._cfg.w_anc * anc_norm
        )
        cri = max(0.0, min(1.0, cri))

        signals = {
            "ces_drift": round(ces_drift, 4),
            "cht_grad": round(cht_grad, 4),
            "anchor_div": round(anchor_div, 4),
            "ces_norm": round(ces_norm, 4),
            "cht_norm": round(cht_norm, 4),
            "anc_norm": round(anc_norm, 4),
        }
        return cri, signals

    def _classify(self, cri: float) -> CollapseStatus:
        cfg = self._cfg
        if cri >= cfg.red_threshold:
            return CollapseStatus.RED
        elif cri >= cfg.orange_threshold:
            return CollapseStatus.ORANGE
        elif cri >= cfg.yellow_threshold:
            return CollapseStatus.YELLOW
        return CollapseStatus.GREEN

    def _log_checkpoint(self, ckpt: Checkpoint, signals: dict) -> None:
        msg = (
            f"CEWS step={ckpt.step} | "
            f"CRI={ckpt.cri:.3f} [{ckpt.status.emoji} {ckpt.status.value}] | "
            f"CES_mean={ckpt.ces_mean:.3f} | "
            f"AncDiv={ckpt.anchor_div:.3f}"
        )
        if ckpt.status == CollapseStatus.RED:
            logger.critical(msg)
        elif ckpt.status == CollapseStatus.ORANGE:
            logger.warning(msg)
        elif ckpt.status == CollapseStatus.YELLOW:
            logger.warning(msg)
        else:
            logger.info(msg)

    def _fire_callbacks(self, ckpt: Checkpoint) -> None:
        if ckpt.status in (CollapseStatus.ORANGE, CollapseStatus.RED):
            if self._on_alert:
                try:
                    self._on_alert(ckpt)
                except Exception as exc:
                    logger.error(f"CEWS on_alert callback raised: {exc}")
        if ckpt.status == CollapseStatus.RED:
            if self._on_pause:
                try:
                    self._on_pause(ckpt)
                except Exception as exc:
                    logger.error(f"CEWS on_pause callback raised: {exc}")
