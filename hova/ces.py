"""
hova/ces.py
Layer 1 — Cognitive Entropy Signature (CES)

CES(D) = H̄_local(D) / H_global(D)

Human writing is locally unpredictable (tangents, pivots) but globally coherent.
AI writing is locally smooth but globally flat.  This ratio captures that distinction
as a continuous signal in (0, ∞) and maps it to a training weight in (0, 1).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Union

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

from hova.config import CESConfig


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """Lowercase whitespace-split tokenisation (no external deps required)."""
    return re.findall(r"\b\w+\b", text.lower())


def _entropy(counts: Counter) -> float:
    """Shannon entropy (bits) of a token count distribution.

    Returns 0.0 for empty or single-token distributions.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def _local_entropy_at(tokens: List[str], k: int, window_size: int) -> float:
    """Local entropy centred at position *k* with half-window *window_size // 2*."""
    half = window_size // 2
    start = max(0, k - half)
    end = min(len(tokens), k + half + 1)
    window = tokens[start:end]
    return _entropy(Counter(window))


def _mean_local_entropy(tokens: List[str], window_size: int) -> float:
    """H̄_local: mean local entropy across all positions in the token list."""
    if not tokens:
        return 0.0
    total = sum(_local_entropy_at(tokens, k, window_size) for k in range(len(tokens)))
    return total / len(tokens)


def _global_entropy(tokens: List[str]) -> float:
    """H_global: Shannon entropy over the full document token distribution."""
    return _entropy(Counter(tokens))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class CESScorer:
    """Score documents using the Cognitive Entropy Signature (Layer 1).

    Parameters
    ----------
    config:
        ``CESConfig`` with hyperparameters.  If ``None``, defaults are used.
    window_size:
        Override for ``config.window_size``.  Convenience parameter.

    Examples
    --------
    >>> scorer = CESScorer()
    >>> ces_value = scorer.score("The quick brown fox jumps over the lazy dog.")
    >>> weight = scorer.weight("Some text here...")
    >>> df = scorer.score_batch(["doc one", "doc two"])
    """

    def __init__(
        self,
        config: Optional[CESConfig] = None,
        *,
        window_size: Optional[int] = None,
    ) -> None:
        self._cfg = config or CESConfig()
        if window_size is not None:
            self._cfg.window_size = window_size

    # ── Core scoring ──────────────────────────────────────────────────────────

    def score(self, text: str) -> float:
        """Compute CES(D) for a single document.

        Parameters
        ----------
        text:
            Raw document string.

        Returns
        -------
        float
            CES value in (0, ∞).  Values > 1.3 indicate strong human signal.
            Returns 0.0 for empty or single-word documents.
        """
        tokens = _tokenise(text)
        if len(tokens) < 2:
            return 0.0
        h_global = _global_entropy(tokens)
        if h_global == 0.0:
            return 0.0
        h_local_mean = _mean_local_entropy(tokens, self._cfg.window_size)
        return h_local_mean / h_global

    def weight(self, text: str) -> float:
        """Map CES → soft training weight in (0, 1).

        weight = sigmoid(alpha * (CES(D) − tau))

        Parameters
        ----------
        text:
            Raw document string.

        Returns
        -------
        float
            Training weight in (0, 1).
        """
        ces = self.score(text)
        return _sigmoid(self._cfg.alpha * (ces - self._cfg.tau))

    def interpret(self, ces_value: float) -> str:
        """Return a human-readable interpretation of a CES value.

        Parameters
        ----------
        ces_value:
            Computed CES score.

        Returns
        -------
        str
            Qualitative label.
        """
        if ces_value >= 1.3:
            return "Strong human signal"
        elif ces_value >= 1.0:
            return "Ambiguous — possible human or high-quality synthetic"
        elif ces_value >= 0.7:
            return "Likely AI-generated — downweight"
        else:
            return "Almost certainly AI-generated — discard"

    def should_discard(self, text: str) -> bool:
        """Return True if the document's CES falls below the discard threshold.

        Parameters
        ----------
        text:
            Raw document string.

        Returns
        -------
        bool
            True if CES < ``config.discard_below``.
        """
        return self.score(text) < self._cfg.discard_below

    # ── Batch scoring ─────────────────────────────────────────────────────────

    def score_batch(
        self,
        documents: Sequence[str],
        ids: Optional[Sequence] = None,
    ) -> "pd.DataFrame":
        """Score a collection of documents and return a summary DataFrame.

        Parameters
        ----------
        documents:
            Iterable of raw text strings.
        ids:
            Optional document identifiers.  If ``None``, integer indices are used.

        Returns
        -------
        pandas.DataFrame
            Columns: ``id``, ``ces``, ``weight``, ``interpretation``, ``discard``.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        if not _PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for score_batch(). "
                "Install with: pip install pandas"
            )
        docs = list(documents)
        _ids = list(ids) if ids is not None else list(range(len(docs)))
        rows = []
        for doc_id, text in zip(_ids, docs):
            ces = self.score(text)
            rows.append(
                {
                    "id": doc_id,
                    "ces": round(ces, 4),
                    "weight": round(self.weight(text), 4),
                    "interpretation": self.interpret(ces),
                    "discard": ces < self._cfg.discard_below,
                }
            )
        return pd.DataFrame(rows)

    def filter_corpus(
        self,
        documents: Sequence[str],
        ids: Optional[Sequence] = None,
    ) -> List[dict]:
        """Filter a corpus, returning only documents that pass the CES threshold.

        Parameters
        ----------
        documents:
            Sequence of raw document strings.
        ids:
            Optional document identifiers.

        Returns
        -------
        list of dict
            Each dict has keys ``id``, ``text``, ``ces``, ``weight``.
        """
        docs = list(documents)
        _ids = list(ids) if ids is not None else list(range(len(docs)))
        results = []
        for doc_id, text in zip(_ids, docs):
            ces = self.score(text)
            if ces >= self._cfg.discard_below:
                results.append(
                    {
                        "id": doc_id,
                        "text": text,
                        "ces": round(ces, 4),
                        "weight": round(_sigmoid(self._cfg.alpha * (ces - self._cfg.tau)), 4),
                    }
                )
        return results

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def explain(self, text: str) -> dict:
        """Return a detailed breakdown of the CES computation.

        Parameters
        ----------
        text:
            Raw document string.

        Returns
        -------
        dict
            Keys: ``tokens``, ``h_local_mean``, ``h_global``, ``ces``, ``weight``,
            ``interpretation``, ``discard``.
        """
        tokens = _tokenise(text)
        if len(tokens) < 2:
            return {
                "tokens": len(tokens),
                "h_local_mean": 0.0,
                "h_global": 0.0,
                "ces": 0.0,
                "weight": 0.0,
                "interpretation": "Too short to score",
                "discard": True,
            }
        h_global = _global_entropy(tokens)
        h_local_mean = _mean_local_entropy(tokens, self._cfg.window_size)
        ces = h_local_mean / h_global if h_global > 0 else 0.0
        w = _sigmoid(self._cfg.alpha * (ces - self._cfg.tau))
        return {
            "tokens": len(tokens),
            "h_local_mean": round(h_local_mean, 4),
            "h_global": round(h_global, 4),
            "ces": round(ces, 4),
            "weight": round(w, 4),
            "interpretation": self.interpret(ces),
            "discard": ces < self._cfg.discard_below,
        }
