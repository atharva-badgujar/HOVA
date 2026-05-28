"""
hova/dps.py
Layer 4 — Disagreement Preservation Sampling (DPS)

Model collapse is a VARIANCE problem — the distribution narrows.
DPS operationalises corpus health as a 4-dimensional tensor (CHT) and
enforces minimum thresholds during every training batch.

    CHT(B) = [Ω_op, Ω_emo, Ω_lex, Ω_cult]
    Sample B* = argmax_B P(B) subject to CHT(B*) ≥ CHT_min

Documents that increase batch variance are preferentially sampled.
"""

from __future__ import annotations

import math
import random
import re
import warnings
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from hova.config import DPSConfig


# ──────────────────────────────────────────────────────────────────────────────
# Optional dependencies
# ──────────────────────────────────────────────────────────────────────────────

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from textblob import TextBlob  # type: ignore
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False

try:
    from langdetect import detect  # type: ignore
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CHTConfig:
    """Minimum thresholds for the Corpus Health Tensor.

    Parameters
    ----------
    min_op:
        Minimum opinion divergence Ω_op (variance of opinion embeddings).
    min_emo:
        Minimum emotional entropy Ω_emo in bits.
    min_lex:
        Minimum lexical diversity Ω_lex (mean pairwise distance).
    min_cult:
        Minimum cultural entropy Ω_cult in bits.
    """

    min_op: float = 0.35
    min_emo: float = 1.5
    min_lex: float = 0.40
    min_cult: float = 1.2

    def as_array(self) -> np.ndarray:
        return np.array([self.min_op, self.min_emo, self.min_lex, self.min_cult])


@dataclass
class CHTVector:
    """Observed Corpus Health Tensor values for a batch.

    Attributes
    ----------
    omega_op:
        Opinion divergence variance.
    omega_emo:
        Emotional entropy in bits.
    omega_lex:
        Mean pairwise lexical distance.
    omega_cult:
        Language-region entropy in bits.
    """

    omega_op: float
    omega_emo: float
    omega_lex: float
    omega_cult: float

    def as_array(self) -> np.ndarray:
        return np.array([self.omega_op, self.omega_emo, self.omega_lex, self.omega_cult])

    def satisfies(self, config: CHTConfig) -> bool:
        """Return True if all CHT components meet their minimums."""
        a = self.as_array()
        b = config.as_array()
        return bool(np.all(a >= b))

    def deficit(self, config: CHTConfig) -> np.ndarray:
        """Return the component-wise shortfall (negative = deficit)."""
        return self.as_array() - config.as_array()

    def to_dict(self) -> dict:
        return {
            "omega_op": round(self.omega_op, 4),
            "omega_emo": round(self.omega_emo, 4),
            "omega_lex": round(self.omega_lex, 4),
            "omega_cult": round(self.omega_cult, 4),
        }

    def status(self, config: CHTConfig) -> str:
        if self.satisfies(config):
            return "✅ Healthy — all CHT constraints satisfied"
        deficits = []
        names = ["Ω_op", "Ω_emo", "Ω_lex", "Ω_cult"]
        for name, obs, mn in zip(
            names,
            self.as_array(),
            config.as_array(),
        ):
            if obs < mn:
                deficits.append(f"{name}={obs:.3f} < {mn:.3f}")
        return "⚠️  CHT deficit: " + ", ".join(deficits)


# ──────────────────────────────────────────────────────────────────────────────
# CHT computation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _entropy(dist: Counter) -> float:
    """Shannon entropy in bits."""
    total = sum(dist.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in dist.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def _sentiment_label(text: str) -> str:
    """Map text to sentiment label: positive / neutral / negative."""
    if _TEXTBLOB_AVAILABLE:
        polarity = TextBlob(text).sentiment.polarity
    else:
        pos_words = {"good", "great", "excellent", "happy", "love", "wonderful", "best"}
        neg_words = {"bad", "terrible", "awful", "sad", "hate", "horrible", "worst"}
        tokens = set(_tokenise(text))
        pos = len(tokens & pos_words)
        neg = len(tokens & neg_words)
        polarity = (pos - neg) / (pos + neg + 1e-8)

    if polarity > 0.05:
        return "positive"
    elif polarity < -0.05:
        return "negative"
    return "neutral"


def _detect_lang(text: str) -> str:
    """Detect language code with fallback to 'unknown'."""
    if not _LANGDETECT_AVAILABLE:
        # Very rough heuristic: check for common character sets
        if any(ord(c) > 127 for c in text[:200]):
            return "non_latin"
        return "en"
    try:
        return detect(text[:500])
    except Exception:
        return "unknown"


def _tfidf_matrix(texts: List[str]) -> Optional[np.ndarray]:
    """Compute TF-IDF matrix for a list of texts."""
    if not _SKLEARN_AVAILABLE or len(texts) < 2:
        return None
    try:
        vec = TfidfVectorizer(max_features=5000, stop_words="english")
        return vec.fit_transform(texts).toarray()
    except Exception:
        return None


def _opinion_embed(text: str) -> np.ndarray:
    """Generate a lightweight opinion embedding using term-presence flags."""
    opinion_words = {
        "should", "must", "never", "always", "think", "believe", "opinion",
        "wrong", "right", "agree", "disagree", "claim", "argue", "support",
        "oppose", "certainly", "probably", "possibly", "doubt"
    }
    tokens = set(_tokenise(text))
    vec = np.array([float(w in tokens) for w in sorted(opinion_words)])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def compute_cht(texts: List[str]) -> CHTVector:
    """Compute the Corpus Health Tensor for a batch of documents.

    Parameters
    ----------
    texts:
        List of document strings to analyse.

    Returns
    -------
    CHTVector
        Observed CHT values for this batch.
    """
    if not texts:
        return CHTVector(0.0, 0.0, 0.0, 0.0)

    # ── Ω_op: Opinion Divergence ────────────────────────────────────────────
    if len(texts) > 1:
        op_vecs = np.stack([_opinion_embed(t) for t in texts])
        # Variance across documents in each opinion dimension, then mean
        omega_op = float(np.mean(np.var(op_vecs, axis=0)))
    else:
        omega_op = 0.0

    # ── Ω_emo: Emotional Entropy ─────────────────────────────────────────────
    sentiment_counts: Counter = Counter(_sentiment_label(t) for t in texts)
    omega_emo = _entropy(sentiment_counts)

    # ── Ω_lex: Lexical Diversity ─────────────────────────────────────────────
    if len(texts) > 1:
        tfidf_mat = _tfidf_matrix(texts)
        if tfidf_mat is not None:
            n = len(texts)
            distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = tfidf_mat[i], tfidf_mat[j]
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    if na > 0 and nb > 0:
                        sim = np.dot(a, b) / (na * nb)
                    else:
                        sim = 0.0
                    distances.append(1.0 - float(sim))
            omega_lex = float(np.mean(distances)) if distances else 0.0
        else:
            # Fallback: vocabulary overlap
            vocabs = [set(_tokenise(t)) for t in texts]
            distances = []
            for i in range(len(vocabs)):
                for j in range(i + 1, len(vocabs)):
                    union = vocabs[i] | vocabs[j]
                    if union:
                        inter = vocabs[i] & vocabs[j]
                        distances.append(1.0 - len(inter) / len(union))
            omega_lex = float(np.mean(distances)) if distances else 0.0
    else:
        omega_lex = 0.0

    # ── Ω_cult: Cultural Entropy ──────────────────────────────────────────────
    lang_counts: Counter = Counter(_detect_lang(t) for t in texts)
    omega_cult = _entropy(lang_counts)

    return CHTVector(omega_op, omega_emo, omega_lex, omega_cult)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class DPSSampler:
    """Sample training batches that enforce CHT diversity constraints.

    DPS uses a priority-queue approach: for each needed slot in a batch,
    it scores candidate documents by how much they increase the current
    batch's CHT, then picks the best candidate.

    Parameters
    ----------
    config:
        ``DPSConfig`` hyperparameters.  Uses defaults if ``None``.
    cht_config:
        ``CHTConfig`` threshold values.  If ``None``, defaults are used.

    Examples
    --------
    >>> sampler = DPSSampler()
    >>> corpus = ["Doc about politics...", "A love story...", "Technical paper..."]
    >>> batch = sampler.sample(corpus, batch_size=2)
    >>> health = sampler.corpus_health(corpus)
    >>> print(health.status(sampler.cht_config))
    """

    def __init__(
        self,
        config: Optional[DPSConfig] = None,
        cht_config: Optional[CHTConfig] = None,
    ) -> None:
        self._cfg = config or DPSConfig()
        self.cht_config = cht_config or CHTConfig(
            min_op=self._cfg.min_opinion_divergence,
            min_emo=self._cfg.min_emotional_entropy,
            min_lex=self._cfg.min_lexical_diversity,
            min_cult=self._cfg.min_cultural_entropy,
        )

    # ── Main sampling ─────────────────────────────────────────────────────────

    def sample(
        self,
        corpus: Sequence[str],
        batch_size: int,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Sample a batch of documents satisfying CHT diversity constraints.

        Parameters
        ----------
        corpus:
            Pool of candidate documents.
        batch_size:
            Number of documents to return.
        seed:
            Optional random seed for reproducibility.

        Returns
        -------
        list of str
            Sampled documents.  If the corpus is smaller than batch_size,
            all documents are returned.

        Notes
        -----
        If the corpus cannot satisfy CHT constraints (e.g., all documents are
        in the same language), a warning is issued and the best-effort batch
        is returned.
        """
        rng = random.Random(seed)
        docs = list(corpus)

        if len(docs) <= batch_size:
            return docs

        # Step 1: candidate pool (random oversample)
        k = min(batch_size * self._cfg.oversampling_factor, len(docs))
        candidates = rng.sample(docs, k)

        # Step 2: Greedy priority-queue construction
        selected: List[str] = []
        remaining = list(candidates)

        for _ in range(batch_size):
            if not remaining:
                break
            best_idx = self._best_candidate(selected, remaining)
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)

        # Step 3: Check CHT and warn if constraints not met
        final_cht = compute_cht(selected)
        if not final_cht.satisfies(self.cht_config):
            warnings.warn(
                f"DPS could not fully satisfy CHT constraints.\n"
                f"{final_cht.status(self.cht_config)}\n"
                "Consider increasing corpus diversity or relaxing thresholds.",
                RuntimeWarning,
                stacklevel=2,
            )

        return selected

    def corpus_health(self, corpus: Sequence[str]) -> CHTVector:
        """Compute CHT for the entire corpus (or a sample).

        Parameters
        ----------
        corpus:
            Sequence of documents.

        Returns
        -------
        CHTVector
            Corpus Health Tensor with all four components.
        """
        docs = list(corpus)
        if len(docs) > 500:
            # Sample for speed
            docs = random.sample(docs, 500)
        return compute_cht(docs)

    def marginal_gain(self, candidate: str, current_batch: List[str]) -> float:
        """Compute how much a candidate increases the batch's total CHT.

        Parameters
        ----------
        candidate:
            Document to evaluate.
        current_batch:
            Documents already selected.

        Returns
        -------
        float
            Sum of increases across all CHT components.  Higher = more diverse.
        """
        if not current_batch:
            test = [candidate]
        else:
            test = current_batch + [candidate]

        cht_new = compute_cht(test)
        if not current_batch:
            # First document: all gain
            return float(np.sum(cht_new.as_array()))

        cht_old = compute_cht(current_batch)
        delta = cht_new.as_array() - cht_old.as_array()
        return float(np.sum(np.clip(delta, 0, None)))  # only count gains

    # ── Private helpers ───────────────────────────────────────────────────────

    def _best_candidate(self, selected: List[str], remaining: List[str]) -> int:
        """Return index of the candidate that maximises marginal CHT gain."""
        best_idx = 0
        best_gain = -1.0
        for i, doc in enumerate(remaining):
            gain = self.marginal_gain(doc, selected)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        return best_idx
