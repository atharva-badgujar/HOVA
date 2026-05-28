"""
hova/tmc.py
Layer 2 — Temporal Mutation Chain (TMC)

Humans change over time: vocabulary evolves, opinions shift, syntax matures.
No AI can fabricate the genuine statistical trace of a life unfolding.
TMC tracks this trace across an author's document history and scores authenticity.

    φ(a, t) = [V_drift, S_vol, T_jump, Syn_mut]
    TMC_score(a) = Var_TMC(a) · ID_cons(a)
    weight_TMC(D) = sigmoid(β · TMC_score(a))
"""

from __future__ import annotations

import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from hova.config import TMCConfig

# ──────────────────────────────────────────────────────────────────────────────
# Optional heavy dependencies — fail gracefully
# ──────────────────────────────────────────────────────────────────────────────

try:
    from textblob import TextBlob  # type: ignore
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False

try:
    import spacy  # type: ignore
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    from gensim import corpora  # type: ignore
    from gensim.models import LdaModel  # type: ignore
    _GENSIM_AVAILABLE = True
except ImportError:
    _GENSIM_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _jaccard_distance(set_a: set, set_b: set) -> float:
    """Jaccard distance = 1 - Jaccard similarity."""
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    return 1.0 - len(intersection) / len(union)


def _sentiment_score(text: str) -> float:
    """Return sentiment polarity in [-1, 1].

    Falls back to 0.0 if TextBlob is unavailable.
    """
    if not _TEXTBLOB_AVAILABLE:
        # Simple heuristic: count positive vs negative words
        positives = {"good", "great", "excellent", "happy", "love", "wonderful"}
        negatives = {"bad", "terrible", "awful", "sad", "hate", "horrible"}
        tokens = set(_tokenise(text))
        pos = len(tokens & positives)
        neg = len(tokens & negatives)
        if pos + neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)
    return TextBlob(text).sentiment.polarity


def _pos_sequence(text: str) -> List[str]:
    """Extract POS tag sequence.  Returns word tokens if spaCy unavailable."""
    if _SPACY_AVAILABLE:
        try:
            nlp = _get_spacy_model()
            doc = nlp(text[:10_000])  # cap to avoid memory issues
            return [token.pos_ for token in doc if not token.is_space]
        except Exception:
            pass
    # Fallback: word-level tokens as a pseudo-POS sequence
    return _tokenise(text)


_spacy_model_cache = None


def _get_spacy_model():
    global _spacy_model_cache
    if _spacy_model_cache is None:
        try:
            _spacy_model_cache = spacy.load("en_core_web_sm")
        except OSError:
            warnings.warn(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm\n"
                "Falling back to token-level POS approximation.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Create a minimal blank model
            _spacy_model_cache = spacy.blank("en")
    return _spacy_model_cache


def _edit_distance_normalised(seq_a: List[str], seq_b: List[str]) -> float:
    """Normalised Levenshtein distance between two sequences."""
    if not seq_a and not seq_b:
        return 0.0
    max_len = max(len(seq_a), len(seq_b))
    if max_len == 0:
        return 0.0

    # Wagner-Fischer DP (memory efficient)
    prev = list(range(len(seq_b) + 1))
    for i, ca in enumerate(seq_a):
        curr = [i + 1] + [0] * len(seq_b)
        for j, cb in enumerate(seq_b):
            curr[j + 1] = min(
                prev[j + 1] + 1,      # deletion
                curr[j] + 1,          # insertion
                prev[j] + (ca != cb), # substitution
            )
        prev = curr
    return prev[-1] / max_len


def _tfidf_vector(text: str) -> np.ndarray:
    """Simple TF-IDF vector for a single document (no fit step)."""
    tokens = _tokenise(text)
    if not tokens:
        return np.zeros(1)
    counts = defaultdict(int)
    for t in tokens:
        counts[t] += 1
    total = len(tokens)
    # TF only (IDF requires corpus); good enough for identity consistency
    vocab = sorted(counts.keys())
    vec = np.array([counts[w] / total for w in vocab], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors of potentially different sizes."""
    # Pad shorter vector with zeros
    size = max(len(a), len(b))
    va = np.pad(a, (0, size - len(a)))
    vb = np.pad(b, (0, size - len(b)))
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


# ──────────────────────────────────────────────────────────────────────────────
# Internal data structure
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _AuthorRecord:
    """Internal record per author."""
    texts: List[str] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)  # POSIX timestamps
    features: List[np.ndarray] = field(default_factory=list)  # φ(a,t) vectors
    style_vecs: List[np.ndarray] = field(default_factory=list)  # for ID_cons


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class TMCTracker:
    """Track per-author document evolution and compute TMC authenticity scores.

    Parameters
    ----------
    config:
        ``TMCConfig`` with hyperparameters.  Uses defaults if ``None``.

    Examples
    --------
    >>> tracker = TMCTracker()
    >>> tracker.add_document("alice", "I love coffee.", "2020-01-01")
    >>> tracker.add_document("alice", "Coffee is terrible. Tea is better.", "2022-06-01")
    >>> tracker.add_document("alice", "I switched to matcha last year.", "2024-03-01")
    >>> score = tracker.get_score("alice")
    >>> weight = tracker.get_weight("alice")
    """

    def __init__(self, config: Optional[TMCConfig] = None) -> None:
        self._cfg = config or TMCConfig()
        self._authors: Dict[str, _AuthorRecord] = defaultdict(_AuthorRecord)

    # ── Document ingestion ────────────────────────────────────────────────────

    def add_document(
        self,
        author_id: str,
        text: str,
        timestamp: Optional[str | float | datetime] = None,
    ) -> None:
        """Register a document for an author.

        Parameters
        ----------
        author_id:
            Unique string identifier for the author.
        text:
            Raw document text.
        timestamp:
            When the document was written.  Accepts ISO date strings,
            POSIX floats, or ``datetime`` objects.  If ``None``, uses
            current time.
        """
        ts = self._parse_timestamp(timestamp)
        record = self._authors[author_id]

        # Keep sorted by timestamp
        insert_idx = len(record.timestamps)
        for i, existing_ts in enumerate(record.timestamps):
            if ts < existing_ts:
                insert_idx = i
                break

        record.texts.insert(insert_idx, text)
        record.timestamps.insert(insert_idx, ts)
        record.style_vecs.insert(insert_idx, _tfidf_vector(text))
        # Features computed lazily in get_score()
        record.features = []  # invalidate cache

    # ── Scoring ───────────────────────────────────────────────────────────────

    def get_score(self, author_id: str) -> float:
        """Compute TMC authenticity score for an author.

        TMC_score(a) = Var_TMC(a) · ID_cons(a)

        Returns
        -------
        float
            Score in [0, ∞).  Higher → more authentic longitudinal pattern.
            Returns 0.0 if the author has fewer documents than
            ``config.min_documents_per_author``.
        """
        record = self._authors.get(author_id)
        if record is None:
            return 0.0
        if len(record.texts) < self._cfg.min_documents_per_author:
            return 0.0

        features = self._compute_feature_sequence(record)
        if len(features) < 2:
            return 0.0

        feature_matrix = np.stack(features)  # shape (T, 4)
        var_tmc = float(np.mean(np.var(feature_matrix, axis=0)))
        id_cons = self._identity_consistency(record)
        return var_tmc * id_cons

    def get_weight(self, author_id: str) -> float:
        """Map TMC score → training weight in (0, 1).

        weight = sigmoid(beta · TMC_score(a))

        Returns
        -------
        float
            Weight in (0, 1).
        """
        return _sigmoid(self._cfg.beta * self.get_score(author_id))

    def get_feature_vector(self, author_id: str, t: int) -> Optional[np.ndarray]:
        """Return φ(a, t) for a specific author at timestep *t*.

        Parameters
        ----------
        author_id:
            Author identifier.
        t:
            Timestep index (0 = first document).

        Returns
        -------
        numpy.ndarray or None
            4-dimensional feature vector, or None if unavailable.
        """
        record = self._authors.get(author_id)
        if record is None or t < 1 or t >= len(record.texts):
            return None
        features = self._compute_feature_sequence(record)
        return features[t - 1] if t - 1 < len(features) else None

    def list_authors(self) -> List[str]:
        """Return all author IDs currently tracked."""
        return list(self._authors.keys())

    def summary(self) -> dict:
        """Return a summary of tracked authors and their scores.

        Returns
        -------
        dict
            Maps ``author_id → {n_docs, tmc_score, weight, status}``.
        """
        result = {}
        for aid, record in self._authors.items():
            n = len(record.texts)
            if n < self._cfg.min_documents_per_author:
                status = f"insufficient ({n}/{self._cfg.min_documents_per_author} docs)"
                score = 0.0
                weight = 0.0
            else:
                score = self.get_score(aid)
                weight = self.get_weight(aid)
                status = "authentic" if score > 0.05 else "suspicious"
            result[aid] = {
                "n_docs": n,
                "tmc_score": round(score, 4),
                "weight": round(weight, 4),
                "status": status,
            }
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_timestamp(ts) -> float:
        if ts is None:
            import time
            return time.time()
        if isinstance(ts, float | int):
            return float(ts)
        if isinstance(ts, datetime):
            return ts.timestamp()
        if isinstance(ts, str):
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d"):
                try:
                    return datetime.strptime(ts, fmt).timestamp()
                except ValueError:
                    continue
        return 0.0

    def _compute_feature_sequence(self, record: _AuthorRecord) -> List[np.ndarray]:
        """Compute φ(a, t) for t = 1 .. T-1 (consecutive document pairs)."""
        if record.features:
            return record.features  # cached

        features = []
        texts = record.texts
        for t in range(1, len(texts)):
            prev, curr = texts[t - 1], texts[t]

            # V_drift: Jaccard distance of vocabularies
            v_drift = _jaccard_distance(
                set(_tokenise(prev)), set(_tokenise(curr))
            )

            # S_vol: absolute sentiment shift
            s_vol = abs(_sentiment_score(curr) - _sentiment_score(prev))

            # T_jump: topic distribution distance (simplified)
            t_jump = self._topic_jump(prev, curr)

            # Syn_mut: normalised edit distance on POS sequences
            pos_prev = _pos_sequence(prev)
            pos_curr = _pos_sequence(curr)
            syn_mut = _edit_distance_normalised(pos_prev[:200], pos_curr[:200])

            features.append(np.array([v_drift, s_vol, t_jump, syn_mut], dtype=float))

        record.features = features
        return features

    def _topic_jump(self, text_a: str, text_b: str) -> float:
        """Estimate topic distribution distance between two documents.

        Uses LDA if gensim is available, otherwise falls back to vocabulary
        overlap as a proxy.
        """
        if not _GENSIM_AVAILABLE:
            # Fallback: symmetric lexical distance
            return _jaccard_distance(set(_tokenise(text_a)), set(_tokenise(text_b)))

        try:
            tokens_a = _tokenise(text_a)
            tokens_b = _tokenise(text_b)
            dictionary = corpora.Dictionary([tokens_a, tokens_b])
            corpus = [dictionary.doc2bow(tokens_a), dictionary.doc2bow(tokens_b)]
            lda = LdaModel(
                corpus,
                num_topics=min(self._cfg.n_topics, 5),
                id2word=dictionary,
                passes=1,
                random_state=42,
            )
            topic_a = dict(lda[corpus[0]])
            topic_b = dict(lda[corpus[1]])
            all_topics = set(topic_a) | set(topic_b)
            diff = sum(abs(topic_a.get(k, 0) - topic_b.get(k, 0)) for k in all_topics)
            return min(diff, 1.0)
        except Exception:
            return _jaccard_distance(set(_tokenise(text_a)), set(_tokenise(text_b)))

    def _identity_consistency(self, record: _AuthorRecord) -> float:
        """ID_cons = cosine_similarity(style_embed(D_1), style_embed(D_T))."""
        if len(record.style_vecs) < 2:
            return 1.0
        return _cosine_sim(record.style_vecs[0], record.style_vecs[-1])
