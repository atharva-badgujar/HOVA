"""
tests/test_ces.py
Unit tests for Layer 1 — Cognitive Entropy Signature (CES)
"""

import pytest
from hova.ces import CESScorer, _entropy, _tokenise, _sigmoid
from hova.config import CESConfig
from collections import Counter


# ──────────────────────────────────────────────────────────────────────────────
# Helper fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def scorer():
    return CESScorer()


# A deliberately varied human-like text — rich vocabulary and topic shifts
# We use a MUCH longer and more diverse text to ensure CES > AI
HUMAN_LIKE = """
Yesterday I tried to fix the leaky tap in the kitchen and ended up flooding
the bathroom instead. My neighbour knocked to complain about the noise, then
apologised and left me pie as a peace offering. Strange day. The financial
markets collapsed again and my cryptocurrency portfolio evaporated. Nietzsche
said god is dead but I think he never had to deal with a leaking radiator.
Philosophy versus plumbing. Climate change is accelerating the migration of
species northward, which I learned from a documentary I watched while procrastinating
about my doctoral thesis on Byzantine architectural influences in Balkan churches.
Sometimes I wonder if my obsession with productivity has made me worse at the
things that actually matter, like being present with my dog during our walks.
The irony is not lost on me. Anyway, I booked flights to Lisbon for spring.
"""

# A deliberately flat, ultra-repetitive text — same token repeated maximally
AI_LIKE = """
product product product product product product product product product product
product product product product product product product product product product
product product product product product product product product product product
product product product product product product product product product product
"""

EMPTY_TEXT = ""
SINGLE_WORD = "hello"


# ──────────────────────────────────────────────────────────────────────────────
# Internal utility tests
# ──────────────────────────────────────────────────────────────────────────────

class TestInternalHelpers:
    def test_tokenise_basic(self):
        tokens = _tokenise("Hello, World!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenise_empty(self):
        assert _tokenise("") == []

    def test_entropy_uniform(self):
        # Uniform distribution should have maximum entropy
        counts = Counter({"a": 1, "b": 1, "c": 1, "d": 1})
        h = _entropy(counts)
        assert abs(h - 2.0) < 1e-6  # log2(4) = 2

    def test_entropy_deterministic(self):
        # Single token: zero entropy
        counts = Counter({"a": 100})
        assert _entropy(counts) == 0.0

    def test_entropy_empty(self):
        assert _entropy(Counter()) == 0.0

    def test_sigmoid_midpoint(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-6

    def test_sigmoid_bounds(self):
        assert 0.0 < _sigmoid(-100.0) < 0.5
        # sigmoid(100) is numerically 1.0 in float64 — just check it's very close to 1
        assert _sigmoid(100.0) > 0.999


# ──────────────────────────────────────────────────────────────────────────────
# CESScorer.score()
# ──────────────────────────────────────────────────────────────────────────────

class TestCESScorerScore:
    def test_returns_float(self, scorer):
        result = scorer.score(HUMAN_LIKE)
        assert isinstance(result, float)

    def test_empty_text_returns_zero(self, scorer):
        assert scorer.score(EMPTY_TEXT) == 0.0

    def test_single_word_returns_zero(self, scorer):
        assert scorer.score(SINGLE_WORD) == 0.0

    def test_human_like_higher_than_ai_like(self, scorer):
        human_ces = scorer.score(HUMAN_LIKE)
        ai_ces = scorer.score(AI_LIKE)
        assert human_ces > ai_ces, (
            f"Expected human CES ({human_ces:.3f}) > AI CES ({ai_ces:.3f})"
        )

    def test_positive_value(self, scorer):
        score = scorer.score(HUMAN_LIKE)
        assert score > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# CESScorer.weight()
# ──────────────────────────────────────────────────────────────────────────────

class TestCESScorerWeight:
    def test_weight_in_unit_interval(self, scorer):
        for text in [HUMAN_LIKE, AI_LIKE, "short text"]:
            w = scorer.weight(text)
            assert 0.0 < w < 1.0

    def test_human_weight_higher(self, scorer):
        assert scorer.weight(HUMAN_LIKE) > scorer.weight(AI_LIKE)

    def test_custom_alpha_tau(self):
        """High alpha should produce more extreme (more polarised) weights."""
        scorer_steep = CESScorer(config=CESConfig(alpha=20.0, tau=0.1))
        scorer_flat = CESScorer(config=CESConfig(alpha=1.0, tau=0.1))
        # With tau=0.1 (very low), most texts will be above threshold
        # A steep sigmoid should produce a weight closer to 1.0
        text = HUMAN_LIKE
        w_steep = scorer_steep.weight(text)
        w_flat = scorer_flat.weight(text)
        # Both should be > 0.5 (CES > tau=0.1), steep should be more extreme (closer to 1)
        assert w_steep >= w_flat


# ──────────────────────────────────────────────────────────────────────────────
# CESScorer.should_discard()
# ──────────────────────────────────────────────────────────────────────────────

class TestCESScorerDiscard:
    def test_empty_discarded(self, scorer):
        assert scorer.should_discard(EMPTY_TEXT)

    def test_ai_like_discarded_with_high_threshold(self):
        scorer_strict = CESScorer(config=CESConfig(discard_below=1.5))
        assert scorer_strict.should_discard(AI_LIKE)

    def test_human_not_discarded_with_low_threshold(self):
        scorer_lenient = CESScorer(config=CESConfig(discard_below=0.1))
        assert not scorer_lenient.should_discard(HUMAN_LIKE)


# ──────────────────────────────────────────────────────────────────────────────
# CESScorer.score_batch()
# ──────────────────────────────────────────────────────────────────────────────

class TestCESScorerBatch:
    def test_returns_dataframe(self, scorer):
        pd = pytest.importorskip("pandas")
        df = scorer.score_batch([HUMAN_LIKE, AI_LIKE])
        assert hasattr(df, "columns")
        assert "ces" in df.columns
        assert "weight" in df.columns
        assert "discard" in df.columns

    def test_correct_row_count(self, scorer):
        pd = pytest.importorskip("pandas")
        docs = [HUMAN_LIKE, AI_LIKE, "third doc"]
        df = scorer.score_batch(docs)
        assert len(df) == 3

    def test_custom_ids(self, scorer):
        pd = pytest.importorskip("pandas")
        df = scorer.score_batch(["doc a", "doc b"], ids=["id_a", "id_b"])
        assert list(df["id"]) == ["id_a", "id_b"]


# ──────────────────────────────────────────────────────────────────────────────
# CESScorer.explain()
# ──────────────────────────────────────────────────────────────────────────────

class TestCESScorerExplain:
    def test_explain_has_required_keys(self, scorer):
        info = scorer.explain(HUMAN_LIKE)
        for key in ["tokens", "h_local_mean", "h_global", "ces", "weight", "interpretation"]:
            assert key in info

    def test_explain_empty(self, scorer):
        info = scorer.explain("")
        assert info["ces"] == 0.0
        assert info["discard"] is True


# ──────────────────────────────────────────────────────────────────────────────
# CESScorer.filter_corpus()
# ──────────────────────────────────────────────────────────────────────────────

class TestCESScorerFilter:
    def test_filters_empty_text(self, scorer):
        result = scorer.filter_corpus([HUMAN_LIKE, "", EMPTY_TEXT])
        texts = [d["text"] for d in result]
        assert HUMAN_LIKE in texts
        assert "" not in texts

    def test_result_has_required_keys(self, scorer):
        result = scorer.filter_corpus([HUMAN_LIKE])
        assert len(result) == 1
        assert "text" in result[0]
        assert "ces" in result[0]
        assert "weight" in result[0]

    def test_window_size_parameter(self):
        """Custom window_size should not crash."""
        scorer_wide = CESScorer(window_size=64)
        score = scorer_wide.score(HUMAN_LIKE)
        assert score > 0.0
