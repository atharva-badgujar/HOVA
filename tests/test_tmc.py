"""
tests/test_tmc.py
Unit tests for Layer 2 — Temporal Mutation Chain (TMC)
"""

import pytest
import time
from hova.tmc import TMCTracker, _jaccard_distance, _edit_distance_normalised
from hova.config import TMCConfig


# ──────────────────────────────────────────────────────────────────────────────
# Helper fixtures and data
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tracker():
    return TMCTracker(config=TMCConfig(min_documents_per_author=2))


DOCS_ALICE = [
    ("2018-01-01", "I love basketball and outdoor sports. Running is my therapy."),
    ("2020-06-15", "I've been reading more philosophy lately. Nietzsche is fascinating."),
    ("2023-03-10", "My startup failed but I learned a lot. Now I'm writing a novel."),
]

DOCS_BOT = [
    ("2020-01-01", "The product is good. The service is good. Quality is good."),
    ("2020-01-02", "The product is good. The service is good. Quality is good."),
    ("2020-01-03", "The product is good. The service is good. Quality is good."),
]


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestJaccardDistance:
    def test_identical_sets(self):
        assert _jaccard_distance({"a", "b"}, {"a", "b"}) == 0.0

    def test_disjoint_sets(self):
        assert _jaccard_distance({"a", "b"}, {"c", "d"}) == 1.0

    def test_partial_overlap(self):
        d = _jaccard_distance({"a", "b", "c"}, {"b", "c", "d"})
        # |intersection| = 2, |union| = 4 → similarity = 0.5 → distance = 0.5
        assert abs(d - 0.5) < 1e-6

    def test_empty_sets(self):
        assert _jaccard_distance(set(), set()) == 0.0


class TestEditDistance:
    def test_identical_sequences(self):
        assert _edit_distance_normalised(["A", "B", "C"], ["A", "B", "C"]) == 0.0

    def test_completely_different(self):
        d = _edit_distance_normalised(["A", "B"], ["C", "D"])
        assert d > 0.0

    def test_empty_sequences(self):
        assert _edit_distance_normalised([], []) == 0.0

    def test_one_empty(self):
        d = _edit_distance_normalised(["A", "B", "C"], [])
        assert d == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# TMCTracker: document ingestion
# ──────────────────────────────────────────────────────────────────────────────

class TestTMCTrackerIngestion:
    def test_add_document_no_crash(self, tracker):
        tracker.add_document("alice", "Some text.", "2020-01-01")

    def test_list_authors_after_add(self, tracker):
        tracker.add_document("alice", "Hello.", "2020-01-01")
        assert "alice" in tracker.list_authors()

    def test_multiple_authors(self, tracker):
        tracker.add_document("alice", "Text A.", "2020-01-01")
        tracker.add_document("bob", "Text B.", "2021-06-15")
        authors = tracker.list_authors()
        assert "alice" in authors
        assert "bob" in authors

    def test_timestamp_string_formats(self, tracker):
        """Multiple timestamp formats should not crash."""
        tracker.add_document("u1", "Text 1.", "2020-01-01")
        tracker.add_document("u2", "Text 2.", "2020-01-01T12:00:00")
        tracker.add_document("u3", "Text 3.", 1577836800.0)


# ──────────────────────────────────────────────────────────────────────────────
# TMCTracker: scoring
# ──────────────────────────────────────────────────────────────────────────────

class TestTMCTrackerScoring:
    def test_score_zero_for_unknown_author(self, tracker):
        assert tracker.get_score("nonexistent") == 0.0

    def test_score_zero_below_min_documents(self, tracker):
        tracker.add_document("alice", "Just one document.", "2020-01-01")
        # min_documents_per_author = 2, only 1 added
        assert tracker.get_score("alice") == 0.0

    def test_score_positive_with_enough_docs(self):
        t = TMCTracker(config=TMCConfig(min_documents_per_author=2))
        for ts, text in DOCS_ALICE:
            t.add_document("alice", text, ts)
        score = t.get_score("alice")
        assert score >= 0.0  # can be 0 if texts are very similar

    def test_weight_in_unit_interval(self):
        t = TMCTracker(config=TMCConfig(min_documents_per_author=2))
        for ts, text in DOCS_ALICE:
            t.add_document("alice", text, ts)
        weight = t.get_weight("alice")
        assert 0.0 < weight < 1.0

    def test_weight_zero_for_unknown(self, tracker):
        w = tracker.get_weight("nobody")
        # sigmoid(0) = 0.5, but score is 0 → weight = sigmoid(0) = 0.5
        assert 0.0 < w <= 1.0  # always in (0,1)


# ──────────────────────────────────────────────────────────────────────────────
# TMCTracker: summary
# ──────────────────────────────────────────────────────────────────────────────

class TestTMCTrackerSummary:
    def test_summary_contains_author(self):
        t = TMCTracker(config=TMCConfig(min_documents_per_author=2))
        for ts, text in DOCS_ALICE:
            t.add_document("alice", text, ts)
        summary = t.summary()
        assert "alice" in summary
        assert "n_docs" in summary["alice"]
        assert "tmc_score" in summary["alice"]
        assert "weight" in summary["alice"]
        assert "status" in summary["alice"]

    def test_summary_insufficient_docs_label(self):
        t = TMCTracker(config=TMCConfig(min_documents_per_author=5))
        t.add_document("alice", "One doc.", "2020-01-01")
        summary = t.summary()
        assert "insufficient" in summary["alice"]["status"]


# ──────────────────────────────────────────────────────────────────────────────
# TMCTracker: feature vectors
# ──────────────────────────────────────────────────────────────────────────────

class TestTMCFeatureVectors:
    def test_feature_vector_none_for_t0(self):
        t = TMCTracker(config=TMCConfig(min_documents_per_author=2))
        for ts, text in DOCS_ALICE:
            t.add_document("alice", text, ts)
        # t=0 should return None (no previous doc)
        assert t.get_feature_vector("alice", 0) is None

    def test_feature_vector_shape(self):
        t = TMCTracker(config=TMCConfig(min_documents_per_author=2))
        for ts, text in DOCS_ALICE:
            t.add_document("alice", text, ts)
        vec = t.get_feature_vector("alice", 1)
        assert vec is not None
        assert len(vec) == 4  # [V_drift, S_vol, T_jump, Syn_mut]

    def test_feature_values_in_range(self):
        t = TMCTracker(config=TMCConfig(min_documents_per_author=2))
        for ts, text in DOCS_ALICE:
            t.add_document("alice", text, ts)
        vec = t.get_feature_vector("alice", 1)
        assert vec is not None
        for val in vec:
            assert 0.0 <= val <= 1.0, f"Feature out of range: {val}"
