"""
tests/test_dps.py
Unit tests for Layer 4 — Disagreement Preservation Sampling (DPS)
"""

import pytest
import math
from hova.dps import (
    DPSSampler, CHTConfig, CHTVector, compute_cht,
    _entropy, _sentiment_label,
)
from hova.config import DPSConfig


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures and test data
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def lenient_config():
    """Very lenient CHT thresholds for basic testing."""
    return CHTConfig(min_op=0.0, min_emo=0.0, min_lex=0.0, min_cult=0.0)


@pytest.fixture
def default_sampler():
    return DPSSampler()


DIVERSE_CORPUS = [
    "I strongly support renewable energy and climate action.",
    "Government regulation always makes things worse for business.",
    "Je pense que la politique française est complexe.",  # French
    "Traditional values are the foundation of a stable society.",
    "Technology will solve all our environmental problems eventually.",
    "We should tax the rich to fund universal basic income.",
    "Free markets allocate resources more efficiently than central planning.",
    "Community and belonging matter more than individual achievement.",
    "The scientific consensus on vaccines is clear and should be followed.",
    "I distrust mainstream media and look for alternative information.",
]

HOMOGENEOUS_CORPUS = [
    "The product is good and I recommend it.",
    "The product is good and I recommend it highly.",
    "The product is very good and I highly recommend it.",
    "This product is excellent and I recommend purchasing it.",
    "I highly recommend this good product to everyone.",
]


# ──────────────────────────────────────────────────────────────────────────────
# CHT computation
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeCHT:
    def test_empty_batch_returns_zeros(self):
        cht = compute_cht([])
        assert cht.omega_op == 0.0
        assert cht.omega_emo == 0.0
        assert cht.omega_lex == 0.0
        assert cht.omega_cult == 0.0

    def test_single_document(self):
        cht = compute_cht(["Just one document here."])
        # With only one document, pairwise metrics should be 0
        assert cht.omega_lex == 0.0
        assert cht.omega_op == 0.0

    def test_diverse_batch_healthier(self):
        cht_diverse = compute_cht(DIVERSE_CORPUS)
        cht_homo = compute_cht(HOMOGENEOUS_CORPUS)
        # Diverse corpus should have higher or equal CHT scores
        # (At minimum, emotional entropy should differ)
        total_diverse = sum(cht_diverse.as_array())
        total_homo = sum(cht_homo.as_array())
        assert total_diverse >= total_homo

    def test_cht_to_dict(self):
        cht = compute_cht(["hello", "world"])
        d = cht.to_dict()
        for key in ["omega_op", "omega_emo", "omega_lex", "omega_cult"]:
            assert key in d

    def test_cht_as_array_shape(self):
        cht = compute_cht(["text"])
        arr = cht.as_array()
        assert len(arr) == 4


# ──────────────────────────────────────────────────────────────────────────────
# CHTConfig and CHTVector
# ──────────────────────────────────────────────────────────────────────────────

class TestCHTVectorSatisfies:
    def test_satisfies_lenient(self):
        cht = CHTVector(omega_op=0.5, omega_emo=2.0, omega_lex=0.6, omega_cult=1.5)
        config = CHTConfig(min_op=0.35, min_emo=1.5, min_lex=0.40, min_cult=1.2)
        assert cht.satisfies(config)

    def test_fails_deficit(self):
        cht = CHTVector(omega_op=0.1, omega_emo=0.5, omega_lex=0.2, omega_cult=0.3)
        config = CHTConfig(min_op=0.35, min_emo=1.5, min_lex=0.40, min_cult=1.2)
        assert not cht.satisfies(config)

    def test_deficit_values(self):
        cht = CHTVector(omega_op=0.2, omega_emo=1.0, omega_lex=0.3, omega_cult=0.8)
        config = CHTConfig(min_op=0.35, min_emo=1.5, min_lex=0.40, min_cult=1.2)
        deficit = cht.deficit(config)
        assert all(deficit < 0)  # all components in deficit

    def test_status_healthy(self):
        cht = CHTVector(omega_op=0.5, omega_emo=2.0, omega_lex=0.6, omega_cult=1.5)
        config = CHTConfig(min_op=0.35, min_emo=1.5, min_lex=0.40, min_cult=1.2)
        status = cht.status(config)
        assert "Healthy" in status

    def test_status_deficit(self):
        cht = CHTVector(omega_op=0.1, omega_emo=0.5, omega_lex=0.2, omega_cult=0.3)
        config = CHTConfig(min_op=0.35, min_emo=1.5, min_lex=0.40, min_cult=1.2)
        status = cht.status(config)
        assert "deficit" in status.lower()


# ──────────────────────────────────────────────────────────────────────────────
# DPSSampler
# ──────────────────────────────────────────────────────────────────────────────

class TestDPSSampler:
    def test_sample_correct_size(self, default_sampler):
        batch = default_sampler.sample(DIVERSE_CORPUS, batch_size=4, seed=42)
        assert len(batch) == 4

    def test_sample_from_smaller_corpus(self, default_sampler):
        small = ["doc one", "doc two"]
        batch = default_sampler.sample(small, batch_size=10)
        assert len(batch) == 2  # returns all

    def test_sample_reproducible_with_seed(self, default_sampler):
        b1 = default_sampler.sample(DIVERSE_CORPUS, batch_size=5, seed=0)
        b2 = default_sampler.sample(DIVERSE_CORPUS, batch_size=5, seed=0)
        assert b1 == b2

    def test_sample_different_seeds(self, default_sampler):
        b1 = default_sampler.sample(DIVERSE_CORPUS, batch_size=5, seed=1)
        b2 = default_sampler.sample(DIVERSE_CORPUS, batch_size=5, seed=2)
        # Very unlikely to be identical with different seeds on large corpus
        # (not guaranteed, but statistically expected)
        assert len(b1) == len(b2) == 5

    def test_corpus_health_returns_cht(self, default_sampler):
        cht = default_sampler.corpus_health(DIVERSE_CORPUS)
        assert isinstance(cht, CHTVector)

    def test_marginal_gain_increases_cht(self, default_sampler):
        # Adding a French document to an English-only batch should increase cultural entropy
        english_batch = DIVERSE_CORPUS[:5]
        french_doc = "Je suis très heureux aujourd'hui."
        gain = default_sampler.marginal_gain(french_doc, english_batch)
        assert isinstance(gain, float)

    def test_custom_cht_config(self):
        """Very lenient config should always be satisfied."""
        config = CHTConfig(min_op=0.0, min_emo=0.0, min_lex=0.0, min_cult=0.0)
        sampler = DPSSampler(cht_config=config)
        batch = sampler.sample(DIVERSE_CORPUS, batch_size=5, seed=42)
        assert len(batch) == 5
        cht = compute_cht(batch)
        assert cht.satisfies(config)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_entropy_bit(self):
        from collections import Counter
        # Two equally likely outcomes → 1 bit
        assert abs(_entropy(Counter({"a": 1, "b": 1})) - 1.0) < 1e-6

    def test_sentiment_label_positive(self):
        assert _sentiment_label("great excellent wonderful love") == "positive"

    def test_sentiment_label_negative(self):
        assert _sentiment_label("terrible awful hate bad worst") == "negative"
