"""
tests/test_pipeline.py
End-to-end integration tests for HOVAPipeline.
"""

import pytest
from hova.pipeline import HOVAPipeline, Document
from hova.config import HOVAConfig, CESConfig, TMCConfig, DPSConfig, CEWSConfig, ANTConfig
from hova.dps import CHTConfig


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

TOY_CORPUS = [
    "Yesterday I tried something new and it completely changed my perspective on cooking.",
    "I strongly disagree with the current government policy on infrastructure spending.",
    "Je pense que la technologie va transformer notre société de manière fondamentale.",
    "The scientific method is the best tool humanity has developed for understanding reality.",
    "Traditional communities provide meaning and belonging that modern life fails to offer.",
    "Markets allocate resources efficiently, but they ignore externalities and inequality.",
    "My grandmother's recipes carry stories of migration, resilience, and love across decades.",
    "We need urgent international cooperation on climate before we pass tipping points.",
    "Freedom of speech includes the right to say things that offend the majority.",
    "The data strongly suggests that early childhood education has enormous long-term returns.",
    # Add some repetitive low-quality content
    "Buy now. Best deal. Great price. Buy today.",
    "Product is good product is good product is good.",
]

TOY_AUTHORS = [
    "alice", "bob", "charlie", "alice", "bob",
    "charlie", "alice", "bob", "charlie", "alice",
    None, None,  # no author for low-quality docs
]

TOY_TIMESTAMPS = [
    "2020-01-01", "2020-02-01", "2020-03-01",
    "2021-01-01", "2021-06-01", "2021-09-01",
    "2022-01-01", "2022-04-01", "2022-11-01",
    "2023-01-01", "2023-02-01", "2023-03-01",
]


@pytest.fixture
def fast_config():
    """Config with lenient thresholds for fast testing."""
    return HOVAConfig(
        ces=CESConfig(discard_below=0.3, window_size=16),
        tmc=TMCConfig(min_documents_per_author=2, lda_passes=1),
        dps=DPSConfig(
            min_opinion_divergence=0.0,
            min_emotional_entropy=0.0,
            min_lexical_diversity=0.0,
            min_cultural_entropy=0.0,
        ),
        cews=CEWSConfig(checkpoint_every=100),  # won't trigger in small tests
        ant=ANTConfig(anchor_model_path=None),
    )


@pytest.fixture
def pipeline(fast_config):
    return HOVAPipeline(config=fast_config)


# ──────────────────────────────────────────────────────────────────────────────
# Basic pipeline.run()
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineRun:
    def test_run_returns_list(self, pipeline):
        result = pipeline.run(TOY_CORPUS)
        assert isinstance(result, list)

    def test_run_returns_documents(self, pipeline):
        result = pipeline.run(TOY_CORPUS)
        for doc in result:
            assert isinstance(doc, Document)

    def test_run_filters_empty(self, pipeline):
        corpus = ["Good text here with variety.", "", "   "]
        result = pipeline.run(corpus)
        # Empty/whitespace should have CES 0 and be discarded or have very low weight
        texts = [d.text for d in result]
        # Empty string CES = 0 < discard_below=0.3 → should be filtered
        assert "" not in texts

    def test_documents_have_ces_scores(self, pipeline):
        result = pipeline.run(TOY_CORPUS)
        for doc in result:
            assert doc.ces >= 0.0

    def test_documents_have_weights(self, pipeline):
        result = pipeline.run(TOY_CORPUS)
        for doc in result:
            assert 0.0 <= doc.ces_weight <= 1.0
            assert 0.0 <= doc.tmc_weight <= 1.0
            assert 0.0 <= doc.combined_weight <= 1.0

    def test_run_with_authors_and_timestamps(self, pipeline):
        result = pipeline.run(
            TOY_CORPUS,
            author_ids=TOY_AUTHORS,
            timestamps=TOY_TIMESTAMPS,
        )
        assert len(result) > 0

    def test_run_with_document_objects(self, pipeline):
        docs = [
            Document(text="Hello world, this is a diverse document.", author_id="u1"),
            Document(text="Another document about science and discovery.", author_id="u2"),
        ]
        result = pipeline.run(docs)
        assert isinstance(result, list)

    def test_run_empty_corpus(self, pipeline):
        result = pipeline.run([])
        assert result == []


# ──────────────────────────────────────────────────────────────────────────────
# pipeline.get_batch()
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineGetBatch:
    def test_get_batch_correct_size(self, pipeline):
        corpus = pipeline.run(TOY_CORPUS)
        if len(corpus) >= 3:
            batch = pipeline.get_batch(corpus, batch_size=3, seed=42)
            assert len(batch) == 3

    def test_get_batch_returns_documents(self, pipeline):
        corpus = pipeline.run(TOY_CORPUS)
        if len(corpus) >= 2:
            batch = pipeline.get_batch(corpus, batch_size=2)
            for doc in batch:
                assert isinstance(doc, Document)

    def test_get_batch_smaller_than_corpus(self, pipeline):
        small_corpus = pipeline.run(["Alpha text here.", "Beta text there."])
        batch = pipeline.get_batch(small_corpus, batch_size=10)
        assert len(batch) <= len(small_corpus)


# ──────────────────────────────────────────────────────────────────────────────
# pipeline.corpus_health()
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineCorpusHealth:
    def test_health_returns_dict(self, pipeline):
        corpus = pipeline.run(TOY_CORPUS)
        health = pipeline.corpus_health(corpus)
        assert isinstance(health, dict)

    def test_health_has_expected_keys(self, pipeline):
        corpus = pipeline.run(TOY_CORPUS)
        health = pipeline.corpus_health(corpus)
        assert "n_documents" in health
        assert "ces" in health
        assert "cht" in health

    def test_health_n_documents_correct(self, pipeline):
        corpus = pipeline.run(TOY_CORPUS)
        health = pipeline.corpus_health(corpus)
        assert health["n_documents"] == len(corpus)


# ──────────────────────────────────────────────────────────────────────────────
# Document.to_dict()
# ──────────────────────────────────────────────────────────────────────────────

class TestDocumentToDict:
    def test_to_dict_has_keys(self):
        doc = Document(text="Some text here.", id="doc1", author_id="alice")
        d = doc.to_dict()
        for key in ["id", "author_id", "ces", "ces_weight", "tmc_weight"]:
            assert key in d

    def test_to_dict_text_preview_truncated(self):
        long_text = "x" * 200
        doc = Document(text=long_text)
        d = doc.to_dict()
        assert len(d["text_preview"]) <= 103  # 100 chars + "..."


# ──────────────────────────────────────────────────────────────────────────────
# from_config factory (YAML)
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineFromConfig:
    def test_from_config_raises_without_yaml_file(self):
        with pytest.raises(FileNotFoundError):
            HOVAPipeline.from_config("nonexistent_file.yaml")

    def test_from_yaml_config_file(self, tmp_path):
        """Write a minimal YAML and load it."""
        yaml_content = """
ces:
  window_size: 16
  discard_below: 0.3
tmc:
  n_topics: 5
  min_documents_per_author: 2
dps:
  min_opinion_divergence: 0.0
  min_emotional_entropy: 0.0
  min_lexical_diversity: 0.0
  min_cultural_entropy: 0.0
cews:
  checkpoint_every: 1000
"""
        yaml_file = tmp_path / "hova_config.yaml"
        yaml_file.write_text(yaml_content)

        pipeline = HOVAPipeline.from_config(str(yaml_file))
        result = pipeline.run(["A test document with varied content and context."])
        assert isinstance(result, list)
