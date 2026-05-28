"""
hova/pipeline.py
HOVAPipeline — Full end-to-end HOVA orchestrator.

Chains all five layers:
    RAW DATA → CES filter → TMC analysis → DPS sampling → Training Loop (ANT + CEWS)

Usage
-----
    from hova import HOVAPipeline
    pipeline = HOVAPipeline.from_config("hova_config.yaml")
    clean_corpus = pipeline.run(raw_corpus)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from hova.ces import CESScorer
from hova.cews import CEWSMonitor, CollapseStatus
from hova.config import HOVAConfig
from hova.dps import CHTConfig, DPSSampler, compute_cht
from hova.tmc import TMCTracker

logger = logging.getLogger("hova.pipeline")


# ──────────────────────────────────────────────────────────────────────────────
# Document type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A corpus document with optional metadata.

    Attributes
    ----------
    text:
        Raw document text.
    id:
        Optional unique identifier.
    author_id:
        Optional author identifier (required for TMC scoring).
    timestamp:
        Optional document timestamp (ISO string, POSIX float, or datetime).
    ces:
        CES score (populated after pipeline.run()).
    ces_weight:
        CES training weight (populated after pipeline.run()).
    tmc_weight:
        TMC training weight (populated after pipeline.run()).
    combined_weight:
        Final combined weight (populated after pipeline.run()).
    """

    text: str
    id: Optional[str] = None
    author_id: Optional[str] = None
    timestamp: Optional[str] = None
    ces: float = 0.0
    ces_weight: float = 0.0
    tmc_weight: float = 0.0
    combined_weight: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "author_id": self.author_id,
            "timestamp": self.timestamp,
            "ces": round(self.ces, 4),
            "ces_weight": round(self.ces_weight, 4),
            "tmc_weight": round(self.tmc_weight, 4),
            "combined_weight": round(self.combined_weight, 4),
            "text_preview": self.text[:100] + ("..." if len(self.text) > 100 else ""),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class HOVAPipeline:
    """Orchestrate the full 7-stage HOVA data cleaning and training pipeline.

    Parameters
    ----------
    config:
        ``HOVAConfig`` bundling all layer configurations.

    Examples
    --------
    **From a config file:**

    >>> pipeline = HOVAPipeline.from_config("hova_config.yaml")
    >>> clean = pipeline.run(raw_corpus)

    **Programmatic setup:**

    >>> from hova import HOVAPipeline, HOVAConfig, CESConfig
    >>> config = HOVAConfig(ces=CESConfig(discard_below=0.8))
    >>> pipeline = HOVAPipeline(config)
    >>> clean = pipeline.run(["doc1 text...", "doc2 text..."])
    """

    def __init__(self, config: Optional[HOVAConfig] = None) -> None:
        self._cfg = config or HOVAConfig()
        self._ces = CESScorer(config=self._cfg.ces)
        self._tmc = TMCTracker(config=self._cfg.tmc)
        self._dps = DPSSampler(
            config=self._cfg.dps,
            cht_config=CHTConfig(
                min_op=self._cfg.dps.min_opinion_divergence,
                min_emo=self._cfg.dps.min_emotional_entropy,
                min_lex=self._cfg.dps.min_lexical_diversity,
                min_cult=self._cfg.dps.min_cultural_entropy,
            ),
        )
        self._cews = CEWSMonitor(config=self._cfg.cews)
        self._ant_loss = None  # loaded lazily

    @classmethod
    def from_config(cls, path: str) -> "HOVAPipeline":
        """Factory: load a pipeline from a YAML config file.

        Parameters
        ----------
        path:
            Path to ``hova_config.yaml``.

        Returns
        -------
        HOVAPipeline
            Configured pipeline instance.
        """
        config = HOVAConfig.from_yaml(path)
        return cls(config)

    # ── Main data pipeline ────────────────────────────────────────────────────

    def run(
        self,
        corpus: Sequence[str | Document],
        author_ids: Optional[Sequence[Optional[str]]] = None,
        timestamps: Optional[Sequence[Optional[str]]] = None,
        doc_ids: Optional[Sequence[Optional[str]]] = None,
    ) -> List[Document]:
        """Run the HOVA data pipeline on a raw corpus.

        Executes stages 1–4:
          Stage 1 — (input already provided)
          Stage 2 — CES scoring and filtering
          Stage 3 — TMC analysis
          Stage 4 — DPS sampling (applied when you call get_batch())

        Parameters
        ----------
        corpus:
            Raw documents as strings or ``Document`` objects.
        author_ids:
            Optional per-document author identifiers.
        timestamps:
            Optional per-document timestamps.
        doc_ids:
            Optional per-document identifiers.

        Returns
        -------
        list of Document
            Filtered and weighted documents, ready for batch sampling.
        """
        docs = self._coerce_documents(corpus, author_ids, timestamps, doc_ids)
        logger.info(f"HOVA Pipeline: received {len(docs)} documents.")

        # ── Stage 2: CES Filter ───────────────────────────────────────────────
        ces_passed = []
        n_discarded = 0
        for doc in docs:
            doc.ces = self._ces.score(doc.text)
            doc.ces_weight = self._ces.weight(doc.text)
            if self._ces.should_discard(doc.text):
                n_discarded += 1
            else:
                ces_passed.append(doc)
        logger.info(
            f"Stage 2 (CES): {n_discarded} discarded, {len(ces_passed)} retained."
        )

        # ── Stage 3: TMC Analysis ─────────────────────────────────────────────
        for doc in ces_passed:
            if doc.author_id:
                self._tmc.add_document(
                    doc.author_id, doc.text, doc.timestamp
                )

        for doc in ces_passed:
            if doc.author_id:
                doc.tmc_weight = self._tmc.get_weight(doc.author_id)
            else:
                doc.tmc_weight = 0.5  # neutral weight if no author info

        # Combined weight: geometric mean of CES and TMC weights
        for doc in ces_passed:
            doc.combined_weight = (doc.ces_weight * doc.tmc_weight) ** 0.5

        logger.info(
            f"Stage 3 (TMC): {len(self._tmc.list_authors())} unique authors tracked."
        )

        return ces_passed

    def get_batch(
        self,
        corpus: List[Document],
        batch_size: int,
        seed: Optional[int] = None,
    ) -> List[Document]:
        """Stage 4: Sample a diversity-preserving batch from the filtered corpus.

        Parameters
        ----------
        corpus:
            List of ``Document`` objects returned by ``run()``.
        batch_size:
            Number of documents in the batch.
        seed:
            Optional random seed.

        Returns
        -------
        list of Document
            A batch satisfying CHT diversity constraints.
        """
        texts = [d.text for d in corpus]
        sampled_texts = self._dps.sample(texts, batch_size=batch_size, seed=seed)
        sampled_set = set(id(s) for s in sampled_texts)

        # Match back to Document objects (by text content)
        text_to_doc = {d.text: d for d in corpus}
        return [text_to_doc[t] for t in sampled_texts if t in text_to_doc]

    # ── Training loop integration ─────────────────────────────────────────────

    def training_loop(
        self,
        model: Any,
        corpus: List[Document],
        epochs: int = 1,
        batch_size: int = 32,
        on_batch: Optional[Callable[[List[Document], int], Tuple[float, float, float]]] = None,
        ant_model_path: Optional[str] = None,
    ) -> List[dict]:
        """Run a HOVA-supervised training loop (Stages 5–7).

        This method orchestrates training with:
        - ANT loss (if PyTorch + Transformers available and anchor configured)
        - CEWS monitoring at every checkpoint interval
        - Automatic pause + alert on RED CRI

        Parameters
        ----------
        model:
            Your model to train.
        corpus:
            Filtered corpus from ``run()``.
        epochs:
            Number of training epochs.
        batch_size:
            Documents per batch.
        on_batch:
            Callback invoked for each batch with signature:
            ``on_batch(batch_docs, step) → (loss, ces_mean, anchor_div)``
            If not provided, a stub that returns mock values is used.
        ant_model_path:
            Path to anchor model.  Overrides ``config.ant.anchor_model_path``.

        Returns
        -------
        list of dict
            CEWS checkpoint history.
        """
        # Lazy ANT initialisation
        effective_anchor = ant_model_path or self._cfg.ant.anchor_model_path
        if effective_anchor and self._ant_loss is None:
            try:
                from hova.ant import AnchorNodeLoss
                self._ant_loss = AnchorNodeLoss(
                    config=self._cfg.ant,
                    anchor_model_path=effective_anchor,
                )
                logger.info(f"ANT: anchor model loaded from {effective_anchor}")
            except ImportError as e:
                logger.warning(f"ANT not available (missing deps): {e}")

        step = 0
        n_docs = len(corpus)

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            for batch_start in range(0, n_docs, batch_size):
                batch = self.get_batch(
                    corpus,
                    batch_size=min(batch_size, n_docs - batch_start),
                    seed=step,
                )
                if not batch:
                    continue

                # ── Stage 5: Training step ───────────────────────────────────
                if on_batch:
                    loss, ces_mean, anchor_div = on_batch(batch, step)
                else:
                    # Placeholder: compute CES statistics for monitoring
                    ces_scores = [d.ces for d in batch]
                    ces_mean = float(np.mean(ces_scores))
                    anchor_div = 0.0
                    loss = 0.0

                # ── CHT for this batch ───────────────────────────────────────
                cht = compute_cht([d.text for d in batch])
                cht_vec = list(cht.as_array())

                # ── Stage 6: CEWS checkpoint ─────────────────────────────────
                ckpt = self._cews.update(
                    ces_mean=ces_mean,
                    cht_vector=cht_vec,
                    anchor_div=anchor_div,
                )
                if ckpt:
                    self._cews.alert()

                # ── Stage 7: Auto-pause on RED ───────────────────────────────
                if self._cews.is_paused():
                    logger.critical(
                        f"CEWS AUTO-PAUSE at step {step}. "
                        "Audit the data pipeline, then call pipeline.resume()."
                    )
                    return self._cews.export_history()

                step += 1

        logger.info(f"Training complete. {step} steps processed.")
        return self._cews.export_history()

    def resume(self) -> None:
        """Resume training after a CEWS-triggered pause (audit complete)."""
        self._cews.resume()

    # ── Corpus health diagnostics ──────────────────────────────────────────────

    def corpus_health(self, corpus: List[Document]) -> dict:
        """Compute CHT and CES statistics for the full corpus.

        Parameters
        ----------
        corpus:
            List of ``Document`` objects.

        Returns
        -------
        dict
            Summary with CES statistics, CHT values, and health status.
        """
        texts = [d.text for d in corpus]
        cht = self._dps.corpus_health(texts)
        ces_scores = [d.ces for d in corpus if d.ces > 0]

        return {
            "n_documents": len(corpus),
            "ces": {
                "mean": round(float(np.mean(ces_scores)), 4) if ces_scores else 0.0,
                "std": round(float(np.std(ces_scores)), 4) if ces_scores else 0.0,
                "min": round(float(np.min(ces_scores)), 4) if ces_scores else 0.0,
                "max": round(float(np.max(ces_scores)), 4) if ces_scores else 0.0,
            },
            "cht": cht.to_dict(),
            "cht_status": cht.status(self._dps.cht_config),
            "tmc_authors": len(self._tmc.list_authors()),
            "cews": self._cews.summary(),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _coerce_documents(
        corpus: Sequence[str | Document],
        author_ids: Optional[Sequence[Optional[str]]],
        timestamps: Optional[Sequence[Optional[str]]],
        doc_ids: Optional[Sequence[Optional[str]]],
    ) -> List[Document]:
        docs = []
        for i, item in enumerate(corpus):
            if isinstance(item, Document):
                docs.append(item)
            else:
                docs.append(
                    Document(
                        text=item,
                        id=doc_ids[i] if doc_ids else str(i),
                        author_id=author_ids[i] if author_ids else None,
                        timestamp=timestamps[i] if timestamps else None,
                    )
                )
        return docs
