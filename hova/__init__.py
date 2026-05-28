"""
hova/__init__.py
Public API for the HOVA library.

Human Origin Verification Architecture — v0.1.0
A framework for preventing recursive model collapse in AI training pipelines.

Quick start
-----------
>>> from hova import CESScorer
>>> scorer = CESScorer()
>>> scorer.score("The human text I want to evaluate...")

>>> from hova import HOVAPipeline
>>> pipeline = HOVAPipeline.from_config("hova_config.yaml")
>>> clean_corpus = pipeline.run(raw_documents)
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hova")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

# ── Layer 1: Cognitive Entropy Signature ─────────────────────────────────────
from hova.ces import CESScorer

# ── Layer 2: Temporal Mutation Chain ─────────────────────────────────────────
from hova.tmc import TMCTracker

# ── Layer 3: Anchor Node Training ────────────────────────────────────────────
# Note: AnchorNodeLoss requires PyTorch + Transformers (pip install hova[training])
try:
    from hova.ant import AnchorNodeLoss
except ImportError:
    AnchorNodeLoss = None  # type: ignore[assignment,misc]

# ── Layer 4: Disagreement Preservation Sampling ───────────────────────────────
from hova.dps import DPSSampler, CHTConfig, CHTVector, compute_cht

# ── Layer 5: Collapse Early Warning System ───────────────────────────────────
from hova.cews import CEWSMonitor, CollapseStatus, Checkpoint

# ── Pipeline orchestrator ─────────────────────────────────────────────────────
from hova.pipeline import HOVAPipeline, Document

# ── Configuration ─────────────────────────────────────────────────────────────
from hova.config import (
    HOVAConfig,
    CESConfig,
    TMCConfig,
    ANTConfig,
    DPSConfig,
    CEWSConfig,
)

__all__ = [
    # Version
    "__version__",
    # Layer 1
    "CESScorer",
    # Layer 2
    "TMCTracker",
    # Layer 3
    "AnchorNodeLoss",
    # Layer 4
    "DPSSampler",
    "CHTConfig",
    "CHTVector",
    "compute_cht",
    # Layer 5
    "CEWSMonitor",
    "CollapseStatus",
    "Checkpoint",
    # Pipeline
    "HOVAPipeline",
    "Document",
    # Config
    "HOVAConfig",
    "CESConfig",
    "TMCConfig",
    "ANTConfig",
    "DPSConfig",
    "CEWSConfig",
]
