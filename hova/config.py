"""
hova/config.py
Configuration dataclasses and YAML loader for all HOVA hyperparameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Layer-specific configuration dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CESConfig:
    """Configuration for the Cognitive Entropy Signature scorer (Layer 1)."""

    window_size: int = 32
    """Number of tokens in each local entropy window."""

    alpha: float = 5.0
    """Sigmoid steepness for weight assignment: sigmoid(alpha * (CES - tau))."""

    tau: float = 1.0
    """Sigmoid midpoint / threshold. CES > tau → upweighted."""

    discard_below: float = 0.7
    """Documents with CES < discard_below are discarded entirely."""


@dataclass
class TMCConfig:
    """Configuration for the Temporal Mutation Chain tracker (Layer 2)."""

    n_topics: int = 20
    """Number of LDA topics for topic-jump dimension."""

    beta: float = 4.0
    """Sigmoid steepness for weight_TMC: sigmoid(beta * TMC_score)."""

    min_documents_per_author: int = 3
    """Minimum documents required before computing a TMC score."""

    lda_passes: int = 5
    """Number of LDA training passes."""

    style_embed_model: str = "paraphrase-MiniLM-L6-v2"
    """Sentence-transformers model for identity consistency embedding."""


@dataclass
class ANTConfig:
    """Configuration for the Anchor Node Training loss term (Layer 3)."""

    anchor_model_path: Optional[str] = None
    """Path to a pre-trained (and frozen) HuggingFace anchor model directory."""

    lambda_ant: float = 0.1
    """Weight of the ANT divergence penalty in the total loss."""

    layers_to_match: List[int] = field(default_factory=lambda: [-1, -2, -3])
    """Which hidden layers to compute KL divergence over (negative = from end)."""


@dataclass
class DPSConfig:
    """Configuration for the Disagreement Preservation Sampler (Layer 4)."""

    min_opinion_divergence: float = 0.35
    """Minimum variance of opinion embeddings in each batch (Ω_op ≥ 0.35)."""

    min_emotional_entropy: float = 1.5
    """Minimum sentiment-label entropy in bits (Ω_emo ≥ 1.5)."""

    min_lexical_diversity: float = 0.40
    """Minimum mean pairwise lexical distance (Ω_lex ≥ 0.40)."""

    min_cultural_entropy: float = 1.2
    """Minimum language-region entropy in bits (Ω_cult ≥ 1.2)."""

    oversampling_factor: int = 4
    """How many candidate documents to consider per needed slot in a batch."""


@dataclass
class CEWSConfig:
    """Configuration for the Collapse Early Warning System (Layer 5)."""

    checkpoint_every: int = 500
    """How many training steps between CRI evaluations."""

    red_threshold: float = 0.75
    """CRI ≥ red → auto-pause training."""

    orange_threshold: float = 0.60
    """CRI ≥ orange → alert human operator."""

    yellow_threshold: float = 0.40
    """CRI ≥ yellow → increase monitoring frequency."""

    w_ces: float = 0.35
    """Weight of CES drift component in CRI."""

    w_cht: float = 0.35
    """Weight of CHT gradient component in CRI."""

    w_anc: float = 0.30
    """Weight of anchor divergence component in CRI."""

    history_window: int = 10
    """Number of past checkpoints used for min-max normalisation of CRI signals."""


@dataclass
class HOVAConfig:
    """Master configuration bundling all layer configs."""

    ces: CESConfig = field(default_factory=CESConfig)
    tmc: TMCConfig = field(default_factory=TMCConfig)
    ant: ANTConfig = field(default_factory=ANTConfig)
    dps: DPSConfig = field(default_factory=DPSConfig)
    cews: CEWSConfig = field(default_factory=CEWSConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HOVAConfig":
        """Load a HOVAConfig from a YAML file.

        Parameters
        ----------
        path:
            Path to a ``hova_config.yaml`` file.

        Returns
        -------
        HOVAConfig
            Populated configuration object.

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        FileNotFoundError
            If the YAML file does not exist.
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to load HOVA config files. "
                "Install with: pip install pyyaml"
            )
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}

        def _load(cls_, data):
            if data is None:
                return cls_()
            valid_keys = {f.name for f in cls_.__dataclass_fields__.values()}
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            return cls_(**filtered)

        return cls(
            ces=_load(CESConfig, raw.get("ces")),
            tmc=_load(TMCConfig, raw.get("tmc")),
            ant=_load(ANTConfig, raw.get("ant")),
            dps=_load(DPSConfig, raw.get("dps")),
            cews=_load(CEWSConfig, raw.get("cews")),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Serialise config to YAML.

        Parameters
        ----------
        path:
            Destination file path.
        """
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is required. pip install pyyaml")
        import dataclasses
        data = dataclasses.asdict(self)
        with open(Path(path), "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
