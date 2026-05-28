"""
hova/ant.py
Layer 3 — Anchor Node Training (ANT)

A frozen reference model — trained exclusively on pre-AI human data — whose
hidden representations act as a gravitational anchor during training.

    D_ANT(M, A*, x) = Σ_l KL( h^M_l(x) || h^{A*}_l(x) )
    L_HOVA = L_standard + λ_ANT · D_ANT(M, A*, x)

The ANT loss requires PyTorch and HuggingFace Transformers.
Install the optional dependencies with:  pip install hova[training]
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Union

from hova.config import ANTConfig

# ──────────────────────────────────────────────────────────────────────────────
# Optional deep learning dependencies
# ──────────────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the ANT loss term.\n"
            "Install with:  pip install hova[training]\n"
            "or:            pip install torch transformers"
        )


def _require_transformers():
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "HuggingFace Transformers is required for the ANT loss term.\n"
            "Install with:  pip install hova[training]\n"
            "or:            pip install transformers"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Utility: KL divergence between hidden state distributions
# ──────────────────────────────────────────────────────────────────────────────

def _kl_divergence_hidden(
    h_main: "torch.Tensor",
    h_anchor: "torch.Tensor",
    eps: float = 1e-8,
) -> "torch.Tensor":
    """Compute KL(h_main || h_anchor) for batched hidden states.

    Hidden states are softmax-normalised before KL computation so they
    represent valid probability distributions over the hidden dimension.

    Parameters
    ----------
    h_main:
        Hidden state from the main model, shape (batch, seq_len, hidden).
    h_anchor:
        Hidden state from the anchor model, shape (batch, seq_len, hidden).
    eps:
        Small constant for numerical stability.

    Returns
    -------
    torch.Tensor
        Scalar mean KL divergence.
    """
    _require_torch()
    p = F.softmax(h_main.float(), dim=-1).clamp(min=eps)
    q = F.softmax(h_anchor.float(), dim=-1).clamp(min=eps)
    kl = (p * (p / q).log()).sum(dim=-1)  # (batch, seq_len)
    return kl.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class AnchorNodeLoss(nn.Module if _TORCH_AVAILABLE else object):
    """Compute the ANT divergence penalty between a training model and the frozen anchor.

    The anchor model is loaded once, frozen, and never updated.  It provides
    a gravitational pull back toward authentic human-grounded representations.

    Parameters
    ----------
    config:
        ``ANTConfig`` with hyperparameters.  Uses defaults if ``None``.
    anchor_model_path:
        HuggingFace model identifier or local directory containing the anchor.
        If ``None``, the config value is used.

    Examples
    --------
    >>> ant = AnchorNodeLoss(anchor_model_path="gpt2")
    >>> loss_term = ant(main_model, input_ids)
    >>> total_loss = ce_loss + loss_term
    """

    def __init__(
        self,
        config: Optional[ANTConfig] = None,
        anchor_model_path: Optional[str] = None,
    ) -> None:
        _require_torch()
        _require_transformers()
        super().__init__()

        self._cfg = config or ANTConfig()
        model_path = anchor_model_path or self._cfg.anchor_model_path

        if model_path is None:
            raise ValueError(
                "anchor_model_path must be provided either in ANTConfig or "
                "as a constructor argument.  Example: AnchorNodeLoss(anchor_model_path='gpt2')"
            )

        # Load and FREEZE the anchor model
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._anchor = AutoModel.from_pretrained(
            model_path,
            output_hidden_states=True,
        )
        for param in self._anchor.parameters():
            param.requires_grad = False
        self._anchor.eval()

        self._lambda = self._cfg.lambda_ant
        self._layers = self._cfg.layers_to_match

    def forward(
        self,
        model: "nn.Module",
        input_ids: "torch.Tensor",
        attention_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Compute the ANT loss term.

        Parameters
        ----------
        model:
            The main model being trained.  Must support ``output_hidden_states=True``.
        input_ids:
            Token IDs, shape (batch, seq_len).
        attention_mask:
            Optional attention mask, shape (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Scalar loss term: ``lambda_ANT · Σ_l KL(h_main_l || h_anchor_l)``.
        """
        _require_torch()

        # Main model hidden states
        main_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        main_hidden = main_out.hidden_states  # tuple of (batch, seq, hidden)

        # Anchor model hidden states (no grad)
        device = input_ids.device
        with torch.no_grad():
            anchor_out = self._anchor(
                input_ids=input_ids.to(self._anchor.device),
                attention_mask=(
                    attention_mask.to(self._anchor.device)
                    if attention_mask is not None
                    else None
                ),
                output_hidden_states=True,
            )
        anchor_hidden = anchor_out.hidden_states

        # Sum KL divergences across selected layers
        total_kl = torch.tensor(0.0, device=device, requires_grad=True)
        n_layers = len(main_hidden)
        for layer_idx in self._layers:
            resolved = layer_idx % n_layers if layer_idx < 0 else layer_idx
            if resolved >= n_layers:
                continue
            h_m = main_hidden[resolved]
            h_a = anchor_hidden[resolved].to(device)
            # Match hidden dimensions if architectures differ
            if h_m.shape[-1] != h_a.shape[-1]:
                h_a = F.interpolate(
                    h_a.unsqueeze(0), size=h_m.shape[-1], mode="linear"
                ).squeeze(0)
            total_kl = total_kl + _kl_divergence_hidden(h_m, h_a)

        return self._lambda * total_kl

    def to_device(self, device: Union[str, "torch.device"]) -> "AnchorNodeLoss":
        """Move the anchor model to a device.

        Parameters
        ----------
        device:
            Target device string (e.g. ``"cuda"``, ``"cpu"``, ``"mps"``).

        Returns
        -------
        AnchorNodeLoss
            Self (for chaining).
        """
        self._anchor = self._anchor.to(device)
        return self

    @property
    def lambda_ant(self) -> float:
        """Current value of λ_ANT."""
        return self._lambda

    @lambda_ant.setter
    def lambda_ant(self, value: float) -> None:
        self._lambda = float(value)

    @classmethod
    def is_available(cls) -> bool:
        """Check whether PyTorch + Transformers are installed."""
        return _TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE
