# HOVA — Human Origin Verification Architecture

> **A five-layer mathematical framework for preventing recursive model collapse in AI training pipelines.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/hova.svg)](https://pypi.org/project/hova/)
[![Tests](https://github.com/your-handle/hova/actions/workflows/ci.yml/badge.svg)](https://github.com/your-handle/hova/actions)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX)

---

## The Problem

As AI-generated content floods the internet, future AI models face a catastrophic training risk known as **model collapse**. When models train recursively on outputs from previous models, they suffer progressive degradation: rare ideas vanish, diversity collapses, and outputs converge toward bland statistical averages.

> **By April 2025, over 74% of newly created webpages contained AI-generated text.** The trajectory is clear: without intervention, future AI training will increasingly feed on its own echo.

Existing solutions share a fundamental flaw — they are **reactive**. They attempt to detect and remove AI-generated content after it has already entered the pipeline. This will fail long-term because AI quality improves continuously, making detection increasingly unreliable.

## The HOVA Philosophy

> *Stop trying to detect what is fake. Start mathematically defining and continuously preserving what is irreplaceably human — then make that the gravitational centre of every training pipeline.*

HOVA takes a fundamentally different approach. Rather than chasing AI content to remove it, HOVA **mathematically characterises human cognitive signal** and builds its protection directly into the training architecture through five independent, composable layers.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     RAW DATA INTAKE                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           LAYER 1: CES — Cognitive Entropy Signature        │
│   CES(D) = H̄_local(D) / H_global(D)                       │
│   Discard CES < 0.7  |  weight = sigmoid(α·(CES−τ))        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           LAYER 2: TMC — Temporal Mutation Chain            │
│   TMC_score = Var_TMC(a) · ID_cons(a)                      │
│   Upweight authors with authentic longitudinal drift        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           LAYER 4: DPS — Disagreement Preservation         │
│   CHT(B) = [Ω_op, Ω_emo, Ω_lex, Ω_cult]                  │
│   Sample B* s.t. CHT(B*) ≥ [0.35, 1.5, 0.40, 1.2]        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│    LAYER 3: ANT — Anchor Node Training Loss                 │
│    L_HOVA = L_standard + 0.1 · Σ_l KL(h^M_l || h^A*_l)   │
│    Frozen anchor model pulls representations toward human   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│    LAYER 5: CEWS — Collapse Early Warning System           │
│    CRI = 0.35·σ(CES_drift) + 0.35·σ(CHT_grad) + 0.30·σ(AnchorDiv) │
│    CRI ≥ 0.75 → AUTO-PAUSE + MANDATORY AUDIT               │
└─────────────────────────────────────────────────────────────┘
```

| Layer | Name | What It Does |
|-------|------|-------------|
| **CES** | Cognitive Entropy Signature | Measures the local/global entropy ratio — humans are locally unpredictable but globally coherent |
| **TMC** | Temporal Mutation Chain | Tracks author vocabulary, topic, and sentiment drift over time — real humans change, bots don't |
| **ANT** | Anchor Node Training | Frozen pre-AI reference model pulls training representations back toward human cognitive space |
| **DPS** | Disagreement Preservation Sampling | Enforces corpus health via diversity tensor — opinions, emotions, lexis, and culture |
| **CEWS** | Collapse Early Warning System | Unifies all signals into a Collapse Risk Index (CRI) with automatic training pause |

---

## Installation

### Core (no GPU required)

```bash
pip install hova
```

### With NLP extras (better TMC accuracy)

```bash
pip install hova[nlp]
python -m spacy download en_core_web_sm
python -m textblob.download_corpora
```

### With training extras (ANT loss term)

```bash
pip install hova[training]
```

### Full installation (everything)

```bash
pip install hova[full]
python -m spacy download en_core_web_sm
python -m textblob.download_corpora
```

### From source (development)

```bash
git clone https://github.com/your-handle/hova.git
cd hova
pip install -e ".[dev]"
python -m textblob.download_corpora
```

---

## Quick Start

### Layer 1: Score individual documents

```python
from hova import CESScorer

scorer = CESScorer(window_size=32)

human_text = "Yesterday I tried to fix the leaky tap and flooded the bathroom instead..."
ai_text = "The product is good. The product is very good. The product is available..."

print(scorer.score(human_text))    # e.g. 1.42 → strong human signal
print(scorer.score(ai_text))       # e.g. 0.61 → likely AI-generated

# Batch scoring → returns a pandas DataFrame
df = scorer.score_batch([human_text, ai_text], ids=["doc1", "doc2"])
print(df[["id", "ces", "weight", "interpretation"]])

# Detailed breakdown
info = scorer.explain(human_text)
# {'tokens': 148, 'h_local_mean': 3.21, 'h_global': 2.26, 'ces': 1.42, ...}
```

### Layer 2: Track author authenticity

```python
from hova import TMCTracker

tracker = TMCTracker(n_topics=20)

# Alice: real human — topics and vocabulary evolve over years
tracker.add_document("alice", "I love basketball.", "2018-01-01")
tracker.add_document("alice", "Reading Nietzsche lately.", "2020-06-01")
tracker.add_document("alice", "My startup failed. Writing a novel.", "2023-03-01")

print(tracker.get_score("alice"))   # high: genuine longitudinal drift
print(tracker.get_weight("alice"))  # e.g. 0.82

# See all author summaries
print(tracker.summary())
```

### Layer 4: Sample diversity-preserving batches

```python
from hova import DPSSampler, CHTConfig

config = CHTConfig(min_op=0.35, min_emo=1.5, min_lex=0.40, min_cult=1.2)
sampler = DPSSampler(cht_config=config)

corpus = [...]  # your documents
batch = sampler.sample(corpus, batch_size=1024, seed=42)

health = sampler.corpus_health(corpus)
print(health.status(config))
# → ✅ Healthy — all CHT constraints satisfied
```

### Layer 3: ANT loss term (requires `pip install hova[training]`)

```python
from hova import AnchorNodeLoss

ant_loss = AnchorNodeLoss(
    anchor_model_path="./my_anchor_model",  # or any HuggingFace model ID
    lambda_ant=0.1,
)

# Inside your training loop:
loss = ce_loss + ant_loss(model, input_ids, attention_mask)
loss.backward()
```

### Layer 5: Collapse monitoring

```python
from hova import CEWSMonitor

monitor = CEWSMonitor(
    checkpoint_every=500,
    red_threshold=0.75,
    orange_threshold=0.60,
)

# Inside your training loop:
checkpoint = monitor.update(
    ces_mean=batch_ces_mean,
    cht_vector=[omega_op, omega_emo, omega_lex, omega_cult],
    anchor_div=kl_from_anchor,
)

if monitor.is_paused():
    monitor.alert()  # Logs structured alert; calls on_pause callback
    # → AUDIT YOUR DATA PIPELINE, then:
    monitor.resume()
```

### Full Pipeline (recommended)

```python
from hova import HOVAPipeline

pipeline = HOVAPipeline.from_config("hova_config.yaml")

# Stages 1-4: filter + weight the corpus
clean_corpus = pipeline.run(
    raw_documents,          # list of strings or Document objects
    author_ids=[...],       # optional: enables TMC scoring
    timestamps=[...],       # optional: ISO date strings
)

# Stage 4: diversity-preserving batch sampling
batch = pipeline.get_batch(clean_corpus, batch_size=1024)

# Corpus health diagnostics
print(pipeline.corpus_health(clean_corpus))

# Stages 5-7: training loop with ANT + CEWS
history = pipeline.training_loop(
    model=my_model,
    corpus=clean_corpus,
    epochs=3,
    batch_size=1024,
    on_batch=my_training_step,  # fn(batch_docs, step) → (loss, ces_mean, anchor_div)
)
```

---

## Configuration

Copy `hova_config.yaml` to your project root and customise:

```yaml
ces:
  window_size: 32        # local entropy window (tokens)
  alpha: 5.0             # sigmoid steepness
  tau: 1.0               # CES threshold midpoint
  discard_below: 0.7     # hard discard threshold

tmc:
  n_topics: 20           # LDA topics for topic-jump
  beta: 4.0              # TMC weight sigmoid steepness
  min_documents_per_author: 3

ant:
  anchor_model_path: null    # HuggingFace model ID or local path
  lambda_ant: 0.1            # penalty weight in total loss

dps:
  min_opinion_divergence: 0.35
  min_emotional_entropy: 1.5
  min_lexical_diversity: 0.40
  min_cultural_entropy: 1.2

cews:
  checkpoint_every: 500
  red_threshold: 0.75     # auto-pause
  orange_threshold: 0.60  # alert
  yellow_threshold: 0.40  # warn
  weights: [0.35, 0.35, 0.30]
```

Load with:

```python
pipeline = HOVAPipeline.from_config("hova_config.yaml")
```

---

## Running Examples

```bash
# Quickstart — all 5 layers in 60 lines
python examples/quickstart.py

# Full 7-stage pipeline demo with diagnostics
python examples/full_pipeline_demo.py
```

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"
python -m textblob.download_corpora

# Run the full test suite
pytest tests/ -v

# With coverage report
pytest tests/ --cov=hova --cov-report=term-missing
```

---

## CES Interpretation Reference

| CES Value | Interpretation | Action |
|-----------|---------------|--------|
| **CES ≥ 1.3** | Strong human signal | Full weight |
| **1.0 ≤ CES < 1.3** | Ambiguous | Moderate weight |
| **0.7 ≤ CES < 1.0** | Likely AI-generated | Downweight |
| **CES < 0.7** | Almost certainly AI-generated | **Discard** |

## CEWS Status Reference

| CRI Range | Status | Action |
|-----------|--------|--------|
| 0.00 – 0.40 | 🟢 **Green** | Normal training |
| 0.40 – 0.60 | 🟡 **Yellow** | Log warning, increase monitoring |
| 0.60 – 0.75 | 🟠 **Orange** | Alert human operator |
| 0.75 – 1.00 | 🔴 **Red** | **Auto-pause + mandatory audit** |

---

## Dependency Matrix

| Feature | Required packages |
|---------|-----------------|
| CES, CEWS | `numpy`, `scipy`, `pandas` *(core)* |
| DPS sampling | `scikit-learn`, `textblob`, `langdetect` *(core)* |
| TMC (basic) | `textblob` *(core)* |
| TMC (full) | `spacy`, `gensim` *(nlp extra)* |
| ANT loss | `torch`, `transformers` *(training extra)* |

---

## Project Structure

```
hova/
├── hova/
│   ├── __init__.py      ← Public API
│   ├── ces.py           ← Layer 1: Cognitive Entropy Signature
│   ├── tmc.py           ← Layer 2: Temporal Mutation Chain
│   ├── ant.py           ← Layer 3: Anchor Node Training loss
│   ├── dps.py           ← Layer 4: Disagreement Preservation Sampling
│   ├── cews.py          ← Layer 5: Collapse Early Warning System
│   ├── pipeline.py      ← HOVAPipeline orchestrator
│   └── config.py        ← Dataclasses + YAML loader
├── tests/               ← pytest unit + integration tests
├── examples/
│   ├── quickstart.py
│   └── full_pipeline_demo.py
├── hova_config.yaml     ← Example configuration
├── pyproject.toml       ← Package metadata and dependencies
└── LICENSE              ← Apache 2.0
```

---

## Novelty

All five HOVA components were verified as novel prior to release:

| Component | Novelty |
|-----------|---------|
| **CES** | Entropy detectors exist as binary classifiers; CES is the first *continuous training weight* |
| **TMC** | **Fully novel** — no prior work uses cross-time author drift as authenticity signal |
| **ANT** | Partially novel — frozen model as *gravitational loss penalty* has no prior equivalent |
| **DPS** | **Fully novel** — first system treating disagreement density as a corpus health metric |
| **CEWS** | **Fully novel** — first unified index of multiple collapse signals with automatic pause |

---

## Roadmap

- **v0.1** — CES + DPS + CEWS (released)
- **v0.5** — TMC + full NLP integration + PyPI release
- **v1.0** — ANT loss + full pipeline + HuggingFace Trainer integration + research paper
- **v2.0** — Multimodal extension, streaming support, web-scale deployment guide

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest tests/`
5. Submit a pull request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon) for detailed guidelines.

---

## Citation

If you use HOVA in your research, please cite:

```bibtex
@article{hova2026,
  title   = {HOVA: Human Origin Verification Architecture for Preventing
             Recursive Model Collapse in Contaminated Training Corpora},
  author  = {HOVA Contributors},
  year    = {2026},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
}
```

---

## Related Work

- Shumailov et al. (2024). *Model Collapse.* Nature. — The foundational proof of collapse
- Gerstgrasser et al. (2024). *Accumulation.* — Primary baseline HOVA improves upon
- Kirchenbauer et al. (2024). *Watermarking.* — Complementary approach (generation-side)

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
