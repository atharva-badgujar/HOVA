# HOVA — Human Origin Verification Architecture
### Technical Design Document · v1.0
**A Framework for Preventing Recursive Model Collapse**

> Status: Draft for Open-Source Development · May 2026

---

## Table of Contents

1. [Overview & Problem Statement](#1-overview--problem-statement)
2. [The HOVA Algorithm — Complete Mathematical Specification](#2-the-hova-algorithm--complete-mathematical-specification)
3. [Complete HOVA Pipeline Architecture](#3-complete-hova-pipeline-architecture)
4. [Open-Source Project Specification](#4-open-source-project-specification)
5. [Development Roadmap](#5-development-roadmap)
6. [Research Paper Outline](#6-research-paper-outline)
7. [Skills, Tools & Resources](#7-skills-tools--resources)
8. [Quick Reference — All Equations](#8-quick-reference--all-equations)

---

## 1. Overview & Problem Statement

### 1.1 The Problem

As AI-generated content increasingly floods the internet, future AI models face a catastrophic training risk known as **model collapse**. When models train recursively on outputs from previous models — rather than original human-authored data — they suffer progressive degradation: rare ideas vanish, diversity collapses, and outputs converge toward bland statistical averages.

This is not theoretical. Peer-reviewed research (Shumailov et al., *Nature* 2024) has proven the phenomenon across LLMs, VAEs, and diffusion models. By April 2025, over 74% of newly created webpages contained AI-generated text. The trajectory is clear: without intervention, future AI training will increasingly feed on its own echo.

### 1.2 Why Existing Solutions Are Insufficient

Current approaches share a fundamental flaw: they are **reactive**. They attempt to detect and remove AI-generated content after it has already entered the data pipeline. This approach will fail long-term because:

- AI generation quality improves continuously, making stylistic detection increasingly unreliable
- Binary filtering creates adversarial dynamics — content is optimised to evade detectors
- No existing system mathematically defines and actively preserves what makes human signal uniquely valuable
- No unified early-warning system exists to detect collapse onset before it becomes irreversible

### 1.3 The HOVA Philosophy

HOVA takes a fundamentally different approach. Rather than chasing AI content to remove it, HOVA **mathematically characterises human cognitive signal** and builds its protection directly into the training architecture.

> **Core Thesis:** Stop trying to detect what is fake. Start mathematically defining and continuously preserving what is irreplaceably human — then make that the gravitational centre of every training pipeline.

### 1.4 Novelty Verification

Prior to designing HOVA, an extensive search of existing literature was conducted. The following components are confirmed novel with no prior equivalent found:

| HOVA Component | Novelty Status |
|---|---|
| Cognitive Entropy Signature (CES) — local/global entropy ratio as continuous weight | New formulation. Entropy detectors exist but only as binary classifiers, not as continuous training weights. |
| Temporal Mutation Chain (TMC) — longitudinal author identity drift scoring | **Fully novel.** No existing research uses cross-time author drift as an authenticity signal for training data. |
| Anchor Node Training (ANT) — frozen human-grounded model as a loss term | Partially novel. Static anchor datasets exist; a frozen model as a gravitational loss penalty does not. |
| Disagreement Preservation Sampling (DPS) — corpus health via variance enforcement | **Fully novel.** No existing work treats disagreement density as a corpus health metric with hard sampling constraints. |
| Collapse Early Warning System (CEWS) — unified CRI score triggering auto-pause | **Fully novel.** No system unifies multiple collapse signals into a single index with automatic training interruption. |

---

## 2. The HOVA Algorithm — Complete Mathematical Specification

HOVA is composed of five interdependent layers. Each layer addresses a distinct failure mode of recursive training. Together they form a closed-loop system that actively maintains human signal integrity throughout the training lifecycle.

---

### 2.1 Layer 1 — Cognitive Entropy Signature (CES)

#### 2.1.1 Motivation

Human writing exhibits a characteristic pattern: it is **locally unpredictable** (tangents, contradictions, emotional pivots) but **globally coherent** (the document has a purpose). AI writing inverts this — locally smooth and probabilistically optimal, globally flat and averaged. CES captures this ratio mathematically and uses it as a continuous quality weight on every document.

#### 2.1.2 Formal Definition

Let `D` be a document composed of `n` tokens `{w₁, w₂, …, wₙ}`. Define a window function `W(k, s)` as the set of `s` tokens centred at position `k`.

**Local Entropy at position k with window size s:**
```
H_local(k, s) = - Σ  p(wᵢ | W(k,s)) · log₂ p(wᵢ | W(k,s))
```

**Mean Local Entropy across the document:**
```
H̄_local(D) = (1/n) · Σₖ H_local(k, s)      [s = 32 tokens, recommended]
```

**Global Entropy over the full document:**
```
H_global(D) = - Σ  p(wᵢ | D) · log₂ p(wᵢ | D)
```

**Cognitive Entropy Signature:**
```
CES(D) = H̄_local(D) / H_global(D)
```

#### 2.1.3 Interpretation

| CES Range | Interpretation |
|---|---|
| CES > 1.3 | Strong human signal — high local variance, global coherence maintained |
| 1.0 ≤ CES ≤ 1.3 | Ambiguous — possible human, possible high-quality synthetic |
| CES < 1.0 | Strong AI signal — locally smooth, globally flat |
| CES < 0.7 | Almost certainly AI-generated — discard or heavily downweight |

#### 2.1.4 Training Weight Assignment

```
weight_CES(D) = sigmoid(α · (CES(D) − τ))     [α=5.0, τ=1.0 recommended]
```

This produces a soft weight in `(0, 1)` that gracefully rewards human-like documents without hard thresholds.

---

### 2.2 Layer 2 — Temporal Mutation Chain (TMC)

#### 2.2.1 Motivation

Humans change. Their vocabulary evolves, their opinions shift, their syntax matures. This is because real time and real experience act on them. No AI system can fabricate the genuine statistical trace of a life unfolding. TMC captures this trace and uses it as a **longitudinal authenticity score**.

#### 2.2.2 Formal Definition

For an author `a` with documents `{D₁, D₂, …, Dₜ}` ordered by timestamp, define a four-dimensional feature vector at each timestep `t`:

```
φ(a, t) = [ V_drift(a,t),  S_vol(a,t),  T_jump(a,t),  Syn_mut(a,t) ]
```

Where each dimension is defined as:

**Vocabulary Drift** — Jaccard distance between consecutive documents' vocabularies:
```
V_drift(a, t) = |vocab(Dₜ) △ vocab(Dₜ₋₁)| / |vocab(Dₜ) ∪ vocab(Dₜ₋₁)|
```

**Sentiment Volatility** — absolute shift in mean sentiment score:
```
S_vol(a, t) = |sentiment(Dₜ) − sentiment(Dₜ₋₁)|
```

**Topic Jump** — proportion of topic distribution that changed (K = number of LDA topics):
```
T_jump(a, t) = |topics(Dₜ) △ topics(Dₜ₋₁)| / K
```

**Syntactic Mutation** — normalised edit distance between POS tag sequences:
```
Syn_mut(a, t) = edit_distance(POS_seq(Dₜ), POS_seq(Dₜ₋₁)) / max_len
```

#### 2.2.3 TMC Authenticity Score

The key insight: a real human has **HIGH variance over time** (life changes) but maintains **CONSISTENT identity markers** (it is still the same person).

```
Var_TMC(a)   = (1/T) · Σₜ ||φ(a,t) − φ̄(a)||²          [temporal variance]

ID_cons(a)   = cos_sim( style_embed(D₁), style_embed(Dₜ) )   [identity consistency]

TMC_score(a) = Var_TMC(a) · ID_cons(a)
```

A genuine human author should have `Var_TMC > 0` (they changed) and `ID_cons ≈ 1` (it's still them). AI-generated fake personas fail one or both.

**Document weighting:**
```
weight_TMC(D_a) = sigmoid(β · TMC_score(a))      [β=4.0 recommended]
```

---

### 2.3 Layer 3 — Anchor Node Training (ANT)

#### 2.3.1 Motivation

Even with CES and TMC filtering, the main model's internal representations may drift from authentic human language over time. ANT introduces a **frozen reference model** — trained exclusively on physically-grounded, pre-AI human data — whose internal representations serve as a gravitational anchor, pulling the main model back toward human cognitive space.

#### 2.3.2 Anchor Node Data Requirements

The anchor model `A*` is trained **once** on data that cannot be AI-generated because it requires physical human presence. It is **frozen after training and never updated**.

- Handwritten document OCR (pre-2015, pre-LLM era)
- Audio-transcribed spontaneous conversations
- Medical and legal handwritten records
- Physical mail and diary digitisation
- Longitudinal personal correspondence archives

#### 2.3.3 ANT Loss Term

During training of the main model `M`, for each layer `l`, we compute the representation divergence from the anchor:

```
D_ANT(M, A*, x) = Σₗ  KL( h^M_l(x) || h^{A*}_l(x) )
```

where `h^M_l(x)` is the hidden state at layer `l` of model `M` for input `x`, and `KL` is Kullback-Leibler divergence.

**The modified training objective:**
```
L_HOVA = L_standard + λ_ANT · D_ANT(M, A*, x)      [λ_ANT = 0.1 recommended]
```

As `λ_ANT` penalises divergence from the anchor, the model is continuously pulled toward human-grounded representations. The larger the synthetic contamination, the stronger this corrective pull.

---

### 2.4 Layer 4 — Disagreement Preservation Sampling (DPS)

#### 2.4.1 Motivation

Model collapse is fundamentally a **variance problem** — the distribution narrows. The most insidious form is not just losing data volume but losing the diversity of human disagreement, perspective, and worldview. DPS operationalises corpus health as a measurable tensor and enforces minimum variance thresholds during every training batch.

#### 2.4.2 Corpus Health Tensor (CHT)

Define the Corpus Health Tensor for a batch `B` as:

```
CHT(B) = [ Ω_op(B),  Ω_emo(B),  Ω_lex(B),  Ω_cult(B) ]
```

**Opinion Divergence** — variance of opinion embeddings across the batch:
```
Ω_op(B)   = Var({ opinion_embed(D) : D ∈ B })
```

**Emotional Polarity Variance** — entropy of the sentiment label distribution:
```
Ω_emo(B)  = H({ sentiment_label(D) : D ∈ B })
```

**Lexical Diversity Index** — mean pairwise lexical distance within the batch:
```
Ω_lex(B)  = (1/|B|²) · Σᵢⱼ  (1 − cos_sim(tfidf(Dᵢ), tfidf(Dⱼ)))
```

**Cultural Conflict Density** — entropy of language-region distribution:
```
Ω_cult(B) = H({ lang_region(D) : D ∈ B })
```

#### 2.4.3 DPS Sampling Constraint

```
Sample B* = argmax_{B ⊆ Corpus} P(B)
subject to:  CHT(B*) ≥ CHT_min  (component-wise)
```

In practice this is a **priority-queue sampler**: documents are scored by how much they increase the current batch's CHT. Documents that increase variance (unusual viewpoints, minority dialects, contrarian arguments) are preferentially sampled.

**Recommended minimum thresholds:**

| CHT Component | Minimum Threshold |
|---|---|
| Ω_op — Opinion divergence | ≥ 0.35 (variance of unit-norm vectors) |
| Ω_emo — Emotional entropy | ≥ 1.5 bits (across sentiment classes) |
| Ω_lex — Lexical diversity | ≥ 0.40 (mean pairwise distance) |
| Ω_cult — Cultural entropy | ≥ 1.2 bits (across language-regions) |

---

### 2.5 Layer 5 — Collapse Early Warning System (CEWS)

#### 2.5.1 Motivation

All upstream layers reduce collapse risk. But if contamination is severe enough to slip through, we need to detect the onset of collapse **before it becomes irreversible**. CEWS monitors three independent signals and combines them into a single **Collapse Risk Index (CRI)** that can automatically pause training.

#### 2.5.2 The Three Monitoring Signals

At every checkpoint `c`, compute:

```
CES_drift(c) = CES̄(c-1) − CES̄(c)         [drop in mean CES over batches]

CHT_grad(c)  = ||CHT(c-1) − CHT(c)||₂      [L2 decline in health tensor]

AnchorDiv(c) = D_ANT(M_c, A*, x_probe)     [divergence from anchor on probe set]
```

#### 2.5.3 Collapse Risk Index (CRI)

```
CRI(c) = w₁ · σ(CES_drift(c)) + w₂ · σ(CHT_grad(c)) + w₃ · σ(AnchorDiv(c))

where σ is min-max normalisation over recent history
      w₁=0.35,  w₂=0.35,  w₃=0.30
```

`CRI ∈ [0, 1]`. **Recommended action thresholds:**

| CRI Value | Action |
|---|---|
| 0.0 – 0.40 | 🟢 Green — normal training, no action required |
| 0.40 – 0.60 | 🟡 Yellow — log warning, increase monitoring frequency |
| 0.60 – 0.75 | 🟠 Orange — alert human operator, flag data batch for audit |
| 0.75 – 1.00 | 🔴 Red — **AUTOMATIC TRAINING PAUSE**, mandatory data pipeline audit |

---

## 3. Complete HOVA Pipeline Architecture

### 3.1 Pipeline Stages

1. **RAW DATA INTAKE** — Web scrape, licensed corpora, user-generated content all enter the raw pool.
2. **CES SCORING** — Every document is scored. CES < 0.7 → discard. CES 0.7–1.0 → weight 0.1–0.4. CES > 1.0 → weight 0.4–1.0.
3. **TMC ANALYSIS** — For any author/source with multiple documents, `TMC_score` is computed and used to upweight authentic longitudinal sources.
4. **DPS SAMPLING** — Batch constructor enforces CHT thresholds. Preferentially pulls documents that increase corpus health variance.
5. **TRAINING LOOP** — Standard cross-entropy loss + ANT divergence penalty. Model is pulled toward anchor representations at every step.
6. **CEWS CHECKPOINT** — At every N steps (N=500 recommended), compute CRI. If CRI > 0.75, pause training and audit the incoming data pipeline.
7. **RESUME** — After audit clears, re-enrich with verified human data, reset CRI history, resume training.

### 3.2 Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     RAW DATA INTAKE                         │
│        (web scrape + licensed corpora + UGC)                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     STAGE 2: CES FILTER                     │
│   Score every document → assign continuous weight (0–1)     │
│   Discard CES < 0.7                                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   STAGE 3: TMC ANALYSIS                     │
│   Track author identity drift over time                     │
│   Upweight sources with authentic longitudinal mutation     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   STAGE 4: DPS SAMPLER                      │
│   Build each batch enforcing CHT ≥ [0.35, 1.5, 0.40, 1.2] │
│   Preferentially sample high-variance, disagreement-rich    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 5: TRAINING LOOP + ANT                   │
│   L_HOVA = L_standard + 0.1 · D_ANT(M, A*, x)             │
│   Frozen anchor model A* pulls representations toward       │
│   authentic human language at every gradient step           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 6: CEWS MONITOR (every 500 steps)        │
│   Compute CRI from CES_drift + CHT_grad + AnchorDiv         │
│   CRI > 0.75 → AUTO-PAUSE + ALERT                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
               ┌──────────┴──────────┐
               │                     │
           CRI ≤ 0.75           CRI > 0.75
               │                     │
               ▼                     ▼
          CONTINUE             STAGE 7: AUDIT
          TRAINING          Re-enrich → Reset → Resume
```

---

## 4. Open-Source Project Specification

### 4.1 Project Identity

| Property | Value |
|---|---|
| Package Name | `hova` |
| Install Command | `pip install hova` |
| Language | Python 3.10+ |
| License | Apache 2.0 (permissive, company-friendly) |
| Repository | `github.com/[your-handle]/hova` |
| Target Users | AI/ML engineers, research labs, companies training LLMs |

### 4.2 Repository Structure

```
hova/
├── hova/
│   ├── __init__.py
│   ├── ces.py              # CES scorer
│   ├── tmc.py              # TMC tracker
│   ├── ant.py              # ANT loss + anchor model utilities
│   ├── dps.py              # DPS sampler
│   ├── cews.py             # CEWS monitor + CRI computation
│   ├── pipeline.py         # Full HOVAPipeline orchestrator
│   └── config.py           # Default hyperparameters
├── tests/
├── examples/
│   ├── quickstart.py
│   └── full_pipeline_demo.py
├── benchmarks/
├── docs/
├── README.md
└── pyproject.toml
```

### 4.3 Public API Design

#### 4.3.1 CES Scorer

```python
from hova import CESScorer

scorer = CESScorer(window_size=32)
score  = scorer.score(text)             # float
weight = scorer.weight(text)            # float in (0, 1)
df     = scorer.score_batch(documents)  # returns DataFrame
```

#### 4.3.2 TMC Tracker

```python
from hova import TMCTracker

tracker = TMCTracker(n_topics=20)
tracker.add_document(author_id, text, timestamp)
score   = tracker.get_score(author_id)    # float
weight  = tracker.get_weight(author_id)   # float in (0, 1)
```

#### 4.3.3 DPS Sampler

```python
from hova import DPSSampler, CHTConfig

config  = CHTConfig(min_op=0.35, min_emo=1.5, min_lex=0.40, min_cult=1.2)
sampler = DPSSampler(cht_config=config)
batch   = sampler.sample(corpus, batch_size=1024)
health  = sampler.corpus_health(corpus)   # returns CHT dict
```

#### 4.3.4 ANT Loss

```python
from hova import AnchorNodeLoss

ant_loss = AnchorNodeLoss(
    anchor_model_path='./anchor_model',
    lambda_ant=0.1
)

# inside training loop:
loss = ce_loss + ant_loss(model, batch_inputs)
```

#### 4.3.5 CEWS Monitor

```python
from hova import CEWSMonitor

monitor = CEWSMonitor(
    checkpoint_every=500,
    red_threshold=0.75,
    orange_threshold=0.60
)

# inside training loop:
monitor.update(ces_mean, cht_vector, anchor_div)
if monitor.is_paused():
    monitor.alert()   # logs + calls user-defined callback
```

#### 4.3.6 Full Pipeline

```python
from hova import HOVAPipeline

pipeline     = HOVAPipeline.from_config('hova_config.yaml')
clean_corpus = pipeline.run(raw_corpus)
pipeline.training_loop(model, clean_corpus, epochs=3)
```

### 4.4 Configuration File (`hova_config.yaml`)

```yaml
ces:
  window_size: 32
  alpha: 5.0
  tau: 1.0
  discard_below: 0.7

tmc:
  n_topics: 20
  beta: 4.0
  min_documents_per_author: 3

ant:
  anchor_model_path: null   # set to path of pre-trained anchor
  lambda_ant: 0.1
  layers_to_match: [-1, -2, -3]   # last 3 layers

dps:
  min_opinion_divergence: 0.35
  min_emotional_entropy: 1.5
  min_lexical_diversity: 0.40
  min_cultural_entropy: 1.2

cews:
  checkpoint_every: 500
  red_threshold: 0.75
  orange_threshold: 0.60
  yellow_threshold: 0.40
  weights: [0.35, 0.35, 0.30]
```

---

## 5. Development Roadmap

### Phase 1 — Foundation (Months 1–2)
- Formalise all mathematics; write proofs for CES and DPS convergence properties
- Set up GitHub repository with Apache 2.0 licence, README, contributing guide
- Implement **CES scorer** — simplest component, most demonstrable value → release as v0.1
- Implement **DPS sampler** — validate CHT thresholds on synthetic contamination experiments
- Begin writing Sections 1–3 of the research paper in parallel

### Phase 2 — Core Implementation (Months 3–4)
- Implement **TMC tracker** — build author longitudinal dataset for validation
- Implement **ANT loss term** — train a small anchor model on pre-2015 OCR/transcription data
- Implement **CEWS monitor** with CRI computation and callback system
- Release v0.5 with CES + DPS + CEWS — publish to PyPI

### Phase 3 — Experiments & Validation (Months 5–6)
- Full pipeline experiments: Wikipedia + C4 corpus, contaminated at 10%, 30%, 50%, 90% synthetic ratios
- Baselines to beat: no filtering / simple accumulation (Gerstgrasser 2024) / entropy-only filtering
- Key metrics: perplexity, lexical diversity, semantic entropy, CRI trajectory over generations
- Release v1.0 (all 5 layers) — write results section of paper

### Phase 4 — Paper & Community (Month 7)
- Submit full paper to **arXiv** (establishes timestamp, invites feedback)
- Submit to **COLM** or **EMNLP** conference
- Write integration guides for HuggingFace Trainer, PyTorch Lightning, and JAX/Flax
- Reach out to open-source training communities (EleutherAI, Together AI, LAION) for adoption

---

## 6. Research Paper Outline

**Title:** *HOVA: Human Origin Verification Architecture for Preventing Recursive Model Collapse in Contaminated Training Corpora*

### Abstract (target: 200 words)
- Problem: Model collapse under recursive synthetic contamination is proven and accelerating
- Gap: All existing solutions are reactive detectors; no system mathematically defines and preserves human signal
- Contribution: HOVA — a five-layer architecture providing continuous authenticity weighting, longitudinal identity tracking, frozen anchor grounding, variance-preserving sampling, and early collapse detection
- Key result: *(to be filled after experiments)* HOVA maintains corpus diversity X% longer than accumulation baseline at Y% contamination

### Section 1 — Introduction
- Motivate with real-world contamination statistics (74% of new pages in 2025)
- State the four contributions of this paper clearly
- Briefly describe each layer and its purpose

### Section 2 — Related Work
- Shumailov et al. 2024 — Nature paper proving collapse
- Gerstgrasser et al. 2024 — Accumulation solution (primary baseline)
- Entropy-based detectors (Lavergne 2008, DetectGPT 2023)
- Watermarking approaches (Kirchenbauer 2024)
- Disagreement-aware NLP (LeWiDi 2023) — note it addresses annotation, not training data health
- **Gap statement:** none of these prevent collapse via signal preservation; all are reactive filters

### Section 3 — The HOVA Framework
- Full mathematical specification of all 5 layers
- Proof sketch: CES separates human/AI token entropy distributions under mild regularity conditions
- Proof sketch: DPS maintains CHT ≥ threshold iff the diversity constraint is satisfiable given corpus
- Proposition: ANT loss bounds KL divergence from anchor model as O(λ_ANT · T) over T training steps

### Section 4 — Experimental Setup
- Datasets: Wikipedia (human baseline), C4 (mixed), synthetically contaminated corpora at {10, 30, 50, 90}%
- Baselines: No filtering / Random sampling / Accumulation (Gerstgrasser) / Entropy-only / Full HOVA
- Models: GPT-2 small (fast iteration), Llama 3.2-1B (scale validation)
- Metrics: Perplexity, Type-Token Ratio, semantic entropy, CRI over N training generations

### Section 5 — Results & Discussion
- *[Experimental results to be completed]*
- Ablation study: each layer removed individually to measure contribution
- Failure cases: when does HOVA struggle? (e.g., very high-quality synthetic data)

### Section 6 — Conclusion
- HOVA reframes the problem: from reactive detection to proactive signal preservation
- Future work: multimodal extension, real-time web-scale deployment, formal convergence proofs

---

## 7. Skills, Tools & Resources

### 7.1 Technical Skills

| Skill | Used For |
|---|---|
| Python 3.10+ | Core library implementation |
| PyTorch or JAX | ANT loss term, model training experiments |
| HuggingFace Transformers | Base models, tokenisers, training loops |
| scikit-learn | CES entropy computation, DPS sampler |
| NLTK / spaCy | TMC: POS tagging, syntactic mutation, topic modelling |
| Gensim | LDA topic modelling for TMC T_jump dimension |
| LaTeX | Research paper writing |
| Git / GitHub Actions | Repository, CI/CD, automated testing |

### 7.2 Compute Requirements

| Experiment Stage | Minimum Compute |
|---|---|
| CES + DPS validation (Phase 1) | Single GPU (RTX 3090 or Colab A100 free tier) |
| Full HOVA pipeline on GPT-2 | Single A100 80GB or equivalent (Lambda Labs ~$1.50/hr) |
| Scale validation on Llama 3.2-1B | 4× A100 80GB (Lambda Labs or RunPod ~$6/hr) |
| Ablation study | Same as full pipeline, 5 runs |

### 7.3 Where to Find Collaborators

- **EleutherAI Discord** — active open-source LLM training community
- **HuggingFace Community Forums** — researchers actively building training pipelines
- **ML Collective** — open research org supporting independent researchers
- **arXiv Twitter/X community** — post your preprint and engage
- **University professors** — a single academic co-author dramatically improves publication acceptance rate

### 7.4 Datasets for Experiments

| Dataset | Purpose |
|---|---|
| Wikipedia English dump | High-quality human baseline |
| C4 (Colossal Clean Crawled Corpus) | Mixed web text, good contamination simulation |
| Project Gutenberg | Pre-AI human text, good for anchor model training |
| OpenSubtitles | Spontaneous human dialogue, good TMC source |
| Reddit dumps (Pushshift) | High disagreement density, ideal for DPS validation |

---

## 8. Quick Reference — All Equations

A single-page reference of every equation in HOVA.

### Layer 1 — CES
```
CES(D)         = H̄_local(D) / H_global(D)
weight_CES(D)  = sigmoid(5.0 · (CES(D) − 1.0))
```

### Layer 2 — TMC
```
φ(a,t)         = [V_drift, S_vol, T_jump, Syn_mut]
TMC_score(a)   = Var_TMC(a) · ID_cons(a)
weight_TMC(D)  = sigmoid(4.0 · TMC_score(a))
```

### Layer 3 — ANT
```
D_ANT(M,A*,x)  = Σₗ KL( h^M_l(x) || h^{A*}_l(x) )
L_HOVA         = L_standard + 0.1 · D_ANT
```

### Layer 4 — DPS
```
CHT(B)         = [Ω_op, Ω_emo, Ω_lex, Ω_cult]
Constraint:      CHT(B*) ≥ [0.35, 1.5, 0.40, 1.2]
```

### Layer 5 — CEWS
```
CRI(c)         = 0.35·σ(CES_drift) + 0.35·σ(CHT_grad) + 0.30·σ(AnchorDiv)
Thresholds:      Green<0.40 | Yellow<0.60 | Orange<0.75 | Red≥0.75
```

---

*HOVA Technical Design Document · v1.0 · May 2026*
*Draft for Open-Source Development*
