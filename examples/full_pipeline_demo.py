"""
examples/full_pipeline_demo.py
HOVA Full Pipeline Demo — runs all 7 stages on a toy corpus.

Demonstrates:
    - HOVAPipeline.run() (Stages 1-4)
    - HOVAPipeline.get_batch() (Stage 4: DPS sampling)
    - HOVAPipeline.corpus_health() (diagnostics)
    - HOVAPipeline.training_loop() (Stages 5-7)

Run with:
    python examples/full_pipeline_demo.py
"""

import json
from hova import HOVAPipeline, HOVAConfig, CESConfig, TMCConfig, DPSConfig, CEWSConfig

print("=" * 70)
print("  HOVA — Full Pipeline Demo")
print("  7-Stage Human Origin Verification Architecture")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────────────
# Configure the pipeline
# ──────────────────────────────────────────────────────────────────────────────

config = HOVAConfig(
    ces=CESConfig(
        window_size=32,
        alpha=5.0,
        tau=1.0,
        discard_below=0.5,    # slightly lenient for demo
    ),
    tmc=TMCConfig(
        n_topics=5,           # small for demo speed
        beta=4.0,
        min_documents_per_author=2,
    ),
    dps=DPSConfig(
        min_opinion_divergence=0.10,
        min_emotional_entropy=0.50,
        min_lexical_diversity=0.20,
        min_cultural_entropy=0.30,
    ),
    cews=CEWSConfig(
        checkpoint_every=3,   # frequent for demo visibility
        red_threshold=0.75,
        orange_threshold=0.60,
        yellow_threshold=0.40,
    ),
)

pipeline = HOVAPipeline(config=config)

# ──────────────────────────────────────────────────────────────────────────────
# Raw corpus — mixed quality (simulates real-world data contamination)
# ──────────────────────────────────────────────────────────────────────────────

raw_corpus = [
    # High quality: diverse human perspectives
    "The collapse of manufacturing in my hometown wasn't inevitable — it was a policy choice that nobody admitted to making.",
    "I changed my mind on immigration after living abroad for three years. Context matters enormously.",
    "Je trouve que la philosophie continentale est souvent mal comprise par les anglophones.",
    "My daughter's disability has taught me more about patience than twenty years of meditation practice ever did.",
    "The irony of climate sceptics using empirical language to reject empirical science is not lost on me.",
    "There's a fundamental tension between liberty and equality that no political ideology has honestly resolved.",
    "Growing up bilingually means never quite feeling at home in either language — a beautiful homelessness.",
    "The evidence for early childhood interventions is overwhelming; the political will is not.",
    "Sometimes I wonder if our obsession with productivity has made us worse at the things that actually matter.",
    "Technology amplifies human nature — it makes the generous more generous and the cruel more cruel.",
    # Low quality: AI-like repetitive content
    "The product is good. The product is available. The product is recommended. The product is great.",
    "Buy now. Great price. Best deal. Order today. Fast shipping. Great value. Buy now.",
    # Very short / empty
    "OK.",
    "",
]

authors = [
    "alice", "bob", "charlie", "alice", "bob",
    "charlie", "alice", "bob", "charlie", "alice",
    None, None, None, None,
]

timestamps = [
    "2018-01-01", "2019-03-15", "2020-07-20",
    "2020-11-01", "2021-04-10", "2021-09-30",
    "2022-02-14", "2022-08-01", "2023-01-01",
    "2023-06-15", "2023-01-01", "2023-02-01",
    "2023-03-01", "2023-04-01",
]

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1–3: CES filtering + TMC analysis
# ──────────────────────────────────────────────────────────────────────────────

print("\n📥 STAGE 1: Ingesting raw corpus...")
print(f"   {len(raw_corpus)} documents received.")

print("\n🔵 STAGES 2–3: CES filtering + TMC analysis...")
clean_corpus = pipeline.run(
    raw_corpus,
    author_ids=authors,
    timestamps=timestamps,
)
print(f"   {len(clean_corpus)} documents passed CES filter.")

print("\n   Top documents by combined weight:")
sorted_docs = sorted(clean_corpus, key=lambda d: d.combined_weight, reverse=True)
for i, doc in enumerate(sorted_docs[:5]):
    preview = doc.text[:65] + ("..." if len(doc.text) > 65 else "")
    print(f"   [{i+1}] weight={doc.combined_weight:.3f}  ces={doc.ces:.3f}  | {preview}")

# ──────────────────────────────────────────────────────────────────────────────
# Stage 4: DPS batch sampling
# ──────────────────────────────────────────────────────────────────────────────

print("\n🟡 STAGE 4: Diversity-preserving batch sampling (DPS)...")
if len(clean_corpus) >= 4:
    batch = pipeline.get_batch(clean_corpus, batch_size=4, seed=42)
    print(f"   Sampled batch of {len(batch)} documents:")
    for i, doc in enumerate(batch):
        print(f"   [{i+1}] {doc.text[:65]}...")
else:
    print(f"   (Corpus too small for meaningful batch: {len(clean_corpus)} docs)")

# ──────────────────────────────────────────────────────────────────────────────
# Corpus health report
# ──────────────────────────────────────────────────────────────────────────────

print("\n📊 CORPUS HEALTH REPORT:")
health = pipeline.corpus_health(clean_corpus)
print(f"   Documents    : {health['n_documents']}")
print(f"   CES mean     : {health['ces']['mean']:.3f}")
print(f"   CES std      : {health['ces']['std']:.3f}")
print(f"   Ω_op         : {health['cht']['omega_op']:.3f}")
print(f"   Ω_emo        : {health['cht']['omega_emo']:.3f}")
print(f"   Ω_lex        : {health['cht']['omega_lex']:.3f}")
print(f"   Ω_cult       : {health['cht']['omega_cult']:.3f}")
print(f"   CHT Status   : {health['cht_status']}")
print(f"   TMC Authors  : {health['tmc_authors']}")

# ──────────────────────────────────────────────────────────────────────────────
# Stage 5–7: Training loop with CEWS monitoring
# (Using a mock callback since we have no real model in this demo)
# ──────────────────────────────────────────────────────────────────────────────

print("\n🔴 STAGES 5–7: Training loop + CEWS monitoring...")
print("   (Using mock training callback — plug in your real model here)")

step_counter = [0]

def mock_on_batch(batch_docs, step):
    """Mock training step. Returns (loss, ces_mean, anchor_div)."""
    from hova.ces import CESScorer
    scorer = CESScorer()
    ces_scores = [scorer.score(d.text) for d in batch_docs]
    ces_mean = sum(ces_scores) / len(ces_scores) if ces_scores else 0.0
    # Simulate gradual quality degradation for demo
    anchor_div = 0.05 + (step * 0.08)
    loss = 2.5 - (step * 0.1)
    step_counter[0] = step
    return max(loss, 0.0), ces_mean, min(anchor_div, 2.0)


class MockModel:
    """Placeholder model object."""
    pass


cews_history = pipeline.training_loop(
    model=MockModel(),
    corpus=clean_corpus,
    epochs=2,
    batch_size=3,
    on_batch=mock_on_batch,
)

print(f"\n   Training complete. {len(cews_history)} CEWS checkpoints recorded.")
if cews_history:
    print("   Checkpoint summary:")
    for ckpt in cews_history:
        status_emoji = {
            "green": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴"
        }.get(ckpt["status"], "⚪")
        print(
            f"     Step {ckpt['step']:4d}: CRI={ckpt['cri']:.3f}  "
            f"{status_emoji} {ckpt['status'].upper()}"
        )

print("\n" + "=" * 70)
print("  ✅ Full pipeline demo complete!")
print("  For production use:")
print("    1. Replace MockModel with your actual model")
print("    2. Implement on_batch() to run real forward/backward passes")
print("    3. Use AnchorNodeLoss for the ANT penalty (pip install hova[training])")
print("    4. Configure hova_config.yaml for your dataset and compute budget")
print("=" * 70)
