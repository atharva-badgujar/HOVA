"""
examples/quickstart.py
HOVA Quickstart — demonstrates all five layers in 60 lines.

Run with:
    python examples/quickstart.py
"""

from hova import (
    CESScorer, TMCTracker, DPSSampler, CHTConfig, CEWSMonitor,
    HOVAConfig, CESConfig, TMCConfig, DPSConfig, CEWSConfig,
)

print("=" * 60)
print("  HOVA — Human Origin Verification Architecture")
print("  Quickstart Demo")
print("=" * 60)

# ──────────────────────────────────────────────────────────────────────────────
# Layer 1: Cognitive Entropy Signature (CES)
# ──────────────────────────────────────────────────────────────────────────────

print("\n🔵 LAYER 1 — Cognitive Entropy Signature (CES)")
print("-" * 50)

scorer = CESScorer()

human_text = """
Yesterday I tried to fix the leaky tap in the kitchen — ended up flooding the
bathroom instead. My neighbour knocked to complain. She left me pie as a peace
offering. Strange. Also the markets crashed, so my portfolio is rubbish.
I need coffee. Not just any coffee — the expensive Kenyan single-origin kind.
Is that extravagant? Probably. Do I care? Not today.
"""

ai_text = """
The product is good. The product is very good. The product is excellent.
Customers enjoy the product. The product is recommended. The product is
available. The product is trusted. The product is a product. Customers
use the product. The service is great. The quality is great. The value is great.
"""

for label, text in [("Human-like", human_text), ("AI-like  ", ai_text)]:
    info = scorer.explain(text)
    print(f"  {label}: CES={info['ces']:.3f}  weight={info['weight']:.3f}  → {info['interpretation']}")

# ──────────────────────────────────────────────────────────────────────────────
# Layer 2: Temporal Mutation Chain (TMC)
# ──────────────────────────────────────────────────────────────────────────────

print("\n🟣 LAYER 2 — Temporal Mutation Chain (TMC)")
print("-" * 50)

tracker = TMCTracker(config=TMCConfig(min_documents_per_author=2))

# Alice: genuine human — vocabulary and topics evolve over years
tracker.add_document("alice", "I love basketball and outdoor sports.", "2018-01-01")
tracker.add_document("alice", "Reading Nietzsche lately. Philosophy is fascinating.", "2020-06-01")
tracker.add_document("alice", "My startup failed. Writing a novel now.", "2023-03-01")

# Bot: same text, trivially identical across time
tracker.add_document("bot", "Buy now best deal great price.", "2020-01-01")
tracker.add_document("bot", "Buy now best deal great price.", "2020-01-02")
tracker.add_document("bot", "Buy now best deal great price.", "2020-01-03")

for author in ["alice", "bot"]:
    score = tracker.get_score(author)
    weight = tracker.get_weight(author)
    print(f"  {author:6s}: TMC_score={score:.4f}  weight={weight:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# Layer 4: Disagreement Preservation Sampling (DPS)
# ──────────────────────────────────────────────────────────────────────────────

print("\n🟡 LAYER 4 — Disagreement Preservation Sampling (DPS)")
print("-" * 50)

corpus = [
    "I strongly support renewable energy and urgent climate action.",
    "Government regulation always makes things worse for business.",
    "Je pense que la politique est très complexe en France.",
    "Traditional values are the foundation of a stable society.",
    "Technology will solve all our environmental problems.",
    "Tax the rich and fund universal basic income now.",
    "Free markets allocate resources better than central planning.",
    "Belonging and community matter more than individual achievement.",
    "Vaccine science is clear and should guide public policy.",
    "Mainstream media cannot be trusted for objective reporting.",
]

sampler = DPSSampler(cht_config=CHTConfig(
    min_op=0.10, min_emo=0.5, min_lex=0.20, min_cult=0.3
))
health = sampler.corpus_health(corpus)
print(f"  Corpus Health Tensor:")
print(f"    Ω_op  (opinion divergence) = {health.omega_op:.3f}")
print(f"    Ω_emo (emotional entropy)  = {health.omega_emo:.3f}")
print(f"    Ω_lex (lexical diversity)  = {health.omega_lex:.3f}")
print(f"    Ω_cult (cultural entropy)  = {health.omega_cult:.3f}")
print(f"  → {health.status(sampler.cht_config)}")

batch = sampler.sample(corpus, batch_size=4, seed=42)
print(f"\n  Sampled batch ({len(batch)} documents):")
for i, doc in enumerate(batch):
    print(f"    [{i+1}] {doc[:70]}{'...' if len(doc) > 70 else ''}")

# ──────────────────────────────────────────────────────────────────────────────
# Layer 5: Collapse Early Warning System (CEWS)
# ──────────────────────────────────────────────────────────────────────────────

print("\n🔴 LAYER 5 — Collapse Early Warning System (CEWS)")
print("-" * 50)

monitor = CEWSMonitor(config=CEWSConfig(
    checkpoint_every=10,
    red_threshold=0.75,
    orange_threshold=0.60,
    yellow_threshold=0.40,
))

# Simulate 20 training steps with gradually degrading signals
print("  Simulating 20 training steps (5 stable → 5 degrading → 10 collapse)...")
for step in range(20):
    if step < 5:
        ces, cht, anc = 1.3, [0.5, 2.0, 0.6, 1.5], 0.05
    elif step < 10:
        ces, cht, anc = 1.0, [0.3, 1.2, 0.4, 0.9], 0.4
    else:
        ces, cht, anc = 0.6, [0.1, 0.4, 0.1, 0.2], 1.8
    ckpt = monitor.update(ces_mean=ces, cht_vector=cht, anchor_div=anc)
    if ckpt:
        print(f"  Step {ckpt.step:3d}: CRI={ckpt.cri:.3f}  {ckpt.status.emoji} {ckpt.status.value.upper()}")

print(f"\n  Final CRI : {monitor.current_cri:.3f}")
print(f"  Status    : {monitor.current_status.emoji} {monitor.current_status.value.upper()}")
print(f"  Paused    : {monitor.is_paused()}")

print("\n" + "=" * 60)
print("  ✅ Quickstart complete!")
print("  Next: python examples/full_pipeline_demo.py")
print("=" * 60)
