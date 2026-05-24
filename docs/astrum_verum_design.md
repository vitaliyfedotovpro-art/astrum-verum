# Astrum Verum — Design & Research Notes

> Geometric and Vector-Symbolic approaches to cognitive memory for AI agents.
> This document is deliberately honest about what is proven, what failed, and why.

---

## Muninn — official definition

**Muninn** (Old Norse *“Memory”*, one of Odin's two ravens) is the name of the
VSA memory engine (`astrum_verum/vsa`, `extract`, `cognitive`).

> **Muninn — Vector-Symbolic Associative Memory (VSAM).** Composition-episodic
> architecture: facts via algebraic role-binding (subject⊗relation⊗object),
> conversations as permutation-encoded episodes. Retrieval is **associative *and*
> structural** — by meaning / a partial or noisy cue (cleanup) and by role
> (unbind), not by an exact key. **Zero persona-prompt: pure memory, not a
> personality** — it returns what is stored; it does not generate or personalize.

Two properties define it, and both are *demonstrated*, not asserted:
- **associative** (content-addressable; recall from a partial/corrupted cue) —
  separates Muninn from a key-value store (no exact key needed);
- **compositional / structural** (algebraic role-binding, “who did what to whom”)
  — separates Muninn from a plain vector DB *and* from generative “neural
  memory”. On role-swapped facts (“A loves B” vs “B loves A”) cosine similarity
  sits at **0.5 (chance)** while Muninn scores **1.0** (Phase 1/2).

Naming: project = **Astrum Verum**; memory engine = **Muninn**; the agent that
uses it = **Óðinn** (Muninn is literally his memory). Personality lives *above*
the memory, never *in* it.

---

## 0. TL;DR

Astrum Verum contains **two layers built in sequence**, and the second exists
because the first did not earn its claims:

1. **Lattice core** (`astrum_verum/lattice`, `engine`, `store`, …) — long-term
   memory organised on perfect root lattices (D₄ → E₈, aiming at the Leech
   lattice Λ₂₄), with spreading activation and Hebbian learning. It is elegant
   and mathematically correct, but it is essentially a **fixed vector quantizer
   whose retrieval-quality thesis was never demonstrated**.

2. **VSA/SDM cognitive memory** (`astrum_verum/vsa`, `extract`, `cognitive`) —
   composition-episodic memory built on Vector Symbolic Architectures and Sparse
   Distributed Memory: role-binding facts, permutation-encoded episodes,
   error-correcting cleanup, entity normalization and persistence. This layer is
   **validated end-to-end** and does something flat vector search structurally
   cannot: answer *who-did-what-to-whom* queries.

The honest headline: **the geometry was beautiful but unproven; the algebra works.**

---

## 1. Layer 1 — The Lattice Core (and why it is a mockup)

### 1.1 The idea

Treat memory as a crystal. Project a text embedding onto the surface of a perfect
root lattice; the lattice vertices are "semantic domains", adjacency between
vertices defines association paths, and retrieval is a wave of *spreading
activation* over that graph, biased by *Hebbian* edge learning and an SO(d)
attention rotation.

- **D₄ (24-cell):** 24 vertices, 96 edges, 8 neighbours/vertex, self-dual.
- **E₈:** 240 roots, 6720 edges, 56 neighbours/vertex; densest packing in 8D
  (Viazovska, 2016).
- **Λ₂₄ (Leech):** the intended endpoint — densest packing in 24D, kissing
  number 196 560, automorphism group Co₀.

The geometry in `lattice/d4.py` and `lattice/e8.py` is **correct** (vertex
construction, adjacency and CVP are all asserted at build time). The pipeline
(`engine.py`) is clean: embed → Concept-Anchored Projection → SO(d) focus
rotation → CVP decode → spreading activation → hybrid scoring → Hebbian update.

### 1.2 Why it is a mockup, honestly

Three problems, none fatal individually, jointly decisive:

1. **The bottleneck is the projection, not the lattice.** Embeddings live in
   384+ dimensions; CAP (`projector.py`) compresses them to 4/8D via a softmax
   over hand-chosen anchors. Almost all semantic information is destroyed *before*
   the lattice is consulted. The lattice then quantizes whatever survives. The
   celebrated properties of E₈/Leech are properties of *their own dimension under
   uniform density* — they do not transfer to a lossy low-D projection of
   anisotropic embedding manifolds.

2. **Adjacency is semantically arbitrary.** Anchors are assigned to vertices by
   list order, so two lattice-adjacent cells carry no guaranteed semantic
   relationship. Spreading activation therefore propagates energy to
   semantically-random neighbours; `topo_boost` is mostly noise until Hebbian
   learning (slowly) reshapes it.

3. **The benchmark proved speed, not recall.** `benchmarks/run_benchmark.py`
   generates embeddings and lattice coordinates *independently*; its own recall
   metric, read honestly, collapses toward chance. It demonstrates that
   restricting candidates is fast — and that doing so on a partition decoupled
   from similarity destroys recall.

### 1.3 Does the Leech lattice rescue it? No.

Λ₂₄ is the natural "most beautiful" endpoint, but:

- Its optimality is about **uniform density in 24D**; embeddings are not uniform
  and 24D is far below where embeddings carry their information.
- It has **no roots**; the analogue of the finite vertex set is its **196 560
  minimal vectors** — i.e. 196 560 hand-authored anchors, which the current
  design (`projector.py` requires `len(anchors) == num_cells`) cannot supply.
  The path D₄ → E₈ → Λ₂₄ is not smooth; at Leech the architecture breaks.

**Conclusion:** a well-built, aesthetically motivated structured quantizer whose
central claim (geometry improves retrieval) is unproven. Kept here as the honest
point of departure.

---

## 2. The Pivot — Vector Symbolic Architectures & Sparse Distributed Memory

The bottleneck in Layer 1 lives *before* the geometry. The fix is not a prettier
lattice but a different lineage of associative memory:

- **VSA / Hyperdimensional Computing** (Kanerva; Plate's HRR; Gayler's MAP):
  represent everything as very high-dimensional vectors with three operations —
  **binding** (role↔filler, dissimilar & invertible), **bundling**
  (superposition, similar to all members), **permutation** (order/sequence).
- **Sparse Distributed Memory** (Kanerva, 1988): content-addressable,
  error-correcting associative memory; recall is decoding the nearest stored
  pattern from a noisy cue — i.e. an attractor.
- **The bridge that gives it teeth:** modern Hopfield networks *are* attention
  (Ramsauer et al., 2020), and SDM read is a special case of transformer
  attention (Bricken & Pehlevan, 2021). So "the hippocampus is SDM, and attention
  is its modern reincarnation" is a defensible thesis, not decoration.

The named risk going in: real embeddings are **not** quasi-orthogonal, so
grounding VSA atoms in them could poison the binding algebra. Testing exactly
this was the point of the validation arc.

---

## 3. Layer 2 — The VSA Cognitive Memory

### 3.1 Algebra (`vsa/core.py`, MAP model, bipolar ±1, D = 10 000)

```
bind(a, b)      = a ⊙ b            (elementwise; self-inverse)
unbind(x, r)    = x ⊙ r
bundle(V)       = sign(Σ V)        (majority superposition)
permute(x, k)   = roll(x, k)       (position in a sequence)
ground(e, P)    = sign(e · P)      (SimHash: embedding → bipolar hypervector)
```

`ground` (random-projection SimHash) maps a 384-D embedding to a D-dim bipolar
atom while **preserving angular similarity** (cos(atoms) ≈ 1 − 2θ/π), so
semantically close concepts get correlated atoms — both the goal and the risk.

### 3.2 Memory engine (`vsa/memory.py` — `VSAMemory`)

- **Facts (role-binding):** `fact = bundle(R_subj⊗a(s), R_rel⊗a(r), R_obj⊗a(o))`.
  Structural query: build a partial probe from the known roles, retrieve the
  nearest fact, `unbind` the target role, clean up against the concept codebook.
  This distinguishes `(X, r, Y)` from `(Y, r, X)` — cosine similarity cannot.
- **Episodes (order):** `episode = bundle(ρ⁰a(e₀), ρ¹a(e₁), …)`. Recover the item
  at a position by `unpermute`+cleanup; `successor(item)` answers "what came after".
- **Cleanup / attractor:** noisy/partial cues are resolved by nearest-pattern
  retrieval (the modern-Hopfield = attention form), which iteratively converges
  to the stored canon.
- **Entity normalization:** dirty surface variants whose embedding cosine ≥
  threshold to an existing concept collapse to one canonical atom.
- **Persistence:** `save`/`load` store the projection, embeddings and roles;
  atoms and fact/episode vectors are reconstructed deterministically.

### 3.3 Extraction (`extract/triples.py`)

Modelled on a production batch extractor: an LLM (DeepSeek → xAI → Groq fallback)
turns free text into **ordered (subject, relation, object) triples** (not flat
facts), with exact-triple dedup so role-swaps survive (`A→B` ≠ `B→A`).

### 3.4 Facade (`cognitive.py` — `OdinnMemory`)

```python
mem = OdinnMemory()
mem.remember("Maya founded Helix. Iris mentored Maya.")  # LLM → triples
mem.recall_object("Maya", "founded")        # → "Helix"
mem.recall_subject("mentored", "Maya")       # → "Iris"   (direction matters)
eid = mem.remember_conversation([...])        # ordered episode
mem.whats_next(eid, "reviewed the results")   # → "scheduled a follow-up call"
mem.save("~/.astrum_verum/odinn")             # survives sessions
```

---

## 4. Validation arc (reproducible under `experiments/vsa_sdm/`)

Each phase had **pass/fail thresholds fixed before running**, and two
self-inflicted measurement bugs were caught and corrected (documented because the
catch matters more than the green number).

| Phase | Question | Result |
|---|---|---|
| **0 — algebra** | Does binding/superposition/attractor work on clean atoms? | Binding holds **100+ pairs at D=10k** (graceful decay after ~250); Hopfield attractor recovers exactly up to **40 %** corruption, and iteration rescues **83 %** at 45 % where single-shot fails. |
| **1 — grounding** | Does the algebra survive *real* embeddings? | SimHash similarity correlation **0.988**; VSA role-query accuracy **1.000**; **grounding drop 0.000**. On role-ambiguous synthetic facts: VSA **1.000** vs cosine-RAG **≈0.50** (chance). |
| **2 — real extraction** | Does it beat cosine on data an LLM extracted? | On live-extracted triples with genuine role-swaps: VSA **1.000** vs cosine-RAG **0.600** (cosine is role-blind on identical concept bags). |
| **3 — full memory** | Do facts + episodes + normalization + persistence work together? | `pytest` 6/6; end-to-end demo PASS (recall, role direction, episode order & successor, variant merge, save→load). |

**Bugs caught (honest):** (a) a pass threshold checked a point absent from the
sweep, and a deterministic `argmax` tie-break inflated the cosine baseline —
replaced with expected `1/k` credit; (b) the naive Kanerva SDM landed in a
fragile regime (concentration of measure: random binary hard locations have no
locality), so the attractor was implemented as a modern Hopfield network — which
is also the attention bridge being claimed.

---

## 5. Limitations (what is *not* proven)

- Demonstrations use small, controlled corpora; **entity extraction and
  normalization on messy real dialogue** is the next real engineering risk.
- VSA is **not a better semantic retriever** than cosine/HNSW — it adds a
  capability (structural, role-sensitive, error-correcting recall), it does not
  replace nearest-neighbour search. A cleanup/ANN layer still sits underneath at
  scale.
- Bundling capacity is finite; long episodes need chunking.
- This is a **research library**, not yet wired into any production agent.

---

## 6. Future work

1. Robust extraction + entity resolution on real conversation logs.
2. Scale & capacity study (D, number of facts, episode length).
3. Learned/orthogonalized atoms to ease the grounding–binding tension.
4. Integration into a live agent memory loop (gated, deliberate).

---

## 7. References

- P. Kanerva, *Sparse Distributed Memory* (1988); *Hyperdimensional Computing* (2009).
- T. Plate, *Holographic Reduced Representations* (1995).
- R. Gayler, *Vector Symbolic Architectures* (MAP) (2003).
- H. Ramsauer et al., *Hopfield Networks is All You Need* (2020) — modern Hopfield = attention.
- T. Bricken, C. Pehlevan, *Attention Approximates Sparse Distributed Memory* (NeurIPS 2021).
- M. Viazovska, sphere packing in 8 dimensions (2016); Cohn–Kumar–Miller–Radchenko–Viazovska, 24 dimensions (2017).
- D. Kleyko et al., surveys of VSA/HDC (2022–2023); `torchhd` (Heddes et al., JMLR 2023).
