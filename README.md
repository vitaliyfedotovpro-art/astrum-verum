# Astrum Verum

**Composition-episodic cognitive memory for AI agents — and an honest record of how it got here.**

Astrum Verum started as an attempt to organise long-term memory on *perfect
lattices* (D₄ → E₈ → Leech). That geometry turned out to be an elegant but
**unproven** vector quantizer, so the project pivoted to **Vector Symbolic
Architectures + Sparse Distributed Memory** — a memory that stores *structure*
(who-did-what-to-whom, in what order) and recovers it from noisy cues.

Both layers ship here. The lattice core is kept as the documented point of
departure; the VSA layer is the part that is **validated end-to-end**.

> Full story, math and results: [`docs/astrum_verum_design.md`](docs/astrum_verum_design.md).

### Muninn — the memory engine

*Muninn* (Old Norse *“Memory”*, one of Odin's ravens) is the name of the VSA memory layer:

> **Muninn — Vector-Symbolic Associative Memory (VSAM)**, composition-episodic.
> Retrieval is *associative* (by meaning / a partial or noisy cue) **and**
> *structural* (by role — “who did what to whom”), not by an exact key.
> **Zero persona-prompt: pure memory, not a personality** — it returns what is
> stored, it does not generate or personalize.

- *associative* → unlike a key-value store (no exact key needed);
- *compositional* → unlike a plain vector DB: on role-swapped facts (“A loves B” vs “B loves A”) cosine sits at **0.5 (chance)**, Muninn at **1.0**.

Names: project **Astrum Verum** · memory **Muninn** · the agent that uses it **Óðinn**.

---

## Why this exists

Flat vector search (embed → cosine/HNSW) is excellent at *similarity* but blind
to *structure*: "Alice trusts Bob" and "Bob trusts Alice" have the same word bag,
so cosine cannot tell them apart. Astrum Verum's VSA layer binds roles to fillers,
so it can — and it recovers facts from corrupted/partial cues like an attractor.

**Headline result (reproducible):** on triples an LLM extracted from real text,
with genuine role ambiguity, the VSA layer scores **1.000** where a cosine-RAG
baseline scores **0.600** (chance on the ambiguous pairs).

---

## Install

```bash
pip install -e ".[dev]"        # core + tests
pip install -e ".[dev,api]"    # + FastAPI service for the lattice layer
```

Python ≥ 3.11. Extraction needs an LLM key (`DEEPSEEK_API_KEY`, or `XAI_/GROQ_`)
in the environment or a local `.env`.

---

## Quick start — the cognitive memory (Layer 2)

```python
from astrum_verum import OdinnMemory

mem = OdinnMemory()

# Remember facts from free text (LLM extracts structured triples)
mem.remember("Maya founded Helix. Iris mentored Maya. Maya mentors the juniors.")

mem.recall_object("Maya", "founded")     # → "Helix"
mem.recall_subject("mentored", "Maya")   # → "Iris"     (direction matters!)
mem.recall_object("Maya", "mentors")     # → "the juniors"

# Episodes: order is first-class
eid = mem.remember_conversation([
    "greeted the user", "reviewed the results", "scheduled a follow-up call",
])
mem.whats_next(eid, "reviewed the results")   # → "scheduled a follow-up call"

mem.save("~/.astrum_verum/odinn")              # persists across sessions
mem2 = OdinnMemory.load("~/.astrum_verum/odinn")
```

You can also add facts directly (no LLM) via `mem.remember_triple(s, r, o)`.

### What it does that cosine search cannot
- **Role-sensitive recall** — distinguishes `(X r Y)` from `(Y r X)`.
- **Error-correcting cleanup** — recovers the canonical fact from a noisy cue.
- **Episodic order** — "what happened, and in what sequence".
- **One-shot writes & persistence** — no reindexing; survives restarts.

---

## The lattice layer (Layer 1, kept for honesty)

```python
from astrum_verum import AstrumEngine
from astrum_verum.lattice import E8Plugin

engine = AstrumEngine(lattice=E8Plugin())   # D₄ by default
engine.add("Photosynthesis converts sunlight into chemical energy")
engine.search("plant biology")
```

This works and the geometry is correct, but see the design doc §1 for why its
retrieval-quality thesis is unproven (the bottleneck is the 384→d projection,
not the lattice). A REST API is available via `uvicorn astrum_verum.api:app`.

---

## Validation (run it yourself)

```bash
pytest tests/test_vsa_memory.py -q                     # VSA layer (no network)
PYTHONPATH=. python experiments/vsa_sdm/phase0_algebra.py    # algebra on clean atoms
PYTHONPATH=. python experiments/vsa_sdm/phase1_grounding.py  # grounding survives real embeddings
PYTHONPATH=. python experiments/vsa_sdm/phase2_pipeline.py   # vs cosine-RAG on extracted triples (needs LLM key)
PYTHONPATH=. python experiments/vsa_sdm/phase3_full.py       # full OdinnMemory end-to-end (needs LLM key)
```

| Phase | Claim tested | Result |
|---|---|---|
| 0 | binding capacity + attractor cleanup | 100+ pairs @ D=10k; exact recovery ≤40 % noise |
| 1 | grounding doesn't break binding | corr 0.988, grounding drop 0.000 |
| 2 | beats cosine on real extracted data | VSA 1.000 vs RAG 0.600 (role-ambiguous) |
| 3 | facts+episodes+normalize+persist | pytest 6/6, demo PASS |

---

## Layout

```
astrum_verum/
  vsa/          # VSA core (MAP) + VSAMemory  ← the validated layer
  extract/      # LLM triple extractor (DeepSeek→xAI→Groq)
  cognitive.py  # OdinnMemory facade
  lattice/      # D₄ / E₈ plugins (Layer 1)
  engine.py …   # lattice pipeline, store, scorer, rotation, API
experiments/vsa_sdm/   # the validation arc (phases 0–3)
docs/astrum_verum_design.md   # full design & honest research notes
```

---

## Status & limitations

Research library, not yet wired into a production agent. VSA **adds** structural
recall — it does not replace nearest-neighbour search. Extraction/normalization
on messy real dialogue is the next open problem. See design doc §5.

## License

MIT — see [LICENSE](LICENSE).
