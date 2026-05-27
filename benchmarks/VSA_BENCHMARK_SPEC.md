# Specification: Statistically Valid VSAMemory Benchmark

## Why
The current project figures (facts recall 1.000@8000, SimHash grounding corr ~0.99,
episode order-recall 0.995@200 → 0.25@1000) were obtained from **single runs**.
They cannot be cited as measured values. We need a benchmark that yields each
point as **mean ± std over replication across multiple seeds**, because the atoms
(`random_atoms`) and the SimHash projection are stochastic.

## Iron-clad Honesty Rules (VIOLATION = task failure)
1. **No cherry-picking.** Report ALL runs, including the worst seeds. For each
   point: mean, std, min, max, n_seeds.
2. **Keep raw per-seed** in CSV — so numbers can be independently verified.
3. **Only real embeddings** (see below). No fabricated numbers.
4. If something fails / hits resource limits — **honestly record in the report**
   what and why, and lower the parameter, rather than fudging the result.
5. Do not touch the working memory code. Only the new benchmark file + result files.

## API (read `astrum_verum/vsa/memory.py` and `astrum_verum/vsa/core.py`)
- `VSAMemory(D=10000, seed=int, normalize_threshold=0.82, embed_fn=callable|None)`
- `.add_triple(subject, relation, obj) -> idx`
- `.query({"subject":s, "relation":r}, "object") -> {"answer","score","triple","fact_idx"}`
  (similarly for recall by subject: `query({"relation":r,"object":o}, "subject")`)
- `.add_episode(list[str], episode_id) -> eid`
- `.recall_at(eid, pos) -> str` ; `.episode_order(eid) -> list[str]`
- `core.ground(emb, proj)`, `core.make_projection(emb_dim, D, rng)` — for grounding test.

## Style and Methodology References (read both)
- `experiments/vsa_sdm/phase0_algebra.py` — template: trials-loop, pre-fixed
  thresholds, `.md` report. Copy the discipline, but this is synthetic — we need real API.
- `experiments/vsa_sdm/phase1_grounding.py` — how SimHash grounding corr was measured (reproduce).

## Embedder / Data
- VSAMemory defaults to loading sentence-transformers
  `paraphrase-multilingual-MiniLM-L12-v2`. **Use a real embedder.**
- To make the seed-sweep fast: precompute embeddings of all unique texts
  ONCE, wrap in an `embed_fn` with a cache (`dict[text] -> np.ndarray`), pass as
  `VSAMemory(embed_fn=cached_fn, seed=…)`. Then when the seed changes, only
  roles/projection are recreated (cheap), while heavy embedding is computed once.
- **CORPUS:** deterministically generate (separate fixed data-seed, DO NOT confuse
  with VSA-seed) a pool of unique triples from real words (names, cities, professions,
  animals, diseases, materials, etc. — combine real lexemes, not `node-0001`).
  We need enough unique (subject, relation, object) for max N. Document
  exactly how the pool was generated.
- If a real embedder is unavailable or the machine can't handle 16000 facts — lower the upper
  bound N (e.g. to 8000) and **honestly state this in the report**.

## Experiments

### E1 — Facts structural recall vs N (capacity scaling)
- `N ∈ {1000, 2000, 4000, 8000, 16000}`, `seeds = 30` different ones.
- For each (N, seed): new `VSAMemory(seed=seed, embed_fn=cached)`, add N
  distinct triples; then on a random subsample (≥200 facts or all if fewer)
  do a `query` (known subject+relation → recover object); accuracy =
  fraction of `answer == true object`.
- Point (N): mean ± std ± min accuracy across 30 seeds.

### E2 — Episode order-recall vs length (saturation + window test)
- `length L ∈ {50, 100, 200, 500, 1000}`, `seeds = 30`.
- `add_episode(L items)`, `recall_at` for all positions; positional recall accuracy
  = fraction of `recalled == true item at this position`. mean ± std across seeds.
- Additionally confirm: bounded window (only last W=150 items as
  episode) gives recall ≈ 1.0 regardless of the full dialogue length. This validates
  the architectural decision "working memory = window".

### E3 — SimHash grounding fidelity (reproduce corr ~0.99)
- Take M real texts (≥500). Compute pairwise cos in embedding-space and cos
  of corresponding `ground()`-atoms in hypervector-space. Pearson + Spearman correlation
  between the two sets of similarities. seeds vary projection (30). mean ± std corr.

## Output
- New file: `benchmarks/vsa_memory_benchmark.py` with argparse:
  `--seeds N` (default 30), `--max-n N`, `--quick` (fast run for testing).
- `benchmarks/vsa_memory_results.md` — mean±std±min tables for E1/E2/E3 + run header
  (date, embedder version, n_seeds, D, CORPUS size, hardware/time).
- `benchmarks/vsa_memory_raw.csv` — one row per (experiment, parameter, seed, metric).
- **Run for real** (first `--quick` for self-check, then full run).
- Return summary: mean±std tables for E1/E2/E3 and file paths.

## Determinism
Fix and log all seeds. Data-seed (CORPUS) separated from VSA-seed (algebra).
