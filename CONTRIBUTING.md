# Contributing to Astrum Verum

Astrum Verum has two layers: a geometric lattice memory (Layer 1, kept as a
documented mockup) and a VSA/SDM composition-episodic memory (Layer 2, the
validated part). See [`docs/astrum_verum_design.md`](docs/astrum_verum_design.md) —
it is candid about what is proven and what is not.

---

## ⚠️ Pull Requests are not accepted at this time

This repo is public for **review, learning, demonstration, and discussion** — but
it is **not** currently open to external code contributions.

**Please do NOT open Pull Requests** — they will be closed unmerged.

You are very welcome to:
- ⭐ **Open Issues** — bugs, questions, observations
- 💬 **Start Discussions** — ideas, design feedback, use cases

**Why:** the codebase is kept 100% authored by the maintainer so it can be freely
reused in other projects (including closed-source/commercial ones) without
contributor-license complications. Taking outside patches without a CLA would
forfeit that, so for now the project simply doesn't accept them.

---

## 🛠️ Development Setup (for running/reviewing locally)

1. **Clone**:
   ```bash
   git clone https://github.com/vitaliyfedotovpro-art/astrum-verum.git
   cd astrum-verum
   ```
2. **Virtualenv**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. **Install** (editable + dev/api extras):
   ```bash
   pip install -e ".[dev,api]"
   ```

## 🧪 Running the Test Suite

Covers both layers: lattice geometry/algorithms and the VSA cognitive memory
(`tests/test_vsa_memory.py` — binding, role-sensitivity, episodes, normalization, persistence).

```bash
pytest -v -m "not slow"   # fast suite
pytest -v                 # full suite (incl. stress test)
```

## 📊 Performance Benchmarks

```bash
python3 benchmarks/run_benchmark.py
```
Measures query latency (ms), throughput (QPS), candidate-reduction vs a flat cosine
baseline. **Caveat:** it measures *speed*, not recall — synthetic setup decouples
embeddings from lattice placement (design doc §1.2). Treat numbers as throughput.

## 🎨 Design Principles

1. **Academic cleanliness** — focused, mathematical modules, minimal third-party deps.
2. **Type safety** — strict type hints across functions.
3. **Isometry & precision** — coordinate ops on $D_4$/$E_8$ preserve norms, angles, distances.
4. **Tested stability** — features come with tests.

---

## License

Astrum Verum is licensed under **AGPL-3.0** (see `LICENSE`). Copyright © 2026 Vitaliy Fedotov.
Copyleft/network terms apply to any use or distribution; the copyright holder retains
the right to license the code separately.
