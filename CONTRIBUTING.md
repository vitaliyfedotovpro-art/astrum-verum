# Contributing to Astrum Verum

Thanks for considering a contribution to **Astrum Verum**. It has two layers: a
geometric lattice memory (Layer 1, kept as a documented mockup) and a VSA/SDM
composition-episodic memory (Layer 2, the validated part). Please read
[`docs/astrum_verum_design.md`](docs/astrum_verum_design.md) first — it is candid
about what is proven and what is not, and contributions should keep that honesty.

---

## 🛠️ Development Setup

To set up a local development environment:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/astrum-verum.git
   cd astrum-verum
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the package in editable mode along with development and API extra dependencies:
   ```bash
   pip install -e ".[dev,api]"
   ```

---

## 🧪 Running the Test Suite

Astrum Verum has a test suite covering both layers: the lattice geometry/algorithms (`tests/test_*` for lattice, store, rotation, scorer, …) and the VSA cognitive memory (`tests/test_vsa_memory.py` — binding, role-sensitivity, episodes, normalization, persistence).

* **Run all tests (excluding slow/stress tests)**:
   ```bash
   pytest -v -m "not slow"
   ```

* **Run the full test suite (including the 50,000 nodes stress test)**:
   ```bash
   pytest -v
   ```

Any pull request must keep the full suite green before being merged.

---

## 📊 Running Performance Benchmarks

Before pushing any computational modifications or optimizations to the spreading activation or lattice quantization code, please run the benchmark to verify there are no speed regressions:

```bash
python3 benchmarks/run_benchmark.py
```

This benchmark verifies query latency (ms), throughput (QPS), and candidate reduction ratios against a flat cosine baseline. **Caveat:** it measures *speed*, not recall — its synthetic setup decouples embeddings from lattice placement (see design doc §1.2). Treat its numbers as throughput, not retrieval quality.

---

## 🎨 Design Principles

1. **Academic Cleanliness**: Keep modules highly focused, mathematical, and free of unnecessary third-party dependencies.
2. **Type Safety**: Strictly apply static type hints (`numpy.ndarray`, Python type annotations) across all functions.
3. **Isometry & Precision**: All coordinate operations on $D_4$ and $E_8$ lattices must strictly preserve Euclidean norms, angular relations, and distance metrics.
4. **Tested Stability**: Any new feature or plugin must be accompanied by corresponding unit and integration tests.
