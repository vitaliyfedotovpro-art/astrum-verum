# Contributing to Astrum Verum

First off, thank you for considering contributing to **Astrum Verum**! It is a pure, academically clean, high-performance geometric vector memory framework, and we welcome contributions that preserve its mathematical elegance and high-speed execution.

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

Astrum Verum has a comprehensive, **203-test bulletproof suite** divided into clear mathematical, algorithmic, and integration contours.

* **Run all tests (excluding slow/stress tests)**:
   ```bash
   pytest -v -m "not slow"
   ```

* **Run the full test suite (including the 50,000 nodes stress test)**:
   ```bash
   pytest -v
   ```

Any pull request must pass all 203 tests with 100% green status before being merged.

---

## 📊 Running Performance Benchmarks

Before pushing any computational modifications or optimizations to the spreading activation or lattice quantization code, please run the benchmark to verify there are no speed regressions:

```bash
python3 benchmarks/run_benchmark.py
```

This benchmark verifies query latency (ms), throughput (QPS), and candidate reduction ratios against standard flat cosine search baselines.

---

## 🎨 Design Principles

1. **Academic Cleanliness**: Keep modules highly focused, mathematical, and free of unnecessary third-party dependencies.
2. **Type Safety**: Strictly apply static type hints (`numpy.ndarray`, Python type annotations) across all functions.
3. **Isometry & Precision**: All coordinate operations on $D_4$ and $E_8$ lattices must strictly preserve Euclidean norms, angular relations, and distance metrics.
4. **Tested Stability**: Any new feature or plugin must be accompanied by corresponding unit and integration tests.
