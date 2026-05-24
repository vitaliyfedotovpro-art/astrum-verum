# Astrum Verum

**Geometric Cognitive Memory Architecture on Perfect Lattices**

> *Искусственный гиппокамп на кристаллических решётках.*

Scalable lattice-structured associative memory for AI agents, using the hierarchy
of perfect root lattices (D₄ → E₈ → Λ₂₄) as a geometric scaffold for knowledge
organization, with neurocognitive mechanisms for navigation, association, and adaptation.

## Features

- **D₄ (24-cell)** — 4D lattice with 24 vertices, 96 edges, 8 neighbors per cell
- **E₈** — 8D exceptional lattice with 240 vertices, 6720 edges, 56 neighbors per cell
- **SO(d) Rotation Engine** — attention focus via inverse rotation (O(d²), not O(N·d²))
- **Spreading Activation** — neurocognitive wave propagation with Hebbian learning
- **Hybrid Scoring** — cosine similarity × topological boost × temporal recency
- **REST API** — FastAPI service with full CRUD and search endpoints

## Install

```bash
pip install -e ".[dev]"

# For the REST API:
pip install -e ".[dev,api]"
```

## Quick Start

```python
from astrum_verum import AstrumEngine

# Create engine (D₄ lattice by default)
engine = AstrumEngine()

# Add memories
engine.add("Photosynthesis converts sunlight into chemical energy in plants")
engine.add("Neural networks are composed of layers of artificial neurons")
engine.add("Chloroplasts contain chlorophyll for photosynthesis")

# Search — topological + semantic + temporal scoring
results = engine.search("plant biology and energy conversion")
for r in results:
    print(f"  [{r.score:.3f}] {r.text}")
```

## Using E₈ (240 semantic domains)

```python
from astrum_verum import AstrumEngine
from astrum_verum.lattice import E8Plugin

engine = AstrumEngine(lattice=E8Plugin())
```

## REST API

```bash
uvicorn astrum_verum.api:app --reload
```

| Endpoint | Method | Description |
|---|---|---|
| `/memory/add` | POST | Add a memory |
| `/memory/search` | POST | Topological search (AstrumSearch) |
| `/memory/state` | GET | Current engine state |
| `/memory/cells` | GET | List cells with node counts |
| `/memory/cell/{id}/nodes` | GET | Nodes in a specific cell |
| `/lattice/info` | GET | Lattice metadata |

## Architecture

See [astrum_verum_design.md](./docs/astrum_verum_design.md) for the full design document.

## Tests

```bash
# Core tests (no ML model needed)
pytest tests/test_lattice_d4.py tests/test_lattice_e8.py tests/test_rotation.py \
       tests/test_store.py tests/test_activation.py tests/test_scorer.py -v

# Full integration tests (downloads sentence-transformers model on first run)
pytest tests/ -v
```
