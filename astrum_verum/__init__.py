"""
Astrum Verum — Geometric Cognitive Memory Architecture on Perfect Lattices.

An artificial hippocampus built on crystallographic lattices (D₄ → E₈ → Λ₂₄).
"""

__version__ = "0.1.0"

from .cognitive import OdinnMemory
from .lattice import D4Plugin, E8Plugin, LatticeInfo, LatticePlugin
from .store import MemoryNode, TopologyStore
from .vsa import VSAMemory

__all__ = [
    "AstrumEngine",
    "D4Plugin",
    "E8Plugin",
    "LatticeInfo",
    "LatticePlugin",
    "MemoryNode",
    "OdinnMemory",
    "TopologyStore",
    "VSAMemory",
]


def __getattr__(name: str):
    """Lazy import for AstrumEngine to defer heavy sentence-transformers load."""
    if name == "AstrumEngine":
        from .engine import AstrumEngine

        return AstrumEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
