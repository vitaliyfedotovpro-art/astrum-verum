"""
AstrumEngine — Full pipeline orchestrator for lattice memory.

Ties together all modules into a single coherent API:
    SemanticProjector → RotationEngine → CVP → SpreadingActivation → HybridScorer

Usage::

    engine = AstrumEngine()                        # D₄ by default
    engine.add("fact about the world")
    results = engine.search("related query")       # → list[SearchResult]

    engine = AstrumEngine(lattice=E8Plugin())      # upgrade to E₈
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from .activation import SpreadingActivation
from .defaults import D4_ANCHORS, E8_ANCHORS
from .lattice.base import LatticePlugin
from .lattice.d4 import D4Plugin
from .projector import SemanticProjector
from .rotation import align_to_axis, compute_focus_vector, inverse_rotate_query
from .scorer import HybridScorer
from .store import MemoryNode, TopologyStore


@dataclass
class SearchResult:
    """A single search result returned by ``AstrumEngine.search()``."""

    node_id: str
    text: str
    score: float
    cell_id: int
    breakdown: dict
    metadata: dict = field(default_factory=dict)


class AstrumEngine:
    """
    Main entry point for Astrum Verum lattice memory.

    Orchestrates the full AstrumSearch pipeline:

    1. Embed & project query to lattice coordinates
    2. Compute focus vector from recent context → SO(d) rotation
    3. Inverse-rotate query (O(d²) instead of O(N·d²))
    4. CVP decoding → start cell
    5. Spreading activation → cell energies
    6. Gather candidate memory nodes from activated cells
    7. Hybrid scoring (cosine × topo_boost × recency)
    8. Hebbian edge weight update
    9. Return top-K results
    """

    def __init__(
        self,
        lattice: LatticePlugin | None = None,
        anchor_labels: list[str] | None = None,
        embedder_model: str = "all-MiniLM-L6-v2",
        spreading_decay: float = 0.6,
        spreading_radius: int = 3,
        scorer_alpha: float = 0.5,
        scorer_beta: float = 0.35,
        scorer_gamma: float = 0.15,
        context_window: int = 5,
        focus_decay: float = 0.8,
        storage_dir: str | None = None,
    ) -> None:
        # Defaults.
        if lattice is None:
            lattice = D4Plugin()
        if anchor_labels is None:
            name = lattice.info().name
            if name == "D4":
                anchor_labels = D4_ANCHORS
            elif name == "E8":
                anchor_labels = E8_ANCHORS
            else:
                raise ValueError(
                    f"No default anchors for lattice {name!r}. "
                    f"Please provide anchor_labels explicitly."
                )

        self._lattice = lattice
        self._projector = SemanticProjector(lattice, anchor_labels, embedder_model)
        self._store = TopologyStore(lattice, storage_dir=storage_dir)
        self._activation = SpreadingActivation(
            self._store,
            decay=spreading_decay,
            radius=spreading_radius,
        )
        self._scorer = HybridScorer(
            alpha=scorer_alpha,
            beta=scorer_beta,
            gamma=scorer_gamma,
        )
        self._context_history: list[np.ndarray] = []
        self._context_window = context_window
        self._focus_decay = focus_decay

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def lattice(self) -> LatticePlugin:
        """The underlying lattice plugin."""
        return self._lattice

    @property
    def store(self) -> TopologyStore:
        """The topology store (for direct inspection / stats)."""
        return self._store

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------
    def add(self, text: str, metadata: dict | None = None) -> str:
        """
        Add a memory to the lattice.

        Returns:
            The generated node ID (UUID string).
        """
        # 1. Embed.
        embedding = self._projector.embed(text)

        # 2. Project to lattice coords.
        lattice_coords = self._projector.project_vector(embedding)

        # 3. Soft cell membership.
        memberships = self._projector.soft_membership_vector(embedding)

        # 4. Create node.
        node = MemoryNode(
            id=str(uuid.uuid4()),
            text=text,
            embedding=embedding,
            lattice_coords=lattice_coords,
            cell_memberships=memberships,
            metadata=metadata or {},
        )

        # 5. Store.
        self._store.add_node(node)

        return node.id

    # ------------------------------------------------------------------
    # Read path — AstrumSearch
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Full AstrumSearch pipeline.

        Returns:
            Up to ``top_k`` search results, sorted by hybrid score descending.
        """
        now = time.time()

        # 1. Embed & project query.
        query_embedding = self._projector.embed(query)
        query_lattice = self._projector.project_vector(query_embedding)

        # 2. Compute focus from context history.
        self._context_history.append(query_lattice)
        if len(self._context_history) > self._context_window:
            self._context_history = self._context_history[-self._context_window :]

        if len(self._context_history) >= 2:
            focus = compute_focus_vector(
                self._context_history, decay=self._focus_decay
            )
            rotation = align_to_axis(focus)
            search_point = inverse_rotate_query(query_lattice, rotation)
        else:
            search_point = query_lattice

        # 3. CVP → start cell.
        start_cell = self._lattice.decode_cell(search_point)

        # 4. Spreading activation.
        cell_energies = self._activation.activate(start_cell)

        # 5. Gather candidates from activated cells.
        candidates: list[tuple[MemoryNode, float]] = []
        for cell_id, energy in cell_energies.items():
            for node in self._store.get_nodes_in_cell(cell_id):
                self._store.touch_node(node.id)
                candidates.append((node, energy))

        if not candidates:
            return []

        # 6. Score & rank.
        scored = self._scorer.rank(query_embedding, candidates, top_k, now)

        # 7. Hebbian update.
        self._activation.hebbian_update(cell_energies)

        # 8. Build results.
        results: list[SearchResult] = []
        for sr in scored:
            primary_cell = max(
                sr.node.cell_memberships,
                key=sr.node.cell_memberships.get,
            )
            results.append(
                SearchResult(
                    node_id=sr.node.id,
                    text=sr.node.text,
                    score=sr.score,
                    cell_id=primary_cell,
                    breakdown=sr.breakdown,
                    metadata=sr.node.metadata,
                )
            )

        return results

    # ------------------------------------------------------------------
    # State & persistence
    # ------------------------------------------------------------------
    def state(self) -> dict:
        """Return a summary of the current engine state."""
        stats = self._store.stats()
        return {
            "lattice": self._lattice.info().name,
            "dimension": self._lattice.info().dimension,
            "context_depth": len(self._context_history),
            **stats,
        }

    def save(self, path: str) -> None:
        """Persist engine state (memory nodes + edge weights) to disk."""
        self._store.save(path)

    def load(self, path: str) -> None:
        """Load engine state from disk."""
        self._store.load(path)
