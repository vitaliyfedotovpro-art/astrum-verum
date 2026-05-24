"""
Topology Store — Memory nodes organized by Voronoi cells on a lattice.

Three-level hierarchy:
  - Vertex:     Lattice root vector (conceptual anchor)
  - Cell:       Voronoi cell (semantic domain), holds references to memory nodes
  - MemoryNode: Concrete fact / fragment of knowledge

The topology graph (networkx) carries dynamic edge weights that evolve via
Hebbian learning during search operations.

Storage backends:
  - **In-memory** (< 50 000 nodes): dict + list, JSON serialization
  - **Persistent** (≥ 50 000 nodes): SQLite + numpy mmap, auto-migrated
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from .lattice.base import LatticePlugin

logger = logging.getLogger(__name__)

_AUTO_PERSIST_THRESHOLD = 50_000
_DEFAULT_STORAGE_DIR = Path.home() / ".astrum_verum" / "data"


@dataclass
class MemoryNode:
    """A single fact / memory stored in the lattice."""

    id: str
    text: str
    embedding: np.ndarray  # original high-dim (ℝᴺ)
    lattice_coords: np.ndarray  # projected (ℝᵈ)
    cell_memberships: dict[int, float]  # cell_id → weight

    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: dict = field(default_factory=dict)

    def touch(self) -> None:
        """Update access timestamp and counter."""
        self.last_accessed = time.time()
        self.access_count += 1


class TopologyStore:
    """
    Topology-aware storage for memory nodes.

    Starts in-memory.  When node count reaches ``auto_persist_threshold``
    (default 50 000), automatically migrates to SQLite + numpy mmap.

    The ``storage_dir`` parameter can force persistent mode from the start.
    """

    def __init__(
        self,
        lattice: LatticePlugin,
        storage_dir: str | Path | None = None,
        auto_persist_threshold: int = _AUTO_PERSIST_THRESHOLD,
    ) -> None:
        self._lattice = lattice
        self._graph = self._build_graph()
        self._auto_persist_threshold = auto_persist_threshold

        # Resolve storage dir.
        if storage_dir is not None:
            self._storage_dir = Path(storage_dir)
        else:
            self._storage_dir = _DEFAULT_STORAGE_DIR

        # Persistent backend (lazy).
        self._backend = None  # SqliteMmapBackend | None
        self._persistent = False

        # In-memory storage.
        self._nodes: dict[str, MemoryNode] = {}
        self._cells: dict[int, list[str]] = {
            i: [] for i in range(lattice.num_cells())
        }

        # If storage_dir was explicitly given, start persistent immediately.
        if storage_dir is not None:
            self._activate_persistent()

    # ------------------------------------------------------------------
    # Graph initialization
    # ------------------------------------------------------------------
    def _build_graph(self) -> nx.Graph:
        """Initialize networkx graph from lattice adjacency."""
        G = nx.Graph()
        info = self._lattice.info()
        adj = self._lattice.adjacency()

        for i in range(info.num_vertices):
            G.add_node(i)

        for i, neighbors in adj.items():
            for j in neighbors:
                if i < j:
                    G.add_edge(i, j, weight=1.0)

        return G

    @property
    def graph(self) -> nx.Graph:
        """The underlying networkx topology graph."""
        return self._graph

    @property
    def is_persistent(self) -> bool:
        """Whether the store is using persistent storage."""
        return self._persistent

    # ------------------------------------------------------------------
    # Persistent backend management
    # ------------------------------------------------------------------
    def _activate_persistent(self) -> None:
        """Initialize the persistent backend."""
        from .persistent import SqliteMmapBackend

        self._backend = SqliteMmapBackend(self._storage_dir)
        self._persistent = True

        # Load edge weights if DB already has them.
        self._backend.load_edge_weights(self._graph)

        logger.info(
            "Persistent backend activated at %s", self._storage_dir
        )

    def _migrate_to_persistent(self) -> None:
        """
        Migrate all in-memory nodes to SQLite + mmap.

        Called automatically when node count reaches the threshold.
        """
        from .persistent import SqliteMmapBackend

        nodes = list(self._nodes.values())
        count = len(nodes)

        logger.info(
            "Auto-migrating %d nodes to persistent storage at %s",
            count,
            self._storage_dir,
        )

        self._backend = SqliteMmapBackend(self._storage_dir)
        self._backend.add_nodes_batch(nodes)
        self._backend.save_edge_weights(self._graph)
        self._persistent = True

        # Clear in-memory storage.
        self._nodes.clear()
        for cell_nodes in self._cells.values():
            cell_nodes.clear()

        logger.info("Migration complete: %d nodes persisted.", count)

    def _check_auto_persist(self) -> None:
        """Trigger migration if threshold is reached."""
        if self._persistent:
            return
        if len(self._nodes) >= self._auto_persist_threshold:
            self._migrate_to_persistent()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add_node(self, node: MemoryNode) -> str:
        """Store a memory node, assigning it to its primary cell."""
        if self._persistent:
            self._backend.add_node(node)
        else:
            self._nodes[node.id] = node
            primary_cell = max(
                node.cell_memberships, key=node.cell_memberships.get
            )
            if node.id not in self._cells[primary_cell]:
                self._cells[primary_cell].append(node.id)
            self._check_auto_persist()

        return node.id

    def get_node(self, node_id: str) -> MemoryNode | None:
        """Retrieve a node by ID, or None if not found."""
        if self._persistent:
            return self._backend.get_node(node_id)
        return self._nodes.get(node_id)

    def get_nodes_in_cell(self, cell_id: int) -> list[MemoryNode]:
        """Get all nodes whose primary cell is ``cell_id``."""
        if self._persistent:
            return self._backend.get_nodes_in_cell(cell_id)
        return [
            self._nodes[nid]
            for nid in self._cells.get(cell_id, [])
            if nid in self._nodes
        ]

    def get_all_nodes(self) -> list[MemoryNode]:
        """Return all stored memory nodes."""
        if self._persistent:
            return self._backend.get_all_nodes()
        return list(self._nodes.values())

    def remove_node(self, node_id: str) -> bool:
        """Remove a node.  Returns True if found and removed."""
        if self._persistent:
            return self._backend.remove_node(node_id)
        node = self._nodes.pop(node_id, None)
        if node is None:
            return False
        for cell_nodes in self._cells.values():
            if node_id in cell_nodes:
                cell_nodes.remove(node_id)
        return True

    def touch_node(self, node_id: str) -> None:
        """Update access timestamp and counter for a node."""
        if self._persistent:
            self._backend.touch_node(node_id)
        else:
            node = self._nodes.get(node_id)
            if node is not None:
                node.touch()

    # ------------------------------------------------------------------
    # Edge weights
    # ------------------------------------------------------------------
    def get_edge_weight(self, i: int, j: int) -> float:
        """Get the weight of edge (i, j).  Returns 0 if no edge."""
        if self._graph.has_edge(i, j):
            return self._graph[i][j]["weight"]
        return 0.0

    def update_edge_weight(self, i: int, j: int, delta: float) -> None:
        """Add ``delta`` to the weight of edge (i, j)."""
        if self._graph.has_edge(i, j):
            self._graph[i][j]["weight"] += delta

    def set_edge_weight(self, i: int, j: int, weight: float) -> None:
        """Set the weight of edge (i, j) to an absolute value."""
        if self._graph.has_edge(i, j):
            self._graph[i][j]["weight"] = weight

    def scale_edge_weight(self, i: int, j: int, factor: float) -> None:
        """Multiply the weight of edge (i, j) by ``factor``."""
        if self._graph.has_edge(i, j):
            self._graph[i][j]["weight"] *= factor

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return summary statistics about the store."""
        weights = [d["weight"] for _, _, d in self._graph.edges(data=True)]

        if self._persistent:
            total_nodes = self._backend.count()
            raw_counts = self._backend.cell_counts()
            num_cells = self._lattice.info().num_vertices
            cell_counts = {i: raw_counts.get(i, 0) for i in range(num_cells)}
        else:
            total_nodes = len(self._nodes)
            cell_counts = {
                cid: len(nids) for cid, nids in self._cells.items()
            }

        return {
            "total_nodes": total_nodes,
            "total_cells": len(cell_counts),
            "occupied_cells": sum(1 for c in cell_counts.values() if c > 0),
            "cell_counts": cell_counts,
            "persistent": self._persistent,
            "storage_dir": str(self._storage_dir) if self._persistent else None,
            "edge_weight_min": min(weights) if weights else 0,
            "edge_weight_max": max(weights) if weights else 0,
            "edge_weight_mean": float(np.mean(weights)) if weights else 0,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Persist the store.

        - **In-memory mode**: serializes everything to a single JSON file.
        - **Persistent mode**: flushes SQLite + mmap and saves edge weights.
          The ``path`` argument is ignored (data lives in ``storage_dir``).
        """
        if self._persistent:
            self._backend.save_edge_weights(self._graph)
            self._backend.flush()
            return

        # In-memory JSON serialization.
        data: dict = {
            "lattice": self._lattice.info().name,
            "nodes": {},
            "edge_weights": {},
        }
        for nid, node in self._nodes.items():
            data["nodes"][nid] = {
                "text": node.text,
                "embedding": node.embedding.tolist(),
                "lattice_coords": node.lattice_coords.tolist(),
                "cell_memberships": {
                    str(k): v for k, v in node.cell_memberships.items()
                },
                "created_at": node.created_at,
                "last_accessed": node.last_accessed,
                "access_count": node.access_count,
                "metadata": node.metadata,
            }
        for u, v, d in self._graph.edges(data=True):
            data["edge_weights"][f"{u}-{v}"] = d["weight"]

        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str) -> None:
        """
        Load state.

        - **In-memory mode**: loads from JSON file.
        - **Persistent mode**: reloads edge weights from SQLite.
          The ``path`` argument is ignored.
        """
        if self._persistent:
            self._backend.load_edge_weights(self._graph)
            return

        data = json.loads(Path(path).read_text())

        self._nodes.clear()
        for cell_nodes in self._cells.values():
            cell_nodes.clear()

        for nid, ndata in data["nodes"].items():
            node = MemoryNode(
                id=nid,
                text=ndata["text"],
                embedding=np.array(ndata["embedding"]),
                lattice_coords=np.array(ndata["lattice_coords"]),
                cell_memberships={
                    int(k): v for k, v in ndata["cell_memberships"].items()
                },
                created_at=ndata["created_at"],
                last_accessed=ndata["last_accessed"],
                access_count=ndata["access_count"],
                metadata=ndata.get("metadata", {}),
            )
            self.add_node(node)

        for key, weight in data.get("edge_weights", {}).items():
            u, v = map(int, key.split("-"))
            self.set_edge_weight(u, v, weight)
