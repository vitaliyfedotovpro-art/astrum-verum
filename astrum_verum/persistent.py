"""
SQLite + numpy mmap persistent backend for TopologyStore.

Activated automatically when node count exceeds 50 000 (configurable).

Architecture:
    - **SQLite (WAL mode)**: text, metadata, timestamps, cell memberships
    - **numpy mmap**: embeddings (ℝᴺ) and lattice coordinates (ℝᵈ)

Mmap files are lazily paged by the OS kernel — only accessed pages occupy
physical RAM.  This means a 1M-node store with 384D embeddings uses ~3 GB
on disk but only a few MB in RAM if you're only searching a few cells.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from .store import MemoryNode

logger = logging.getLogger(__name__)

_GROWTH_FACTOR = 4  # quadruple mmap capacity on resize
_DEFAULT_INITIAL_CAPACITY = 100_000


class SqliteMmapBackend:
    """Persistent node storage: SQLite metadata + numpy mmap vectors."""

    def __init__(
        self,
        storage_dir: str | Path,
        embedding_dim: int | None = None,
        lattice_dim: int | None = None,
        initial_capacity: int = _DEFAULT_INITIAL_CAPACITY,
    ) -> None:
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        # SQLite -------------------------------------------------------
        self._db = sqlite3.connect(
            str(self._dir / "nodes.db"),
            check_same_thread=False,
        )
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._init_tables()

        # Resolve dimensions -------------------------------------------
        stored = self._load_meta()
        if embedding_dim is None:
            embedding_dim = stored.get("embedding_dim")
        if lattice_dim is None:
            lattice_dim = stored.get("lattice_dim")

        if embedding_dim is None or lattice_dim is None:
            # Will be set on first add_node().
            self._embedding_dim: int | None = None
            self._lattice_dim: int | None = None
            self._capacity = initial_capacity
            self._embeddings: np.memmap | None = None
            self._lattice_coords: np.memmap | None = None
        else:
            self._embedding_dim = embedding_dim
            self._lattice_dim = lattice_dim
            self._capacity = max(initial_capacity, self._count_rows() + 1)
            self._save_meta()
            self._embeddings = self._open_mmap("embeddings.dat", embedding_dim)
            self._lattice_coords = self._open_mmap("lattice_coords.dat", lattice_dim)

        self._next_row: int = self._count_rows()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_tables(self) -> None:
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS nodes (
                id             TEXT PRIMARY KEY,
                text           TEXT NOT NULL,
                row_idx        INTEGER UNIQUE NOT NULL,
                created_at     REAL NOT NULL,
                last_accessed  REAL NOT NULL,
                access_count   INTEGER DEFAULT 0,
                metadata_json  TEXT DEFAULT '{}',
                primary_cell   INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cell_memberships (
                node_id  TEXT    NOT NULL,
                cell_id  INTEGER NOT NULL,
                weight   REAL    NOT NULL,
                PRIMARY KEY (node_id, cell_id),
                FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS edge_weights (
                u      INTEGER NOT NULL,
                v      INTEGER NOT NULL,
                weight REAL    NOT NULL,
                PRIMARY KEY (u, v)
            );
            CREATE INDEX IF NOT EXISTS idx_primary_cell
                ON nodes(primary_cell);
        """)
        self._db.commit()

    # ------------------------------------------------------------------
    # Meta helpers
    # ------------------------------------------------------------------
    def _load_meta(self) -> dict:
        rows = self._db.execute("SELECT key, value FROM meta").fetchall()
        meta = {}
        for k, v in rows:
            try:
                meta[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                meta[k] = v
        return meta

    def _save_meta(self) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("embedding_dim", json.dumps(self._embedding_dim)),
        )
        self._db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("lattice_dim", json.dumps(self._lattice_dim)),
        )
        self._db.commit()

    def _count_rows(self) -> int:
        r = self._db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        return r

    # ------------------------------------------------------------------
    # Mmap helpers
    # ------------------------------------------------------------------
    def _open_mmap(self, filename: str, dim: int) -> np.memmap:
        path = self._dir / filename
        if path.exists() and path.stat().st_size > 0:
            return np.memmap(
                str(path), dtype=np.float64, mode="r+",
                shape=(self._capacity, dim),
            )
        mm = np.memmap(
            str(path), dtype=np.float64, mode="w+",
            shape=(self._capacity, dim),
        )
        return mm

    def _ensure_capacity(self) -> None:
        if self._next_row < self._capacity:
            return
        new_cap = self._capacity * _GROWTH_FACTOR
        logger.info(
            "Growing mmap capacity %d → %d", self._capacity, new_cap
        )
        self._embeddings = self._grow_mmap(
            "embeddings.dat", self._embedding_dim, new_cap
        )
        self._lattice_coords = self._grow_mmap(
            "lattice_coords.dat", self._lattice_dim, new_cap
        )
        self._capacity = new_cap

    def _grow_mmap(
        self, filename: str, dim: int, new_cap: int
    ) -> np.memmap:
        path = self._dir / filename
        # Flush and release old mmap.
        old = getattr(self, "_embeddings" if "emb" in filename else "_lattice_coords")
        if old is not None:
            old.flush()
            del old

        # Grow the file (zero-padded).
        new_size = new_cap * dim * np.dtype(np.float64).itemsize
        os.truncate(str(path), new_size)

        return np.memmap(
            str(path), dtype=np.float64, mode="r+", shape=(new_cap, dim)
        )

    def _ensure_mmap_initialized(self, node: MemoryNode) -> None:
        """Lazily initialize mmap files on first node."""
        if self._embeddings is not None:
            return
        self._embedding_dim = len(node.embedding)
        self._lattice_dim = len(node.lattice_coords)
        self._save_meta()
        self._embeddings = self._open_mmap("embeddings.dat", self._embedding_dim)
        self._lattice_coords = self._open_mmap(
            "lattice_coords.dat", self._lattice_dim
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add_node(self, node: MemoryNode) -> str:
        self._ensure_mmap_initialized(node)
        self._ensure_capacity()

        row = self._next_row
        self._next_row += 1

        primary_cell = max(node.cell_memberships, key=node.cell_memberships.get)

        self._db.execute(
            "INSERT INTO nodes "
            "(id, text, row_idx, created_at, last_accessed, "
            " access_count, metadata_json, primary_cell) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                node.id, node.text, row,
                node.created_at, node.last_accessed,
                node.access_count, json.dumps(node.metadata),
                primary_cell,
            ),
        )
        for cell_id, weight in node.cell_memberships.items():
            self._db.execute(
                "INSERT INTO cell_memberships (node_id, cell_id, weight) "
                "VALUES (?, ?, ?)",
                (node.id, int(cell_id), weight),
            )
        self._db.commit()

        # Write vectors to mmap.
        self._embeddings[row] = node.embedding
        self._lattice_coords[row] = node.lattice_coords

        return node.id

    def add_nodes_batch(self, nodes: list[MemoryNode]) -> None:
        """Bulk insert (used during migration)."""
        if not nodes:
            return
        self._ensure_mmap_initialized(nodes[0])

        for node in nodes:
            self._ensure_capacity()
            row = self._next_row
            self._next_row += 1

            primary_cell = max(
                node.cell_memberships, key=node.cell_memberships.get
            )
            self._db.execute(
                "INSERT INTO nodes "
                "(id, text, row_idx, created_at, last_accessed, "
                " access_count, metadata_json, primary_cell) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    node.id, node.text, row,
                    node.created_at, node.last_accessed,
                    node.access_count, json.dumps(node.metadata),
                    primary_cell,
                ),
            )
            for cell_id, weight in node.cell_memberships.items():
                self._db.execute(
                    "INSERT INTO cell_memberships (node_id, cell_id, weight) "
                    "VALUES (?, ?, ?)",
                    (node.id, int(cell_id), weight),
                )

            self._embeddings[row] = node.embedding
            self._lattice_coords[row] = node.lattice_coords

        self._db.commit()
        self._embeddings.flush()
        self._lattice_coords.flush()

    def get_node(self, node_id: str) -> MemoryNode | None:
        row = self._db.execute(
            "SELECT id, text, row_idx, created_at, last_accessed, "
            "       access_count, metadata_json "
            "FROM nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def get_nodes_in_cell(self, cell_id: int) -> list[MemoryNode]:
        rows = self._db.execute(
            "SELECT id, text, row_idx, created_at, last_accessed, "
            "       access_count, metadata_json "
            "FROM nodes WHERE primary_cell = ?",
            (cell_id,),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_all_nodes(self) -> list[MemoryNode]:
        rows = self._db.execute(
            "SELECT id, text, row_idx, created_at, last_accessed, "
            "       access_count, metadata_json "
            "FROM nodes"
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def remove_node(self, node_id: str) -> bool:
        cur = self._db.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self._db.execute(
            "DELETE FROM cell_memberships WHERE node_id = ?", (node_id,)
        )
        self._db.commit()
        return cur.rowcount > 0

    def touch_node(self, node_id: str) -> None:
        """Update access timestamp and counter."""
        now = time.time()
        self._db.execute(
            "UPDATE nodes SET last_accessed = ?, access_count = access_count + 1 "
            "WHERE id = ?",
            (now, node_id),
        )
        # Commit batched externally or here for safety.
        self._db.commit()

    def count(self) -> int:
        return self._count_rows()

    def cell_counts(self) -> dict[int, int]:
        rows = self._db.execute(
            "SELECT primary_cell, COUNT(*) FROM nodes GROUP BY primary_cell"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------
    # Edge weights
    # ------------------------------------------------------------------
    def save_edge_weights(self, graph) -> None:
        """Persist networkx edge weights to SQLite."""
        self._db.execute("DELETE FROM edge_weights")
        for u, v, d in graph.edges(data=True):
            self._db.execute(
                "INSERT INTO edge_weights (u, v, weight) VALUES (?, ?, ?)",
                (int(u), int(v), d["weight"]),
            )
        self._db.commit()

    def load_edge_weights(self, graph) -> None:
        """Restore edge weights from SQLite into networkx graph."""
        rows = self._db.execute(
            "SELECT u, v, weight FROM edge_weights"
        ).fetchall()
        for u, v, w in rows:
            if graph.has_edge(u, v):
                graph[u][v]["weight"] = w

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------
    def flush(self) -> None:
        """Flush all pending writes to disk."""
        self._db.commit()
        if self._embeddings is not None:
            self._embeddings.flush()
        if self._lattice_coords is not None:
            self._lattice_coords.flush()

    def close(self) -> None:
        """Flush and close all handles."""
        self.flush()
        self._db.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _row_to_node(self, row: tuple) -> MemoryNode:
        nid, text, row_idx, created_at, last_accessed, access_count, meta_json = row

        memberships = {
            r[0]: r[1]
            for r in self._db.execute(
                "SELECT cell_id, weight FROM cell_memberships WHERE node_id = ?",
                (nid,),
            )
        }

        return MemoryNode(
            id=nid,
            text=text,
            embedding=np.array(self._embeddings[row_idx]),
            lattice_coords=np.array(self._lattice_coords[row_idx]),
            cell_memberships=memberships,
            created_at=created_at,
            last_accessed=last_accessed,
            access_count=access_count,
            metadata=json.loads(meta_json),
        )
