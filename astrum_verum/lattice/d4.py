"""
D₄ Lattice Plugin — The 24-cell (Icositetrachoron).

The 24-cell is the unique regular self-dual polytope in 4D.
It has 24 vertices, 96 edges, and each vertex has exactly 8 neighbors.
Its symmetry group W(D₄) has order 1152 and exhibits *triality* — a
unique order-3 outer automorphism not found in any other root system.

This is the prototype lattice for Astrum Verum.  It provides:
- 24 Voronoi cells (semantic domains)
- 8 neighbors per cell (spreading activation paths)
- Exact CVP via brute-force (fast at only 24 vertices)
- SO(4) rotation support
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .base import LatticeInfo, LatticePlugin

# Minimal distance between adjacent vertices of the 24-cell on the unit sphere.
# Two vertices V_i, V_j are neighbors iff ‖V_i − V_j‖ == _EDGE_LENGTH.
_EDGE_LENGTH = 1.0
_EDGE_LENGTH_TOLERANCE = 1e-9


class D4Plugin(LatticePlugin):
    """Concrete lattice plugin for the D₄ root system (24-cell)."""

    def __init__(self) -> None:
        self._vertices = self._build_vertices()
        self._cell_centers_arr = self._vertices.copy()  # self-dual
        self._adjacency = self._build_adjacency()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def info(self) -> LatticeInfo:
        return LatticeInfo(
            name="D4",
            dimension=4,
            num_vertices=24,
            num_edges=96,
            neighbors_per_vertex=8,
            symmetry_group_order=1152,
        )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def vertices(self) -> np.ndarray:
        return self._vertices.copy()

    def cell_centers(self) -> np.ndarray:
        # D₄ is self-dual: cell centers coincide with vertices.
        return self._cell_centers_arr.copy()

    def adjacency(self) -> dict[int, list[int]]:
        return {k: list(v) for k, v in self._adjacency.items()}

    # ------------------------------------------------------------------
    # CVP
    # ------------------------------------------------------------------
    def closest_vertex(self, point: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Brute-force CVP: compute distance to all 24 vertices.

        At only 24 vertices this is O(24·4) = O(96) operations — trivially fast.
        For E₈ (240 vertices) we'll need a smarter decoder, but for D₄ brute
        force is optimal.
        """
        point = np.asarray(point, dtype=np.float64)
        # Normalize point onto the unit 3-sphere so distances are meaningful.
        norm = np.linalg.norm(point)
        if norm < 1e-12:
            return 0, self._vertices[0].copy()
        point_normalized = point / norm

        # Cosine similarity is equivalent to dot product on unit vectors.
        # Maximize dot product == minimize angular distance.
        dots = self._vertices @ point_normalized
        idx = int(np.argmax(dots))
        return idx, self._vertices[idx].copy()

    def decode_cell(self, point: np.ndarray) -> int:
        """For a self-dual polytope, cell index == closest vertex index."""
        idx, _ = self.closest_vertex(point)
        return idx

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _build_vertices() -> np.ndarray:
        """
        Construct the 24 vertices of the 24-cell on the unit 3-sphere.

        8 axial vertices:  (±1, 0, 0, 0) and permutations.
        16 diagonal vertices: (±½, ±½, ±½, ±½) all sign combinations.
        """
        verts: list[list[float]] = []

        # 8 axial vertices
        for axis in range(4):
            for sign in (+1.0, -1.0):
                v = [0.0, 0.0, 0.0, 0.0]
                v[axis] = sign
                verts.append(v)

        # 16 diagonal vertices
        for bits in range(16):
            v = [
                0.5 if (bits >> i) & 1 == 0 else -0.5
                for i in range(4)
            ]
            verts.append(v)

        arr = np.array(verts, dtype=np.float64)
        assert arr.shape == (24, 4), f"Expected 24 vertices, got {arr.shape[0]}"

        # Verify all lie on the unit sphere.
        norms = np.linalg.norm(arr, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

        return arr

    def _build_adjacency(self) -> dict[int, list[int]]:
        """
        Build adjacency list.

        Two vertices are adjacent iff their Euclidean distance == 1.0.
        Each vertex should have exactly 8 neighbors; total edges == 96.
        """
        n = len(self._vertices)
        adj: dict[int, list[int]] = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(self._vertices[i] - self._vertices[j])
                if abs(dist - _EDGE_LENGTH) < _EDGE_LENGTH_TOLERANCE:
                    adj[i].append(j)
                    adj[j].append(i)

        # Validate: every vertex must have exactly 8 neighbors.
        for i, neighbors in adj.items():
            assert len(neighbors) == 8, (
                f"Vertex {i} has {len(neighbors)} neighbors, expected 8"
            )

        # Validate: total edges = sum of degrees / 2 = 24*8/2 = 96.
        total_edges = sum(len(v) for v in adj.values()) // 2
        assert total_edges == 96, f"Expected 96 edges, got {total_edges}"

        return adj

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return "D4Plugin(24-cell, dim=4, vertices=24, edges=96)"
