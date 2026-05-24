"""
E₈ Lattice Plugin — The exceptional root lattice in 8 dimensions.

E₈ is the unique even unimodular lattice in 8D.  Maryna Viazovska proved
(2016, Fields Medal) that it gives the densest sphere packing in 8 dimensions.

240 root vectors:
  - 112 of type (±1, ±1, 0, 0, 0, 0, 0, 0) — choose 2 of 8 positions × 4 signs
  - 128 of type (±½, ±½, ±½, ±½, ±½, ±½, ±½, ±½) — even number of minus signs

All normalized to the unit 7-sphere.  Each vertex has 56 neighbors in the
root system adjacency graph (edge iff Euclidean distance = 1.0 on unit sphere,
equivalently dot product = 0.5).  Total edges: 6 720.
"""

from __future__ import annotations

import numpy as np

from .base import LatticeInfo, LatticePlugin

# Two normalized root vectors are adjacent iff their dot product ≈ 0.5.
_DOT_PRODUCT_ADJACENT = 0.5
_TOLERANCE = 1e-9


class E8Plugin(LatticePlugin):
    """Concrete lattice plugin for the E₈ root system."""

    def __init__(self) -> None:
        self._vertices = self._build_vertices()
        self._cell_centers_arr = self._vertices.copy()
        self._adjacency = self._build_adjacency()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def info(self) -> LatticeInfo:
        return LatticeInfo(
            name="E8",
            dimension=8,
            num_vertices=240,
            num_edges=6720,
            neighbors_per_vertex=56,
            symmetry_group_order=696_729_600,
        )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def vertices(self) -> np.ndarray:
        return self._vertices.copy()

    def cell_centers(self) -> np.ndarray:
        return self._cell_centers_arr.copy()

    def adjacency(self) -> dict[int, list[int]]:
        return {k: list(v) for k, v in self._adjacency.items()}

    # ------------------------------------------------------------------
    # CVP
    # ------------------------------------------------------------------
    def closest_vertex(self, point: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Brute-force CVP against 240 vertices.

        At 240 × 8 = 1 920 multiply-adds this is still trivially fast.
        """
        point = np.asarray(point, dtype=np.float64)
        norm = np.linalg.norm(point)
        if norm < 1e-12:
            return 0, self._vertices[0].copy()
        point_normalized = point / norm

        dots = self._vertices @ point_normalized
        idx = int(np.argmax(dots))
        return idx, self._vertices[idx].copy()

    def decode_cell(self, point: np.ndarray) -> int:
        idx, _ = self.closest_vertex(point)
        return idx

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _build_vertices() -> np.ndarray:
        """
        Construct the 240 root vectors of E₈, normalized to unit sphere.

        Type 1 (112 vectors): (±1, ±1, 0, …, 0) — C(8,2) × 2² = 112.
        Type 2 (128 vectors): (±½)⁸ with even number of minus signs.
        """
        verts: list[list[float]] = []

        # Type 1: 112 vectors — choose 2 positions, assign ±1 to each.
        for i in range(8):
            for j in range(i + 1, 8):
                for si in (+1.0, -1.0):
                    for sj in (+1.0, -1.0):
                        v = [0.0] * 8
                        v[i] = si
                        v[j] = sj
                        verts.append(v)

        # Type 2: 128 vectors — all (±½)⁸ with even number of negatives.
        for bits in range(256):
            signs = [1.0 if (bits >> k) & 1 == 0 else -1.0 for k in range(8)]
            num_neg = sum(1 for s in signs if s < 0)
            if num_neg % 2 == 0:
                verts.append([s * 0.5 for s in signs])

        arr = np.array(verts, dtype=np.float64)
        assert arr.shape == (240, 8), f"Expected 240 vertices, got {arr.shape[0]}"

        # All roots have norm √2 before normalization.
        raw_norms = np.linalg.norm(arr, axis=1)
        np.testing.assert_allclose(raw_norms, np.sqrt(2.0), atol=1e-12)

        # Normalize to unit sphere.
        arr /= raw_norms[:, np.newaxis]

        return arr

    def _build_adjacency(self) -> dict[int, list[int]]:
        """
        Build adjacency list via dot-product matrix.

        Two unit root vectors are adjacent iff dot product ≈ 0.5
        (equivalently, Euclidean distance ≈ 1.0).
        Each vertex has exactly 56 neighbors; total edges = 6 720.
        """
        n = len(self._vertices)
        adj: dict[int, list[int]] = {i: [] for i in range(n)}

        # Full dot product matrix — O(240² × 8) ≈ 460 K ops, trivially fast.
        dots = self._vertices @ self._vertices.T

        for i in range(n):
            for j in range(i + 1, n):
                if abs(dots[i, j] - _DOT_PRODUCT_ADJACENT) < _TOLERANCE:
                    adj[i].append(j)
                    adj[j].append(i)

        # Validate: every vertex must have exactly 56 neighbors.
        for i, neighbors in adj.items():
            assert len(neighbors) == 56, (
                f"Vertex {i} has {len(neighbors)} neighbors, expected 56"
            )

        # Validate: total edges = 240 × 56 / 2 = 6720.
        total_edges = sum(len(v) for v in adj.values()) // 2
        assert total_edges == 6720, f"Expected 6720 edges, got {total_edges}"

        return adj

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return "E8Plugin(E₈, dim=8, vertices=240, edges=6720)"
