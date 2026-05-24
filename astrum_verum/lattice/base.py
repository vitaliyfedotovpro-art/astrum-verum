"""
Abstract Lattice Plugin interface.

Every lattice (D₄, E₈, Λ₂₄) implements this interface.
Swapping the plugin is all that's needed to upgrade the memory geometry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatticeInfo:
    """Immutable metadata describing a lattice."""

    name: str
    dimension: int
    num_vertices: int
    num_edges: int
    neighbors_per_vertex: int
    symmetry_group_order: int


class LatticePlugin(ABC):
    """
    Abstract base class for a lattice geometry plugin.

    A lattice plugin provides:
    - The set of root vectors (vertices)
    - Cell centers (dual lattice / Voronoi centers)
    - Adjacency graph
    - Closest Vector Problem (CVP) decoder
    - SO(d) rotation utilities
    """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @abstractmethod
    def info(self) -> LatticeInfo:
        """Return immutable metadata about this lattice."""
        ...

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @abstractmethod
    def vertices(self) -> np.ndarray:
        """
        Root vectors of the lattice, normalized to unit sphere.

        Returns:
            ndarray of shape ``(num_vertices, dimension)``.
        """
        ...

    @abstractmethod
    def cell_centers(self) -> np.ndarray:
        """
        Centers of Voronoi cells (dual lattice points), normalized.

        For a self-dual lattice like D₄ these coincide with vertices.

        Returns:
            ndarray of shape ``(num_cells, dimension)``.
        """
        ...

    @abstractmethod
    def adjacency(self) -> dict[int, list[int]]:
        """
        Vertex adjacency graph.

        Returns:
            Mapping ``vertex_id → [neighbor_ids]`` where two vertices are
            adjacent iff their Euclidean distance equals the lattice minimal
            distance.
        """
        ...

    # ------------------------------------------------------------------
    # CVP — Closest Vector Problem
    # ------------------------------------------------------------------
    @abstractmethod
    def closest_vertex(self, point: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Find the closest lattice vertex to an arbitrary point in ℝ^d.

        Args:
            point: Array of shape ``(dimension,)``.

        Returns:
            ``(vertex_index, vertex_coordinates)``
        """
        ...

    @abstractmethod
    def decode_cell(self, point: np.ndarray) -> int:
        """
        Determine which Voronoi cell a point belongs to.

        Args:
            point: Array of shape ``(dimension,)``.

        Returns:
            Cell index (int).
        """
        ...

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def num_cells(self) -> int:
        """Number of Voronoi cells (== number of cell centers)."""
        return len(self.cell_centers())

    def dimension(self) -> int:
        """Dimensionality of the lattice."""
        return self.info().dimension
