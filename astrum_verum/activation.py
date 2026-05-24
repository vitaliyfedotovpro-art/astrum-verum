"""
Spreading Activation — Neurocognitive wave propagation.

Inspired by ACT-R (Adaptive Control of Thought — Rational).  Energy
propagates from a start cell outward through lattice edges, modulated by
dynamic edge weights and exponential decay.

After each search, Hebbian learning updates the edge weights:
  - Co-activated edges are strengthened
  - Non-co-activated edges slowly decay
This creates an *adaptive topology* — the crystal literally learns to
think like its user.
"""

from __future__ import annotations

from .store import TopologyStore


class SpreadingActivation:
    """Spreading activation over the lattice topology graph."""

    def __init__(
        self,
        store: TopologyStore,
        decay: float = 0.6,
        radius: int = 3,
        min_energy: float = 0.01,
    ) -> None:
        """
        Args:
            store:  The topology store holding the lattice graph.
            decay:  Exponential decay factor per step (λ in design doc).
            radius: Maximum BFS depth (number of propagation steps).
            min_energy: Cells with energy below this threshold are pruned.
        """
        self._store = store
        self._decay = decay
        self._radius = radius
        self._min_energy = min_energy

    def activate(self, start_cell: int) -> dict[int, float]:
        """
        Propagate activation wave from ``start_cell``.

        Returns:
            Mapping ``{cell_id: energy}`` for all cells above threshold.
        """
        graph = self._store.graph
        energies: dict[int, float] = {start_cell: 1.0}
        frontier: set[int] = {start_cell}

        for step in range(1, self._radius + 1):
            next_frontier: set[int] = set()
            for cell in frontier:
                for neighbor in graph.neighbors(cell):
                    weight = graph[cell][neighbor].get("weight", 1.0)
                    energy = energies[cell] * weight * (self._decay ** step)

                    if energy < self._min_energy:
                        continue

                    if neighbor in energies:
                        energies[neighbor] = max(energies[neighbor], energy)
                    else:
                        energies[neighbor] = energy
                        next_frontier.add(neighbor)

            frontier = next_frontier

        return energies

    def hebbian_update(
        self,
        activated_cells: dict[int, float],
        learning_rate: float = 0.05,
        forget_rate: float = 0.001,
    ) -> None:
        """
        Hebbian edge weight update.

        Co-activated edges (both endpoints in ``activated_cells``):
            W(i,j) += η · E(i) · E(j)

        Non-co-activated edges:
            W(i,j) *= (1 − δ)

        Args:
            activated_cells: Output of ``activate()``.
            learning_rate:   η — how fast co-activated edges strengthen.
            forget_rate:     δ — how fast unused edges decay.
        """
        graph = self._store.graph
        activated_set = set(activated_cells.keys())

        for u, v in graph.edges():
            if u in activated_set and v in activated_set:
                delta = learning_rate * activated_cells[u] * activated_cells[v]
                self._store.update_edge_weight(u, v, delta)
            else:
                self._store.scale_edge_weight(u, v, 1.0 - forget_rate)
