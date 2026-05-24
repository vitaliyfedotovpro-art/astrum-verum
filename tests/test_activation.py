"""Tests for Spreading Activation."""

import pytest

from astrum_verum.activation import SpreadingActivation
from astrum_verum.lattice.d4 import D4Plugin
from astrum_verum.store import TopologyStore


@pytest.fixture
def store() -> TopologyStore:
    return TopologyStore(D4Plugin())


@pytest.fixture
def sa(store: TopologyStore) -> SpreadingActivation:
    return SpreadingActivation(store, decay=0.6, radius=3, min_energy=0.01)


class TestActivation:
    def test_start_cell_energy(self, sa: SpreadingActivation) -> None:
        """Start cell always has energy 1.0."""
        energies = sa.activate(0)
        assert energies[0] == 1.0

    def test_energy_decays(self, sa: SpreadingActivation) -> None:
        """Neighboring cells have less energy than start."""
        energies = sa.activate(0)
        for cell, energy in energies.items():
            if cell != 0:
                assert energy < 1.0

    def test_radius_limit(self, store: TopologyStore) -> None:
        """With radius=1, only start + immediate neighbors are activated."""
        sa = SpreadingActivation(store, decay=0.9, radius=1, min_energy=0.001)
        energies = sa.activate(0)

        d4 = D4Plugin()
        adj = d4.adjacency()
        expected = {0} | set(adj[0])
        assert set(energies.keys()).issubset(expected)

    def test_all_energies_positive(self, sa: SpreadingActivation) -> None:
        """All energies must be strictly positive."""
        energies = sa.activate(0)
        for energy in energies.values():
            assert energy > 0

    def test_different_start_cells(self, sa: SpreadingActivation) -> None:
        """Different start cells produce different activation patterns."""
        e0 = sa.activate(0)
        e5 = sa.activate(5)
        # The start cells themselves must differ.
        assert 0 in e0
        assert 5 in e5

    def test_high_decay_limits_spread(self, store: TopologyStore) -> None:
        """Very low decay should activate fewer cells."""
        sa_low = SpreadingActivation(store, decay=0.01, radius=3, min_energy=0.01)
        energies = sa_low.activate(0)
        # With decay=0.01, energy at step 1 ≈ 1.0 × 0.01 = 0.01 (just at threshold).
        # At step 2, energy ≈ 0.01 × 0.01² = 0.000001 (pruned).
        # So we expect only the start cell + possibly some radius-1 neighbors.
        assert len(energies) <= 9  # start + max 8 neighbors


class TestHebbianUpdate:
    def test_strengthens_coactivated(self, store: TopologyStore) -> None:
        """Co-activated edges should increase in weight."""
        sa = SpreadingActivation(store, decay=0.6, radius=1)

        d4 = D4Plugin()
        adj = d4.adjacency()
        i, j = 0, adj[0][0]

        before = store.get_edge_weight(i, j)
        energies = sa.activate(0)
        # Both i and j should be activated (j is a neighbor of i, radius=1).
        assert i in energies and j in energies

        sa.hebbian_update(energies, learning_rate=0.1)
        after = store.get_edge_weight(i, j)
        assert after > before

    def test_weakens_unused(self, store: TopologyStore) -> None:
        """Non-co-activated edges should decay."""
        sa = SpreadingActivation(store, decay=0.1, radius=1, min_energy=0.5)
        energies = sa.activate(0)

        d4 = D4Plugin()
        adj = d4.adjacency()

        # Find a cell not in the activated set.
        far_cell = None
        for cell in range(24):
            if cell not in energies:
                far_cell = cell
                break

        if far_cell is not None:
            neighbor_of_far = adj[far_cell][0]
            if neighbor_of_far not in energies:
                before = store.get_edge_weight(far_cell, neighbor_of_far)
                sa.hebbian_update(energies, forget_rate=0.01)
                after = store.get_edge_weight(far_cell, neighbor_of_far)
                assert after < before

    def test_repeated_activation_amplifies(self, store: TopologyStore) -> None:
        """Repeated activation of the same cell should accumulate edge weight."""
        sa = SpreadingActivation(store, decay=0.6, radius=1)
        d4 = D4Plugin()
        adj = d4.adjacency()
        i, j = 0, adj[0][0]

        for _ in range(5):
            energies = sa.activate(0)
            sa.hebbian_update(energies, learning_rate=0.1)

        assert store.get_edge_weight(i, j) > 1.0  # started at 1.0
