"""
Графовый контур.

Проверка динамики смыслов и алгоритма затухания.
Тестируется:
    - BFS-волна Spreading Activation (корректное распространение + затухание)
    - Хеббовское обучение (пластичность весов рёбер)

Всё тестируется на абстрактной топологии без ИИ-контекста.
"""

from __future__ import annotations

import numpy as np
import pytest

from astrum_verum.activation import SpreadingActivation
from astrum_verum.lattice.d4 import D4Plugin
from astrum_verum.lattice.e8 import E8Plugin
from astrum_verum.store import TopologyStore


# =====================================================================
# 1. BFS-волна (Spreading Activation)
# =====================================================================
class TestBFSWave:
    """
    Проверка, что при активации узла A волна корректно расходится
    по смежным рёбрам и полностью затухает на заданном радиусе.
    """

    @pytest.fixture
    def d4_store(self) -> TopologyStore:
        return TopologyStore(D4Plugin())

    @pytest.fixture
    def e8_store(self) -> TopologyStore:
        return TopologyStore(E8Plugin())

    def test_epicenter_always_1(self, d4_store: TopologyStore) -> None:
        """Эпицентр волны (стартовый узел) всегда имеет E = 1.0."""
        sa = SpreadingActivation(d4_store, decay=0.6, radius=3)
        for start in range(24):
            energies = sa.activate(start)
            assert energies[start] == 1.0, (
                f"Start cell {start} has energy {energies[start]}"
            )

    def test_monotone_decay_from_center(
        self, d4_store: TopologyStore
    ) -> None:
        """
        Энергия должна монотонно убывать с удалением от центра.
        Для ближайших соседей: E < 1.0.
        """
        sa = SpreadingActivation(d4_store, decay=0.6, radius=3)
        adj = D4Plugin().adjacency()

        for start in range(24):
            energies = sa.activate(start)
            for neighbor in adj[start]:
                if neighbor in energies:
                    assert energies[neighbor] < energies[start], (
                        f"Neighbor {neighbor} of {start} has energy "
                        f"{energies[neighbor]} >= {energies[start]}"
                    )

    def test_wave_reaches_neighbors(
        self, d4_store: TopologyStore
    ) -> None:
        """
        При radius ≥ 1, все непосредственные соседи должны быть
        активированы (если decay достаточно высок).
        """
        sa = SpreadingActivation(
            d4_store, decay=0.9, radius=1, min_energy=0.001
        )
        adj = D4Plugin().adjacency()

        energies = sa.activate(0)
        for neighbor in adj[0]:
            assert neighbor in energies, (
                f"Direct neighbor {neighbor} of cell 0 was not activated"
            )

    def test_complete_extinction_at_low_decay(
        self, d4_store: TopologyStore
    ) -> None:
        """
        При очень низком decay волна должна затухнуть за 1 шаг.
        Только стартовый узел остаётся активным.
        """
        sa = SpreadingActivation(
            d4_store, decay=0.001, radius=5, min_energy=0.01
        )
        energies = sa.activate(0)

        # Только стартовый узел.
        assert len(energies) == 1
        assert 0 in energies

    def test_radius_zero_only_start(
        self, d4_store: TopologyStore
    ) -> None:
        """radius=0 → только стартовый узел."""
        sa = SpreadingActivation(d4_store, decay=0.9, radius=0)
        energies = sa.activate(0)
        assert energies == {0: 1.0}

    def test_wave_symmetry_on_regular_graph(
        self, d4_store: TopologyStore
    ) -> None:
        """
        На графе с одинаковыми весами рёбер, все соседи стартового
        узла должны иметь одинаковую энергию.
        """
        sa = SpreadingActivation(
            d4_store, decay=0.6, radius=1, min_energy=0.001
        )
        energies = sa.activate(0)
        adj = D4Plugin().adjacency()

        neighbor_energies = [
            energies[n] for n in adj[0] if n in energies
        ]
        if len(neighbor_energies) > 1:
            # Все должны быть равны.
            for e in neighbor_energies[1:]:
                np.testing.assert_allclose(
                    e, neighbor_energies[0], atol=1e-12,
                    err_msg="Neighbor energies should be equal on uniform graph",
                )

    def test_all_energies_strictly_positive(
        self, d4_store: TopologyStore
    ) -> None:
        """Все энергии в выходном словаре > 0."""
        sa = SpreadingActivation(d4_store, decay=0.6, radius=3)
        energies = sa.activate(0)
        for cell, energy in energies.items():
            assert energy > 0, f"Cell {cell} has non-positive energy {energy}"

    def test_e8_wave_degree_56(self, e8_store: TopologyStore) -> None:
        """
        На E₈ волна при radius=1 должна активировать ровно 57 узлов
        (1 центр + 56 соседей).
        """
        sa = SpreadingActivation(
            e8_store, decay=0.9, radius=1, min_energy=0.001
        )
        energies = sa.activate(0)
        assert len(energies) == 57, (
            f"Expected 57 activated cells (1 + 56 neighbors), "
            f"got {len(energies)}"
        )

    def test_weighted_propagation(self, d4_store: TopologyStore) -> None:
        """
        Если усилить вес одного ребра, энергия по нему
        должна быть больше, чем по стандартным рёбрам.
        """
        adj = D4Plugin().adjacency()
        boosted_neighbor = adj[0][0]
        weak_neighbor = adj[0][1]

        # Усиливаем одно ребро.
        d4_store.set_edge_weight(0, boosted_neighbor, 5.0)

        sa = SpreadingActivation(
            d4_store, decay=0.6, radius=1, min_energy=0.001
        )
        energies = sa.activate(0)

        assert energies[boosted_neighbor] > energies[weak_neighbor], (
            f"Boosted edge ({energies[boosted_neighbor]:.4f}) should "
            f"transmit more energy than weak ({energies[weak_neighbor]:.4f})"
        )


# =====================================================================
# 2. Хеббовское обучение (Plasticity)
# =====================================================================
class TestHebbianPlasticity:
    """
    Имитация частого совместного вызова двух абстрактных узлов.
    Проверка, что веса рёбер адаптируются.
    """

    @pytest.fixture
    def d4_store(self) -> TopologyStore:
        return TopologyStore(D4Plugin())

    def test_coactivation_strengthens_edge(
        self, d4_store: TopologyStore
    ) -> None:
        """
        «Neurons that fire together wire together».
        Вес ребра между коактивированными узлами должен расти.
        """
        sa = SpreadingActivation(d4_store, decay=0.9, radius=1)
        adj = D4Plugin().adjacency()
        i, j = 0, adj[0][0]

        weight_before = d4_store.get_edge_weight(i, j)

        for _ in range(10):
            energies = sa.activate(i)
            assert j in energies, "Neighbor must be co-activated"
            sa.hebbian_update(energies, learning_rate=0.1)

        weight_after = d4_store.get_edge_weight(i, j)
        assert weight_after > weight_before, (
            f"Edge weight did not increase: {weight_before} → {weight_after}"
        )

    def test_unused_edges_decay(self, d4_store: TopologyStore) -> None:
        """
        Веса неиспользуемых рёбер должны экспоненциально затухать.
        """
        sa = SpreadingActivation(
            d4_store, decay=0.1, radius=1, min_energy=0.5
        )
        adj = D4Plugin().adjacency()

        # Находим ребро, которое не будет коактивировано.
        energies = sa.activate(0)
        far_cells = [c for c in range(24) if c not in energies]
        assert len(far_cells) > 0, "Need at least one inactive cell"

        far = far_cells[0]
        far_neighbor = adj[far][0]

        # Оба конца ребра должны быть вне зоны активации.
        if far_neighbor not in energies:
            weight_before = d4_store.get_edge_weight(far, far_neighbor)
            sa.hebbian_update(energies, forget_rate=0.05)
            weight_after = d4_store.get_edge_weight(far, far_neighbor)
            assert weight_after < weight_before, (
                f"Unused edge weight did not decay: "
                f"{weight_before} → {weight_after}"
            )

    def test_plasticity_convergence(
        self, d4_store: TopologyStore
    ) -> None:
        """
        После 100 циклов активации одного и того же узла,
        связанные рёбра должны быть значительно сильнее
        несвязанных.
        """
        sa = SpreadingActivation(d4_store, decay=0.6, radius=1)
        adj = D4Plugin().adjacency()

        # 100 циклов активации узла 0.
        for _ in range(100):
            energies = sa.activate(0)
            sa.hebbian_update(
                energies, learning_rate=0.05, forget_rate=0.001
            )

        # Рёбра из узла 0 должны быть сильнее.
        local_weight = d4_store.get_edge_weight(0, adj[0][0])

        # Удалённое ребро (не из 0).
        far = 1  # D₄: узел 1 далеко от 0
        far_weight = d4_store.get_edge_weight(far, adj[far][0])

        assert local_weight > far_weight, (
            f"Local edge ({local_weight:.3f}) should be stronger "
            f"than far edge ({far_weight:.3f}) after 100 activations"
        )

    def test_forget_rate_zero_no_decay(
        self, d4_store: TopologyStore
    ) -> None:
        """При forget_rate=0 неиспользуемые рёбра не затухают."""
        sa = SpreadingActivation(
            d4_store, decay=0.1, radius=1, min_energy=0.5
        )
        adj = D4Plugin().adjacency()

        energies = sa.activate(0)
        far_cells = [c for c in range(24) if c not in energies]

        if len(far_cells) >= 2:
            far = far_cells[0]
            far_neighbor = adj[far][0]
            if far_neighbor not in energies:
                weight_before = d4_store.get_edge_weight(far, far_neighbor)
                sa.hebbian_update(energies, forget_rate=0.0)
                weight_after = d4_store.get_edge_weight(far, far_neighbor)
                assert weight_after == weight_before

    def test_learning_rate_proportionality(
        self, d4_store: TopologyStore
    ) -> None:
        """
        Удвоение learning_rate должно примерно удвоить
        прирост веса за один шаг.
        """
        sa = SpreadingActivation(d4_store, decay=0.9, radius=1)
        adj = D4Plugin().adjacency()
        i, j = 0, adj[0][0]

        # Шаг с lr=0.1.
        energies = sa.activate(i)
        w0 = d4_store.get_edge_weight(i, j)
        sa.hebbian_update(energies, learning_rate=0.1, forget_rate=0.0)
        delta_1 = d4_store.get_edge_weight(i, j) - w0

        # Сброс.
        d4_store.set_edge_weight(i, j, w0)

        # Шаг с lr=0.2.
        sa.hebbian_update(energies, learning_rate=0.2, forget_rate=0.0)
        delta_2 = d4_store.get_edge_weight(i, j) - w0

        # delta_2 ≈ 2 × delta_1.
        np.testing.assert_allclose(
            delta_2, 2 * delta_1, rtol=0.01,
            err_msg="Weight delta should scale linearly with learning_rate",
        )
