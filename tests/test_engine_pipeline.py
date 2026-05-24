"""
Контур Интеграционного Пайплайна.

Сквозная проверка фреймворка через абстрактный интерфейс.
Используются синтетические эмбеддинги — тесты не зависят от ML-модели
и работают мгновенно.

Тестируется:
    - "Слепой ввод" — полный цикл пайплайна на абстрактных данных
    - Гибридный скоринг — topo-boost поднимает слабого кандидата

Если эти тесты красные — сломана интеграция между модулями.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from astrum_verum.activation import SpreadingActivation
from astrum_verum.lattice.d4 import D4Plugin
from astrum_verum.lattice.e8 import E8Plugin
from astrum_verum.rotation import align_to_axis, compute_focus_vector, inverse_rotate_query
from astrum_verum.scorer import HybridScorer, ScoredResult
from astrum_verum.store import MemoryNode, TopologyStore


def _make_node(
    text: str,
    embedding: np.ndarray,
    cell: int,
    last_accessed: float | None = None,
) -> MemoryNode:
    """Фабрика MemoryNode с контролируемыми параметрами."""
    return MemoryNode(
        id=f"node-{text}",
        text=text,
        embedding=embedding,
        lattice_coords=np.zeros(4),  # не используется напрямую в скоринге
        cell_memberships={cell: 1.0},
        last_accessed=last_accessed or time.time(),
    )


# =====================================================================
# 1. "Слепой ввод" — Abstract Semantic Projection
# =====================================================================
class TestAbstractPipeline:
    """
    Подача абстрактной матрицы якорных концептов и текстового
    эмбеддинга. Пайплайн должен без ошибок пройти полный цикл:

        Проекция → Вращение → CVP → Активация графа → Кандидаты
    """

    @pytest.fixture
    def d4(self) -> D4Plugin:
        return D4Plugin()

    @pytest.fixture
    def e8(self) -> E8Plugin:
        return E8Plugin()

    def test_full_cycle_d4(self, d4: D4Plugin) -> None:
        """Полный цикл на D₄ с синтетическими данными."""
        store = TopologyStore(d4)
        sa = SpreadingActivation(store, decay=0.6, radius=2)
        scorer = HybridScorer()

        rng = np.random.default_rng(42)

        # Заполняем хранилище.
        for i in range(24):
            emb = rng.standard_normal(384)
            emb /= np.linalg.norm(emb)
            node = _make_node(f"fact-{i}", emb, cell=i)
            store.add_node(node)

        # Синтетический запрос.
        query_emb = rng.standard_normal(384)
        query_emb /= np.linalg.norm(query_emb)

        # Проекция → CVP.
        # Используем прямой dot product с вершинами (обход projector).
        verts = d4.vertices()
        query_lattice = rng.standard_normal(4)
        query_lattice /= np.linalg.norm(query_lattice)

        # Вращение.
        history = [rng.standard_normal(4) for _ in range(3)]
        history = [h / np.linalg.norm(h) for h in history]
        focus = compute_focus_vector(history, decay=0.8)
        R = align_to_axis(focus)
        rotated_query = inverse_rotate_query(query_lattice, R)

        # CVP.
        start_cell = d4.decode_cell(rotated_query)
        assert 0 <= start_cell < 24

        # Spreading Activation.
        energies = sa.activate(start_cell)
        assert len(energies) > 0
        assert energies[start_cell] == 1.0

        # Сбор кандидатов и скоринг.
        candidates = []
        for cell_id, energy in energies.items():
            for node in store.get_nodes_in_cell(cell_id):
                candidates.append((node, energy))

        if candidates:
            results = scorer.rank(query_emb, candidates, top_k=5)
            assert len(results) > 0
            assert all(isinstance(r, ScoredResult) for r in results)
            # Scores должны быть убывающими.
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score

        # Hebbian update не падает.
        sa.hebbian_update(energies)

    def test_full_cycle_e8(self, e8: E8Plugin) -> None:
        """Полный цикл на E₈ — 240 ячеек."""
        store = TopologyStore(e8)
        sa = SpreadingActivation(store, decay=0.6, radius=2)
        scorer = HybridScorer()
        rng = np.random.default_rng(42)

        # Заполняем 240 ячеек.
        for i in range(240):
            emb = rng.standard_normal(384)
            emb /= np.linalg.norm(emb)
            node = _make_node(f"e8-fact-{i}", emb, cell=i)
            store.add_node(node)

        query_emb = rng.standard_normal(384)
        query_emb /= np.linalg.norm(query_emb)

        query_lattice = rng.standard_normal(8)
        query_lattice /= np.linalg.norm(query_lattice)

        start_cell = e8.decode_cell(query_lattice)
        assert 0 <= start_cell < 240

        energies = sa.activate(start_cell)
        assert start_cell in energies

        candidates = []
        for cell_id, energy in energies.items():
            for node in store.get_nodes_in_cell(cell_id):
                candidates.append((node, energy))

        results = scorer.rank(query_emb, candidates, top_k=10)
        assert len(results) > 0

    def test_pipeline_with_empty_store(self, d4: D4Plugin) -> None:
        """Пайплайн на пустом хранилище не падает, возвращает 0 кандидатов."""
        store = TopologyStore(d4)
        sa = SpreadingActivation(store, decay=0.6, radius=2)
        scorer = HybridScorer()

        query_lattice = np.array([1.0, 0.0, 0.0, 0.0])
        start_cell = d4.decode_cell(query_lattice)
        energies = sa.activate(start_cell)

        candidates = []
        for cell_id, energy in energies.items():
            for node in store.get_nodes_in_cell(cell_id):
                candidates.append((node, energy))

        assert candidates == []
        results = scorer.rank(np.zeros(384), candidates, top_k=5)
        assert results == []

    def test_pipeline_determinism(self, d4: D4Plugin) -> None:
        """
        Два прогона с одинаковым seed должны дать
        идентичные результаты.
        """
        def run_pipeline(seed: int) -> list[float]:
            store = TopologyStore(d4)
            sa = SpreadingActivation(store, decay=0.6, radius=2)
            scorer = HybridScorer()
            rng = np.random.default_rng(seed)

            for i in range(24):
                emb = rng.standard_normal(384)
                emb /= np.linalg.norm(emb)
                store.add_node(_make_node(f"det-{i}", emb, cell=i))

            query_emb = rng.standard_normal(384)
            query_emb /= np.linalg.norm(query_emb)

            query_lattice = rng.standard_normal(4)
            query_lattice /= np.linalg.norm(query_lattice)
            start_cell = d4.decode_cell(query_lattice)

            energies = sa.activate(start_cell)
            candidates = []
            for cell_id, energy in energies.items():
                for node in store.get_nodes_in_cell(cell_id):
                    candidates.append((node, energy))

            now = 1000000.0  # fixed time
            results = scorer.rank(query_emb, candidates, top_k=5, now=now)
            return [r.score for r in results]

        scores_a = run_pipeline(seed=42)
        scores_b = run_pipeline(seed=42)
        np.testing.assert_array_equal(scores_a, scores_b)


# =====================================================================
# 2. Гибридный скоринг — Topo-Boost
# =====================================================================
class TestHybridScoringTopoBoost:
    """
    Проверка формулы:  α·cos + β·topo + γ·recency

    Узел с МЕНЬШИМ косинусным сходством, но БОЛЬШИМ topo-boost,
    должен подняться в топ выдачи.
    """

    def test_topo_boost_overrides_cosine(self) -> None:
        """
        Нода A: cos=0.8, topo=0.1
        Нода B: cos=0.7, topo=1.0

        С α=0.5, β=0.35: score_A = 0.5×0.8 + 0.35×0.1 = 0.435
                          score_B = 0.5×0.7 + 0.35×1.0 = 0.700

        B должен победить, несмотря на меньший косинус.
        """
        scorer = HybridScorer(alpha=0.5, beta=0.35, gamma=0.15)
        now = time.time()

        # Конструируем эмбеддинги с контролируемым косинусом.
        query = np.zeros(384)
        query[0] = 1.0  # единичный вектор по оси 0

        emb_a = np.zeros(384)
        emb_a[0] = 0.8
        emb_a[1] = 0.6  # cos(query, emb_a) = 0.8 / norm(emb_a) = 0.8

        emb_b = np.zeros(384)
        emb_b[0] = 0.7
        emb_b[1] = 0.7141  # cos ≈ 0.7

        node_a = _make_node("high-cos", emb_a, cell=0, last_accessed=now)
        node_b = _make_node("low-cos", emb_b, cell=0, last_accessed=now)

        result_a = scorer.score(query, node_a, cell_energy=0.1, now=now)
        result_b = scorer.score(query, node_b, cell_energy=1.0, now=now)

        assert result_b.score > result_a.score, (
            f"Topo-boosted node B ({result_b.score:.4f}) should rank "
            f"higher than node A ({result_a.score:.4f})"
        )

    def test_recency_boost_overrides_cosine(self) -> None:
        """
        Нода A: cos=0.9, last_accessed = 1 час назад
        Нода B: cos=0.85, last_accessed = только что

        С достаточным γ и recency_decay, B может победить.
        """
        scorer = HybridScorer(
            alpha=0.4, beta=0.1, gamma=0.5, recency_decay=0.01
        )
        now = time.time()

        query = np.zeros(384)
        query[0] = 1.0

        emb_a = np.zeros(384)
        emb_a[0] = 0.9
        emb_a[1] = 0.436  # cos ≈ 0.9

        emb_b = np.zeros(384)
        emb_b[0] = 0.85
        emb_b[1] = 0.527  # cos ≈ 0.85

        node_a = _make_node(
            "old", emb_a, cell=0, last_accessed=now - 3600
        )
        node_b = _make_node("fresh", emb_b, cell=0, last_accessed=now)

        result_a = scorer.score(query, node_a, cell_energy=0.5, now=now)
        result_b = scorer.score(query, node_b, cell_energy=0.5, now=now)

        assert result_b.score > result_a.score, (
            f"Recent node B ({result_b.score:.4f}) should rank higher "
            f"than old node A ({result_a.score:.4f})"
        )

    def test_pure_cosine_mode(self) -> None:
        """С α=1, β=0, γ=0 скоринг сводится к чистому косинусу."""
        scorer = HybridScorer(alpha=1.0, beta=0.0, gamma=0.0)
        now = time.time()

        query = np.array([1.0, 0.0, 0.0])
        node = _make_node(
            "test", np.array([0.6, 0.8, 0.0]), cell=0, last_accessed=now
        )

        result = scorer.score(query, node, cell_energy=999.0, now=now)

        expected_cos = 0.6 / 1.0  # dot / (norm_q * norm_e) = 0.6 / 1.0
        np.testing.assert_allclose(
            result.score, expected_cos, atol=1e-10,
            err_msg="With α=1, score should equal cosine similarity",
        )

    def test_pure_topo_mode(self) -> None:
        """С α=0, β=1, γ=0 скоринг сводится к topo_boost."""
        scorer = HybridScorer(alpha=0.0, beta=1.0, gamma=0.0)
        now = time.time()

        query = np.array([1.0, 0.0, 0.0])
        node = _make_node(
            "test", np.array([0.0, 1.0, 0.0]), cell=0, last_accessed=now
        )

        result = scorer.score(query, node, cell_energy=0.42, now=now)

        np.testing.assert_allclose(
            result.score, 0.42, atol=1e-10,
            err_msg="With β=1, score should equal cell_energy",
        )

    def test_ranking_integrates_all_signals(self) -> None:
        """
        Ранжирование с 3 нодами, каждая доминирует по одному сигналу.
        Финальный ранг определяется суммой.
        """
        scorer = HybridScorer(alpha=0.4, beta=0.35, gamma=0.25)
        now = time.time()

        query = np.zeros(384)
        query[0] = 1.0

        # Нода A: высокий косинус, низкие остальные.
        emb_a = np.zeros(384)
        emb_a[0] = 1.0
        node_a = _make_node("cos-king", emb_a, cell=0, last_accessed=now - 3600)

        # Нода B: средний косинус, высокий topo.
        emb_b = np.zeros(384)
        emb_b[0] = 0.5
        emb_b[1] = 0.866
        node_b = _make_node("topo-king", emb_b, cell=0, last_accessed=now - 3600)

        # Нода C: низкий косинус, высокая рецентность.
        emb_c = np.zeros(384)
        emb_c[0] = 0.3
        emb_c[1] = 0.954
        node_c = _make_node("recency-king", emb_c, cell=0, last_accessed=now)

        candidates = [
            (node_a, 0.1),   # low topo
            (node_b, 1.0),   # high topo
            (node_c, 0.1),   # low topo, high recency
        ]

        results = scorer.rank(query, candidates, top_k=3, now=now)

        # Все 3 должны быть в результатах.
        assert len(results) == 3

        # Результаты отсортированы по убыванию.
        for i in range(2):
            assert results[i].score >= results[i + 1].score

        # Breakdown должен содержать все компоненты.
        for r in results:
            assert "cosine_similarity" in r.breakdown
            assert "topo_boost" in r.breakdown
            assert "recency" in r.breakdown
