"""Tests for the HybridScorer."""

import time

import numpy as np
import pytest

from astrum_verum.scorer import HybridScorer
from astrum_verum.store import MemoryNode


def _make_node(
    embedding: np.ndarray,
    last_accessed: float | None = None,
) -> MemoryNode:
    """Helper to create a MemoryNode for scoring tests."""
    return MemoryNode(
        id="test",
        text="test",
        embedding=embedding,
        lattice_coords=np.array([1.0, 0.0, 0.0, 0.0]),
        cell_memberships={0: 1.0},
        last_accessed=last_accessed or time.time(),
    )


class TestHybridScorer:
    def test_weights_must_sum_to_one(self) -> None:
        """Constructor should reject weights that don't sum to 1."""
        HybridScorer(alpha=0.5, beta=0.3, gamma=0.2)  # OK
        with pytest.raises(ValueError):
            HybridScorer(alpha=0.5, beta=0.5, gamma=0.5)

    def test_identical_vectors_high_cosine(self) -> None:
        scorer = HybridScorer()
        v = np.array([1.0, 0.0, 0.0])
        node = _make_node(v)
        result = scorer.score(v, node, cell_energy=1.0)
        assert result.breakdown["cosine_similarity"] > 0.99

    def test_orthogonal_vectors_zero_cosine(self) -> None:
        scorer = HybridScorer()
        q = np.array([1.0, 0.0, 0.0])
        e = np.array([0.0, 1.0, 0.0])
        node = _make_node(e)
        result = scorer.score(q, node, cell_energy=0.0)
        assert abs(result.breakdown["cosine_similarity"]) < 1e-10

    def test_opposite_vectors_negative_cosine(self) -> None:
        scorer = HybridScorer()
        q = np.array([1.0, 0.0, 0.0])
        e = np.array([-1.0, 0.0, 0.0])
        node = _make_node(e)
        result = scorer.score(q, node, cell_energy=0.0)
        assert result.breakdown["cosine_similarity"] < -0.99

    def test_recency_decays_over_time(self) -> None:
        scorer = HybridScorer(recency_decay=1.0)
        v = np.array([1.0, 0.0, 0.0])
        now = time.time()

        recent_node = _make_node(v, last_accessed=now)
        old_node = _make_node(v, last_accessed=now - 10)

        r_recent = scorer.score(v, recent_node, 1.0, now=now)
        r_old = scorer.score(v, old_node, 1.0, now=now)

        assert r_recent.breakdown["recency"] > r_old.breakdown["recency"]

    def test_topo_boost_affects_score(self) -> None:
        scorer = HybridScorer()
        v = np.array([1.0, 0.0, 0.0])
        node = _make_node(v)

        high = scorer.score(v, node, cell_energy=1.0)
        low = scorer.score(v, node, cell_energy=0.0)

        assert high.score > low.score

    def test_score_breakdown_keys(self) -> None:
        scorer = HybridScorer()
        v = np.array([1.0, 0.0, 0.0])
        node = _make_node(v)
        result = scorer.score(v, node, cell_energy=0.5)

        expected_keys = {
            "cosine_similarity",
            "topo_boost",
            "recency",
            "alpha",
            "beta",
            "gamma",
        }
        assert set(result.breakdown.keys()) == expected_keys


class TestRanking:
    def test_rank_order(self) -> None:
        """Higher cosine similarity should rank first (alpha=1.0)."""
        scorer = HybridScorer(alpha=1.0, beta=0.0, gamma=0.0)
        q = np.array([1.0, 0.0, 0.0])

        good = _make_node(np.array([0.9, 0.1, 0.0]))
        bad = _make_node(np.array([0.0, 0.0, 1.0]))

        results = scorer.rank(q, [(good, 1.0), (bad, 1.0)], top_k=2)
        assert results[0].node is good

    def test_top_k_limits(self) -> None:
        scorer = HybridScorer()
        q = np.array([1.0, 0.0, 0.0])
        rng = np.random.default_rng(0)
        nodes = [
            (_make_node(rng.standard_normal(3)), 0.5)
            for _ in range(10)
        ]

        results = scorer.rank(q, nodes, top_k=3)
        assert len(results) == 3

    def test_empty_candidates(self) -> None:
        scorer = HybridScorer()
        q = np.array([1.0, 0.0, 0.0])
        results = scorer.rank(q, [], top_k=5)
        assert results == []
