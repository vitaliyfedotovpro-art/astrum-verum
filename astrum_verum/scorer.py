"""
Hybrid Scorer — Combines cosine similarity, topological boost, and recency.

    score(doc) = α · cos_sim(q, doc) + β · topo_boost(doc) + γ · recency(doc)

where:
    cos_sim    — classical cosine similarity in the original N-dimensional space
    topo_boost — energy of the document's cell after spreading activation
    recency    — exp(−μ · Δt) exponential decay from last access time

Default weights: α=0.5, β=0.35, γ=0.15  (sum to 1.0).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .store import MemoryNode


@dataclass
class ScoredResult:
    """A memory node with its computed score and breakdown."""

    node: MemoryNode
    score: float
    breakdown: dict


class HybridScorer:
    """Three-signal hybrid scorer."""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.35,
        gamma: float = 0.15,
        recency_decay: float = 0.001,
    ) -> None:
        if abs(alpha + beta + gamma - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {alpha + beta + gamma:.6f}"
            )
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._recency_decay = recency_decay

    def score(
        self,
        query_embedding: np.ndarray,
        node: MemoryNode,
        cell_energy: float,
        now: float | None = None,
    ) -> ScoredResult:
        """Score a single node against a query embedding."""
        if now is None:
            now = time.time()

        # Cosine similarity.
        q = np.asarray(query_embedding, dtype=np.float64)
        e = np.asarray(node.embedding, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        e_norm = np.linalg.norm(e)

        if q_norm < 1e-12 or e_norm < 1e-12:
            cos_sim = 0.0
        else:
            cos_sim = float(np.dot(q, e) / (q_norm * e_norm))

        # Topological boost.
        topo_boost = cell_energy

        # Recency.
        dt = max(0.0, now - node.last_accessed)
        recency = float(np.exp(-self._recency_decay * dt))

        # Combined score.
        total = (
            self._alpha * cos_sim
            + self._beta * topo_boost
            + self._gamma * recency
        )

        return ScoredResult(
            node=node,
            score=total,
            breakdown={
                "cosine_similarity": cos_sim,
                "topo_boost": topo_boost,
                "recency": recency,
                "alpha": self._alpha,
                "beta": self._beta,
                "gamma": self._gamma,
            },
        )

    def rank(
        self,
        query_embedding: np.ndarray,
        candidates: list[tuple[MemoryNode, float]],
        top_k: int = 5,
        now: float | None = None,
    ) -> list[ScoredResult]:
        """
        Score and rank candidate nodes.

        Args:
            query_embedding: Query vector in ℝᴺ.
            candidates:      List of ``(node, cell_energy)`` tuples.
            top_k:           Number of results to return.
            now:             Current timestamp (defaults to ``time.time()``).
        """
        if now is None:
            now = time.time()

        scored = [
            self.score(query_embedding, node, energy, now)
            for node, energy in candidates
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
