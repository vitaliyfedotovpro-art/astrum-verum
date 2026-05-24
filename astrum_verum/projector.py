"""
Semantic Projector — Concept-Anchored Projection (CAP).

Projects high-dimensional embeddings (384D / 768D / 1536D) down to the
lattice dimension (4D / 8D) via semantic anchor concepts.

Algorithm:
    1. Embed text → ℝᴺ  (via sentence-transformers)
    2. Cosine similarity to K anchor embeddings → ℝᴷ
    3. Softmax(similarity) × cell_centers → weighted sum in ℝᵈ
    4. Normalize onto the unit (d−1)-sphere

The model is loaded lazily on first use to keep import time fast.
"""

from __future__ import annotations

import numpy as np

from .lattice.base import LatticePlugin


class SemanticProjector:
    """Projects text into lattice space via Concept-Anchored Projection."""

    def __init__(
        self,
        lattice: LatticePlugin,
        anchor_labels: list[str],
        embedder_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._lattice = lattice
        self._anchor_labels = anchor_labels
        self._embedder_model = embedder_model
        self._model = None
        self._anchor_embeddings: np.ndarray | None = None
        self._cell_centers = lattice.cell_centers()

        if len(anchor_labels) != lattice.num_cells():
            raise ValueError(
                f"Need {lattice.num_cells()} anchors for "
                f"{lattice.info().name}, got {len(anchor_labels)}"
            )

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        """Load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._embedder_model)
            # Pre-compute anchor embeddings (normalized).
            self._anchor_embeddings = self._model.encode(
                self._anchor_labels, normalize_embeddings=True
            )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text into ℝᴺ (normalized)."""
        self._ensure_model()
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts into ℝᴺ (normalized)."""
        self._ensure_model()
        return self._model.encode(texts, normalize_embeddings=True)

    # ------------------------------------------------------------------
    # CAP Projection
    # ------------------------------------------------------------------
    def project(self, text: str) -> np.ndarray:
        """
        Full CAP pipeline: text → ℝᴺ → ℝᵈ (lattice dimension).
        """
        embedding = self.embed(text)
        return self.project_vector(embedding)

    def project_vector(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project a pre-computed embedding to lattice coordinates.

        1. Cosine similarity to K anchors
        2. Softmax → weights
        3. Weighted sum of cell centers
        4. Normalize to unit sphere
        """
        self._ensure_model()

        embedding = np.asarray(embedding, dtype=np.float64)
        norm = np.linalg.norm(embedding)
        if norm > 1e-12:
            embedding = embedding / norm

        # Cosine similarity (both sides normalized ⇒ dot product).
        similarities = self._anchor_embeddings @ embedding  # shape (K,)

        # Softmax weights.
        weights = _softmax(similarities)

        # Weighted sum of cell centers.
        projected = weights @ self._cell_centers  # shape (d,)

        # Normalize to unit sphere.
        proj_norm = np.linalg.norm(projected)
        if proj_norm < 1e-12:
            return self._cell_centers[0].copy()

        return projected / proj_norm

    # ------------------------------------------------------------------
    # Soft cell membership
    # ------------------------------------------------------------------
    def soft_membership(
        self,
        text: str,
        temperature: float = 5.0,
    ) -> dict[int, float]:
        """Compute soft cell membership with temperature scaling."""
        embedding = self.embed(text)
        return self.soft_membership_vector(embedding, temperature)

    def soft_membership_vector(
        self,
        embedding: np.ndarray,
        temperature: float = 5.0,
    ) -> dict[int, float]:
        """Soft membership from a pre-computed embedding."""
        self._ensure_model()

        embedding = np.asarray(embedding, dtype=np.float64)
        norm = np.linalg.norm(embedding)
        if norm > 1e-12:
            embedding = embedding / norm

        similarities = self._anchor_embeddings @ embedding
        weights = _softmax(similarities * temperature)

        return {i: float(w) for i, w in enumerate(weights)}


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()
