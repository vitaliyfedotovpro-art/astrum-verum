"""Tests for the SemanticProjector (requires sentence-transformers)."""

import numpy as np
import pytest

from astrum_verum.lattice.d4 import D4Plugin
from astrum_verum.defaults import D4_ANCHORS

pytest.importorskip("sentence_transformers")

from astrum_verum.projector import SemanticProjector  # noqa: E402


@pytest.fixture(scope="module")
def projector() -> SemanticProjector:
    """Module-scoped fixture to avoid reloading the model for every test."""
    return SemanticProjector(
        lattice=D4Plugin(),
        anchor_labels=D4_ANCHORS,
    )


class TestProjector:
    def test_output_shape(self, projector: SemanticProjector) -> None:
        result = projector.project("Hello world")
        assert result.shape == (4,)

    def test_unit_norm(self, projector: SemanticProjector) -> None:
        result = projector.project("quantum mechanics")
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-8)

    def test_deterministic(self, projector: SemanticProjector) -> None:
        """Same input should produce same output."""
        a = projector.project("test input")
        b = projector.project("test input")
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_soft_membership_sums_to_one(
        self, projector: SemanticProjector
    ) -> None:
        memberships = projector.soft_membership("biology and genetics")
        total = sum(memberships.values())
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_soft_membership_has_all_cells(
        self, projector: SemanticProjector
    ) -> None:
        memberships = projector.soft_membership("test")
        assert len(memberships) == 24  # D₄ has 24 cells

    def test_similar_texts_closer_in_embedding(
        self, projector: SemanticProjector
    ) -> None:
        """Related texts should have higher cosine similarity in embedding space."""
        # CAP projection onto 24 cells is lossy, so we test at the
        # embedding level (which feeds the projection) rather than
        # comparing projected lattice coordinates directly.
        a = projector.embed("quantum mechanics and wave functions")
        b = projector.embed("quantum physics and particles")
        c = projector.embed("cooking recipes for pasta")

        sim_ab = float(np.dot(a, b))  # both unit-normalized
        sim_ac = float(np.dot(a, c))

        assert sim_ab > sim_ac

    def test_anchor_count_mismatch(self) -> None:
        """Wrong number of anchors should raise ValueError."""
        with pytest.raises(ValueError):
            SemanticProjector(
                lattice=D4Plugin(),
                anchor_labels=["only one anchor"],
            )

    def test_embed_returns_vector(self, projector: SemanticProjector) -> None:
        """embed() should return a high-dimensional unit vector."""
        v = projector.embed("test")
        assert v.ndim == 1
        assert len(v) > 4  # much higher than lattice dim
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-6)

    def test_project_vector_matches_project(
        self, projector: SemanticProjector
    ) -> None:
        """project_vector(embed(text)) should equal project(text)."""
        text = "astrophysics research"
        via_project = projector.project(text)
        via_manual = projector.project_vector(projector.embed(text))
        np.testing.assert_allclose(via_project, via_manual, atol=1e-10)
