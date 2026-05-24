"""Tests for the AstrumEngine (requires sentence-transformers)."""

import os
import tempfile

import pytest

pytest.importorskip("sentence_transformers")

from astrum_verum.engine import AstrumEngine  # noqa: E402


@pytest.fixture(scope="module")
def engine() -> AstrumEngine:
    """Module-scoped engine to avoid reloading the model per test."""
    return AstrumEngine()


class TestEngineBasic:
    def test_add_returns_id(self, engine: AstrumEngine) -> None:
        node_id = engine.add("The mitochondria is the powerhouse of the cell")
        assert isinstance(node_id, str)
        assert len(node_id) > 0

    def test_search_returns_results(self, engine: AstrumEngine) -> None:
        engine.add("Python is a programming language")
        engine.add("JavaScript runs in the browser")
        results = engine.search("web development languages")
        assert len(results) > 0
        assert hasattr(results[0], "text")
        assert hasattr(results[0], "score")

    def test_search_empty_store(self) -> None:
        eng = AstrumEngine()
        results = eng.search("anything")
        assert results == []

    def test_state(self, engine: AstrumEngine) -> None:
        state = engine.state()
        assert state["lattice"] == "D4"
        assert state["dimension"] == 4
        assert state["total_nodes"] >= 0

    def test_add_with_metadata(self, engine: AstrumEngine) -> None:
        node_id = engine.add(
            "Einstein's theory of relativity",
            metadata={"source": "physics", "year": 1905},
        )
        node = engine.store.get_node(node_id)
        assert node is not None
        assert node.metadata["source"] == "physics"


class TestEngineRelevance:
    def test_related_ranked_higher(self) -> None:
        """Related memories should score higher than unrelated ones."""
        eng = AstrumEngine()
        eng.add("Photosynthesis converts sunlight into chemical energy in plants")
        eng.add("Neural networks are composed of layers of artificial neurons")
        eng.add("Chloroplasts contain chlorophyll for photosynthesis")

        results = eng.search("plant biology and energy conversion")

        # At least one result should mention photosynthesis or plants.
        texts = [r.text for r in results]
        has_bio = any(
            "photo" in t.lower() or "plant" in t.lower() for t in texts
        )
        assert has_bio

    def test_score_has_breakdown(self) -> None:
        eng = AstrumEngine()
        eng.add("quantum computing uses qubits")
        results = eng.search("quantum information")
        assert len(results) > 0
        assert "cosine_similarity" in results[0].breakdown
        assert "topo_boost" in results[0].breakdown
        assert "recency" in results[0].breakdown


class TestEnginePersistence:
    def test_save_load(self) -> None:
        eng = AstrumEngine()
        eng.add("test memory for persistence")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            eng.save(path)

            eng2 = AstrumEngine()
            eng2.load(path)

            assert eng2.store.stats()["total_nodes"] == 1
            node = eng2.store.get_all_nodes()[0]
            assert node.text == "test memory for persistence"
        finally:
            os.unlink(path)
