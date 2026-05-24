"""Tests for the TopologyStore."""

import os
import tempfile

import numpy as np
import pytest

from astrum_verum.lattice.d4 import D4Plugin
from astrum_verum.store import MemoryNode, TopologyStore


@pytest.fixture
def store() -> TopologyStore:
    return TopologyStore(D4Plugin())


def _make_node(
    text: str = "test",
    cell: int = 0,
    embedding_seed: int = 42,
) -> MemoryNode:
    """Helper to create a MemoryNode for testing."""
    rng = np.random.default_rng(embedding_seed)
    return MemoryNode(
        id=f"node-{text}",
        text=text,
        embedding=rng.standard_normal(384),
        lattice_coords=np.array([1.0, 0.0, 0.0, 0.0]),
        cell_memberships={cell: 0.8, (cell + 1) % 24: 0.2},
    )


class TestTopologyStoreBasic:
    def test_add_and_get(self, store: TopologyStore) -> None:
        node = _make_node("hello")
        store.add_node(node)
        retrieved = store.get_node("node-hello")
        assert retrieved is not None
        assert retrieved.text == "hello"

    def test_get_nonexistent(self, store: TopologyStore) -> None:
        assert store.get_node("nope") is None

    def test_get_nodes_in_cell(self, store: TopologyStore) -> None:
        store.add_node(_make_node("a", cell=0, embedding_seed=1))
        store.add_node(_make_node("b", cell=0, embedding_seed=2))
        store.add_node(_make_node("c", cell=5, embedding_seed=3))

        cell0 = store.get_nodes_in_cell(0)
        assert len(cell0) == 2
        assert {n.text for n in cell0} == {"a", "b"}

    def test_get_all_nodes(self, store: TopologyStore) -> None:
        store.add_node(_make_node("x", embedding_seed=10))
        store.add_node(_make_node("y", embedding_seed=11))
        assert len(store.get_all_nodes()) == 2

    def test_remove_node(self, store: TopologyStore) -> None:
        store.add_node(_make_node("rm"))
        assert store.remove_node("node-rm") is True
        assert store.get_node("node-rm") is None
        assert store.remove_node("node-rm") is False

    def test_remove_clears_from_cell(self, store: TopologyStore) -> None:
        store.add_node(_make_node("cell-rm", cell=3))
        store.remove_node("node-cell-rm")
        assert len(store.get_nodes_in_cell(3)) == 0


class TestGraphInitialization:
    def test_graph_nodes(self, store: TopologyStore) -> None:
        """Graph should have 24 lattice vertices."""
        assert store.graph.number_of_nodes() == 24

    def test_graph_edges(self, store: TopologyStore) -> None:
        """Graph should have 96 edges (D₄)."""
        assert store.graph.number_of_edges() == 96


class TestEdgeWeights:
    def test_default_weight(self, store: TopologyStore) -> None:
        adj = D4Plugin().adjacency()
        i, j = 0, adj[0][0]
        assert store.get_edge_weight(i, j) == 1.0

    def test_update_weight(self, store: TopologyStore) -> None:
        adj = D4Plugin().adjacency()
        i, j = 0, adj[0][0]
        store.update_edge_weight(i, j, 0.5)
        assert abs(store.get_edge_weight(i, j) - 1.5) < 1e-10

    def test_set_weight(self, store: TopologyStore) -> None:
        adj = D4Plugin().adjacency()
        i, j = 0, adj[0][0]
        store.set_edge_weight(i, j, 7.7)
        assert abs(store.get_edge_weight(i, j) - 7.7) < 1e-10

    def test_scale_weight(self, store: TopologyStore) -> None:
        adj = D4Plugin().adjacency()
        i, j = 0, adj[0][0]
        store.scale_edge_weight(i, j, 0.9)
        assert abs(store.get_edge_weight(i, j) - 0.9) < 1e-10

    def test_nonexistent_edge(self, store: TopologyStore) -> None:
        assert store.get_edge_weight(999, 998) == 0.0


class TestPersistence:
    def test_save_load_roundtrip(self, store: TopologyStore) -> None:
        store.add_node(_make_node("persist", cell=3))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            store.save(path)

            store2 = TopologyStore(D4Plugin())
            store2.load(path)

            node = store2.get_node("node-persist")
            assert node is not None
            assert node.text == "persist"
            np.testing.assert_allclose(
                node.embedding, store.get_node("node-persist").embedding
            )
        finally:
            os.unlink(path)

    def test_edge_weights_persisted(self, store: TopologyStore) -> None:
        adj = D4Plugin().adjacency()
        i, j = 0, adj[0][0]
        store.set_edge_weight(i, j, 42.0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            store.save(path)
            store2 = TopologyStore(D4Plugin())
            store2.load(path)
            assert abs(store2.get_edge_weight(i, j) - 42.0) < 1e-10
        finally:
            os.unlink(path)


class TestStats:
    def test_stats(self, store: TopologyStore) -> None:
        store.add_node(_make_node("s1", cell=0, embedding_seed=1))
        store.add_node(_make_node("s2", cell=0, embedding_seed=2))
        store.add_node(_make_node("s3", cell=5, embedding_seed=3))

        stats = store.stats()
        assert stats["total_nodes"] == 3
        assert stats["occupied_cells"] == 2
        assert stats["total_cells"] == 24

    def test_touch_updates_access(self) -> None:
        node = _make_node("touch")
        old_count = node.access_count
        node.touch()
        assert node.access_count == old_count + 1
