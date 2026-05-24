"""Tests for SQLite + numpy mmap persistent backend and auto-migration."""

import shutil
import tempfile

import numpy as np
import pytest

from astrum_verum.lattice.d4 import D4Plugin
from astrum_verum.persistent import SqliteMmapBackend
from astrum_verum.store import MemoryNode, TopologyStore


def _make_node(
    text: str = "test",
    cell: int = 0,
    seed: int = 42,
    node_id: str | None = None,
) -> MemoryNode:
    rng = np.random.default_rng(seed)
    return MemoryNode(
        id=node_id or f"node-{text}-{seed}",
        text=text,
        embedding=rng.standard_normal(384),
        lattice_coords=rng.standard_normal(4),
        cell_memberships={cell: 0.8, (cell + 1) % 24: 0.2},
    )


# =====================================================================
# SqliteMmapBackend — direct tests
# =====================================================================
class TestSqliteMmapBackend:
    @pytest.fixture
    def tmpdir(self):
        d = tempfile.mkdtemp(prefix="astrum_test_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def backend(self, tmpdir) -> SqliteMmapBackend:
        return SqliteMmapBackend(tmpdir, embedding_dim=384, lattice_dim=4)

    def test_add_and_get(self, backend: SqliteMmapBackend) -> None:
        node = _make_node("hello", seed=1)
        backend.add_node(node)
        got = backend.get_node(node.id)
        assert got is not None
        assert got.text == "hello"
        np.testing.assert_allclose(got.embedding, node.embedding)
        np.testing.assert_allclose(got.lattice_coords, node.lattice_coords)

    def test_get_nonexistent(self, backend: SqliteMmapBackend) -> None:
        assert backend.get_node("nope") is None

    def test_get_nodes_in_cell(self, backend: SqliteMmapBackend) -> None:
        backend.add_node(_make_node("a", cell=3, seed=1))
        backend.add_node(_make_node("b", cell=3, seed=2))
        backend.add_node(_make_node("c", cell=7, seed=3))
        nodes = backend.get_nodes_in_cell(3)
        assert len(nodes) == 2
        assert {n.text for n in nodes} == {"a", "b"}

    def test_get_all_nodes(self, backend: SqliteMmapBackend) -> None:
        backend.add_node(_make_node("x", seed=10))
        backend.add_node(_make_node("y", seed=11))
        assert len(backend.get_all_nodes()) == 2

    def test_remove_node(self, backend: SqliteMmapBackend) -> None:
        node = _make_node("rm", seed=99)
        backend.add_node(node)
        assert backend.remove_node(node.id) is True
        assert backend.get_node(node.id) is None
        assert backend.remove_node(node.id) is False

    def test_touch_node(self, backend: SqliteMmapBackend) -> None:
        node = _make_node("touch", seed=50)
        backend.add_node(node)
        old_count = node.access_count

        backend.touch_node(node.id)
        got = backend.get_node(node.id)
        assert got.access_count == old_count + 1
        assert got.last_accessed >= node.last_accessed

    def test_count(self, backend: SqliteMmapBackend) -> None:
        assert backend.count() == 0
        backend.add_node(_make_node(seed=1))
        backend.add_node(_make_node(seed=2))
        assert backend.count() == 2

    def test_cell_counts(self, backend: SqliteMmapBackend) -> None:
        backend.add_node(_make_node("a", cell=0, seed=1))
        backend.add_node(_make_node("b", cell=0, seed=2))
        backend.add_node(_make_node("c", cell=5, seed=3))
        counts = backend.cell_counts()
        assert counts.get(0) == 2
        assert counts.get(5) == 1

    def test_batch_insert(self, backend: SqliteMmapBackend) -> None:
        nodes = [_make_node(f"batch-{i}", cell=i % 24, seed=i) for i in range(100)]
        backend.add_nodes_batch(nodes)
        assert backend.count() == 100
        assert backend.get_node(nodes[50].id) is not None

    def test_edge_weights_roundtrip(self, backend: SqliteMmapBackend) -> None:
        import networkx as nx

        G = nx.Graph()
        G.add_edge(0, 1, weight=3.14)
        G.add_edge(2, 3, weight=2.71)

        backend.save_edge_weights(G)

        G2 = nx.Graph()
        G2.add_edge(0, 1, weight=1.0)
        G2.add_edge(2, 3, weight=1.0)
        backend.load_edge_weights(G2)

        assert abs(G2[0][1]["weight"] - 3.14) < 1e-10
        assert abs(G2[2][3]["weight"] - 2.71) < 1e-10

    def test_embedding_integrity(self, backend: SqliteMmapBackend) -> None:
        """Embeddings should survive write → flush → read cycle."""
        node = _make_node("integrity", seed=77)
        backend.add_node(node)
        backend.flush()

        got = backend.get_node(node.id)
        np.testing.assert_allclose(got.embedding, node.embedding, atol=1e-15)

    def test_lazy_init_without_dims(self, tmpdir) -> None:
        """Backend without explicit dims should init on first add."""
        backend = SqliteMmapBackend(tmpdir)
        node = _make_node("lazy", seed=1)
        backend.add_node(node)
        assert backend.count() == 1
        got = backend.get_node(node.id)
        np.testing.assert_allclose(got.embedding, node.embedding)

    def test_reopen_existing_db(self, tmpdir) -> None:
        """Data should survive backend close + reopen."""
        b1 = SqliteMmapBackend(tmpdir, embedding_dim=384, lattice_dim=4)
        node = _make_node("persist", seed=42)
        b1.add_node(node)
        b1.close()

        b2 = SqliteMmapBackend(tmpdir)
        assert b2.count() == 1
        got = b2.get_node(node.id)
        assert got.text == "persist"
        np.testing.assert_allclose(got.embedding, node.embedding)
        b2.close()


# =====================================================================
# TopologyStore — auto-migration tests
# =====================================================================
class TestAutoMigration:
    @pytest.fixture
    def tmpdir(self):
        d = tempfile.mkdtemp(prefix="astrum_migrate_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_starts_in_memory(self) -> None:
        store = TopologyStore(D4Plugin())
        assert not store.is_persistent

    def test_explicit_persistent(self, tmpdir) -> None:
        store = TopologyStore(D4Plugin(), storage_dir=tmpdir)
        assert store.is_persistent

    def test_auto_migrate_at_threshold(self, tmpdir) -> None:
        """Store should auto-migrate when threshold is reached."""
        threshold = 100  # small for testing
        store = TopologyStore(
            D4Plugin(),
            storage_dir=None,
            auto_persist_threshold=threshold,
        )
        # Override storage dir to tmpdir for test isolation.
        store._storage_dir = tmpdir

        assert not store.is_persistent

        # Add nodes up to threshold.
        for i in range(threshold):
            store.add_node(_make_node(f"n{i}", cell=i % 24, seed=i))

        # Should have migrated.
        assert store.is_persistent
        assert store.stats()["total_nodes"] == threshold

    def test_data_survives_migration(self, tmpdir) -> None:
        """All nodes should be accessible after migration."""
        threshold = 50
        store = TopologyStore(
            D4Plugin(),
            auto_persist_threshold=threshold,
        )
        store._storage_dir = tmpdir

        nodes = []
        for i in range(threshold):
            node = _make_node(f"surv{i}", cell=i % 24, seed=i + 1000)
            store.add_node(node)
            nodes.append(node)

        # Verify all nodes are still accessible.
        for node in nodes:
            got = store.get_node(node.id)
            assert got is not None, f"Node {node.id} lost after migration"
            assert got.text == node.text

    def test_add_after_migration(self, tmpdir) -> None:
        """Nodes added after migration should go to persistent backend."""
        threshold = 20
        store = TopologyStore(
            D4Plugin(),
            auto_persist_threshold=threshold,
        )
        store._storage_dir = tmpdir

        for i in range(threshold):
            store.add_node(_make_node(f"pre{i}", cell=i % 24, seed=i))

        # Add more after migration.
        post_node = _make_node("post", cell=5, seed=9999)
        store.add_node(post_node)

        got = store.get_node(post_node.id)
        assert got is not None
        assert got.text == "post"
        assert store.stats()["total_nodes"] == threshold + 1

    def test_edge_weights_preserved_after_migration(self, tmpdir) -> None:
        """Edge weights should survive migration."""
        threshold = 20
        store = TopologyStore(
            D4Plugin(),
            auto_persist_threshold=threshold,
        )
        store._storage_dir = tmpdir

        adj = D4Plugin().adjacency()
        i, j = 0, adj[0][0]
        store.set_edge_weight(i, j, 42.0)

        for k in range(threshold):
            store.add_node(_make_node(f"ew{k}", cell=k % 24, seed=k))

        # Edge weight should still be 42.0.
        assert abs(store.get_edge_weight(i, j) - 42.0) < 1e-10

    def test_touch_node_persistent(self, tmpdir) -> None:
        """touch_node should work in persistent mode."""
        store = TopologyStore(D4Plugin(), storage_dir=tmpdir)
        node = _make_node("touchp", seed=1)
        store.add_node(node)

        store.touch_node(node.id)
        got = store.get_node(node.id)
        assert got.access_count == 1

    def test_touch_node_in_memory(self) -> None:
        """touch_node should work in in-memory mode."""
        store = TopologyStore(D4Plugin())
        node = _make_node("touchm", seed=1)
        store.add_node(node)

        store.touch_node(node.id)
        got = store.get_node(node.id)
        assert got.access_count == 1

    def test_stats_persistent(self, tmpdir) -> None:
        store = TopologyStore(D4Plugin(), storage_dir=tmpdir)
        store.add_node(_make_node("s1", cell=0, seed=1))
        store.add_node(_make_node("s2", cell=0, seed=2))
        store.add_node(_make_node("s3", cell=5, seed=3))

        stats = store.stats()
        assert stats["total_nodes"] == 3
        assert stats["persistent"] is True
        assert stats["occupied_cells"] == 2

    def test_remove_persistent(self, tmpdir) -> None:
        store = TopologyStore(D4Plugin(), storage_dir=tmpdir)
        node = _make_node("rmp", seed=1)
        store.add_node(node)
        assert store.remove_node(node.id) is True
        assert store.get_node(node.id) is None
