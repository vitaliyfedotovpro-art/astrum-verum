"""Tests for the E₈ lattice plugin."""

import numpy as np
import pytest

from astrum_verum.lattice.e8 import E8Plugin


@pytest.fixture
def e8() -> E8Plugin:
    return E8Plugin()


class TestE8Metadata:
    def test_info(self, e8: E8Plugin) -> None:
        info = e8.info()
        assert info.name == "E8"
        assert info.dimension == 8
        assert info.num_vertices == 240
        assert info.num_edges == 6720
        assert info.neighbors_per_vertex == 56
        assert info.symmetry_group_order == 696_729_600

    def test_repr(self, e8: E8Plugin) -> None:
        r = repr(e8)
        assert "E8" in r or "E₈" in r


class TestE8Vertices:
    def test_shape(self, e8: E8Plugin) -> None:
        v = e8.vertices()
        assert v.shape == (240, 8)

    def test_on_unit_sphere(self, e8: E8Plugin) -> None:
        """All 240 vertices must lie on the unit 7-sphere."""
        v = e8.vertices()
        norms = np.linalg.norm(v, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_unique(self, e8: E8Plugin) -> None:
        """All 240 vertices must be distinct."""
        v = e8.vertices()
        for i in range(240):
            for j in range(i + 1, 240):
                dist = np.linalg.norm(v[i] - v[j])
                assert dist > 1e-10, f"Vertices {i} and {j} are duplicates"

    def test_returns_copy(self, e8: E8Plugin) -> None:
        """Modifying returned array must not affect internal state."""
        v1 = e8.vertices()
        v1[0] = 999
        v2 = e8.vertices()
        assert not np.allclose(v1[0], v2[0])

    def test_type1_count(self, e8: E8Plugin) -> None:
        """There should be 112 type-1 vectors (exactly 2 nonzero coords)."""
        v = e8.vertices()
        # After normalization, type-1 vectors have 2 nonzero coords ≈ ±1/√2.
        type1_count = 0
        for i in range(240):
            nonzero = np.sum(np.abs(v[i]) > 0.01)
            if nonzero == 2:
                type1_count += 1
        assert type1_count == 112

    def test_type2_count(self, e8: E8Plugin) -> None:
        """There should be 128 type-2 vectors (all 8 coords nonzero)."""
        v = e8.vertices()
        type2_count = 0
        for i in range(240):
            nonzero = np.sum(np.abs(v[i]) > 0.01)
            if nonzero == 8:
                type2_count += 1
        assert type2_count == 128


class TestE8Adjacency:
    def test_all_vertices_present(self, e8: E8Plugin) -> None:
        adj = e8.adjacency()
        assert set(adj.keys()) == set(range(240))

    def test_degree_56(self, e8: E8Plugin) -> None:
        """Every vertex in E₈ has exactly 56 neighbors."""
        adj = e8.adjacency()
        for i, neighbors in adj.items():
            assert len(neighbors) == 56, (
                f"Vertex {i}: degree {len(neighbors)} != 56"
            )

    def test_symmetric(self, e8: E8Plugin) -> None:
        """If i is neighbor of j, then j must be neighbor of i."""
        adj = e8.adjacency()
        for i, neighbors in adj.items():
            for j in neighbors:
                assert i in adj[j], f"Edge ({i},{j}) is not symmetric"

    def test_edge_length(self, e8: E8Plugin) -> None:
        """Adjacent vertices must have Euclidean distance ≈ 1.0."""
        v = e8.vertices()
        adj = e8.adjacency()
        # Sample every 20th vertex for speed.
        for i in range(0, 240, 20):
            for j in adj[i]:
                dist = np.linalg.norm(v[i] - v[j])
                np.testing.assert_allclose(dist, 1.0, atol=1e-10)

    def test_total_edges_6720(self, e8: E8Plugin) -> None:
        adj = e8.adjacency()
        total = sum(len(v) for v in adj.values()) // 2
        assert total == 6720


class TestE8CVP:
    def test_vertex_maps_to_itself(self, e8: E8Plugin) -> None:
        """CVP on an exact vertex must return that vertex."""
        v = e8.vertices()
        for i in range(0, 240, 10):  # sample for speed
            idx, coords = e8.closest_vertex(v[i])
            assert idx == i
            np.testing.assert_allclose(coords, v[i])

    def test_scaled_vertex(self, e8: E8Plugin) -> None:
        """A scaled vertex should still map to that vertex."""
        v = e8.vertices()
        for i in range(0, 240, 10):
            idx, _ = e8.closest_vertex(v[i] * 5.3)
            assert idx == i

    def test_random_points(self, e8: E8Plugin) -> None:
        """CVP on random 8D points should return a valid vertex index."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            point = rng.standard_normal(8)
            idx, coords = e8.closest_vertex(point)
            assert 0 <= idx < 240
            np.testing.assert_allclose(coords, e8.vertices()[idx])

    def test_zero_point(self, e8: E8Plugin) -> None:
        """CVP on the origin should return vertex 0 (fallback)."""
        idx, _ = e8.closest_vertex(np.zeros(8))
        assert idx == 0


class TestE8CellDecode:
    def test_decode_matches_cvp(self, e8: E8Plugin) -> None:
        """decode_cell must match closest_vertex index."""
        rng = np.random.default_rng(123)
        for _ in range(50):
            point = rng.standard_normal(8)
            cell = e8.decode_cell(point)
            idx, _ = e8.closest_vertex(point)
            assert cell == idx
