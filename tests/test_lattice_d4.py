"""Tests for the D₄ (24-cell) lattice plugin."""

import numpy as np
import pytest

from astrum_verum.lattice.d4 import D4Plugin


@pytest.fixture
def d4() -> D4Plugin:
    return D4Plugin()


class TestD4Metadata:
    def test_info(self, d4: D4Plugin) -> None:
        info = d4.info()
        assert info.name == "D4"
        assert info.dimension == 4
        assert info.num_vertices == 24
        assert info.num_edges == 96
        assert info.neighbors_per_vertex == 8

    def test_repr(self, d4: D4Plugin) -> None:
        assert "D4" in repr(d4)


class TestD4Vertices:
    def test_shape(self, d4: D4Plugin) -> None:
        v = d4.vertices()
        assert v.shape == (24, 4)

    def test_on_unit_sphere(self, d4: D4Plugin) -> None:
        """All vertices must lie on the unit 3-sphere."""
        v = d4.vertices()
        norms = np.linalg.norm(v, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_unique(self, d4: D4Plugin) -> None:
        """All 24 vertices must be distinct."""
        v = d4.vertices()
        for i in range(24):
            for j in range(i + 1, 24):
                dist = np.linalg.norm(v[i] - v[j])
                assert dist > 1e-10, f"Vertices {i} and {j} are duplicates"

    def test_returns_copy(self, d4: D4Plugin) -> None:
        """Modifying returned array must not affect internal state."""
        v1 = d4.vertices()
        v1[0] = [999, 999, 999, 999]
        v2 = d4.vertices()
        assert not np.allclose(v1[0], v2[0])


class TestD4Adjacency:
    def test_all_vertices_present(self, d4: D4Plugin) -> None:
        adj = d4.adjacency()
        assert set(adj.keys()) == set(range(24))

    def test_degree_8(self, d4: D4Plugin) -> None:
        """Every vertex in the 24-cell has exactly 8 neighbors."""
        adj = d4.adjacency()
        for i, neighbors in adj.items():
            assert len(neighbors) == 8, f"Vertex {i}: degree {len(neighbors)} != 8"

    def test_symmetric(self, d4: D4Plugin) -> None:
        """If i is neighbor of j, then j must be neighbor of i."""
        adj = d4.adjacency()
        for i, neighbors in adj.items():
            for j in neighbors:
                assert i in adj[j], f"Edge ({i},{j}) is not symmetric"

    def test_edge_length(self, d4: D4Plugin) -> None:
        """All edges must have Euclidean length == 1.0."""
        v = d4.vertices()
        adj = d4.adjacency()
        for i, neighbors in adj.items():
            for j in neighbors:
                dist = np.linalg.norm(v[i] - v[j])
                np.testing.assert_allclose(dist, 1.0, atol=1e-10)

    def test_total_edges_96(self, d4: D4Plugin) -> None:
        adj = d4.adjacency()
        total = sum(len(v) for v in adj.values()) // 2
        assert total == 96


class TestD4CVP:
    def test_vertex_maps_to_itself(self, d4: D4Plugin) -> None:
        """CVP on an exact vertex must return that vertex."""
        v = d4.vertices()
        for i in range(24):
            idx, coords = d4.closest_vertex(v[i])
            assert idx == i
            np.testing.assert_allclose(coords, v[i])

    def test_scaled_vertex(self, d4: D4Plugin) -> None:
        """A scaled version of a vertex should still map to that vertex."""
        v = d4.vertices()
        for i in range(24):
            idx, _ = d4.closest_vertex(v[i] * 3.7)
            assert idx == i

    def test_midpoint_returns_one_of_two(self, d4: D4Plugin) -> None:
        """Midpoint of two adjacent vertices should map to one of them."""
        v = d4.vertices()
        adj = d4.adjacency()
        i = 0
        j = adj[0][0]
        midpoint = (v[i] + v[j]) / 2.0
        idx, _ = d4.closest_vertex(midpoint)
        assert idx in (i, j)

    def test_random_points(self, d4: D4Plugin) -> None:
        """CVP on random points should return a valid vertex index."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            point = rng.standard_normal(4)
            idx, coords = d4.closest_vertex(point)
            assert 0 <= idx < 24
            np.testing.assert_allclose(coords, d4.vertices()[idx])


class TestD4CellDecode:
    def test_decode_matches_cvp(self, d4: D4Plugin) -> None:
        """For self-dual D₄, decode_cell must match closest_vertex index."""
        rng = np.random.default_rng(123)
        for _ in range(50):
            point = rng.standard_normal(4)
            cell = d4.decode_cell(point)
            idx, _ = d4.closest_vertex(point)
            assert cell == idx
