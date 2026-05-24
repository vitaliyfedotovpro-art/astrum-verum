"""Tests for the SO(d) rotation engine."""

import numpy as np
import pytest

from astrum_verum.rotation import (
    align_to_axis,
    compute_focus_vector,
    inverse_rotate_query,
    random_rotation,
)


class TestAlignToAxis:
    def test_identity_when_already_aligned(self) -> None:
        """If focus == target axis, R should be identity."""
        focus = np.array([0, 0, 0, 1.0])
        R = align_to_axis(focus, axis=-1)
        np.testing.assert_allclose(R, np.eye(4), atol=1e-10)

    def test_maps_focus_to_axis(self) -> None:
        """R @ focus should equal the target axis unit vector."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            focus = rng.standard_normal(4)
            focus /= np.linalg.norm(focus)
            R = align_to_axis(focus, axis=-1)
            result = R @ focus
            expected = np.array([0, 0, 0, 1.0])
            np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_orthogonal(self) -> None:
        """R must be orthogonal: R⊺R = I."""
        rng = np.random.default_rng(99)
        for _ in range(50):
            focus = rng.standard_normal(4)
            focus /= np.linalg.norm(focus)
            R = align_to_axis(focus)
            np.testing.assert_allclose(R.T @ R, np.eye(4), atol=1e-10)

    def test_determinant_plus_one(self) -> None:
        """R must be a proper rotation: det(R) = +1."""
        rng = np.random.default_rng(7)
        for _ in range(50):
            focus = rng.standard_normal(4)
            focus /= np.linalg.norm(focus)
            R = align_to_axis(focus)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-8)

    def test_preserves_distances(self) -> None:
        """Rotation must preserve Euclidean distances between any two points."""
        rng = np.random.default_rng(55)
        focus = rng.standard_normal(4)
        focus /= np.linalg.norm(focus)
        R = align_to_axis(focus)

        a = rng.standard_normal(4)
        b = rng.standard_normal(4)
        dist_before = np.linalg.norm(a - b)
        dist_after = np.linalg.norm(R @ a - R @ b)
        np.testing.assert_allclose(dist_before, dist_after, atol=1e-10)

    def test_anti_aligned(self) -> None:
        """Focus pointing opposite to target should still produce valid SO(4)."""
        focus = np.array([0, 0, 0, -1.0])
        R = align_to_axis(focus, axis=-1)
        result = R @ focus
        np.testing.assert_allclose(result, [0, 0, 0, 1.0], atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-8)

    def test_different_dimensions(self) -> None:
        """Should work for any dimensionality, not just 4."""
        for d in (3, 4, 8, 24):
            rng = np.random.default_rng(d)
            focus = rng.standard_normal(d)
            focus /= np.linalg.norm(focus)
            R = align_to_axis(focus, axis=-1)
            result = R @ focus
            expected = np.zeros(d)
            expected[-1] = 1.0
            np.testing.assert_allclose(result, expected, atol=1e-7)


class TestInverseRotateQuery:
    def test_inverse_rotation(self) -> None:
        """R⊺ @ (R @ x) should give x back."""
        rng = np.random.default_rng(13)
        focus = rng.standard_normal(4)
        focus /= np.linalg.norm(focus)
        R = align_to_axis(focus)

        query = rng.standard_normal(4)
        rotated = R @ query
        recovered = inverse_rotate_query(rotated, R)
        np.testing.assert_allclose(recovered, query, atol=1e-10)


class TestComputeFocusVector:
    def test_single_vector(self) -> None:
        """With one vector, focus should equal that vector normalized."""
        v = np.array([3.0, 0.0, 4.0, 0.0])
        focus = compute_focus_vector([v])
        expected = v / np.linalg.norm(v)
        np.testing.assert_allclose(focus, expected, atol=1e-10)

    def test_unit_norm(self) -> None:
        """Focus vector must always be unit norm."""
        rng = np.random.default_rng(21)
        vecs = [rng.standard_normal(4) for _ in range(5)]
        focus = compute_focus_vector(vecs)
        np.testing.assert_allclose(np.linalg.norm(focus), 1.0, atol=1e-10)

    def test_recent_has_more_weight(self) -> None:
        """The most recent vector should dominate the focus."""
        old = np.array([1.0, 0, 0, 0])
        new = np.array([0, 0, 0, 1.0])
        focus = compute_focus_vector([old, new], decay=0.1)
        # With decay=0.1, old gets weight 0.1, new gets weight 1.0.
        # Focus should be much closer to `new`.
        assert focus[3] > focus[0]

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_focus_vector([])


class TestRandomRotation:
    def test_orthogonal(self) -> None:
        R = random_rotation(4, rng=np.random.default_rng(0))
        np.testing.assert_allclose(R.T @ R, np.eye(4), atol=1e-10)

    def test_det_plus_one(self) -> None:
        R = random_rotation(4, rng=np.random.default_rng(1))
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-8)

    def test_different_seeds_different_rotations(self) -> None:
        R1 = random_rotation(4, rng=np.random.default_rng(0))
        R2 = random_rotation(4, rng=np.random.default_rng(1))
        assert not np.allclose(R1, R2)
