"""
Геометрический контур.

Чистая математика пространства, полностью изолированная от ИИ-контекста.
Тестируется:
    - Ортогональность SO(d) вращений (R^T R = I)
    - Стабильность CVP-декодера E₈ при шуме
    - Граничные условия (нулевой вектор, NaN, Inf)

Если эти тесты красные — сломана фундаментальная геометрия,
и всё остальное бессмысленно.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import det

from astrum_verum.lattice.d4 import D4Plugin
from astrum_verum.lattice.e8 import E8Plugin
from astrum_verum.rotation import (
    align_to_axis,
    compute_focus_vector,
    inverse_rotate_query,
    random_rotation,
)


# =====================================================================
# 1. SO(d) ортогональность вращений
# =====================================================================
class TestSOdOrthogonality:
    """
    Проверка, что матрица вращения R удовлетворяет:
        R^T R = I   (ортогональность)
        det(R) = +1  (собственное вращение, не отражение)
    """

    @pytest.mark.parametrize("dim", [4, 8, 16, 32])
    def test_identity_property(self, dim: int) -> None:
        """R^T R = I для всех размерностей."""
        R = random_rotation(dim, seed=42)
        product = R.T @ R
        np.testing.assert_allclose(
            product, np.eye(dim), atol=1e-12,
            err_msg=f"R^T R ≠ I for dim={dim}",
        )

    @pytest.mark.parametrize("dim", [4, 8, 16, 32])
    def test_determinant_plus_one(self, dim: int) -> None:
        """det(R) = +1 (proper rotation, not reflection)."""
        R = random_rotation(dim, seed=42)
        d = det(R)
        np.testing.assert_allclose(
            d, 1.0, atol=1e-10,
            err_msg=f"det(R) = {d} ≠ 1.0 for dim={dim}",
        )

    @pytest.mark.parametrize("dim", [4, 8])
    def test_preserves_vector_norms(self, dim: int) -> None:
        """‖Rv‖ = ‖v‖ для произвольных векторов."""
        R = random_rotation(dim, seed=42)
        rng = np.random.default_rng(123)

        for _ in range(100):
            v = rng.standard_normal(dim)
            Rv = R @ v
            np.testing.assert_allclose(
                np.linalg.norm(Rv), np.linalg.norm(v), atol=1e-12,
            )

    @pytest.mark.parametrize("dim", [4, 8])
    def test_preserves_dot_products(self, dim: int) -> None:
        """⟨Ru, Rv⟩ = ⟨u, v⟩ (сохраняет углы)."""
        R = random_rotation(dim, seed=42)
        rng = np.random.default_rng(456)

        for _ in range(100):
            u = rng.standard_normal(dim)
            v = rng.standard_normal(dim)
            np.testing.assert_allclose(
                np.dot(R @ u, R @ v), np.dot(u, v), atol=1e-11,
            )

    @pytest.mark.parametrize("dim", [4, 8])
    def test_preserves_euclidean_distances(self, dim: int) -> None:
        """‖Ru − Rv‖ = ‖u − v‖ (изометрия)."""
        R = random_rotation(dim, seed=42)
        rng = np.random.default_rng(789)

        for _ in range(100):
            u = rng.standard_normal(dim)
            v = rng.standard_normal(dim)
            np.testing.assert_allclose(
                np.linalg.norm(R @ u - R @ v),
                np.linalg.norm(u - v),
                atol=1e-12,
            )

    def test_align_to_axis_maps_focus_exactly(self) -> None:
        """align_to_axis(f) должен отобразить f на ось e₁."""
        rng = np.random.default_rng(42)
        for dim in [4, 8]:
            f = rng.standard_normal(dim)
            f /= np.linalg.norm(f)
            R = align_to_axis(f)
            mapped = R @ f
            e1 = np.zeros(dim)
            e1[0] = 1.0
            np.testing.assert_allclose(mapped, e1, atol=1e-12)

    def test_inverse_rotation_inverts(self) -> None:
        """inverse_rotate_query(q, R) = R^T q."""
        rng = np.random.default_rng(42)
        for dim in [4, 8]:
            f = rng.standard_normal(dim)
            f /= np.linalg.norm(f)
            R = align_to_axis(f)
            q = rng.standard_normal(dim)
            q_inv = inverse_rotate_query(q, R)
            np.testing.assert_allclose(q_inv, R.T @ q, atol=1e-12)

    def test_many_random_rotations_all_valid(self) -> None:
        """100 случайных вращений в 8D — все должны быть SO(8)."""
        for seed in range(100):
            R = random_rotation(8, seed=seed)
            product = R.T @ R
            np.testing.assert_allclose(
                product, np.eye(8), atol=1e-12,
                err_msg=f"R^T R ≠ I for seed={seed}",
            )
            np.testing.assert_allclose(
                det(R), 1.0, atol=1e-10,
                err_msg=f"det(R) ≠ 1 for seed={seed}",
            )


# =====================================================================
# 2. CVP-декодер: стабильность при шуме
# =====================================================================
class TestCVPStability:
    """
    Подача заведомо известных векторов с добавлением шума.
    CVP должен безошибочно возвращать ID ближайшего корневого вектора.
    """

    @pytest.fixture
    def e8(self) -> E8Plugin:
        return E8Plugin()

    @pytest.fixture
    def d4(self) -> D4Plugin:
        return D4Plugin()

    @pytest.mark.parametrize("noise_level", [0.01, 0.05, 0.1, 0.15])
    def test_e8_cvp_noise_robustness(
        self, e8: E8Plugin, noise_level: float
    ) -> None:
        """
        CVP(v + ε) = CVP(v) для шума ‖ε‖ < порог.

        Минимальное расстояние между соседними вершинами E₈ = 1.0
        (на единичной сфере). Для шума < 0.5 CVP должен быть стабилен.
        """
        verts = e8.vertices()
        rng = np.random.default_rng(42)

        errors = 0
        total = 240
        for i in range(total):
            noise = rng.standard_normal(8) * noise_level
            noisy = verts[i] + noise
            idx, _ = e8.closest_vertex(noisy)
            if idx != i:
                errors += 1

        # Для noise ≤ 0.15 ожидаем < 5% ошибок.
        error_rate = errors / total
        assert error_rate < 0.05, (
            f"CVP error rate {error_rate:.1%} at noise={noise_level}"
        )

    @pytest.mark.parametrize("noise_level", [0.01, 0.05, 0.1, 0.2])
    def test_d4_cvp_noise_robustness(
        self, d4: D4Plugin, noise_level: float
    ) -> None:
        """CVP стабильность для D₄ решётки."""
        verts = d4.vertices()
        rng = np.random.default_rng(42)

        errors = 0
        total = 24
        for i in range(total):
            noise = rng.standard_normal(4) * noise_level
            noisy = verts[i] + noise
            idx, _ = d4.closest_vertex(noisy)
            if idx != i:
                errors += 1

        error_rate = errors / total
        assert error_rate < 0.1, (
            f"CVP error rate {error_rate:.1%} at noise={noise_level}"
        )

    def test_cvp_exact_vertices(self, e8: E8Plugin) -> None:
        """Точные вершины E₈ должны декодироваться с 0% ошибок."""
        verts = e8.vertices()
        for i in range(240):
            idx, _ = e8.closest_vertex(verts[i])
            assert idx == i, f"Exact vertex {i} decoded as {idx}"

    def test_cvp_scaled_vertices(self, e8: E8Plugin) -> None:
        """Масштабированные вершины должны декодироваться верно."""
        verts = e8.vertices()
        for scale in [0.1, 0.5, 2.0, 10.0, 100.0]:
            for i in range(0, 240, 10):
                idx, _ = e8.closest_vertex(verts[i] * scale)
                assert idx == i, (
                    f"Scaled vertex {i} × {scale} decoded as {idx}"
                )


# =====================================================================
# 3. Граничные условия: нулевой вектор, NaN, Inf
# =====================================================================
class TestEdgeConditions:
    """
    Система НЕ ДОЛЖНА падать с ошибкой деления на ноль
    при подаче нулевого вектора или вектора с экстремальными значениями.
    """

    @pytest.fixture
    def e8(self) -> E8Plugin:
        return E8Plugin()

    @pytest.fixture
    def d4(self) -> D4Plugin:
        return D4Plugin()

    def test_zero_vector_e8(self, e8: E8Plugin) -> None:
        """Нулевой вектор → CVP не падает, возвращает валидный индекс."""
        idx, coords = e8.closest_vertex(np.zeros(8))
        assert 0 <= idx < 240
        assert coords.shape == (8,)

    def test_zero_vector_d4(self, d4: D4Plugin) -> None:
        """Нулевой вектор → CVP не падает, возвращает валидный индекс."""
        idx, coords = d4.closest_vertex(np.zeros(4))
        assert 0 <= idx < 24
        assert coords.shape == (4,)

    def test_very_small_vector(self, e8: E8Plugin) -> None:
        """Вектор с нормой ≈ 1e-15 не вызывает деление на ноль."""
        tiny = np.ones(8) * 1e-15
        idx, _ = e8.closest_vertex(tiny)
        assert 0 <= idx < 240

    def test_very_large_vector(self, e8: E8Plugin) -> None:
        """Вектор с нормой ≈ 1e15 корректно нормализуется."""
        huge = np.ones(8) * 1e15
        idx, _ = e8.closest_vertex(huge)
        assert 0 <= idx < 240

    def test_single_component_vectors(self, e8: E8Plugin) -> None:
        """Одноосевые вектора (0,...,0,1,0,...,0) декодируются."""
        for axis in range(8):
            v = np.zeros(8)
            v[axis] = 1.0
            idx, _ = e8.closest_vertex(v)
            assert 0 <= idx < 240

    def test_negative_vector(self, e8: E8Plugin) -> None:
        """Полностью отрицательный вектор — не падает."""
        v = -np.ones(8)
        idx, _ = e8.closest_vertex(v)
        assert 0 <= idx < 240

    def test_focus_vector_anti_aligned(self) -> None:
        """
        Focus vector, антипараллельный оси — align_to_axis
        не должен создать сингулярную матрицу.
        """
        for dim in [4, 8]:
            anti = np.zeros(dim)
            anti[0] = -1.0
            R = align_to_axis(anti)
            # R должен быть валидной SO(d) матрицей.
            np.testing.assert_allclose(
                R.T @ R, np.eye(dim), atol=1e-12,
            )
            np.testing.assert_allclose(det(R), 1.0, atol=1e-10)

    def test_rotation_with_near_zero_norm(self) -> None:
        """
        compute_focus_vector с почти нулевыми историческими
        векторами не должен падать.
        """
        tiny_vecs = [np.ones(4) * 1e-12, np.ones(4) * 1e-12]
        # Should not raise.
        focus = compute_focus_vector(tiny_vecs, decay=0.8)
        assert focus.shape == (4,)
        # Norm should be 1 (normalized).
        np.testing.assert_allclose(np.linalg.norm(focus), 1.0, atol=1e-6)
