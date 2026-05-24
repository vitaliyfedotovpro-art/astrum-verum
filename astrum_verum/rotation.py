"""
SO(d) Rotation Engine for Astrum Verum.

Handles rotation of the attention focus vector in d-dimensional space.
Uses the optimized *inverse rotation* strategy: instead of rotating all N
memory nodes (O(N·d²)), we rotate only the query vector in the opposite
direction (O(d²)), achieving the same ranking result.

Mathematical basis:
- In 4D, rotations belong to SO(4) and are characterized by *two*
  independent rotation angles in two orthogonal planes (e.g. XY and ZW).
- We use the Gram–Schmidt-based alignment method: given a focus vector F,
  compute R ∈ SO(d) such that R·F = (0, 0, ..., 0, 1) (the search axis).
- For search, apply R⊺ to the query instead: q' = R⊺·q.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def align_to_axis(
    focus: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """
    Compute rotation matrix R ∈ SO(d) that maps ``focus`` to the unit vector
    along ``axis`` (default: last axis, i.e. the W-axis in 4D).

    Uses the Rodrigues / Givens approach: construct the rotation in the 2D
    plane spanned by ``focus`` and ``e_axis`` by angle θ where cos θ = ⟨focus, e⟩.
    This always yields a proper rotation (det = +1) and exactly maps focus → e.

    Args:
        focus: Vector in ℝ^d (need not be unit), shape ``(d,)``.
        axis: Which canonical axis to align to (default -1 = last).

    Returns:
        Rotation matrix R of shape ``(d, d)`` such that ``R @ focus ≈ e_axis``.
    """
    d = len(focus)
    focus = np.asarray(focus, dtype=np.float64)
    norm = np.linalg.norm(focus)
    if norm < 1e-12:
        return np.eye(d)
    f = focus / norm  # unit focus

    # Target axis unit vector.
    e = np.zeros(d, dtype=np.float64)
    e[axis % d] = 1.0

    cos_theta = float(np.dot(f, e))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Already aligned.
    if cos_theta > 1.0 - 1e-12:
        return np.eye(d)

    # Anti-aligned: 180° rotation in the plane containing e and any
    # orthogonal vector.
    if cos_theta < -1.0 + 1e-12:
        # Pick an arbitrary vector not parallel to e.
        perp = np.zeros(d, dtype=np.float64)
        alt = 0 if (axis % d) != 0 else 1
        perp[alt] = 1.0
        # Gram-Schmidt to make it orthogonal to e.
        perp = perp - np.dot(perp, e) * e
        perp /= np.linalg.norm(perp)
        # 180° rotation in the (e, perp) plane: R = I - 2(ee⊺ + pp⊺)... 
        # Simpler: flip e and perp.
        R = np.eye(d) - 2.0 * np.outer(e, e) - 2.0 * np.outer(perp, perp)
        # This flips both e and perp, which is a 180° rotation (det = +1).
        return R

    # General case: rotation in the (f, e) plane by angle θ.
    # Decompose: find the component of f orthogonal to e.
    f_perp = f - cos_theta * e
    f_perp_norm = np.linalg.norm(f_perp)
    if f_perp_norm < 1e-12:
        return np.eye(d)
    u = f_perp / f_perp_norm  # unit vector orthogonal to e, in the rotation plane

    sin_theta = float(np.sqrt(1.0 - cos_theta ** 2))

    # The rotation in the 2D plane spanned by (u, e) that maps f → e:
    #   R = I + (cos θ - 1)(uu⊺ + ee⊺) + sin θ (eu⊺ - ue⊺)
    # where θ is the angle FROM f TO e.
    R = (
        np.eye(d)
        + (cos_theta - 1.0) * (np.outer(u, u) + np.outer(e, e))
        + sin_theta * (np.outer(e, u) - np.outer(u, e))
    )

    return R


def inverse_rotate_query(
    query_4d: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """
    Apply the *inverse* rotation to a query vector.

    Instead of rotating all memory node coordinates by R (O(N·d²)),
    we rotate the query by R⊺ (O(d²)) and search in the original space.

    Args:
        query_4d: Query point in ℝ^d, shape ``(d,)``.
        rotation: Rotation matrix R ∈ SO(d), shape ``(d, d)``.

    Returns:
        Rotated query vector R⊺ · q, shape ``(d,)``.
    """
    return rotation.T @ query_4d


def compute_focus_vector(
    recent_vectors: list[np.ndarray],
    decay: float = 0.8,
) -> np.ndarray:
    """
    Compute the aggregate focus vector from recent context vectors.

    Uses exponentially decaying weights: the most recent vector has
    weight 1.0, the one before it has weight `decay`, then `decay²`, etc.

    Args:
        recent_vectors: List of d-dimensional vectors (most recent last).
        decay: Exponential decay factor in (0, 1).

    Returns:
        Normalized focus vector in ℝ^d.
    """
    if not recent_vectors:
        raise ValueError("Need at least one context vector to compute focus.")

    d = len(recent_vectors[0])
    n = len(recent_vectors)

    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    weights /= weights.sum()

    focus = np.zeros(d, dtype=np.float64)
    for w, v in zip(weights, recent_vectors):
        focus += w * np.asarray(v, dtype=np.float64)

    norm = np.linalg.norm(focus)
    if norm < 1e-12:
        # Degenerate: all vectors cancelled out.  Default to last vector.
        return np.asarray(recent_vectors[-1], dtype=np.float64) / np.linalg.norm(
            recent_vectors[-1]
        )

    return focus / norm


def random_rotation(
    d: int,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a random rotation matrix in SO(d) via QR decomposition
    of a random Gaussian matrix (Haar measure on SO(d)).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    Z = rng.standard_normal((d, d))
    Q, R_ = qr(Z)
    # Ensure det = +1 (QR can give det = -1).
    Q = Q @ np.diag(np.sign(np.diag(R_)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q
