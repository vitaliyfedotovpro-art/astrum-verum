"""
VSA core — примитивы Vector Symbolic Architecture (модель MAP, биполярная).

Промотировано из killer-экспериментов Phase 0/1 (оба PASS): алгебра связывания
держит 100+ пар при D=10k и переживает grounding из реальных эмбеддингов
(SimHash, corr 0.988). Здесь — переиспользуемое ядро.

  bind(a, b)   = a ⊙ b            (поэлементное умножение; self-inverse)
  unbind(x, r) = x ⊙ r            (та же операция — MAP биполярна)
  bundle(V)    = sign(Σ V)        (суперпозиция с majority-знаком)
  ground(e, P) = sign(e · P)      (SimHash: embedding → биполярный гипервектор)
"""

from __future__ import annotations

import numpy as np

DEFAULT_D = 10_000


def random_atoms(n: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """n случайных биполярных гипервекторов {-1,+1}^D (квазиортогональны)."""
    return rng.integers(0, 2, size=(n, D), dtype=np.int8).astype(np.float32) * 2.0 - 1.0


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def unbind(x: np.ndarray, role: np.ndarray) -> np.ndarray:
    return x * role  # MAP: связывание самообратно


def permute(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Циклический сдвиг ρ^k — кодирование позиции в последовательности (эпизоды)."""
    return np.roll(x, k)


def unpermute(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Обратная перестановка ρ^{-k}."""
    return np.roll(x, -k)


def bundle(vectors: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Суперпозиция с majority-знаком; ничьи (сумма==0) разбиваются случайно."""
    s = vectors.sum(axis=0)
    out = np.sign(s)
    ties = out == 0
    if ties.any():
        r = rng if rng is not None else np.random.default_rng(0)
        out[ties] = r.integers(0, 2, size=int(ties.sum())) * 2.0 - 1.0
    return out.astype(np.float32)


def make_projection(emb_dim: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """Фиксированная случайная проекция для SimHash-grounding (emb_dim → D)."""
    return rng.standard_normal((emb_dim, D)).astype(np.float32)


def ground(embedding: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """embedding → биполярный гипервектор через SimHash (sign случайной проекции).
    Сохраняет угловую близость: cos(atoms) ≈ 1 − 2·θ/π."""
    atom = np.sign(embedding @ projection).astype(np.float32)
    atom[atom == 0] = 1.0
    return atom


def cosine_to_codebook(vec: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Косинус биполярного vec ко всем строкам codebook (= dot/D на ±1)."""
    return codebook @ vec
