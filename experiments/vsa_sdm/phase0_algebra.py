"""
Astrum Verum — Phase 0 killer-эксперимент (чистый numpy).

Цель: ДЁШЕВО подтвердить или убить алгебру VSA/SDM на ЧИСТЫХ (случайных) атомах,
ДО борьбы с semantic-grounding'ом. Если учебник не работает даже на идеальных
квазиортогональных атомах — дальше идти незачем.

Два теста, пороги pass/fail зафиксированы ЗАРАНЕЕ (см. THRESHOLDS):

  A. Ёмкость binding (MAP, биполярная {-1,+1}).
     Role-filler факты, свёрнутые в одну запись; восстановление филлера
     разсвязыванием + cleanup. Это «композиционная память» — то, чего
     косинус-RAG не умеет (у него нет алгебры).

  B. Cleanup-память — error-correcting аттрактор.
     Восстановление точного паттерна из ПОВРЕЖДЁННОГО ключа с итеративным
     чтением (вход = выход). Реализована как modern Hopfield = attention
     (Ramsauer 2020): retrieve = sign(Xᵀ·softmax(X·s/√n)). Это «гиппокамп»:
     сходимость к канону, прямой мост к attention (Bricken & Pehlevan, 2021).

     [Прим. Phase 0: наивный SDM Канервы на равномерно-случайных hard-локациях
      попадает в хрупкий режим (концентрация меры в бинарном кубе → нет
      локальности, активное множество = тонкий хвост, сдвиг ключа его рушит).
      Современная и более мощная инстанциация той же идеи аттрактора — Hopfield;
      её и берём, она же — мост к attention.]

Запуск:  python experiments/vsa_sdm/phase0_algebra.py
Пишет:   experiments/vsa_sdm/phase0_results.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Пороги pass/fail — зафиксированы до запуска.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "A_acc_at_P10": 0.95,   # точность восстановления филлера при 10 парах в записи
    "B_recovery_at_f20": 0.95,  # точное восстановление при 20% повреждении (iterated)
}

OUT_MD = Path(__file__).with_name("phase0_results.md")


# ===========================================================================
# VSA-примитивы (модель MAP: Multiply-Add-Permute, биполярная)
# ===========================================================================
def make_atoms(n: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """n случайных квазиортогональных гипервекторов в {-1,+1}^D."""
    return rng.integers(0, 2, size=(n, D), dtype=np.int8).astype(np.float32) * 2.0 - 1.0


def bundle_sign(vectors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Суперпозиция (bundling) с majority-знаком; ничьи (сумма==0) — случайно."""
    s = vectors.sum(axis=0)
    out = np.sign(s)
    ties = out == 0
    if ties.any():
        out[ties] = rng.integers(0, 2, size=int(ties.sum())) * 2.0 - 1.0
    return out.astype(np.float32)


# ===========================================================================
# Эксперимент A — ёмкость binding
# ===========================================================================
def experiment_a(
    D: int = 10_000,
    R: int = 512,
    N: int = 512,
    Ps: tuple[int, ...] = (1, 3, 5, 8, 10, 20, 40, 80, 150, 250, 400, 500),
    trials: int = 60,
    seed: int = 0,
) -> dict[int, float]:
    rng = np.random.default_rng(seed)
    roles = make_atoms(R, D, rng)
    fillers = make_atoms(N, D, rng)

    acc_by_p: dict[int, float] = {}
    for P in Ps:
        if P > R or P > N:
            continue
        correct = 0
        total = 0
        for _ in range(trials):
            role_idx = rng.choice(R, P, replace=False)
            filler_idx = rng.choice(N, P, replace=False)
            # факт = role ⊗ filler;  запись = bundle всех P фактов
            pairs = roles[role_idx] * fillers[filler_idx]          # (P, D)
            record = bundle_sign(pairs, rng)                       # (D,)
            # запрос: record ⊗ role_j  ≈ filler_j + шум  → cleanup
            queries = record[None, :] * roles[role_idx]            # (P, D)
            sims = queries @ fillers.T                             # (P, N)
            preds = np.argmax(sims, axis=1)                        # (P,)
            correct += int(np.sum(preds == filler_idx))
            total += P
        acc_by_p[P] = correct / total
    return acc_by_p


def crossing(acc_by_p: dict[int, float], level: float) -> str:
    """Последнее P, при котором точность ещё >= level (граница развала)."""
    last = None
    for p in sorted(acc_by_p):
        if acc_by_p[p] >= level:
            last = p
    return str(last) if last is not None else f"<{min(acc_by_p)}"


# ===========================================================================
# Эксперимент B — modern Hopfield cleanup (= attention), аттрактор
# ===========================================================================
def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def experiment_b(
    n: int = 1000,
    M: int = 200,
    fs: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45),
    iters: int = 3,
    seed: int = 1,
) -> tuple[dict[float, dict[str, float]], int]:
    rng = np.random.default_rng(seed)

    # Хранимые паттерны (биполярные). X — это keys=values в attention.
    X = (rng.integers(0, 2, size=(M, n)).astype(np.float32) * 2.0 - 1.0)
    scale = 1.0 / np.sqrt(n)  # масштабирование как в scaled dot-product attention

    def retrieve(s: np.ndarray) -> np.ndarray:
        # одно обновление modern Hopfield == один шаг attention:
        #   a = softmax( (X · s)/√n );   s' = sign(Xᵀ · a)
        a = _softmax_rows((X @ s) * scale)      # (M,) веса внимания по паттернам
        out = a @ X                              # (n,) взвешенная сумма
        return np.sign(out).astype(np.float32)

    results: dict[float, dict[str, float]] = {}
    for f in fs:
        k = int(round(f * n))
        agg = {"se": 0, "ie": 0, "sh": 0, "ih": 0}
        for w in X:
            flip = rng.choice(n, k, replace=False)
            cue = w.copy()
            cue[flip] = -cue[flip]  # инверсия биполярного бита

            r1 = retrieve(cue)
            agg["se"] += int(np.array_equal(r1, w))
            agg["sh"] += int(np.sum(r1 != w))

            ri = cue
            for _ in range(iters):
                ri = retrieve(ri)
            agg["ie"] += int(np.array_equal(ri, w))
            agg["ih"] += int(np.sum(ri != w))

        results[f] = {
            "single_exact": agg["se"] / M,
            "iter_exact": agg["ie"] / M,
            "single_ham": agg["sh"] / M,
            "iter_ham": agg["ih"] / M,
            "iter_bitacc": 1.0 - agg["ih"] / M / n,
            "cue_ham": float(k),
        }
    return results, M


# ===========================================================================
# Отчёт
# ===========================================================================
def main() -> None:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 68)
    emit("  ASTRUM VERUM — PHASE 0  (VSA/SDM, чистые атомы, numpy)")
    emit("=" * 68)

    # ---- A ----
    emit("\n## A. Ёмкость binding (MAP, D=10 000, cleanup по 512 филлерам)")
    acc_a = experiment_a()
    emit(f"\n{'P (пар в записи)':>18} | {'точность recall':>16}")
    emit(f"{'-'*18}-+-{'-'*16}")
    for p in sorted(acc_a):
        mark = "  <- порог P=10" if p == 10 else ""
        emit(f"{p:>18} | {acc_a[p]:>15.3f}{mark}")
    p90 = crossing(acc_a, 0.90)
    p50 = crossing(acc_a, 0.50)
    a_at10 = acc_a.get(10, float("nan"))
    a_pass = a_at10 >= THRESHOLDS["A_acc_at_P10"]
    emit(f"\n  Граница: точность >=0.90 до P={p90};  >=0.50 до P={p50}")
    emit(
        f"  A: точность@P10 = {a_at10:.3f} "
        f"(порог >= {THRESHOLDS['A_acc_at_P10']}) -> "
        f"{'PASS ✓' if a_pass else 'FAIL ✗'}"
    )

    # ---- B ----
    emit("\n## B. Modern Hopfield cleanup = attention (n=1000, M=200 паттернов)")
    res_b, n_stored = experiment_b()
    emit(f"\n  Хранимых паттернов (keys=values): {n_stored};  retrieve = 1 шаг attention")
    emit(
        f"\n{'f (бит)':>8} | {'exact':>6}/{'exact':>6} | "
        f"{'Hamming до цели':>22} | {'iter bit-acc':>11}"
    )
    emit(f"{'повр.':>8} | {'single':>6}/{'iter':>6} | {'cue → single → iter':>22} |")
    emit(f"{'-'*8}-+-{'-'*13}-+-{'-'*22}-+-{'-'*11}")
    for f in sorted(res_b):
        b = res_b[f]
        mark = "  <- f=0.20" if abs(f - 0.20) < 1e-9 else ""
        ham = f"{b['cue_ham']:.0f} → {b['single_ham']:.1f} → {b['iter_ham']:.1f}"
        emit(
            f"{f:>8.2f} | {b['single_exact']:>6.2f}/{b['iter_exact']:>6.2f} | "
            f"{ham:>22} | {b['iter_bitacc']:>11.4f}{mark}"
        )
    b20 = res_b[0.20]
    b_iter_at20 = b20["iter_exact"]
    b_attractor = b20["iter_ham"] <= b20["single_ham"] <= b20["cue_ham"]
    b_pass = b_iter_at20 >= THRESHOLDS["B_recovery_at_f20"] and b_attractor
    emit(
        f"\n  B: iterated exact@f0.20 = {b_iter_at20:.3f} "
        f"(порог >= {THRESHOLDS['B_recovery_at_f20']}); "
        f"bit-acc = {b20['iter_bitacc']:.4f}"
    )
    emit(
        f"  Аттрактор: Hamming cue {b20['cue_ham']:.0f} → single "
        f"{b20['single_ham']:.1f} → iter {b20['iter_ham']:.1f}  "
        f"({'чистит ✓' if b_attractor else 'не чистит ✗'})"
    )
    emit(f"  B -> {'PASS ✓' if b_pass else 'FAIL ✗'}")

    # ---- вердикт ----
    emit("\n" + "=" * 68)
    verdict = "PASS ✓✓ — алгебра держит, идём в Phase 1 (grounding)" if (
        a_pass and b_pass
    ) else "ЧАСТИЧНО/FAIL — см. цифры; враг локализован дёшево"
    emit(f"  ВЕРДИКТ PHASE 0:  {verdict}")
    emit("=" * 68)

    OUT_MD.write_text(
        "# Phase 0 — результаты (VSA/SDM, чистые атомы)\n\n```\n"
        + "\n".join(lines)
        + "\n```\n"
    )
    print(f"\n[отчёт записан] {OUT_MD}")


if __name__ == "__main__":
    main()
