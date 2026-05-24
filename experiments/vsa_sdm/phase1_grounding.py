"""
Astrum Verum — Phase 1: grounding + дискриминатор против косинус-RAG.

Главный вопрос пивота: переживёт ли VSA-алгебра переход на РЕАЛЬНЫЕ эмбеддинги
(которые НЕ квазиортогональны — это риск, названный с самого начала), и делает
ли она то, чего косинус-RAG не может в принципе?

Тест — каноническая «проблема связывания» (variable binding). Факты — это
структурированные тройки ролей: {agent, patient, location} ← концепты.
Берём пары-близнецы, отличающиеся ТОЛЬКО назначением ролей:
    f1 = (agent=a, patient=p, location=l)
    f2 = (agent=p, patient=a, location=l)   # тот же набор концептов!
Косинус-similarity по содержимому их различить не может (одинаковый «мешок»
концептов). VSA связывает роль⊗филлер и различает точно.

Атомы концептов GROUNDED: embedding(384) → SimHash (sign случайной проекции) →
биполярный гипервектор D=10k (сохраняет угловую близость → семантически близкие
концепты получают КОРРЕЛИРОВАННЫЕ атомы = стресс для cleanup и binding).
Роли — случайные (структурные, не семантические).

Бейзлайны RAG (честные, oracle-извлечение: «правильно» ⟺ извлёк нужный факт):
    bag-RAG      — факт = среднее эмбеддингов концептов (чистый similarity).
    sentence-RAG — факт = эмбеддинг фразы с ролями-метками (order-aware, сильнее).

Запуск:  python experiments/vsa_sdm/phase1_grounding.py
Пишет:   experiments/vsa_sdm/phase1_results.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

D = 10_000
OUT_MD = Path(__file__).with_name("phase1_results.md")

# Пороги pass/fail — зафиксированы до запуска.
THRESHOLDS = {
    "vsa_overall": 0.90,        # общая точность VSA на role-запросах
    "vsa_minus_rag_ambig": 0.30,  # отрыв VSA от лучшего RAG на ambiguous-запросах
    "grounding_drop_max": 0.15,   # насколько grounded-VSA может отстать от random-VSA
}

# Концепты со скрытой кластерной структурой → реальная градация близости.
CLUSTERS = {
    "animals": ["dog", "cat", "wolf", "lion", "eagle", "shark", "rabbit", "horse", "bear"],
    "cities": ["Paris", "Tokyo", "London", "Berlin", "Cairo", "Moscow", "Madrid", "Rome", "Boston"],
    "professions": ["doctor", "nurse", "engineer", "teacher", "lawyer", "pilot", "chef", "farmer", "artist"],
    "tools": ["hammer", "wrench", "drill", "saw", "knife", "axe", "shovel", "ladder", "brush"],
    "foods": ["bread", "cheese", "apple", "rice", "soup", "cake", "fish", "salad", "honey"],
    "emotions": ["joy", "anger", "fear", "sadness", "hope", "pride", "envy", "calm", "trust"],
    "vehicles": ["car", "train", "plane", "boat", "bike", "truck", "bus", "ferry", "rocket"],
}
CONCEPTS = [c for group in CLUSTERS.values() for c in group]


# ---------------------------------------------------------------------------
# VSA-примитивы
# ---------------------------------------------------------------------------
def bundle_sign(vectors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    s = vectors.sum(axis=0)
    out = np.sign(s)
    ties = out == 0
    if ties.any():
        out[ties] = rng.integers(0, 2, size=int(ties.sum())) * 2.0 - 1.0
    return out.astype(np.float32)


def random_atoms(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(n, D), dtype=np.int8).astype(np.float32) * 2.0 - 1.0


def ground_atoms(embeddings: np.ndarray, seed: int = 7) -> np.ndarray:
    """embedding(N) → биполярный гипервектор(D) через SimHash (sign случайной
    проекции). Сохраняет угловую близость: cos(atom_i,atom_j) ≈ 1 − 2·θ_ij/π."""
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((embeddings.shape[1], D)).astype(np.float32)
    atoms = np.sign(embeddings @ proj).astype(np.float32)
    atoms[atoms == 0] = 1.0
    return atoms


# ---------------------------------------------------------------------------
# Генерация базы фактов (роли: 0=agent, 1=patient, 2=location)
# ---------------------------------------------------------------------------
def build_facts(C: int, F: int, n_twins: int, rng: np.random.Generator):
    """Возвращает (facts, ambiguous_flag). facts[i] = (a,p,l) индексы концептов.
    Близнец факта (a,p,l) = (p,a,l) — тот же набор концептов, роли agent/patient
    переставлены. (patient,location) уникальны среди base-фактов → запрос
    однозначен."""
    facts: list[tuple[int, int, int]] = []
    ambiguous: list[bool] = []
    used_pl: set[tuple[int, int]] = set()

    def fresh_triple() -> tuple[int, int, int]:
        while True:
            a, p, l = rng.choice(C, 3, replace=False)
            if (p, l) not in used_pl and (a, l) not in used_pl:
                return int(a), int(p), int(l)

    n_base = F - n_twins
    for i in range(n_base):
        a, p, l = fresh_triple()
        facts.append((a, p, l))
        used_pl.add((p, l))
        if i < n_twins:  # навешиваем близнеца на первые n_twins фактов
            facts.append((p, a, l))      # роли agent/patient переставлены
            used_pl.add((a, l))
            ambiguous.append(True)       # ОБА члена пары неразличимы по «мешку»
            ambiguous.append(True)
        else:
            ambiguous.append(False)
    return facts, ambiguous


# ---------------------------------------------------------------------------
# VSA: кодирование и role-запрос (agent given patient+location)
# ---------------------------------------------------------------------------
def vsa_encode(facts, roles, concept_atoms, rng):
    mem = np.empty((len(facts), D), dtype=np.float32)
    for i, (a, p, l) in enumerate(facts):
        mem[i] = bundle_sign(
            np.stack([
                roles[0] * concept_atoms[a],
                roles[1] * concept_atoms[p],
                roles[2] * concept_atoms[l],
            ]),
            rng,
        )
    return mem


def vsa_query_agent(facts, mem, roles, concept_atoms):
    """Для каждого факта: probe(patient,location) → retrieve → unbind agent →
    cleanup. correct ⟺ восстановлен верный концепт-agent."""
    correct = np.zeros(len(facts), dtype=bool)
    rng = np.random.default_rng(0)
    for i, (a, p, l) in enumerate(facts):
        probe = bundle_sign(
            np.stack([roles[1] * concept_atoms[p], roles[2] * concept_atoms[l]]), rng
        )
        idx = int(np.argmax(mem @ probe))          # извлечение факта
        unbound = mem[idx] * roles[0]              # unbind роли agent
        pred = int(np.argmax(concept_atoms @ unbound))  # cleanup по концептам
        correct[i] = pred == a
    return correct


# ---------------------------------------------------------------------------
# RAG-бейзлайны (oracle-извлечение: «правильно» ⟺ извлечён нужный факт).
# Ничьи (идентичные эмбеддинги role-swapped фактов) считаем как ожидаемую
# точность 1/k при случайном tie-break — честная модель «косинус не различает».
# ---------------------------------------------------------------------------
def _credit(sims: np.ndarray, target: int) -> float:
    mx = sims.max()
    tied = np.flatnonzero(sims >= mx - 1e-6)
    return (1.0 / len(tied)) if target in tied else 0.0


def rag_bag(facts, emb):
    fact_vec = np.stack([
        (emb[a] + emb[p] + emb[l]) / 3.0 for (a, p, l) in facts
    ])
    fact_vec /= np.linalg.norm(fact_vec, axis=1, keepdims=True)
    out = np.zeros(len(facts), dtype=np.float64)
    for i, (a, p, l) in enumerate(facts):
        q = (emb[p] + emb[l]) / 2.0
        q /= np.linalg.norm(q)
        out[i] = _credit(fact_vec @ q, i)
    return out


def rag_sentence(facts, model):
    names = CONCEPTS
    fact_txt = [f"agent {names[a]}, patient {names[p]}, location {names[l]}" for a, p, l in facts]
    query_txt = [f"patient {names[p]}, location {names[l]}, who is the agent" for a, p, l in facts]
    fv = model.encode(fact_txt, normalize_embeddings=True)
    qv = model.encode(query_txt, normalize_embeddings=True)
    out = np.zeros(len(facts), dtype=np.float64)
    for i in range(len(facts)):
        out[i] = _credit(fv @ qv[i], i)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 70)
    emit("  ASTRUM VERUM — PHASE 1  (grounding + дискриминатор vs косинус-RAG)")
    emit("=" * 70)

    from sentence_transformers import SentenceTransformer

    emit("\n[*] Эмбеддинг концептов (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(CONCEPTS, normalize_embeddings=True).astype(np.float32)
    C = len(CONCEPTS)

    # --- проверка качества grounding ---
    concept_atoms = ground_atoms(emb)
    e_cos = emb @ emb.T
    a_cos = (concept_atoms @ concept_atoms.T) / D
    iu = np.triu_indices(C, k=1)
    corr = float(np.corrcoef(e_cos[iu], a_cos[iu])[0, 1])
    emit(
        f"[*] Grounding (SimHash D={D}): корреляция cos(emb) vs cos(atom) = {corr:.3f}"
    )
    emit("    (близко к 1.0 → семантика концептов сохранена в гипервекторах)")

    rng = np.random.default_rng(42)
    roles = random_atoms(3, rng)  # agent / patient / location — структурные
    rand_concept_atoms = random_atoms(C, rng)  # для контроля «без grounding»

    F, n_twins = 200, 60
    facts, ambiguous = build_facts(C, F, n_twins, rng)
    amb = np.array(ambiguous)
    emit(
        f"\n[*] База: {len(facts)} фактов, из них {int(amb.sum())} ambiguous "
        f"(есть близнец с переставленными ролями)."
    )

    # --- VSA (grounded) ---
    mem_g = vsa_encode(facts, roles, concept_atoms, rng)
    vsa_g = vsa_query_agent(facts, mem_g, roles, concept_atoms)

    # --- VSA (random atoms, контроль grounding) ---
    mem_r = vsa_encode(facts, roles, rand_concept_atoms, rng)
    vsa_r = vsa_query_agent(facts, mem_r, roles, rand_concept_atoms)

    # --- RAG ---
    rag_b = rag_bag(facts, emb)
    rag_s = rag_sentence(facts, model)

    def acc(mask, sub=None):
        m = mask if sub is None else mask[sub]
        return float(np.mean(m)) if len(m) else float("nan")

    base = ~amb  # будем считать раздельно ambiguous vs unique по ВСЕМ фактам
    emit("\n## Точность role-запроса «agent given (patient, location)»")
    emit(f"\n{'метод':>22} | {'все':>7} | {'ambiguous':>10} | {'unique':>7}")
    emit(f"{'-'*22}-+-{'-'*7}-+-{'-'*10}-+-{'-'*7}")
    rows = [
        ("VSA (grounded)", vsa_g),
        ("VSA (random atoms)", vsa_r),
        ("RAG (bag-of-concepts)", rag_b),
        ("RAG (sentence, order)", rag_s),
    ]
    for name, c in rows:
        emit(
            f"{name:>22} | {acc(c):>7.3f} | {acc(c, amb):>10.3f} | {acc(c, base):>7.3f}"
        )

    # --- кривая ёмкости VSA (grounded) ---
    emit("\n## Ёмкость VSA (grounded): точность vs число фактов")
    emit(f"\n{'F фактов':>10} | {'точность':>9}")
    emit(f"{'-'*10}-+-{'-'*9}")
    for Ftest in (50, 100, 200, 400, 800):
        f2, _ = build_facts(C, Ftest, Ftest // 3, np.random.default_rng(100 + Ftest))
        m2 = vsa_encode(f2, roles, concept_atoms, rng)
        c2 = vsa_query_agent(f2, m2, roles, concept_atoms)
        emit(f"{Ftest:>10} | {acc(c2):>9.3f}")

    # --- вердикт ---
    vsa_overall = acc(vsa_g)
    best_rag_amb = max(acc(rag_b, amb), acc(rag_s, amb))
    margin = acc(vsa_g, amb) - best_rag_amb
    grounding_drop = acc(vsa_r) - acc(vsa_g)

    p1 = vsa_overall >= THRESHOLDS["vsa_overall"]
    p2 = margin >= THRESHOLDS["vsa_minus_rag_ambig"]
    p3 = grounding_drop <= THRESHOLDS["grounding_drop_max"]

    emit("\n" + "=" * 70)
    emit("  ПРОВЕРКА ПОРОГОВ:")
    emit(
        f"   VSA overall = {vsa_overall:.3f} (>= {THRESHOLDS['vsa_overall']}) "
        f"-> {'✓' if p1 else '✗'}"
    )
    emit(
        f"   отрыв VSA−RAG на ambiguous = {margin:+.3f} "
        f"(>= {THRESHOLDS['vsa_minus_rag_ambig']}) -> {'✓' if p2 else '✗'}"
    )
    emit(
        f"   просадка grounding (random−grounded) = {grounding_drop:+.3f} "
        f"(<= {THRESHOLDS['grounding_drop_max']}) -> {'✓' if p3 else '✗'}"
    )
    verdict = (
        "PASS ✓✓ — алгебра пережила grounding и бьёт RAG на структуре. Тезис доказан."
        if (p1 and p2 and p3)
        else "ЧАСТИЧНО/FAIL — см. цифры; узкое место локализовано."
    )
    emit(f"\n  ВЕРДИКТ PHASE 1:  {verdict}")
    emit("=" * 70)

    OUT_MD.write_text(
        "# Phase 1 — результаты (grounding + дискриминатор)\n\n```\n"
        + "\n".join(lines)
        + "\n```\n"
    )
    print(f"\n[отчёт записан] {OUT_MD}")


if __name__ == "__main__":
    main()
