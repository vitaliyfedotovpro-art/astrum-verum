"""
Astrum Verum — Phase 2: end-to-end на РЕАЛЬНОМ извлечении.

text → extract_triples() (живой LLM, паттерн deep_parse) → VSAMemory (role-binding)
→ структурные запросы. Плюс честный косинус-RAG-бейзлайн на role-sensitive запросе,
чтобы проверить: сохраняется ли преимущество VSA на ГРЯЗНЫХ, реально извлечённых
данных (а не на синтетике Phase 1).

Запуск:  PYTHONPATH=. python experiments/vsa_sdm/phase2_pipeline.py
Пишет:   experiments/vsa_sdm/phase2_results.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from astrum_verum.extract import extract_triples
from astrum_verum.vsa import VSAMemory

OUT_MD = Path(__file__).with_name("phase2_results.md")

# Текст с явными отношениями и НАМЕРЕННОЙ перестановкой ролей ("trusts"/"manages"
# в обе стороны) — чтобы проверить role-sensitivity на реально извлечённых триплетах.
TEXT = """\
Maya is a botanist and works in Lisbon. She founded the lab Helix.
Helix uses the Leech lattice for memory. Maya owns a cat named Pixel.
Alice trusts Bob. Bob trusts Alice. Carol manages Dave. Dave manages Carol.
"""
# NB: "Alice trusts Bob" и "Bob trusts Alice" — идентичный «мешок» концептов,
# различаются ТОЛЬКО ролями. Косинус-RAG тут слепнет (≈0.5), VSA различает по
# role-binding. Это и есть честная нагрузка на дискриминатор на реальном извлечении.


def main() -> None:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 70)
    emit("  ASTRUM VERUM — PHASE 2  (text → extract → VSA, на реальном LLM)")
    emit("=" * 70)

    # --- 1. Извлечение триплетов (живой провайдер) ---
    emit("\n[*] Извлечение триплетов из текста (DeepSeek→xAI→Groq)...")
    triples, provider = extract_triples(TEXT)
    if not triples:
        emit("  ОШИБКА: ни одного триплета (нет ключей / провайдер недоступен).")
        OUT_MD.write_text("# Phase 2\n\n```\n" + "\n".join(lines) + "\n```\n")
        return
    emit(f"  провайдер: {provider};  извлечено триплетов: {len(triples)}")
    emit("")
    for t in triples:
        emit(f"    ({t['subject']}) --[{t['relation']}]--> ({t['object']})")

    # --- 2. Загрузка в VSA-память ---
    mem = VSAMemory()
    for t in triples:
        mem.add_triple(t["subject"], t["relation"], t["object"])

    # --- 3. Round-trip recall: object при (subject, relation) ---
    emit("\n## Round-trip: object при (subject, relation)")
    rt_ok = 0
    for t in triples:
        res = mem.query({"subject": t["subject"], "relation": t["relation"]}, "object")
        ok = res["answer"].strip().lower() == t["object"].strip().lower()
        rt_ok += ok
        emit(
            f"  q(subj={t['subject']!r}, rel={t['relation']!r}) → "
            f"{res['answer']!r}  [{'✓' if ok else '✗ ждали '+t['object']!r}]  "
            f"(score {res['score']:.2f})"
        )
    rt_acc = rt_ok / len(triples)
    emit(f"\n  Round-trip recall: {rt_ok}/{len(triples)} = {rt_acc:.3f}")

    # --- 4. Role-sensitivity: сущности, играющие РАЗНЫЕ роли ---
    subj_set = {t["subject"].strip().lower() for t in triples}
    obj_set = {t["object"].strip().lower() for t in triples}
    pivots = subj_set & obj_set  # сущность встречается и как subject, и как object
    emit("\n## Role-sensitivity (сущности в разных ролях)")
    if not pivots:
        emit("  В извлечении нет сущности, играющей обе роли — role-swap не на чем "
             "проверить. (Round-trip выше уже использует role-binding.)")
    else:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("all-MiniLM-L6-v2")

        def emb(s: str) -> np.ndarray:
            return st.encode(s, normalize_embeddings=True).astype(np.float32)

        # bag-RAG: факт = среднее эмбеддингов (subj,rel,obj); запрос = (subj,rel).
        fact_vecs = np.stack([
            (emb(t["subject"]) + emb(t["relation"]) + emb(t["object"])) / 3.0
            for t in triples
        ])
        fact_vecs /= np.linalg.norm(fact_vecs, axis=1, keepdims=True)

        vsa_ok = rag_ok = n = 0
        for piv in sorted(pivots):
            for ti, t in enumerate(triples):
                if t["subject"].strip().lower() != piv:
                    continue
                n += 1
                # VSA
                r = mem.query({"subject": t["subject"], "relation": t["relation"]}, "object")
                v_ok = r["answer"].strip().lower() == t["object"].strip().lower()
                vsa_ok += v_ok
                # bag-RAG (oracle: правильно ⟺ извлёк нужный факт)
                q = (emb(t["subject"]) + emb(t["relation"])) / 2.0
                q /= np.linalg.norm(q)
                sims = fact_vecs @ q
                mx = sims.max()
                tied = np.flatnonzero(sims >= mx - 1e-6)
                r_credit = (1.0 / len(tied)) if ti in tied else 0.0
                rag_ok += r_credit
                emit(
                    f"  pivot={piv!r}: q(subj={t['subject']!r}, rel={t['relation']!r})"
                    f" → VSA {r['answer']!r} [{'✓' if v_ok else '✗'}], "
                    f"RAG credit {r_credit:.2f}"
                )
        if n:
            emit(f"\n  role-sensitive: VSA {vsa_ok}/{n} = {vsa_ok/n:.3f} | "
                 f"RAG {rag_ok:.2f}/{n} = {rag_ok/n:.3f}")

    # --- вердикт ---
    emit("\n" + "=" * 70)
    p = rt_acc >= 0.90
    emit(f"  Round-trip recall {rt_acc:.3f} (>= 0.90) -> {'PASS ✓' if p else 'FAIL ✗'}")
    emit("  ВЕРДИКТ PHASE 2: " + (
        "PASS — реальное извлечение заведено в VSA, role-binding работает на грязных данных."
        if p else
        "ЧАСТИЧНО — см. промахи (вероятно нормализация сущностей / near-synonym cleanup)."
    ))
    emit("=" * 70)

    OUT_MD.write_text("# Phase 2 — результаты (real extraction → VSA)\n\n```\n"
                      + "\n".join(lines) + "\n```\n")
    print(f"\n[отчёт записан] {OUT_MD}")


if __name__ == "__main__":
    main()
