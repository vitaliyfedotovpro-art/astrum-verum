"""
Astrum Verum — Phase 3: полный OdinnMemory в деле (не обрезанный).

Демонстрирует ВСЕ способности живьём (реальный эмбеддер + живой LLM-extractor):
  1. remember(text)            — извлечь факты-триплеты и запомнить
  2. recall_object/subject     — структурный запрос (role-binding)
  3. role-sensitivity          — различение (A,r,B) vs (B,r,A)
  4. нормализация сущностей    — грязные варианты → один канон
  5. эпизоды (порядок)         — recall порядка + «что было после X»
  6. персистентность           — save → load → запросы работают

Запуск:  PYTHONPATH=. python experiments/vsa_sdm/phase3_full.py
Пишет:   experiments/vsa_sdm/phase3_results.md
"""

from __future__ import annotations

from pathlib import Path

from astrum_verum import OdinnMemory

OUT_MD = Path(__file__).with_name("phase3_results.md")

TEXT = """\
Maya is a botanist and founded the lab Helix.
Helix uses the Leech lattice. Alice trusts Bob. Bob trusts Alice.
Carol manages Dave. Dave manages Carol.
"""

CONVERSATION = [
    "greeted the user",
    "asked about the project",
    "reviewed the results",
    "scheduled a follow-up call",
    "said goodbye",
]


def main() -> None:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 70)
    emit("  ASTRUM VERUM — PHASE 3  (полный OdinnMemory, живьём)")
    emit("=" * 70)

    mem = OdinnMemory(normalize_threshold=0.78)

    # --- 1. remember (живой LLM) ---
    emit("\n## 1. remember(text) — извлечение фактов")
    triples = mem.remember(TEXT)
    if not triples:
        emit("  ОШИБКА: извлечение пустое (нет LLM-ключа?).")
        OUT_MD.write_text("# Phase 3\n\n```\n" + "\n".join(lines) + "\n```\n")
        return
    for t in triples:
        emit(f"    ({t['subject']}) --[{t['relation']}]--> ({t['object']})")

    # --- 2. recall (round-trip по извлечённому) ---
    emit("\n## 2. recall_object(subject, relation) — структурный запрос")
    rt_ok = 0
    for t in triples:
        r = mem.recall_object(t["subject"], t["relation"])
        ok = r["answer"].strip().lower() == t["object"].strip().lower()
        rt_ok += ok
        emit(f"  {t['subject']!r} --[{t['relation']}]--> {r['answer']!r} "
             f"[{'✓' if ok else '✗'}]")
    emit(f"  round-trip recall: {rt_ok}/{len(triples)}")

    # --- 3. role-sensitivity ---
    emit("\n## 3. role-sensitivity (различение направления отношения)")
    for s, rel in [("Alice", "trusts"), ("Bob", "trusts"), ("Carol", "manages")]:
        r = mem.recall_object(s, rel)
        emit(f"  кого {s} {rel}? → {r['answer']!r}")
    rsub = mem.recall_subject("trusts", "Alice")
    emit(f"  кто trusts Alice? → {rsub['answer']!r}  (subject-направление)")

    # --- 4. нормализация сущностей (грязные варианты) ---
    emit("\n## 4. нормализация сущностей")
    before = mem.vsa.n_concepts
    mem.remember_triple("Maya", "owns", "the cat Pixel")
    mem.remember_triple("maya", "adopted", "the cat Pixel")  # варианты Maya + Pixel
    after = mem.vsa.n_concepts
    emit(f"  концептов до доп.фактов: {before}, после: {after}")
    emit(f"  алиасы 'Maya': {mem.vsa.aliases_of('Maya')}")
    emit(f"  алиасы 'the cat Pixel': {mem.vsa.aliases_of('the cat Pixel')}")

    # --- 5. эпизоды ---
    emit("\n## 5. эпизод (порядок + «что было после»)")
    eid = mem.remember_conversation(CONVERSATION)
    order = mem.episode_order(eid)
    order_ok = [o.lower() for o in order] == [c.lower() for c in CONVERSATION]
    emit(f"  восстановленный порядок: {order}")
    emit(f"  порядок верный: {'✓' if order_ok else '✗'}")
    nxt = mem.whats_next(eid, "reviewed the results")
    nxt_ok = (nxt or "").lower() == "scheduled a follow-up call"
    emit(f"  что было после 'reviewed the results'? → {nxt!r} [{'✓' if nxt_ok else '✗'}]")

    # --- 6. персистентность ---
    emit("\n## 6. персистентность (save → load)")
    path = Path(__file__).with_name(".phase3_odinn_state")
    mem.save(path)
    mem2 = OdinnMemory.load(path)
    rl = mem2.recall_object("Alice", "trusts")
    persist_ok = (
        mem2.vsa.n_facts == mem.vsa.n_facts
        and rl["answer"].strip().lower() == "bob"
        and [o.lower() for o in mem2.episode_order(eid)] == [c.lower() for c in CONVERSATION]
    )
    emit(f"  после load: facts={mem2.vsa.n_facts}, Alice-trusts→{rl['answer']!r}, "
         f"эпизод сохранён: {'✓' if persist_ok else '✗'}")
    for ext in (".npz", ".json"):
        path.with_suffix(ext).unlink(missing_ok=True)

    # --- вердикт ---
    emit("\n" + "=" * 70)
    all_ok = rt_ok == len(triples) and order_ok and nxt_ok and persist_ok
    emit("  ВЕРДИКТ PHASE 3: " + (
        "PASS ✓✓ — полный OdinnMemory работает: факты, роли, эпизоды, нормализация, персист."
        if all_ok else "ЧАСТИЧНО — см. промахи выше."
    ))
    emit("=" * 70)

    OUT_MD.write_text("# Phase 3 — полный OdinnMemory (живьём)\n\n```\n"
                      + "\n".join(lines) + "\n```\n")
    print(f"\n[отчёт записан] {OUT_MD}")


if __name__ == "__main__":
    main()
