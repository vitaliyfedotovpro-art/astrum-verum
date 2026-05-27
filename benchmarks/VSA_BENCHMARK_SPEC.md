# Спецификация: статистически валидный бенчмарк VSAMemory

## Зачем
Текущие цифры проекта (facts recall 1.000@8000, SimHash grounding corr ~0.99,
episode order-recall 0.995@200 → 0.25@1000) получены **одиночными прогонами**.
Их нельзя приводить как измеренные величины. Нужен бенчмарк, который даёт каждую
точку как **mean ± std по репликации над несколькими seed**, потому что атомы
(`random_atoms`) и SimHash-проекция стохастичны.

## Железные правила честности (НАРУШЕНИЕ = провал задачи)
1. **Никакого cherry-pick.** Репортить ВСЕ прогоны, включая худшие seed. Для каждой
   точки: mean, std, min, max, n_seeds.
2. **Сохранять raw per-seed** в CSV — чтобы цифры можно было перепроверить.
3. **Только реальные эмбеддинги** (см. ниже). Никаких выдуманных чисел.
4. Если что-то не запустилось / упёрлось в ресурсы — **честно записать в отчёт**,
   что и почему, и снизить параметр, а не подгонять результат.
5. Не трогать рабочий код памяти. Только новый файл бенчмарка + файлы результатов.

## API (прочитай `astrum_verum/vsa/memory.py` и `astrum_verum/vsa/core.py`)
- `VSAMemory(D=10000, seed=int, normalize_threshold=0.82, embed_fn=callable|None)`
- `.add_triple(subject, relation, obj) -> idx`
- `.query({"subject":s, "relation":r}, "object") -> {"answer","score","triple","fact_idx"}`
  (аналогично recall по subject: `query({"relation":r,"object":o}, "subject")`)
- `.add_episode(list[str], episode_id) -> eid`
- `.recall_at(eid, pos) -> str` ; `.episode_order(eid) -> list[str]`
- `core.ground(emb, proj)`, `core.make_projection(emb_dim, D, rng)` — для grounding-теста.

## Референсы стиля и методологии (прочитай оба)
- `experiments/vsa_sdm/phase0_algebra.py` — образец: trials-loop, заранее зафиксированные
  пороги, отчёт в `.md`. Скопируй дисциплину, но это синтетика — нам нужен реальный API.
- `experiments/vsa_sdm/phase1_grounding.py` — как мерили SimHash grounding corr (воспроизвести).

## Эмбеддер / данные
- VSAMemory по умолчанию грузит sentence-transformers
  `paraphrase-multilingual-MiniLM-L12-v2`. **Использовать реальный эмбеддер.**
- Чтобы seed-свип был быстрым: предпосчитай эмбеддинги всех уникальных текстов
  ОДИН раз, оберни в `embed_fn` с кэшем (`dict[text] -> np.ndarray`), передавай как
  `VSAMemory(embed_fn=cached_fn, seed=…)`. Тогда при смене seed пересоздаются только
  roles/projection (дёшево), а тяжёлый эмбеддинг считается один раз.
- **CORPUS:** сгенерируй детерминированно (отдельный фиксированный data-seed, НЕ путать
  с VSA-seed) пул уникальных триплетов из реальных слов (имена, города, профессии,
  животные, болезни, материалы и т.п. — комбинируй реальные лексемы, не `node-0001`).
  Нужно достаточно уникальных (subject, relation, object) для max N. Задокументируй,
  как именно генерил пул.
- Если реальный эмбеддер недоступен или 16000 фактов не тянет машина — снизь верхнюю
  границу N (например до 8000) и **честно зафиксируй это в отчёте**.

## Эксперименты

### E1 — Facts structural recall vs N (масштабирование ёмкости)
- `N ∈ {1000, 2000, 4000, 8000, 16000}`, `seeds = 30` различных.
- Для каждого (N, seed): новый `VSAMemory(seed=seed, embed_fn=cached)`, добавить N
  различных триплетов; затем на случайной подвыборке (≥200 фактов или все, если меньше)
  сделать `query` (известны subject+relation → восстановить object); точность =
  доля `answer == истинный object`.
- Точка (N): mean ± std ± min точности по 30 seeds.

### E2 — Episode order-recall vs length (saturation + проверка окна)
- `length L ∈ {50, 100, 200, 500, 1000}`, `seeds = 30`.
- `add_episode(L items)`, `recall_at` по всем позициям; точность позиционного recall
  = доля `recalled == истинный item на этой позиции`. mean ± std по seeds.
- Дополнительно подтвердить: bounded window (только последние W=150 элементов как
  эпизод) даёт recall ≈ 1.0 независимо от полной длины диалога. Это валидирует
  архитектурное решение «рабочая память = окно».

### E3 — SimHash grounding fidelity (воспроизвести corr ~0.99)
- Взять M реальных текстов (≥500). Посчитать попарные cos в embedding-space и cos
  соответствующих `ground()`-атомов в hypervector-space. Корреляция Pearson + Spearman
  между двумя наборами сходств. seeds варьируют projection (30). mean ± std corr.

## Вывод
- Новый файл: `benchmarks/vsa_memory_benchmark.py` с argparse:
  `--seeds N` (default 30), `--max-n N`, `--quick` (быстрый прогон для проверки).
- `benchmarks/vsa_memory_results.md` — таблицы mean±std±min по E1/E2/E3 + шапка прогона
  (дата, версия эмбеддера, n_seeds, D, размер CORPUS, железо/время).
- `benchmarks/vsa_memory_raw.csv` — строка на каждый (эксперимент, параметр, seed, метрика).
- **Прогнать реально** (сначала `--quick` для самопроверки, потом полный прогон).
- Вернуть сводку: таблицы mean±std по E1/E2/E3 и пути к файлам.

## Детерминизм
Все seed фиксировать и логировать. Data-seed (CORPUS) отделить от VSA-seed (алгебра).
