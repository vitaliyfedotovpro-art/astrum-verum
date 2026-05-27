#!/usr/bin/env python3
"""
VSA Memory Benchmark — статистически валидный бенчмарк VSAMemory.

Эксперименты:
  E1 — Facts structural recall vs N (масштабирование ёмкости)
  E2 — Episode order-recall vs length (насыщение + проверка окна W=150)
  E3 — SimHash grounding fidelity (Pearson/Spearman corr)

Запуск:
  python benchmarks/vsa_memory_benchmark.py --quick          # самопроверка
  python benchmarks/vsa_memory_benchmark.py                  # полный прогон
  python benchmarks/vsa_memory_benchmark.py --seeds 10 --max-n 4000

Пишет:
  benchmarks/vsa_memory_results.md   — отчёт с таблицами
  benchmarks/vsa_memory_raw.csv      — per-seed сырые данные
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "vsa_memory_results.md"
OUT_CSV = HERE / "vsa_memory_raw.csv"

# ---------------------------------------------------------------------------
# Константы (фиксированы до запуска)
# ---------------------------------------------------------------------------
D = 10_000
DATA_SEED = 12345          # детерминированная генерация CORPUS
EMBEDDER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
NORMALIZE_THRESHOLD = 0.82

# Эксперименты
E1_NS = (1000, 2000, 4000, 8000, 16000)       # N фактов
E1_QUERY_SAMPLE = 200                           # подвыборка для query
E2_LS = (50, 100, 200, 500, 1000)              # длины эпизодов
E2_WINDOW = 150                                 # окно рабочей памяти
E3_M = 500                                      # текстов для grounding

# Quick-режим
QUICK_SEEDS = 3
QUICK_MAX_N = 2000
QUICK_NS = (500, 1000, 2000)
QUICK_LS = (30, 60, 120, 250, 500)
QUICK_E3_M = 100


# ===========================================================================
# CORPUS — детерминированная генерация пула триплетов из реальных слов
# ===========================================================================
def _build_word_lists():
    """Возвращает (entities, relations) — списки реальных лексем."""
    entities = [
        # Имена (30)
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
        "Iris", "Jack", "Kate", "Leo", "Maria", "Nick", "Olivia", "Paul",
        "Quinn", "Rose", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
        "Yara", "Zack", "Amir", "Bella", "Cyrus", "Dahlia",
        # Города (30)
        "Paris", "Tokyo", "London", "Berlin", "Cairo", "Moscow", "Madrid", "Rome",
        "Boston", "Seoul", "Delhi", "Sydney", "Toronto", "Dubai", "Lagos", "Lima",
        "Havana", "Vienna", "Athens", "Oslo", "Kyiv", "Hanoi", "Quito", "Riga",
        "Mumbai", "Jakarta", "Nairobi", "Bangkok", "Lisbon", "Zurich",
        # Страны (25)
        "France", "Japan", "Germany", "Brazil", "Canada", "India", "Australia",
        "Norway", "Kenya", "Chile", "Thailand", "Morocco", "Sweden", "Vietnam",
        "Portugal", "Greece", "Peru", "Finland", "Turkey", "Poland",
        "Mexico", "Egypt", "Argentina", "Colombia", "Ethiopia",
        # Профессии (25)
        "doctor", "nurse", "engineer", "teacher", "lawyer", "pilot", "chef",
        "farmer", "artist", "writer", "judge", "architect", "scientist", "musician",
        "carpenter", "plumber", "electrician", "dentist", "pharmacist", "veterinarian",
        "surgeon", "astronomer", "geologist", "librarian", "diplomat",
        # Животные (25)
        "wolf", "eagle", "shark", "rabbit", "horse", "bear", "dolphin", "panther",
        "falcon", "salmon", "turtle", "cobra", "raven", "bison", "lemur", "jaguar",
        "penguin", "octopus", "gazelle", "condor", "elephant", "gorilla", "crocodile",
        "cheetah", "walrus",
        # Болезни (20)
        "malaria", "cholera", "typhoid", "measles", "polio", "rabies", "tetanus",
        "anthrax", "leprosy", "asthma", "diabetes", "migraine", "anemia", "eczema",
        "tuberculosis", "pneumonia", "hepatitis", "influenza", "diphtheria", "scurvy",
        # Материалы (20)
        "steel", "copper", "bronze", "marble", "granite", "bamboo", "silk", "wool",
        "rubber", "graphite", "ceramic", "titanium", "platinum", "obsidian", "aluminum",
        "quartz", "diamond", "sandstone", "porcelain", "carbon_fiber",
        # Растения (20)
        "oak", "maple", "birch", "cedar", "willow", "bamboo_tree", "cypress",
        "lavender", "rosemary", "thyme", "basil", "ginger", "saffron", "vanilla",
        "orchid", "jasmine", "cactus", "fern", "lotus", "mangrove",
        # Инструменты (17)
        "hammer", "wrench", "chisel", "compass", "microscope", "telescope", "scalpel",
        "lathe", "screwdriver", "saw", "drill", "pliers", "level", "protractor",
        "caliper", "soldering_iron", "multimeter",
        # Еда (20)
        "bread", "cheese", "honey", "olive", "almond", "garlic", "cinnamon",
        "vinegar", "yogurt", "chocolate", "coffee", "salmon_food", "quinoa",
        "avocado", "walnut", "caviar", "truffle", "paprika", "saffron_spice", "maple_syrup",
        # Абстракции (20)
        "justice", "freedom", "wisdom", "courage", "harmony", "chaos", "truth",
        "beauty", "honor", "loyalty", "curiosity", "serenity", "ambition", "gratitude",
        "compassion", "integrity", "resilience", "humility", "patience", "discipline",
        # Географические объекты (20)
        "ocean", "mountain", "river", "desert", "forest", "volcano", "glacier",
        "island", "peninsula", "waterfall", "canyon", "reef", "plateau", "tundra",
        "savanna", "delta", "fjord", "archipelago", "estuary", "geyser",
        # Технологии (20)
        "internet", "satellite", "laser", "transistor", "algorithm", "database",
        "encryption", "sensor", "actuator", "robotics", "blockchain", "nanotech",
        "quantum_computer", "neural_network", "solar_panel", "wind_turbine",
        "fusion_reactor", "superconductor", "microprocessor", "antenna",
        # Спорт / игры (20)
        "soccer", "tennis", "archery", "fencing", "rowing", "gymnastics", "boxing",
        "skiing", "surfing", "polo", "cricket", "karate", "badminton", "golf",
        "hockey", "baseball", "volleyball", "wrestling", "judo", "cycling",
        # Языки (15)
        "English", "Mandarin", "Spanish", "Arabic", "Russian", "Swahili", "Latin",
        "Sanskrit", "Hebrew", "Korean", "Turkish", "Persian", "Bengali", "Tamil", "Zulu",
        # Музыкальные инструменты (15)
        "piano", "violin", "cello", "flute", "trumpet", "harp", "clarinet", "guitar",
        "drum", "saxophone", "organ", "accordion", "bassoon", "oboe", "xylophone",
        # Цвета (15)
        "crimson", "indigo", "amber", "teal", "scarlet", "azure", "emerald", "violet",
        "magenta", "cyan", "maroon", "olive", "coral", "turquoise", "burgundy",
        # Метеорология (15)
        "hurricane", "tornado", "monsoon", "drought", "blizzard", "typhoon", "tsunami",
        "avalanche", "earthquake", "cyclone", "fog", "dew", "frost", "hailstorm", "rainbow",
        # Архитектурные сооружения (20)
        "bridge", "tower", "cathedral", "temple", "mosque", "pyramid", "lighthouse",
        "castle", "fortress", "dam", "canal", "aqueduct", "skyscraper", "stadium",
        "museum", "library", "observatory", "monastery", "colosseum", "pagoda",
        # Минералы (15)
        "amethyst", "sapphire", "ruby", "emerald_gem", "topaz", "opal", "jade",
        "garnet", "onyx", "jasper", "malachite", "turquoise_mineral", "agate",
        "beryl", "fluorite",
        # Транспорт (15)
        "train", "airplane", "helicopter", "submarine", "bicycle", "motorcycle",
        "scooter", "tram", "ferry", "yacht", "glider", "zeppelin", "rickshaw",
        "snowmobile", "hovercraft",
        # Физические понятия (15)
        "gravity", "entropy", "velocity", "friction", "momentum", "pressure",
        "density", "viscosity", "capacitance", "inductance", "impedance",
        "refraction", "diffusion", "turbulence", "resonance",
        # Анатомия (20)
        "heart", "brain", "liver", "kidney", "lungs", "stomach", "intestine",
        "spleen", "pancreas", "thyroid", "retina", "cornea", "cartilage",
        "tendon", "ligament", "artery", "neuron", "synapse", "cortex", "marrow",
        # Философские концепты (15)
        "ethics", "aesthetics", "epistemology", "ontology", "phenomenology",
        "dialectic", "syllogism", "paradox", "dogma", "doctrine", "maxim",
        "axiom", "theorem", "postulate", "corollary",
        # Химические элементы (15)
        "hydrogen", "oxygen", "nitrogen", "carbon", "sodium", "calcium", "iron",
        "zinc", "silver", "gold", "mercury", "iodine", "helium", "lithium", "silicon",
        # Экономические термины (15)
        "inflation", "recession", "monopoly", "tariff", "subsidy", "dividend",
        "liquidity", "equity", "bond", "mortgage", "interest", "capital",
        "commodity", "derivative", "arbitrage",
    ]

    relations = [
        "lives_in", "works_as", "travels_to", "treats", "eats", "builds_from",
        "studies", "teaches", "rules_over", "exports_to", "imports_from",
        "founded_by", "located_near", "similar_to", "depends_on", "produces",
        "transforms_into", "communicates_with", "protects", "destroys",
        "belongs_to", "evolves_from", "measures", "contains", "symbolizes",
        "derives_from", "opposes", "supports", "precedes", "follows",
        "competes_with", "cooperates_with", "influences", "replaces",
        "connects_to", "separates_from", "amplifies", "dampens",
        "originates_in", "terminates_at", "overlaps_with", "penetrates",
        "reflects", "absorbs", "emits", "catalyzes", "inhibits",
        "activates", "regulates", "codes_for",
        # Дополнительные отношения
        "supervises", "mentors", "advises", "funds", "audits",
        "licenses", "certifies", "inspects", "repairs", "cleans",
        "filters", "synthesizes", "decomposes", "isolates", "combines",
        "migrates_to", "hibernates_in", "pollinates", "germinates_in",
        "weathers", "erodes", "sediments", "crystallizes", "ionizes",
        "magnetizes", "polarizes", "resonates_with", "interferes_with",
        "diffracts", "refracts",
    ]

    return entities, relations


def build_corpus(data_seed: int, max_n: int):
    """
    Генерирует детерминированный пул уникальных триплетов.

    Алгоритм:
      1. Фиксируем списки entities/relations.
      2. data_seed задаёт перемешивание (shuffle) entities и relations.
      3. Генерируем комбинации (s, r, o) с гарантией уникальности triple.
      4. Берём первые max_n + запас для эпизодов.
    """
    entities, relations = _build_word_lists()
    rng = np.random.default_rng(data_seed)

    # Детерминированно перемешиваем
    ents = list(entities)
    rels = list(relations)
    rng.shuffle(ents)
    rng.shuffle(rels)

    triples = []
    seen = set()

    # Генерируем triplets: используем data_seed RNG для выбора индексов —
    # гарантирует разнообразие всех трёх компонент (s, r, o).
    e_len = len(ents)
    r_len = len(rels)
    rng_gen = np.random.default_rng(data_seed)

    # Генерируем с запасом (self-loops + повторы отсеются)
    safety = 0
    while len(triples) < max_n:
        si = int(rng_gen.integers(0, e_len))
        ri = int(rng_gen.integers(0, r_len))
        oi = int(rng_gen.integers(0, e_len))
        if si == oi:  # избегаем self-loop
            continue
        key = (ents[si], rels[ri], ents[oi])
        if key not in seen:
            seen.add(key)
            triples.append(key)
        safety += 1
        if safety > max_n * 20:
            break

    # Собираем уникальные тексты (для эмбеддинга)
    unique_texts = set()
    for s, r, o in triples:
        unique_texts.add(s)
        unique_texts.add(r)
        unique_texts.add(o)

    # Добавляем дополнительные тексты для эпизодов (берём из entities)
    episode_texts = list(ents[:500]) if len(ents) >= 500 else list(ents)

    for t in episode_texts:
        unique_texts.add(t)

    return {
        "triples": triples,
        "unique_texts": sorted(unique_texts),
        "entities": ents,
        "relations": rels,
        "episode_items": episode_texts,
    }


# ===========================================================================
# Эмбеддинг с кэшем
# ===========================================================================
def build_cached_embedder(texts: list[str], model_name: str = EMBEDDER_MODEL):
    """
    Предсчитывает эмбеддинги всех texts ОДИН раз.
    Возвращает embed_fn(text) -> np.ndarray (из кэша).
    """
    from sentence_transformers import SentenceTransformer

    print(f"  [embed] Загрузка модели {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"  [embed] Эмбеддинг {len(texts)} уникальных текстов...")
    t0 = time.perf_counter()
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    elapsed = time.perf_counter() - t0
    print(f"  [embed] Готово за {elapsed:.1f} с, dim={embs.shape[1]}.")

    cache: dict[str, np.ndarray] = {}
    for text, emb in zip(texts, embs):
        cache[text] = emb.astype(np.float32)
    # также по lower (VSAMemory использует strip().lower() для ключей)
    for text, emb in zip(texts, embs):
        key = text.strip().lower()
        if key not in cache:
            cache[key] = emb.astype(np.float32)

    def embed_fn(text: str) -> np.ndarray:
        k = text.strip().lower()
        if k in cache:
            return cache[k]
        # fallback (не должно случаться)
        e = model.encode(text, normalize_embeddings=True).astype(np.float32)
        cache[k] = e
        return e

    return embed_fn, embs.shape[1], cache


# ===========================================================================
# E1 — Facts structural recall vs N
# ===========================================================================
def run_e1(
    cached_fn: Callable[[str], np.ndarray],
    triples: list[tuple[str, str, str]],
    ns: tuple[int, ...],
    seeds: list[int],
    query_sample: int,
) -> list[dict]:
    """E1: масштабирование ёмкости фактов."""
    from astrum_verum.vsa.memory import VSAMemory

    rows: list[dict] = []
    total = len(seeds) * len(ns)
    done = 0

    for seed in seeds:
        # Для каждого seed создаём отдельные VSAMemory для каждого N
        # (следуем спецификации буквально)
        for n in ns:
            t0 = time.perf_counter()
            mem = VSAMemory(
                D=D,
                seed=seed,
                normalize_threshold=NORMALIZE_THRESHOLD,
                embed_fn=cached_fn,
            )
            # Добавляем n триплетов
            for s, r, o in triples[:n]:
                mem.add_triple(s, r, o)

            # Случайная подвыборка для query (≥ query_sample или все)
            n_query = min(query_sample, n)
            query_rng = np.random.default_rng(seed * 1000 + n)
            query_indices = query_rng.choice(n, n_query, replace=False)

            correct = 0
            for qi in query_indices:
                s, r, o = triples[qi]
                result = mem.query({"subject": s, "relation": r}, "object")
                if result["answer"] == o:
                    correct += 1

            accuracy = correct / n_query
            elapsed = time.perf_counter() - t0
            done += 1

            rows.append({
                "experiment": "E1",
                "parameter": "N",
                "param_value": n,
                "seed": seed,
                "metric": "accuracy",
                "value": accuracy,
                "n_facts": n,
                "n_queried": n_query,
                "correct": correct,
                "elapsed_s": round(elapsed, 2),
            })
            print(f"  E1 N={n:>5} seed={seed:>3}  acc={accuracy:.4f}  "
                  f"({done}/{total}, {elapsed:.1f}s)")

    return rows


# ===========================================================================
# E2 — Episode order-recall vs length
# ===========================================================================
def run_e2(
    cached_fn: Callable[[str], np.ndarray],
    episode_items: list[str],
    ls: tuple[int, ...],
    seeds: list[int],
    window: int,
) -> list[dict]:
    """E2: позиционный recall эпизодов + проверка bounded window."""
    from astrum_verum.vsa.memory import VSAMemory

    rows: list[dict] = []
    total = len(seeds) * len(ls) * 2  # *2: full + window
    done = 0

    for seed in seeds:
        for L in ls:
            t0 = time.perf_counter()

            # --- Full episode ---
            # Берём L уникальных items (циклически, если пул меньше L)
            items = []
            pool = list(episode_items)
            for i in range(L):
                items.append(pool[i % len(pool)])

            mem = VSAMemory(
                D=D,
                seed=seed,
                normalize_threshold=NORMALIZE_THRESHOLD,
                embed_fn=cached_fn,
            )
            eid = mem.add_episode(items, episode_id="ep_full")

            pos_correct = 0
            for pos in range(L):
                recalled = mem.recall_at(eid, pos)
                if recalled == items[pos]:
                    pos_correct += 1
            full_accuracy = pos_correct / L

            elapsed_full = time.perf_counter() - t0
            done += 1

            rows.append({
                "experiment": "E2",
                "parameter": "L",
                "param_value": L,
                "seed": seed,
                "metric": "accuracy_full",
                "value": full_accuracy,
                "length": L,
                "correct_positions": pos_correct,
                "elapsed_s": round(elapsed_full, 2),
            })
            print(f"  E2 L={L:>4} seed={seed:>3}  full_acc={full_accuracy:.4f}  "
                  f"({done}/{total}, {elapsed_full:.1f}s)")

            # --- Bounded window (W=150) ---
            t1 = time.perf_counter()
            window_items = items[-window:] if L > window else items
            actual_window = len(window_items)

            mem_w = VSAMemory(
                D=D,
                seed=seed,
                normalize_threshold=NORMALIZE_THRESHOLD,
                embed_fn=cached_fn,
            )
            eid_w = mem_w.add_episode(window_items, episode_id="ep_window")

            w_correct = 0
            for pos in range(actual_window):
                recalled = mem_w.recall_at(eid_w, pos)
                if recalled == window_items[pos]:
                    w_correct += 1
            window_accuracy = w_correct / actual_window if actual_window > 0 else 1.0

            elapsed_window = time.perf_counter() - t1
            done += 1

            rows.append({
                "experiment": "E2",
                "parameter": "L",
                "param_value": L,
                "seed": seed,
                "metric": "accuracy_window_W150",
                "value": window_accuracy,
                "length": L,
                "window_size": actual_window,
                "correct_positions": w_correct,
                "elapsed_s": round(elapsed_window, 2),
            })
            print(f"  E2 L={L:>4} seed={seed:>3}  window_acc(W={actual_window})={window_accuracy:.4f}  "
                  f"({done}/{total}, {elapsed_window:.1f}s)")

    return rows


# ===========================================================================
# E3 — SimHash grounding fidelity
# ===========================================================================
def run_e3(
    cached_fn: Callable[[str], np.ndarray],
    texts: list[str],
    seeds: list[int],
    m: int,
) -> list[dict]:
    """E3: корреляция cos(emb) vs cos(ground(emb)) для M текстов."""
    from astrum_verum.vsa.core import ground, make_projection

    rows: list[dict] = []

    # Берём первые M текстов
    selected = texts[:m]
    actual_m = len(selected)

    # Предсчитываем эмбеддинги
    print(f"  [E3] Предсчёт эмбеддингов {actual_m} текстов...")
    embs = np.stack([cached_fn(t) for t in selected])  # (M, emb_dim)
    emb_dim = embs.shape[1]

    # Попарные косинусы в embedding-space (один раз)
    e_cos = embs @ embs.T  # (M, M)
    iu = np.triu_indices(actual_m, k=1)

    total = len(seeds)
    done = 0
    for seed in seeds:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        proj = make_projection(emb_dim, D, rng)

        # Ground все эмбеддинги
        atoms = np.stack([ground(e, proj) for e in embs])  # (M, D)
        a_cos = (atoms @ atoms.T) / D  # косинус в hypervector-space

        # Корреляции
        e_flat = e_cos[iu]
        a_flat = a_cos[iu]

        pearson = float(np.corrcoef(e_flat, a_flat)[0, 1])
        # Spearman через scipy (если есть) или numpy-реализацию
        try:
            from scipy.stats import spearmanr
            spearman, _ = spearmanr(e_flat, a_flat)
            spearman = float(spearman)
        except ImportError:
            # Ручная реализация Spearman rank correlation
            def rankdata(x):
                order = np.argsort(x)
                ranks = np.empty_like(order, dtype=np.float64)
                ranks[order] = np.arange(1, len(x) + 1)
                # средний ранг для ties
                return ranks

            r_e = rankdata(e_flat)
            r_a = rankdata(a_flat)
            spearman = float(np.corrcoef(r_e, r_a)[0, 1])

        elapsed = time.perf_counter() - t0
        done += 1

        rows.append({
            "experiment": "E3",
            "parameter": "M",
            "param_value": actual_m,
            "seed": seed,
            "metric": "pearson",
            "value": pearson,
            "elapsed_s": round(elapsed, 2),
        })
        rows.append({
            "experiment": "E3",
            "parameter": "M",
            "param_value": actual_m,
            "seed": seed,
            "metric": "spearman",
            "value": spearman,
            "elapsed_s": round(elapsed, 2),
        })
        print(f"  E3 M={actual_m} seed={seed:>3}  pearson={pearson:.6f}  "
              f"spearman={spearman:.6f}  ({done}/{total}, {elapsed:.1f}s)")

    return rows


# ===========================================================================
# Агрегация и отчёт
# ===========================================================================
def aggregate(rows: list[dict]) -> dict:
    """Группирует per-seed rows → сводка (experiment, param, metric) → stats."""
    groups: dict[tuple[str, str, int, str], list[float]] = {}
    for r in rows:
        key = (r["experiment"], r["parameter"], r["param_value"], r["metric"])
        groups.setdefault(key, []).append(r["value"])

    stats = {}
    for k, vals in groups.items():
        arr = np.array(vals)
        stats[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n_seeds": len(arr),
        }
    return stats


def write_results(rows: list[dict], stats: dict, meta: dict) -> None:
    """Пишет OUT_MD и OUT_CSV."""

    # --- CSV ---
    fieldnames = [
        "experiment", "parameter", "param_value", "seed", "metric",
        "value", "n_facts", "n_queried", "length", "window_size",
        "correct", "correct_positions", "elapsed_s",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[CSV]  {OUT_CSV}  ({len(rows)} rows)")

    # --- MD ---
    lines: list[str] = []
    def emit(s: str = "") -> None:
        lines.append(s)

    emit("# VSA Memory Benchmark — результаты")
    emit()
    emit("## Параметры прогона")
    emit()
    for k, v in meta.items():
        emit(f"- **{k}**: {v}")
    emit()

    # E1
    emit("## E1 — Facts structural recall vs N (ёмкость)")
    emit()
    emit("| N (фактов) | Mean Acc | Std | Min | Max | n_seeds |")
    emit("|------------|----------|-----|-----|-----|---------|")
    e1_ordered = sorted(
        [k for k in stats if k[0] == "E1"],
        key=lambda k: k[2],
    )
    for k in e1_ordered:
        s = stats[k]
        emit(f"| {k[2]:>5} | {s['mean']:.4f} | {s['std']:.4f} | "
             f"{s['min']:.4f} | {s['max']:.4f} | {s['n_seeds']} |")
    emit()

    # E2
    emit("## E2 — Episode order-recall vs length")
    emit()
    emit("### Полный эпизод (accuracy_full)")
    emit()
    emit("| L (длина) | Mean Acc | Std | Min | Max | n_seeds |")
    emit("|-----------|----------|-----|-----|-----|---------|")
    e2_full = sorted(
        [k for k in stats if k[0] == "E2" and k[3] == "accuracy_full"],
        key=lambda k: k[2],
    )
    for k in e2_full:
        s = stats[k]
        emit(f"| {k[2]:>5} | {s['mean']:.4f} | {s['std']:.4f} | "
             f"{s['min']:.4f} | {s['max']:.4f} | {s['n_seeds']} |")
    emit()
    emit("### Окно W=150 (accuracy_window_W150)")
    emit()
    emit("| L (полная длина) | Mean Acc | Std | Min | Max | n_seeds |")
    emit("|------------------|----------|-----|-----|-----|---------|")
    e2_win = sorted(
        [k for k in stats if k[0] == "E2" and k[3] == "accuracy_window_W150"],
        key=lambda k: k[2],
    )
    for k in e2_win:
        s = stats[k]
        emit(f"| {k[2]:>5} | {s['mean']:.4f} | {s['std']:.4f} | "
             f"{s['min']:.4f} | {s['max']:.4f} | {s['n_seeds']} |")
    emit()

    # E3
    emit("## E3 — SimHash grounding fidelity")
    emit()
    emit(f"| Метрика | Mean Corr | Std | Min | Max | n_seeds |")
    emit(f"|---------|-----------|-----|-----|-----|---------|")
    e3_keys = sorted(
        [k for k in stats if k[0] == "E3"],
        key=lambda k: (k[3], k[2]),
    )
    for k in e3_keys:
        s = stats[k]
        emit(f"| {k[3]} | {s['mean']:.6f} | {s['std']:.6f} | "
             f"{s['min']:.6f} | {s['max']:.6f} | {s['n_seeds']} |")
    emit()

    # Честность
    emit("## Замечания о честности")
    emit()
    emit("- Все seed (включая худшие) включены в отчёт.")
    emit("- Raw per-seed данные записаны в CSV для перепроверки.")
    emit("- Эмбеддинги: реальные, через sentence-transformers.")
    emit(f"- CORPUS: детерминированно сгенерирован из реальных слов (data_seed={DATA_SEED}).")
    emit("- Код VSAMemory не модифицирован.")
    emit()

    report = "\n".join(lines)
    OUT_MD.write_text(report)
    print(f"[MD]   {OUT_MD}")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="VSA Memory Benchmark — статистически валидный бенчмарк"
    )
    parser.add_argument(
        "--seeds", type=int, default=None,
        help=f"Число seed (default: {QUICK_SEEDS} для --quick, иначе 30)",
    )
    parser.add_argument(
        "--max-n", type=int, default=None,
        help=f"Максимальное N фактов (default: {QUICK_MAX_N} для --quick, иначе {E1_NS[-1]})",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Быстрый прогон для самопроверки (мало seeds, мало N)",
    )
    parser.add_argument(
        "--skip-e1", action="store_true", help="Пропустить E1"
    )
    parser.add_argument(
        "--skip-e2", action="store_true", help="Пропустить E2"
    )
    parser.add_argument(
        "--skip-e3", action="store_true", help="Пропустить E3"
    )
    args = parser.parse_args()

    # Режим
    quick = args.quick
    n_seeds = args.seeds or (QUICK_SEEDS if quick else 30)
    max_n = args.max_n or (QUICK_MAX_N if quick else E1_NS[-1])

    if quick:
        ns = tuple(n for n in QUICK_NS if n <= max_n)
        ls = tuple(l for l in QUICK_LS if l <= max_n)
        e3_m = QUICK_E3_M
    else:
        ns = tuple(n for n in E1_NS if n <= max_n)
        ls = E2_LS
        e3_m = E3_M

    seeds = list(range(n_seeds))

    # Метаинформация
    start_time = datetime.now(timezone.utc)
    meta = {
        "date_utc": start_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "embedder_model": EMBEDDER_MODEL,
        "n_seeds": n_seeds,
        "D": D,
        "max_N": max_n,
        "normalize_threshold": NORMALIZE_THRESHOLD,
        "data_seed": DATA_SEED,
        "quick_mode": quick,
        "ns": ns,
        "ls": ls,
        "e3_m": e3_m,
        "machine": "Apple M4, 16 GB RAM, arm64",
    }

    print("=" * 68)
    print("  VSA MEMORY BENCHMARK")
    print("=" * 68)
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print()

    # --- CORPUS ---
    print("[1/5] Генерация CORPUS...")
    t0 = time.perf_counter()
    corpus = build_corpus(DATA_SEED, max_n)
    meta["corpus_triples"] = len(corpus["triples"])
    meta["corpus_unique_texts"] = len(corpus["unique_texts"])
    meta["corpus_entities"] = len(corpus["entities"])
    meta["corpus_relations"] = len(corpus["relations"])
    print(f"  Триплетов: {meta['corpus_triples']}, "
          f"уникальных текстов: {meta['corpus_unique_texts']}, "
          f"({time.perf_counter() - t0:.1f}s)")

    # --- Эмбеддинги ---
    print("\n[2/5] Предсчёт эмбеддингов...")
    t0 = time.perf_counter()
    cached_fn, emb_dim, emb_cache = build_cached_embedder(
        corpus["unique_texts"], EMBEDDER_MODEL
    )
    meta["emb_dim"] = emb_dim
    meta["embedding_time_s"] = round(time.perf_counter() - t0, 1)
    print(f"  dim={emb_dim}, time={meta['embedding_time_s']}s")

    all_rows: list[dict] = []

    # --- E1 ---
    if not args.skip_e1:
        print(f"\n[3/5] E1 — Facts structural recall vs N "
              f"(N∈{ns}, seeds={n_seeds})...")
        t0 = time.perf_counter()
        e1_rows = run_e1(cached_fn, corpus["triples"], ns, seeds, E1_QUERY_SAMPLE)
        meta["e1_time_s"] = round(time.perf_counter() - t0, 1)
        all_rows.extend(e1_rows)
        print(f"  E1 done: {len(e1_rows)} rows, {meta['e1_time_s']}s")
    else:
        print("\n[3/5] E1 — SKIPPED")

    # --- E2 ---
    if not args.skip_e2:
        print(f"\n[4/5] E2 — Episode order-recall vs length "
              f"(L∈{ls}, seeds={n_seeds})...")
        t0 = time.perf_counter()
        e2_rows = run_e2(cached_fn, corpus["episode_items"], ls, seeds, E2_WINDOW)
        meta["e2_time_s"] = round(time.perf_counter() - t0, 1)
        all_rows.extend(e2_rows)
        print(f"  E2 done: {len(e2_rows)} rows, {meta['e2_time_s']}s")
    else:
        print("\n[4/5] E2 — SKIPPED")

    # --- E3 ---
    if not args.skip_e3:
        print(f"\n[5/5] E3 — SimHash grounding fidelity "
              f"(M={e3_m}, seeds={n_seeds})...")
        t0 = time.perf_counter()
        e3_rows = run_e3(cached_fn, corpus["unique_texts"], seeds, e3_m)
        meta["e3_time_s"] = round(time.perf_counter() - t0, 1)
        all_rows.extend(e3_rows)
        print(f"  E3 done: {len(e3_rows)} rows, {meta['e3_time_s']}s")
    else:
        print("\n[5/5] E3 — SKIPPED")

    # --- Агрегация и отчёт ---
    print("\n[AGG] Агрегация и запись отчёта...")
    meta["total_time_s"] = round(
        (datetime.now(timezone.utc) - start_time).total_seconds(), 1
    )
    stats = aggregate(all_rows)
    write_results(all_rows, stats, meta)

    # --- Сводка в консоль ---
    print("\n" + "=" * 68)
    print("  СВОДКА")
    print("=" * 68)

    # E1
    if not args.skip_e1:
        print("\n## E1 — Facts structural recall vs N")
        print(f"  {'N':>6} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        for k in sorted([k for k in stats if k[0] == "E1"], key=lambda k: k[2]):
            s = stats[k]
            print(f"  {k[2]:>6} | {s['mean']:8.4f} | {s['std']:8.4f} | "
                  f"{s['min']:8.4f} | {s['max']:8.4f}")

    # E2
    if not args.skip_e2:
        print("\n## E2 — Episode order-recall (full)")
        print(f"  {'L':>6} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        for k in sorted(
            [k for k in stats if k[0] == "E2" and k[3] == "accuracy_full"],
            key=lambda k: k[2],
        ):
            s = stats[k]
            print(f"  {k[2]:>6} | {s['mean']:8.4f} | {s['std']:8.4f} | "
                  f"{s['min']:8.4f} | {s['max']:8.4f}")
        print("\n## E2 — Episode order-recall (window W=150)")
        print(f"  {'L':>6} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        for k in sorted(
            [k for k in stats if k[0] == "E2" and k[3] == "accuracy_window_W150"],
            key=lambda k: k[2],
        ):
            s = stats[k]
            print(f"  {k[2]:>6} | {s['mean']:8.4f} | {s['std']:8.4f} | "
                  f"{s['min']:8.4f} | {s['max']:8.4f}")

    # E3
    if not args.skip_e3:
        print("\n## E3 — SimHash grounding fidelity")
        print(f"  {'Metric':>12} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
        print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        for k in sorted([k for k in stats if k[0] == "E3"], key=lambda k: (k[3], k[2])):
            s = stats[k]
            print(f"  {k[3]:>12} | {s['mean']:8.6f} | {s['std']:8.6f} | "
                  f"{s['min']:8.6f} | {s['max']:8.6f}")

    print(f"\n  Файлы:")
    print(f"    {OUT_MD}")
    print(f"    {OUT_CSV}")
    print(f"  Общее время: {meta['total_time_s']}s")
    print("=" * 68)


if __name__ == "__main__":
    main()
