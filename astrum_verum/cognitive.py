"""
CognitiveMemory — единый фасад композиционно-эпизодической памяти.

Связывает извлечение (LLM-триплеты) и VSA-слой в одну юзабельную память:

    mem = CognitiveMemory()
    mem.remember("Maya founded Helix. Iris mentored Maya.")
    mem.recall_object("Maya", "founded")        # → "Helix"
    mem.recall_subject("mentored", "Maya")      # → "Iris"
    mem.remember_conversation(["greeting", "asked about USG", "scheduled call"])
    mem.whats_next(episode_id, "asked about USG")  # → "scheduled call"
    mem.save("~/.astrum_verum/memory_state")       # переживает сессии

Извлечение (`remember`) требует LLM-ключ (DeepSeek→xAI→Groq из env/.env).
Запросы и эпизоды работают без сети. Это НЕ обрезанная версия: факты +
порядок + нормализация сущностей + персистентность.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .extract import extract_triples
from .vsa import VSAMemory


class CognitiveMemory:
    def __init__(
        self,
        D: int = 10_000,
        seed: int = 0,
        normalize_threshold: float = 0.82,
        embed_fn: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.vsa = VSAMemory(
            D=D, seed=seed, normalize_threshold=normalize_threshold, embed_fn=embed_fn
        )

    # ---- запись ----
    def remember(self, text: str) -> list[dict]:
        """Извлечь факты-триплеты из текста (живой LLM) и положить в память."""
        triples, _provider = extract_triples(text)
        for t in triples:
            self.vsa.add_triple(t["subject"], t["relation"], t["object"])
        return triples

    def remember_triple(self, subject: str, relation: str, obj: str) -> int:
        """Положить факт напрямую, без LLM (например, из готового графа)."""
        return self.vsa.add_triple(subject, relation, obj)

    def remember_conversation(self, turns: list[str], episode_id: str | None = None) -> str:
        """Запомнить упорядоченный эпизод (реплики/события)."""
        return self.vsa.add_episode(turns, episode_id)

    # ---- запросы (без сети) ----
    def recall_object(self, subject: str, relation: str) -> dict:
        return self.vsa.query({"subject": subject, "relation": relation}, "object")

    def recall_subject(self, relation: str, obj: str) -> dict:
        return self.vsa.query({"relation": relation, "object": obj}, "subject")

    def whats_next(self, episode_id: str, item: str) -> str | None:
        return self.vsa.successor(episode_id, item)

    def episode_order(self, episode_id: str) -> list[str]:
        return self.vsa.episode_order(episode_id)

    # ---- персистентность ----
    def save(self, path: str | Path) -> None:
        self.vsa.save(path)

    @classmethod
    def load(cls, path: str | Path, embed_fn: Callable[[str], np.ndarray] | None = None) -> "CognitiveMemory":
        obj = cls.__new__(cls)
        obj.vsa = VSAMemory.load(path, embed_fn=embed_fn)
        return obj

    def __repr__(self) -> str:
        return f"CognitiveMemory({self.vsa!r})"
