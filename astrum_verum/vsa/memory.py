"""
VSAMemory — полная композиционно-эпизодическая когнитивная память (Phase 3).

Не демо: факты (role-binding) + эпизоды (порядок через permutation) +
нормализация сущностей (грязные варианты → один канон) + персистентность
(переживает сессии). Валидировано Phase 0/1/2 (алгебра держит, grounding
выжил, дискриминатор бьёт косинус на реальном извлечении).

Геометрия (MAP, биполярная, D=10k):
    fact      = bundle( R_subj⊗a(s), R_rel⊗a(r), R_obj⊗a(o) )
    episode   = bundle( ρ⁰a(e₀), ρ¹a(e₁), …, ρⁿa(eₙ) )   (ρ = циклический сдвиг)
    a(concept)= ground(embedding) — SimHash; близкие концепты → близкие атомы.

Эмбеддер инъектируемый (`embed_fn`) — для детерминированных тестов без модели;
по умолчанию sentence-transformers (lazy).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from . import core


class VSAMemory:
    ROLE_NAMES = ("subject", "relation", "object")

    def __init__(
        self,
        D: int = core.DEFAULT_D,
        embedder_model: str = "all-MiniLM-L6-v2",
        seed: int = 0,
        normalize_threshold: float = 0.82,
        embed_fn: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.D = D
        self._embedder_model = embedder_model
        self._normalize_threshold = float(normalize_threshold)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._embed_fn = embed_fn
        self._model = None
        self._emb_dim: int | None = None
        self._proj: np.ndarray | None = None

        self._roles = {n: core.random_atoms(1, D, self._rng)[0] for n in self.ROLE_NAMES}

        # Канонический кодбук концептов.
        self._names: list[str] = []
        self._kinds: list[str] = []          # "entity" | "relation" | "event"
        self._atoms: list[np.ndarray] = []
        self._embs: list[np.ndarray] = []
        self._index: dict[str, int] = {}     # surface (lower) → canonical idx
        self._aliases: dict[int, list[str]] = {}

        # Факты и эпизоды.
        self._fact_idx: list[tuple[int, int, int]] = []
        self._fact_vecs: list[np.ndarray] = []
        self._fact_meta: list[dict] = []
        self._episodes: dict[str, dict] = {}
        self._ep_counter = 0

    # ------------------------------------------------------------------
    # Эмбеддинг / grounding
    # ------------------------------------------------------------------
    def _ensure_proj(self, emb_dim: int) -> None:
        if self._proj is None:
            self._emb_dim = emb_dim
            self._proj = core.make_projection(emb_dim, self.D, self._rng)

    def _embed(self, text: str) -> np.ndarray:
        if self._embed_fn is not None:
            e = np.asarray(self._embed_fn(text), dtype=np.float32)
            n = np.linalg.norm(e)
            e = e / n if n > 0 else e
        else:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._embedder_model)
            e = self._model.encode(text, normalize_embeddings=True).astype(np.float32)
        self._ensure_proj(e.shape[0])
        return e

    # ------------------------------------------------------------------
    # Кодбук с нормализацией сущностей
    # ------------------------------------------------------------------
    def _concept_index(self, concept: str, kind: str) -> int:
        """Канонический индекс концепта. Грязные варианты, чей эмбеддинг ближе
        normalize_threshold к существующему концепту того же типа, сливаются."""
        key = concept.strip().lower()
        if key in self._index:
            return self._index[key]
        emb = self._embed(concept)

        same = [i for i, k in enumerate(self._kinds) if k == kind]
        if same:
            E = np.stack([self._embs[i] for i in same])
            sims = E @ emb
            j = int(np.argmax(sims))
            if float(sims[j]) >= self._normalize_threshold:
                canon = same[j]
                self._index[key] = canon
                if key != self._names[canon].strip().lower():
                    self._aliases.setdefault(canon, []).append(concept)
                return canon

        idx = len(self._names)
        self._names.append(concept)
        self._kinds.append(kind)
        self._atoms.append(core.ground(emb, self._proj))
        self._embs.append(emb)
        self._index[key] = idx
        return idx

    def _cleanup(self, vec: np.ndarray, kind: str) -> tuple[str, float, int]:
        idxs = [i for i, k in enumerate(self._kinds) if k == kind]
        if not idxs:
            return "", 0.0, -1
        cb = np.stack([self._atoms[i] for i in idxs])
        sims = (cb @ vec) / self.D
        b = int(np.argmax(sims))
        return self._names[idxs[b]], float(sims[b]), idxs[b]

    # ------------------------------------------------------------------
    # Факты
    # ------------------------------------------------------------------
    def add_triple(self, subject: str, relation: str, obj: str, meta: dict | None = None) -> int:
        si = self._concept_index(subject, "entity")
        ri = self._concept_index(relation, "relation")
        oi = self._concept_index(obj, "entity")
        fact = core.bundle(np.stack([
            core.bind(self._roles["subject"], self._atoms[si]),
            core.bind(self._roles["relation"], self._atoms[ri]),
            core.bind(self._roles["object"], self._atoms[oi]),
        ]), self._rng)
        self._fact_idx.append((si, ri, oi))
        self._fact_vecs.append(fact)
        self._fact_meta.append(meta or {})
        return len(self._fact_vecs) - 1

    def query(self, known: dict[str, str], target_role: str) -> dict:
        """known: {role: concept}; возвращает восстановленный концепт target_role
        + извлечённый триплет. Различает (X,r,Y) и (Y,r,X) по РОЛЯМ."""
        if not self._fact_vecs:
            return {"answer": None, "score": 0.0, "triple": None, "fact_idx": -1}
        mem = np.stack(self._fact_vecs)
        terms = []
        for role, concept in known.items():
            kind = "relation" if role == "relation" else "entity"
            ci = self._concept_index(concept, kind)
            terms.append(core.bind(self._roles[role], self._atoms[ci]))
        probe = core.bundle(np.stack(terms), self._rng)
        fidx = int(np.argmax((mem @ probe) / self.D))
        kind = "relation" if target_role == "relation" else "entity"
        unbound = core.unbind(self._fact_vecs[fidx], self._roles[target_role])
        name, score, _ = self._cleanup(unbound, kind)
        si, ri, oi = self._fact_idx[fidx]
        return {
            "answer": name,
            "score": score,
            "triple": (self._names[si], self._names[ri], self._names[oi]),
            "fact_idx": fidx,
        }

    # ------------------------------------------------------------------
    # Эпизоды (порядок через permutation)
    # ------------------------------------------------------------------
    def add_episode(self, items: list[str], episode_id: str | None = None) -> str:
        idxs = [self._concept_index(it, "event") for it in items]
        vec = core.bundle(
            np.stack([core.permute(self._atoms[i], pos) for pos, i in enumerate(idxs)]),
            self._rng,
        )
        eid = episode_id or f"ep{self._ep_counter}"
        self._ep_counter += 1
        self._episodes[eid] = {"item_idx": idxs, "vec": vec}
        return eid

    def recall_at(self, episode_id: str, pos: int) -> str:
        ep = self._episodes[episode_id]
        if pos < 0 or pos >= len(ep["item_idx"]):
            return ""
        return self._cleanup(core.unpermute(ep["vec"], pos), "event")[0]

    def episode_order(self, episode_id: str) -> list[str]:
        ep = self._episodes[episode_id]
        return [self.recall_at(episode_id, p) for p in range(len(ep["item_idx"]))]

    def successor(self, episode_id: str, item: str) -> str | None:
        ep = self._episodes[episode_id]
        n = len(ep["item_idx"])
        a = self._atoms[self._concept_index(item, "event")]
        sims = [(core.unpermute(ep["vec"], p) @ a) / self.D for p in range(n)]
        pos = int(np.argmax(sims))
        return self.recall_at(episode_id, pos + 1) if pos + 1 < n else None

    # ------------------------------------------------------------------
    # Персистентность (память переживает сессии)
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            base.with_suffix(".npz"),
            proj=self._proj if self._proj is not None else np.zeros((0, 0), np.float32),
            roles=np.stack([self._roles[n] for n in self.ROLE_NAMES]).astype(np.int8),
            embs=(np.stack(self._embs) if self._embs
                  else np.zeros((0, self._emb_dim or 1), np.float32)),
        )
        meta = {
            "D": self.D, "seed": self._seed, "emb_dim": self._emb_dim,
            "embedder_model": self._embedder_model,
            "normalize_threshold": self._normalize_threshold,
            "role_names": list(self.ROLE_NAMES),
            "names": self._names, "kinds": self._kinds,
            "index": self._index,
            "aliases": {str(k): v for k, v in self._aliases.items()},
            "fact_idx": self._fact_idx, "fact_meta": self._fact_meta,
            "episodes": {k: v["item_idx"] for k, v in self._episodes.items()},
            "ep_counter": self._ep_counter,
        }
        base.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path, embed_fn: Callable[[str], np.ndarray] | None = None) -> "VSAMemory":
        base = Path(path)
        arr = np.load(base.with_suffix(".npz"))
        meta = json.loads(base.with_suffix(".json").read_text())

        m = cls(D=meta["D"], embedder_model=meta.get("embedder_model", "all-MiniLM-L6-v2"),
                seed=meta["seed"], normalize_threshold=meta["normalize_threshold"],
                embed_fn=embed_fn)
        m._emb_dim = meta["emb_dim"]
        m._proj = arr["proj"]
        roles = arr["roles"].astype(np.float32)
        m._roles = {n: roles[i] for i, n in enumerate(meta["role_names"])}
        m._names = list(meta["names"])
        m._kinds = list(meta["kinds"])
        m._index = {k: int(v) for k, v in meta["index"].items()}
        m._aliases = {int(k): v for k, v in meta["aliases"].items()}
        embs = arr["embs"]
        m._embs = [embs[i] for i in range(len(m._names))]
        m._atoms = [core.ground(e, m._proj) for e in m._embs]  # детерминированно

        m._fact_idx = [tuple(t) for t in meta["fact_idx"]]
        m._fact_meta = meta["fact_meta"]
        m._fact_vecs = [
            core.bundle(np.stack([
                core.bind(m._roles["subject"], m._atoms[si]),
                core.bind(m._roles["relation"], m._atoms[ri]),
                core.bind(m._roles["object"], m._atoms[oi]),
            ]), m._rng)
            for (si, ri, oi) in m._fact_idx
        ]
        m._episodes = {}
        for eid, idxs in meta["episodes"].items():
            idxs = list(idxs)
            vec = core.bundle(
                np.stack([core.permute(m._atoms[i], pos) for pos, i in enumerate(idxs)]),
                m._rng,
            )
            m._episodes[eid] = {"item_idx": idxs, "vec": vec}
        m._ep_counter = meta["ep_counter"]
        return m

    # ------------------------------------------------------------------
    @property
    def n_facts(self) -> int:
        return len(self._fact_vecs)

    @property
    def n_concepts(self) -> int:
        return len(self._names)

    @property
    def triples(self) -> list[tuple[str, str, str]]:
        return [(self._names[s], self._names[r], self._names[o]) for s, r, o in self._fact_idx]

    def aliases_of(self, concept: str) -> list[str]:
        key = concept.strip().lower()
        idx = self._index.get(key)
        return self._aliases.get(idx, []) if idx is not None else []

    def __repr__(self) -> str:
        return (f"VSAMemory(D={self.D}, facts={self.n_facts}, "
                f"concepts={self.n_concepts}, episodes={len(self._episodes)})")
