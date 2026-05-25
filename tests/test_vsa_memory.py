"""Тесты VSA-слоя (Phase 3): факты, роли, эпизоды, нормализация, персистентность.

Большинство тестов детерминированы через инъекцию фейкового эмбеддера (без сети).
Тест нормализации — на реальной модели (skip, если sentence-transformers нет).
"""

import hashlib

import numpy as np
import pytest

from astrum_verum.vsa import VSAMemory

_EMB_DIM = 64


def fake_embed(text: str) -> np.ndarray:
    """Детерминированный псевдо-эмбеддинг: квазиортогональные единичные векторы."""
    h = int(hashlib.sha1(text.strip().lower().encode()).hexdigest(), 16) % (2**32)
    v = np.random.default_rng(h).standard_normal(_EMB_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def mem() -> VSAMemory:
    return VSAMemory(D=10_000, seed=1, embed_fn=fake_embed)


class TestBinding:
    def test_roundtrip(self):
        m = mem()
        m.add_triple("maya", "founded", "helix")
        r = m.query({"subject": "maya", "relation": "founded"}, "object")
        assert r["answer"].lower() == "helix"

    def test_role_sensitivity(self):
        """(A,r,B) и (B,r,A) — идентичный набор концептов, различаются ролями."""
        m = mem()
        m.add_triple("alice", "trusts", "bob")
        m.add_triple("bob", "trusts", "alice")
        assert m.query({"subject": "alice", "relation": "trusts"}, "object")["answer"].lower() == "bob"
        assert m.query({"subject": "bob", "relation": "trusts"}, "object")["answer"].lower() == "alice"
        assert m.query({"relation": "trusts", "object": "bob"}, "subject")["answer"].lower() == "alice"

    def test_capacity(self):
        m = mem()
        rng = np.random.default_rng(0)
        ents = [f"e{i}" for i in range(150)]
        rels = [f"r{i}" for i in range(20)]
        facts = []
        for i in range(100):  # уникальный subject → (subject,relation) однозначен
            s = f"subj{i}"
            r = rels[int(rng.integers(20))]
            o = ents[int(rng.integers(150))]
            m.add_triple(s, r, o)
            facts.append((s, r, o))
        ok = sum(
            m.query({"subject": s, "relation": r}, "object")["answer"].lower() == o.lower()
            for s, r, o in facts
        )
        assert ok / len(facts) >= 0.95


class TestEpisodes:
    def test_order_and_successor(self):
        m = mem()
        items = ["woke up", "drank coffee", "wrote code", "ran tests", "committed"]
        eid = m.add_episode(items)
        order = [x.lower() for x in m.episode_order(eid)]
        assert order == [i.lower() for i in items]
        assert m.successor(eid, "wrote code").lower() == "ran tests"
        assert m.successor(eid, "committed") is None  # последний


class TestNormalization:
    def test_merge_variants(self):
        pytest.importorskip("sentence_transformers")
        m = VSAMemory(D=10_000, seed=1, normalize_threshold=0.75)  # реальный эмбеддер
        i_exact_a = m._concept_index("Maya", "entity")
        i_exact_b = m._concept_index("maya", "entity")  # тот же ключ
        assert i_exact_a == i_exact_b
        a = m._concept_index("junior doctors", "entity")
        b = m._concept_index("the junior doctors", "entity")  # near-dup → слияние
        assert a == b
        c = m._concept_index("hammer", "entity")  # далёкий → отдельный
        assert c != a


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        m = mem()
        m.add_triple("maya", "founded", "helix")
        m.add_triple("alice", "trusts", "bob")
        m.add_triple("bob", "trusts", "alice")
        eid = m.add_episode(["a", "b", "c", "d"])

        p = tmp_path / "memory_state"
        m.save(p)
        m2 = VSAMemory.load(p, embed_fn=fake_embed)

        assert m2.n_facts == m.n_facts
        assert m2.n_concepts == m.n_concepts
        assert m2.query({"subject": "alice", "relation": "trusts"}, "object")["answer"].lower() == "bob"
        assert m2.query({"subject": "bob", "relation": "trusts"}, "object")["answer"].lower() == "alice"
        assert [x.lower() for x in m2.episode_order(eid)] == ["a", "b", "c", "d"]
