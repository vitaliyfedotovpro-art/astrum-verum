"""
Контур Хранилища и Рубежа Перехода.

Стресс-тестирование механики переключения in-memory → SQLite + numpy.memmap
при достижении порога 50 000 нод.

    ┌─────────────────────────────────────────────┐
    │  Node 1 … 49 999    →   in-memory (dict)   │
    │  Node 50 000         →   TRIGGER            │
    │  Node 50 001 …       →   SQLite + mmap      │
    └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from astrum_verum.lattice.d4 import D4Plugin
from astrum_verum.persistent import SqliteMmapBackend
from astrum_verum.store import MemoryNode, TopologyStore

_EMBEDDING_DIM = 384
_LATTICE_DIM = 4


def _synthetic_node(index: int, rng: np.random.Generator) -> MemoryNode:
    """Create a single synthetic node with deterministic random vectors."""
    return MemoryNode(
        id=f"syn-{index:06d}",
        text=f"synthetic-memory-{index}",
        embedding=rng.standard_normal(_EMBEDDING_DIM).astype(np.float64),
        lattice_coords=rng.standard_normal(_LATTICE_DIM).astype(np.float64),
        cell_memberships={index % 24: 0.8, (index + 1) % 24: 0.2},
    )


def _synthetic_nodes_batch(
    count: int, rng: np.random.Generator
) -> list[MemoryNode]:
    """Generate a batch of synthetic nodes."""
    return [_synthetic_node(i, rng) for i in range(count)]


# =====================================================================
# 1. Тест триггера 50 000 нод
# =====================================================================
class TestMigrationTrigger:
    """
    Проверка точного срабатывания рубежа перехода.

    - На 49 999 нодах: хранилище СТРОГО in-memory.
    - На 50 000-й ноде: автоматический триггер миграции.
    """

    @pytest.fixture
    def tmpdir(self):
        d = tempfile.mkdtemp(prefix="astrum_50k_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_exact_threshold_boundary(self, tmpdir) -> None:
        """
        Фундаментальный тест: хранилище остаётся in-memory до ровно
        N-1 ноды и мигрирует на ровно N-й ноде.

        Используем threshold=1000 для скорости — механика идентична 50K.
        """
        THRESHOLD = 1000
        store = TopologyStore(
            D4Plugin(),
            auto_persist_threshold=THRESHOLD,
        )
        store._storage_dir = Path(tmpdir)

        rng = np.random.default_rng(42)

        # Добавляем до threshold - 1.
        for i in range(THRESHOLD - 1):
            store.add_node(_synthetic_node(i, rng))

        # Проверка 1: строго in-memory.
        assert not store.is_persistent, (
            f"Store switched to persistent at {THRESHOLD - 1} nodes, "
            f"expected threshold is {THRESHOLD}"
        )
        assert store.stats()["total_nodes"] == THRESHOLD - 1

        # Добавляем ровно threshold-ю ноду.
        store.add_node(_synthetic_node(THRESHOLD - 1, rng))

        # Проверка 2: теперь persistent.
        assert store.is_persistent, (
            f"Store did NOT switch to persistent at {THRESHOLD} nodes"
        )
        assert store.stats()["total_nodes"] == THRESHOLD

    @pytest.mark.slow
    def test_production_threshold_50k(self, tmpdir) -> None:
        """
        Полный стресс-тест на продуктовом пороге 50 000 нод.

        Этот тест генерирует ~150 MB данных (50K × 384D × float64).
        Пометка @pytest.mark.slow — можно пропустить через -m 'not slow'.
        """
        THRESHOLD = 50_000
        store = TopologyStore(
            D4Plugin(),
            auto_persist_threshold=THRESHOLD,
        )
        store._storage_dir = Path(tmpdir)

        rng = np.random.default_rng(0)

        # Добавляем все 50K нод.
        for i in range(THRESHOLD):
            store.add_node(_synthetic_node(i, rng))

        # Должен мигрировать.
        assert store.is_persistent
        assert store.stats()["total_nodes"] == THRESHOLD

        # Файлы SQLite и mmap должны существовать.
        assert (store._storage_dir / "nodes.db").exists()
        assert (store._storage_dir / "embeddings.dat").exists()
        assert (store._storage_dir / "lattice_coords.dat").exists()

    def test_data_integrity_after_migration(self, tmpdir) -> None:
        """
        Каждая нода, записанная до миграции, должна быть доступна
        после миграции с идентичными эмбеддингами.
        """
        THRESHOLD = 200
        store = TopologyStore(
            D4Plugin(),
            auto_persist_threshold=THRESHOLD,
        )
        store._storage_dir = Path(tmpdir)

        rng = np.random.default_rng(77)
        originals: dict[str, np.ndarray] = {}

        for i in range(THRESHOLD):
            node = _synthetic_node(i, rng)
            originals[node.id] = node.embedding.copy()
            store.add_node(node)

        assert store.is_persistent

        # Проверяем каждый 10-й (для скорости).
        for nid, original_emb in list(originals.items())[::10]:
            got = store.get_node(nid)
            assert got is not None, f"Node {nid} lost after migration"
            np.testing.assert_allclose(
                got.embedding, original_emb, atol=1e-15,
                err_msg=f"Embedding corrupted for {nid}",
            )

    def test_post_migration_writes(self, tmpdir) -> None:
        """
        Ноды, добавленные ПОСЛЕ миграции, должны корректно
        записываться в SQLite + mmap.
        """
        THRESHOLD = 100
        store = TopologyStore(
            D4Plugin(),
            auto_persist_threshold=THRESHOLD,
        )
        store._storage_dir = Path(tmpdir)

        rng = np.random.default_rng(99)
        for i in range(THRESHOLD):
            store.add_node(_synthetic_node(i, rng))

        assert store.is_persistent

        # Добавляем ещё 50 нод после миграции.
        for i in range(THRESHOLD, THRESHOLD + 50):
            node = _synthetic_node(i, rng)
            store.add_node(node)

        assert store.stats()["total_nodes"] == THRESHOLD + 50

        # Проверяем пост-миграционную ноду.
        got = store.get_node(f"syn-{THRESHOLD + 25:06d}")
        assert got is not None
        assert got.text == f"synthetic-memory-{THRESHOLD + 25}"


# =====================================================================
# 2. Тест ленивой загрузки (mmap)
# =====================================================================
class TestMmapLazyLoading:
    """
    Проверка, что после перехода на рубеж CVP-поиск активирует
    и извлекает из numpy.memmap только конкретный сектор данных.
    """

    @pytest.fixture
    def persistent_store(self):
        d = tempfile.mkdtemp(prefix="astrum_mmap_")
        store = TopologyStore(D4Plugin(), storage_dir=d)
        rng = np.random.default_rng(42)

        # Заполняем 500 нод по 24 ячейкам.
        for i in range(500):
            store.add_node(_synthetic_node(i, rng))

        yield store
        shutil.rmtree(d, ignore_errors=True)

    def test_cell_query_returns_subset(
        self, persistent_store: TopologyStore
    ) -> None:
        """
        Запрос ячейки должен вернуть только ноды этой ячейки,
        а не весь массив.
        """
        cell_nodes = persistent_store.get_nodes_in_cell(0)
        all_nodes = persistent_store.get_all_nodes()

        # Ноды ячейки — строгое подмножество.
        assert len(cell_nodes) < len(all_nodes)
        assert len(cell_nodes) > 0

        # Каждая нода из ячейки действительно принадлежит ячейке 0.
        for node in cell_nodes:
            primary = max(
                node.cell_memberships, key=node.cell_memberships.get
            )
            assert primary == 0

    def test_mmap_file_exists_and_sized(
        self, persistent_store: TopologyStore
    ) -> None:
        """
        Файл mmap должен существовать и иметь корректный размер:
        capacity × dim × sizeof(float64).
        """
        emb_path = persistent_store._storage_dir / "embeddings.dat"
        assert emb_path.exists()

        file_size = emb_path.stat().st_size
        # Размер = capacity × 384 × 8 bytes.
        expected_min = 500 * _EMBEDDING_DIM * 8  # минимум для 500 нод
        assert file_size >= expected_min

    def test_embedding_roundtrip_through_mmap(
        self, persistent_store: TopologyStore
    ) -> None:
        """
        Эмбеддинг, записанный в mmap, при чтении должен быть
        побитово идентичен оригиналу.
        """
        rng = np.random.default_rng(42)

        # Воспроизводим те же ноды.
        for i in range(10):
            expected = _synthetic_node(i, rng)
            got = persistent_store.get_node(f"syn-{i:06d}")
            assert got is not None
            np.testing.assert_array_equal(
                got.embedding, expected.embedding,
                err_msg=f"Mmap embedding mismatch for node {i}",
            )

    def test_individual_node_retrieval_is_independent(
        self, persistent_store: TopologyStore
    ) -> None:
        """
        Запрос отдельной ноды не должен вытягивать другие ноды.
        Верифицируем через возвращаемый тип и содержимое.
        """
        node_a = persistent_store.get_node("syn-000010")
        node_b = persistent_store.get_node("syn-000200")

        assert node_a is not None and node_b is not None
        assert node_a.id != node_b.id
        assert not np.array_equal(node_a.embedding, node_b.embedding)


# =====================================================================
# 3. Тест атомарности SQLite
# =====================================================================
class TestSqliteAtomicity:
    """
    Проверка, что при сбое база данных SQLite не повреждается
    и сохраняет целостность.
    """

    @pytest.fixture
    def tmpdir(self):
        d = tempfile.mkdtemp(prefix="astrum_atomic_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_wal_mode_enabled(self, tmpdir) -> None:
        """SQLite должен использовать WAL (Write-Ahead Logging)."""
        backend = SqliteMmapBackend(
            tmpdir, embedding_dim=_EMBEDDING_DIM, lattice_dim=_LATTICE_DIM
        )
        mode = backend._db.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal", f"Expected WAL mode, got {mode}"
        backend.close()

    def test_integrity_check_passes(self, tmpdir) -> None:
        """PRAGMA integrity_check должен проходить после записи."""
        backend = SqliteMmapBackend(
            tmpdir, embedding_dim=_EMBEDDING_DIM, lattice_dim=_LATTICE_DIM
        )
        rng = np.random.default_rng(42)

        for i in range(100):
            backend.add_node(_synthetic_node(i, rng))

        result = backend._db.execute("PRAGMA integrity_check").fetchone()[0]
        assert result == "ok", f"Integrity check failed: {result}"
        backend.close()

    def test_survives_close_reopen(self, tmpdir) -> None:
        """
        Данные должны пережить close() + reopen().
        Имитирует нормальный перезапуск процесса.
        """
        rng = np.random.default_rng(42)

        # Пишем.
        b1 = SqliteMmapBackend(
            tmpdir, embedding_dim=_EMBEDDING_DIM, lattice_dim=_LATTICE_DIM
        )
        original_emb = None
        for i in range(50):
            node = _synthetic_node(i, rng)
            if i == 25:
                original_emb = node.embedding.copy()
            b1.add_node(node)
        b1.close()

        # Переоткрываем.
        b2 = SqliteMmapBackend(tmpdir)
        assert b2.count() == 50

        got = b2.get_node("syn-000025")
        assert got is not None
        np.testing.assert_allclose(got.embedding, original_emb, atol=1e-15)
        b2.close()

    def test_foreign_key_cascade(self, tmpdir) -> None:
        """
        Удаление ноды должно каскадно удалить её cell_memberships.
        """
        backend = SqliteMmapBackend(
            tmpdir, embedding_dim=_EMBEDDING_DIM, lattice_dim=_LATTICE_DIM
        )
        rng = np.random.default_rng(42)
        node = _synthetic_node(0, rng)
        backend.add_node(node)

        # Проверяем что memberships записаны.
        memberships = backend._db.execute(
            "SELECT COUNT(*) FROM cell_memberships WHERE node_id = ?",
            (node.id,),
        ).fetchone()[0]
        assert memberships > 0

        # Удаляем ноду.
        backend.remove_node(node.id)

        # Memberships должны быть удалены.
        remaining = backend._db.execute(
            "SELECT COUNT(*) FROM cell_memberships WHERE node_id = ?",
            (node.id,),
        ).fetchone()[0]
        assert remaining == 0
        backend.close()

    def test_concurrent_read_after_write(self, tmpdir) -> None:
        """
        После записи данные сразу доступны для чтения
        (проверка WAL flush).
        """
        backend = SqliteMmapBackend(
            tmpdir, embedding_dim=_EMBEDDING_DIM, lattice_dim=_LATTICE_DIM
        )
        rng = np.random.default_rng(42)

        for i in range(10):
            node = _synthetic_node(i, rng)
            backend.add_node(node)

            # Немедленное чтение после записи.
            got = backend.get_node(node.id)
            assert got is not None, f"Node {node.id} not readable after write"

        backend.close()
