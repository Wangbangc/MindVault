"""P0-1 跨进程丢更新回归测试。

锁定缺陷:MemoryManager 启动时整份读入 memories.json，之后从不 reload，
每次写又整列表覆写。两个进程(Streamlit + MCP)各持独立实例指向同一目录时，
谁后写谁就用陈旧内存把对方的新记忆静默清空。

修复:写盘前在跨进程文件锁内 reload-merge(按 id 合并),配合 session
tombstone 防止删除/归档被对方旧内存复活。

测试用 MemoryManager.__new__() 绕过重 __init__(不加载 bge / 不连 LLM),
手动装配字段。两个实例指向同一 tmp dir 即模拟两个进程。
"""

import os
import json

import numpy as np

from memory import MemoryManager
from filelock import FileLock


class _StubEmbed:
    """可控的假嵌入器:available=True 但产出确定向量,避免下载模型。"""

    available = True

    def embed(self, texts):
        # 形状对齐即可(merge/save 路径不依赖具体值)。
        return np.zeros((len(texts), 4)) if texts else np.array([])

    def embed_query(self, query):
        return np.zeros(4)


def _make_mm(data_dir):
    """构造一个最小可用的 MemoryManager,指向 data_dir。"""
    mm = MemoryManager.__new__(MemoryManager)
    mm._data_dir = data_dir
    os.makedirs(data_dir, exist_ok=True)
    mm._memories_path = os.path.join(data_dir, "memories.json")
    mm._archive_path = os.path.join(data_dir, "memories_archive.json")
    mm._memories = []
    mm._archive = {"archived_at": None, "memories": []}
    mm._memory_vectors = None
    mm._embedder = _StubEmbed()
    mm._pending_access_writes = 0
    mm._access_flush_threshold = 20
    mm._decay_threshold = 0.15
    mm._write_lock = FileLock(os.path.join(data_dir, ".lock"))
    mm._active_removed_ids = set()
    mm._archive_removed_ids = set()
    return mm


def _read_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return {m["id"] for m in json.load(f)}


def _mem(mid, content="x", category="general"):
    return {
        "id": mid,
        "content": content,
        "category": category,
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_accessed_at": "2026-01-01T00:00:00+00:00",
        "access_count": 0,
        "decay_score": 1.0,
    }


def test_concurrent_add_no_lost_update(tmp_path):
    """核心回归:两进程各加一条不同记忆,谁也不该被对方覆盖清空。

    旧代码(整列表覆写、无 reload)下:B 后写会用只含 b 的快照覆盖磁盘,
    A 写入的 a 永久丢失 → 断言失败。新代码 reload-merge → 两条都在。
    """
    d = str(tmp_path / "data")
    a = _make_mm(d)
    b = _make_mm(d)

    a._memories = [_mem("a")]
    a._save_memories()                      # 磁盘 = {a}
    b._memories = [_mem("b")]               # b 启动时磁盘还没有 a 之后才写? 模拟陈旧:b 不含 a
    b._save_memories()                      # 旧代码: 覆写成 {b}; 新代码: merge 成 {a,b}

    assert _read_ids(a._memories_path) == {"a", "b"}


def test_deleted_memory_not_resurrected(tmp_path):
    """A 删除 x 后,另一进程 B 把 x 写回磁盘;A 再落盘不得复活 x。

    这是 tombstone 的核心价值:reload-merge 会看到磁盘上 B 写回的 x
    (磁盘有、A 内存无),若无 tombstone 就会把它当“其他进程的新增”并回。
    无 tombstone 时本条失败。
    """
    d = str(tmp_path / "data")
    a = _make_mm(d)
    b = _make_mm(d)

    a._memories = [_mem("x"), _mem("y")]
    a._save_memories()                      # 磁盘 = {x, y}

    a.delete_memory("x")                    # A 删 x → tombstone(x), 磁盘 = {y}

    b._memories = [_mem("x"), _mem("y")]    # B 陈旧:仍持有 x
    b._save_memories()                      # B 把 x 写回 → 磁盘 = {x, y}

    a._save_memories()                      # reload 看到磁盘的 x;tombstone 应排除
    assert "x" not in _read_ids(a._memories_path)
    assert "x" not in {m["id"] for m in a._memories}
    assert "y" in _read_ids(a._memories_path)


def test_archive_no_lost_update(tmp_path):
    """两进程各归档不同记忆,归档文件不得互相覆盖丢失。"""
    d = str(tmp_path / "data")
    a = _make_mm(d)
    b = _make_mm(d)

    a._archive = {"archived_at": "t1", "memories": [_mem("m1")]}
    a._save_archive()                       # 磁盘 archive = {m1}
    b._archive = {"archived_at": "t2", "memories": [_mem("m2")]}  # B 陈旧:只有 m2
    b._save_archive()                       # 旧代码覆写丢 m1; 新代码 union → {m1, m2}

    with open(a._archive_path, "r", encoding="utf-8") as f:
        ids = {m["id"] for m in json.load(f)["memories"]}
    assert ids == {"m1", "m2"}


def test_restored_memory_not_pushed_back_to_archive(tmp_path):
    """A restore r 到 active 后,另一进程 B 把 r 写回 archive;A 再落盘不得复活到归档。

    archive 侧 tombstone 的核心价值:无它时 reload-merge 会把 B 写回的 r
    并回归档,造成 r 同时存在于 active 和 archive。无 tombstone 时本条失败。
    """
    d = str(tmp_path / "data")
    a = _make_mm(d)
    b = _make_mm(d)

    a._archive = {"archived_at": "t", "memories": [_mem("r")]}
    a._save_archive()                       # 磁盘 archive = {r}
    a._memories = []
    a.restore_from_archive("r")             # r → active, archive_removed_ids={r}, 磁盘 archive = {}

    b._archive = {"archived_at": "t", "memories": [_mem("r")]}  # B 陈旧:archive 仍含 r
    b._save_archive()                       # B 把 r 写回 → 磁盘 archive = {r}

    a._save_archive()                       # reload 看到磁盘的 r;archive tombstone 应排除
    with open(a._archive_path, "r", encoding="utf-8") as f:
        arch_ids = {m["id"] for m in json.load(f)["memories"]}
    assert "r" not in arch_ids
    assert "r" in {m["id"] for m in a._memories}


def test_rebuild_skipped_when_no_remote_merge(tmp_path, monkeypatch):
    """无 remote-only 记忆并入时,_save_memories 不应触发 _rebuild_embeddings。"""
    d = str(tmp_path / "data")
    a = _make_mm(d)
    a._memories = [_mem("a")]
    a._save_memories()                      # 磁盘 = {a}

    calls = {"n": 0}
    monkeypatch.setattr(a, "_rebuild_embeddings", lambda: calls.__setitem__("n", calls["n"] + 1))

    a._save_memories()                      # 磁盘已是 {a},无新增 → 不重建
    assert calls["n"] == 0

    # 制造一次真正的 remote 并入:直接往磁盘塞一条 a 不知道的记忆
    with open(a._memories_path, "w", encoding="utf-8") as f:
        json.dump([_mem("a"), _mem("remote")], f)
    a._save_memories()                      # 并入 remote → 重建一次
    assert calls["n"] == 1
