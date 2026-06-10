"""检索写放大修复的回归测试。

验证 search_memories 不再每次命中都全量写盘，而是按阈值批量落盘；
access_count/last_accessed_at 的内存语义不变；flush 保证最终一致。

用 __new__ 绕过重型 __init__，embedder 不可用走纯关键词/计数路径，
并把 _save_memories 替换为计数器，统计真实落盘次数。
"""

from datetime import datetime, timezone

import numpy as np

from memory import MemoryManager


class _StubEmbed:
    """可用的 stub embedder：查询向量与记忆向量点积≈1，确保走 hybrid RRF 路径。"""

    available = True

    def embed_query(self, query):
        return np.array([1.0, 0.0, 0.0])


def _make_mm(threshold=20):
    mm = MemoryManager.__new__(MemoryManager)
    mm._memories = []
    mm._memory_vectors = None
    mm._pending_access_writes = 0
    mm._access_flush_threshold = threshold
    mm._data_dir = "/tmp/_mm_test_unused"
    mm._memories_path = "/tmp/_mm_test_unused/memories.json"
    mm._embedder = _StubEmbed()

    # 用计数器替换真实落盘，记录写盘次数
    mm._save_calls = 0

    def _fake_save():
        mm._save_calls += 1
        mm._pending_access_writes = 0  # 复刻真实 _save_memories 的清零语义

    mm._save_memories = _fake_save
    return mm


def _seed(mm, n=3):
    now = datetime.now(timezone.utc).isoformat()
    for i in range(n):
        mm._memories.append({
            "id": f"m{i}",
            "content": f"记忆内容关键词{i} python 编程",
            "category": "general",
            "created_at": now,
            "last_accessed_at": now,
            "access_count": 0,
            "decay_score": 1.0,
        })
    # 设置与查询向量同向的记忆向量矩阵，保证 vec_results 非空 → 走 hybrid 路径
    mm._memory_vectors = np.array([[1.0, 0.0, 0.0] for _ in range(n)])


def test_search_does_not_write_every_call():
    """阈值内的多次搜索不应触发任何全量落盘。"""
    mm = _make_mm(threshold=20)
    _seed(mm)
    for _ in range(19):
        mm.search_memories("python", top_k=3, hybrid=True)
    assert mm._save_calls == 0, f"阈值内不应落盘，实际落盘 {mm._save_calls} 次"


def test_search_flushes_at_threshold():
    """累计到阈值时落盘一次，并重置脏计数。"""
    mm = _make_mm(threshold=20)
    _seed(mm)
    for _ in range(20):
        mm.search_memories("python", top_k=3, hybrid=True)
    assert mm._save_calls == 1, f"到阈值应落盘 1 次，实际 {mm._save_calls}"
    assert mm._pending_access_writes == 0, "落盘后脏计数应清零"


def test_access_count_still_increments_in_memory():
    """access_count 的内存语义不变：每次命中仍 +1。"""
    mm = _make_mm(threshold=20)
    _seed(mm)
    for _ in range(5):
        mm.search_memories("python", top_k=3, hybrid=True)
    total = sum(m["access_count"] for m in mm._memories)
    assert total > 0, "命中记忆的 access_count 应被累加"


def test_flush_access_stats_persists_pending():
    """flush_access_stats 在有未落盘统计时强制写盘一次。"""
    mm = _make_mm(threshold=20)
    _seed(mm)
    mm.search_memories("python", top_k=3, hybrid=True)
    assert mm._pending_access_writes > 0
    mm.flush_access_stats()
    assert mm._save_calls == 1
    assert mm._pending_access_writes == 0


def test_flush_noop_when_nothing_pending():
    """没有未落盘统计时 flush 不应写盘。"""
    mm = _make_mm(threshold=20)
    _seed(mm)
    mm.flush_access_stats()
    assert mm._save_calls == 0
