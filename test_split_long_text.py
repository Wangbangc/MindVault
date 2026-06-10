"""_split_long_text 硬分割兜底分支的回归测试（issue #4）。

无换行、无句号的超长连续文本会走到字符硬分割分支，旧实现末段可能切出
一个长度 <= chunk_overlap 的冗余碎片。本测试锁定该行为已被修复。

不加载 embedding / ChromaDB——用 __new__ 绕过 __init__，只测纯切分逻辑。
"""

from knowledge_base import KnowledgeBase


def _make_kb(chunk_size=500, chunk_overlap=50):
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.chunk_size = chunk_size
    kb.chunk_overlap = chunk_overlap
    return kb


def test_hard_split_drops_redundant_trailing_fragment():
    """950 = 450*2 + 50：旧逻辑末段恰好 50 字符（==overlap）的碎片，应被丢弃。"""
    kb = _make_kb(chunk_size=500, chunk_overlap=50)
    parts = kb._split_long_text("甲" * 950)
    lens = [len(p) for p in parts]
    assert lens == [500, 500], f"应丢弃冗余末段，实际 {lens}"


def test_hard_split_no_fragment_across_sizes():
    """多种长度下，硬分割结果都不得含 <= chunk_overlap 的碎片。"""
    kb = _make_kb(chunk_size=500, chunk_overlap=50)
    for n in (901, 950, 1000, 1350, 1400):
        lens = [len(p) for p in kb._split_long_text("甲" * n)]
        assert all(l > kb.chunk_overlap for l in lens), f"n={n} 出现碎片: {lens}"


def test_hard_split_keeps_legit_trailing_segment():
    """末段长度 > overlap 时是合法块，不得被误删（回归保护）。"""
    kb = _make_kb(chunk_size=500, chunk_overlap=50)
    # 951 -> [500, 500, 51]，末段 51 > 50，保留
    assert [len(p) for p in kb._split_long_text("乙" * 951)] == [500, 500, 51]


def test_hard_split_no_content_loss():
    """丢弃的末段是纯冗余（被前块完整覆盖），不得丢失真实内容。"""
    kb = _make_kb(chunk_size=500, chunk_overlap=50)
    parts = kb._split_long_text("甲" * 950)
    # 全部由 '甲' 构成，且总覆盖 >= 原长（相邻块按 step=450 重叠）
    assert all(set(p) == {"甲"} for p in parts)
    assert sum(len(p) for p in parts) >= 950


def test_hard_split_single_chunk_not_dropped():
    """只切出一块时不得被 pop（边界：len(chunks) > 1 才丢弃）。"""
    kb = _make_kb(chunk_size=500, chunk_overlap=50)
    # 恰好 500 不进 _split_long_text 的硬分割（<=chunk_size 直接返回）；
    # 用一个略超 size 但第二块 > overlap 的长度验证不会清空到 0 块。
    parts = kb._split_long_text("甲" * 540)
    assert len(parts) >= 1 and parts  # 永不返回空
