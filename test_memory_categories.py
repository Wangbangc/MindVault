"""记忆分类单一事实源（FACT_TYPES）的回归测试。

锁定 MCP server、UI 编辑、add_memory 共用同一套分类，杜绝“幽灵分类”
导致 UI 编辑时 list.index() 抛 ValueError 整页崩溃的问题。
"""

from memory import MemoryManager


def test_category_index_known_returns_position():
    for i, cat in enumerate(MemoryManager.FACT_TYPES):
        assert MemoryManager.category_index(cat) == i


def test_category_index_unknown_falls_back_to_default():
    """未知 / 历史遗留分类不得抛异常，应回退到默认分类下标。"""
    idx = MemoryManager.category_index("does-not-exist")
    assert idx == MemoryManager.FACT_TYPES.index(MemoryManager.DEFAULT_CATEGORY)


def test_default_category_is_in_fact_types():
    assert MemoryManager.DEFAULT_CATEGORY in MemoryManager.FACT_TYPES


def test_fact_types_covers_legacy_mcp_categories():
    """修复前 MCP 暴露的分类必须全部并入 FACT_TYPES，否则旧数据仍会割裂。"""
    legacy_mcp = {"general", "personal", "work", "code", "project"}
    legacy_core = {"preference", "fact", "plan", "relationship", "general"}
    missing = (legacy_mcp | legacy_core) - set(MemoryManager.FACT_TYPES)
    assert not missing, f"FACT_TYPES 漏掉历史分类: {missing}"


def test_add_memory_normalizes_unknown_category():
    """add_memory 收到未知分类时归一化为默认，合法分类保留。

    用 __new__ 绕过重型 __init__，embedder 不可用以跳过去重分支，
    直接断言写入 self._memories 的对象 category。
    """
    mm = MemoryManager.__new__(MemoryManager)
    mm._memories = []
    mm._memory_vectors = None

    class _NoEmbed:
        available = False

    mm._embedder = _NoEmbed()
    mm._add_embedding = lambda content: None
    mm._save_memories = lambda: None

    mm.add_memory("一条来自 MCP 的记忆", category="work")
    assert mm._memories[-1]["category"] == "work", "合法分类应被保留"

    mm.add_memory("未知分类的记忆", category="不存在的分类")
    assert mm._memories[-1]["category"] == MemoryManager.DEFAULT_CATEGORY, (
        "未知分类应归一化为默认"
    )
