"""MemoryManager LLM 懒初始化的回归测试。

确保纯检索 / CRUD 场景（含 MCP server 暴露的全部能力）无需 OPENAI_API_KEY
即可工作——LLM 仅在真正调用 extract_facts 时才被构造。

用 monkeypatch 把 ChatOpenAI 换成探针，统计构造次数，避免真实网络 / 凭证。
"""

import memory
from memory import MemoryManager


class _LLMProbe:
    """记录被构造的次数，并提供 invoke 占位。"""

    instances = 0

    def __init__(self, *args, **kwargs):
        type(self).instances += 1

    def invoke(self, prompt):
        class _R:
            content = "无"
        return _R()


def _make_mm(monkeypatch, tmp_path):
    """构造 MemoryManager，ChatOpenAI 被探针替换、embedder 置为不可用。"""
    monkeypatch.setattr(memory, "ChatOpenAI", _LLMProbe)
    _LLMProbe.instances = 0

    # 让 MemoryEmbedder 不加载真实模型
    class _NoEmbed:
        available = False
        dimension = 384

        def embed(self, texts):
            import numpy as np
            return np.array([])

        def embed_query(self, q):
            import numpy as np
            return np.array([])

    monkeypatch.setattr(memory, "MemoryEmbedder", lambda *a, **k: _NoEmbed())
    return MemoryManager(data_dir=str(tmp_path / "mem"))


def test_init_does_not_construct_llm(monkeypatch, tmp_path):
    """初始化不得构造 LLM（这是 MCP 无 key 启动的关键）。"""
    _make_mm(monkeypatch, tmp_path)
    assert _LLMProbe.instances == 0, "初始化时不应构造 ChatOpenAI"


def test_crud_and_search_do_not_construct_llm(monkeypatch, tmp_path):
    """add / search / stats 全程不触发 LLM 构造。"""
    mm = _make_mm(monkeypatch, tmp_path)
    mm.add_memory("用户用 Python 写后端", category="general")
    mm.search_memories("Python", top_k=3)
    mm.get_memory_stats()
    mm.list_by_category()
    assert _LLMProbe.instances == 0, "检索 / CRUD 不应构造 LLM"


def test_extract_facts_constructs_llm_lazily(monkeypatch, tmp_path):
    """只有 extract_facts 才懒构造 LLM，且只构造一次。"""
    mm = _make_mm(monkeypatch, tmp_path)
    assert _LLMProbe.instances == 0
    mm.extract_facts("User: 我喜欢猫\nAI: 好的")
    assert _LLMProbe.instances == 1, "首次用到时应构造一次"
    mm.extract_facts("User: 我也喜欢狗\nAI: 收到")
    assert _LLMProbe.instances == 1, "后续复用同一个 LLM 实例，不应重复构造"


def test_llm_property_returns_same_instance(monkeypatch, tmp_path):
    """llm property 多次访问返回同一个懒构造实例。"""
    mm = _make_mm(monkeypatch, tmp_path)
    first = mm.llm
    second = mm.llm
    assert first is second
    assert _LLMProbe.instances == 1
