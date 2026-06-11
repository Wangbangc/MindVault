"""tools.py 共享 KnowledgeBase 注入的回归测试。

验证 search_knowledge 工具用的是被注入的实例，而不是自建的第二个实例
（否则 UI 上传的文档刷新不到工具侧的 BM25 索引，混合检索退化）。

用轻量 stub 冒充 KnowledgeBase，避免加载 embedding 模型 / ChromaDB。
"""

import tools


class _StubKB:
    """冒充 KnowledgeBase，记录 search 调用。"""

    def __init__(self, label):
        self.label = label
        self.search_calls = []

    def search(self, query):
        self.search_calls.append(query)
        return [{"text": f"hit from {self.label}", "source": "doc.md"}]


def setup_function(_):
    # 每个用例前清空模块级注入状态，避免相互污染
    tools._kb = None


def test_injected_instance_is_used():
    """注入后，get_knowledge_base 必须返回同一个实例，不另建。"""
    kb = _StubKB("injected")
    tools.set_knowledge_base(kb)
    assert tools.get_knowledge_base() is kb


def test_search_knowledge_queries_injected_kb():
    """search_knowledge 工具应命中被注入的实例。"""
    kb = _StubKB("injected")
    tools.set_knowledge_base(kb)
    # @tool 包装后用 .invoke 调用
    out = tools.search_knowledge.invoke({"query": "向量检索"})
    assert kb.search_calls == ["向量检索"], "工具没有查询注入的实例"
    assert "hit from injected" in out


def test_set_overrides_previous_instance():
    """再次注入应覆盖旧实例（单一事实源）。"""
    first = _StubKB("first")
    second = _StubKB("second")
    tools.set_knowledge_base(first)
    tools.set_knowledge_base(second)
    assert tools.get_knowledge_base() is second


def test_no_duplicate_instance_after_injection():
    """注入后多次 get 始终是同一个对象，绝不触发懒加载新建。"""
    kb = _StubKB("injected")
    tools.set_knowledge_base(kb)
    assert tools.get_knowledge_base() is tools.get_knowledge_base() is kb
