"""情景记忆进入 LLM 上下文的集成回归测试（修复“第三层记忆形同虚设”）。

此前上次会话摘要仅在 UI 显示，从未进入发给 LLM 的 system prompt。
本测试锁定：build_messages_with_memory 会把指定 thread 的情景摘要注入
system prompt；call_react 会从 LangGraph config 取出 thread_id 并传入。

注：导入 agent 会真实加载 embedding 模型（较慢）。设置 dummy key 以便在
master 分支（LLM 尚未懒初始化）上也能导入。
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-for-import")

import agent  # noqa: E402
from memory import EpisodicMemory  # noqa: E402


def test_episode_summary_reaches_system_prompt(tmp_path, monkeypatch):
    ep = EpisodicMemory(episodes_dir=str(tmp_path / "episodes"))
    ep.save("user_a", "上次我们确定了用 Milvus 替换 Chroma 的迁移方案")
    monkeypatch.setattr(agent, "episodic_memory", ep)

    # state 不含 HumanMessage → last_user_msg 为空 → 跳过向量检索，只验情景注入
    msgs = agent.build_messages_with_memory({"messages": []}, thread_id="user_a")
    system_text = msgs[0].content
    assert "上次会话回忆" in system_text, "情景摘要未注入 system prompt"
    assert "Milvus 替换 Chroma" in system_text


def test_unknown_thread_no_episode_injection(tmp_path, monkeypatch):
    ep = EpisodicMemory(episodes_dir=str(tmp_path / "episodes"))
    monkeypatch.setattr(agent, "episodic_memory", ep)
    msgs = agent.build_messages_with_memory({"messages": []}, thread_id="nobody")
    assert "上次会话回忆" not in msgs[0].content


def test_empty_thread_id_is_backward_compatible(tmp_path, monkeypatch):
    """不传 thread_id 时不应崩，也不注入（向后兼容）。"""
    ep = EpisodicMemory(episodes_dir=str(tmp_path / "episodes"))
    ep.save("user_a", "一些摘要")
    monkeypatch.setattr(agent, "episodic_memory", ep)
    msgs = agent.build_messages_with_memory({"messages": []}, thread_id="")
    assert "上次会话回忆" not in msgs[0].content


def test_call_react_extracts_thread_id_from_config(monkeypatch):
    """call_react 应从 LangGraph config.configurable.thread_id 取出并传入。"""
    captured = {}

    def _fake_build(state, thread_id=""):
        captured["thread_id"] = thread_id
        from langchain_core.messages import SystemMessage
        return [SystemMessage(content="x")]

    class _FakeAgent:
        def invoke(self, payload, config=None):
            return {"messages": payload["messages"]}

    monkeypatch.setattr(agent, "build_messages_with_memory", _fake_build)
    monkeypatch.setattr(agent, "react_agent", _FakeAgent())

    agent.call_react(
        {"messages": []},
        config={"configurable": {"thread_id": "user_xyz"}},
    )
    assert captured["thread_id"] == "user_xyz"


def test_call_react_handles_missing_config(monkeypatch):
    """config 为 None 时不崩，thread_id 退化为空。"""
    captured = {}

    def _fake_build(state, thread_id=""):
        captured["thread_id"] = thread_id
        from langchain_core.messages import SystemMessage
        return [SystemMessage(content="x")]

    class _FakeAgent:
        def invoke(self, payload, config=None):
            return {"messages": []}

    monkeypatch.setattr(agent, "build_messages_with_memory", _fake_build)
    monkeypatch.setattr(agent, "react_agent", _FakeAgent())

    agent.call_react({"messages": []}, config=None)
    assert captured["thread_id"] == ""
