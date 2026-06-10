"""
ReAct Agent — 标准 Reasoning + Acting 循环
LLM 自主决定何时调用工具，支持多步工具调用
"""

import os
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from memory import MemoryManager, EpisodicMemory
from knowledge_base import KnowledgeBase
from retriever import UnifiedRetriever
from context_manager import ContextManager
from tools import TOOLS, set_knowledge_base

load_dotenv()


# ==================== 状态定义 ====================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ==================== 实例 ====================

memory_manager = MemoryManager(
    data_dir="./memory_data",
    llm_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
)

episodic_memory = EpisodicMemory(episodes_dir="./episodes")

knowledge_base = KnowledgeBase()

# 注入到 tools 模块，确保 search_knowledge 工具与此处、UI 操作同一个实例
set_knowledge_base(knowledge_base)

context_mgr = ContextManager(max_turns=20)

retriever = UnifiedRetriever(knowledge_base, memory_manager)

# LLM（streaming=True 用于流式输出）
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0.7,
    streaming=True,
)


# ==================== ReAct Agent ====================

SYSTEM_PROMPT = """你是一个个人知识库助手，帮助用户管理知识、回答问题、制定计划。
你会记住用户的历史偏好和重要信息，像一个贴心的助手一样提供个性化帮助。
你有以下工具可用，请根据需要自主决定是否调用：
- search_knowledge: 搜索已导入的文档知识库
- get_current_time: 获取当前北京时间
- calculate: 计算数学表达式
- web_search: 搜索互联网获取实时信息（当前事件、实时数据等）
- web_fetch: 从指定URL获取并提取网页内容
请用中文回复。"""

# 创建 ReAct Agent（LangGraph 标准实现）
react_agent = create_react_agent(llm, TOOLS)


def build_messages_with_memory(state: AgentState) -> list:
    """构建注入记忆上下文的消息列表"""
    messages = state["messages"]

    # 获取最后一条用户消息
    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    system_parts = [SYSTEM_PROMPT]

    # 统一检索：融合知识库和记忆
    if last_user_msg:
        unified_results = retriever.search(last_user_msg, top_k=5)
        context_block = retriever.format_for_prompt(unified_results)
        if context_block:
            system_parts.append(context_block)

    # 知识库状态提示
    if knowledge_base.count() > 0:
        system_parts.append(
            f"[知识库状态] 已导入 {knowledge_base.count()} 个文档分块，"
            f"来源：{', '.join(knowledge_base.list_sources())}。"
            "用户提问时可调用 search_knowledge 工具检索。"
        )

    # 构建完整消息列表
    full_messages = [SystemMessage(content="\n\n".join(system_parts))]
    full_messages.extend(messages)
    return full_messages


# ==================== 图构建 ====================

# max_turns=10 → 每轮 2 个 super-step（reason + act）→ recursion_limit = 10*2+1
MAX_RECURSION = 21


def call_react(state: AgentState) -> dict:
    """注入记忆后调用 ReAct Agent"""
    full_messages = build_messages_with_memory(state)
    result = react_agent.invoke(
        {"messages": full_messages},
        config={"recursion_limit": MAX_RECURSION},
    )
    return {"messages": result["messages"]}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_react)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


agent_graph = build_graph()
