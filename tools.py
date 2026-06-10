"""
ReAct Agent 工具定义
使用 @tool decorator，供 create_react_agent 绑定
"""

import ast
import operator
import os
from datetime import datetime, timezone, timedelta

from langchain_core.tools import tool

from knowledge_base import KnowledgeBase

# ─── 单例 KB ─────────────────────────────────────────────

_kb = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


# ─── 工具定义 ─────────────────────────────────────────────

@tool
def search_knowledge(query: str) -> str:
    """搜索文档知识库，查找与用户问题相关的文档内容。
    当用户询问已导入文档中的信息时使用此工具。"""
    kb = get_knowledge_base()
    results = kb.search(query)
    if not results:
        return "未在知识库中找到相关信息。"
    context = "\n\n---\n\n".join([r["text"] for r in results])
    sources = list({r["source"] for r in results})
    return f"找到 {len(results)} 条相关段落（来源：{', '.join(sources)}）：\n\n{context}"


@tool
def get_current_time() -> str:
    """获取当前北京时间。当用户询问当前时间、现在几点时使用此工具。"""
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S 北京时间")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。当用户要求进行数学计算时使用此工具。
    支持加减乘除、括号、幂运算等基本运算。"""
    # 安全运算符映射
    ALLOWED_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPS:
            left = _eval(node.left)
            right = _eval(node.right)
            return ALLOWED_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPS:
            return ALLOWED_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"不支持的表达式: {ast.dump(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
        # 整数结果不显示小数点
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return str(result)
    except ZeroDivisionError:
        return "错误：除数不能为零"
    except Exception as e:
        return f"计算错误：{e}"


@tool
def web_search(query: str) -> str:
    """搜索互联网获取实时信息。当需要查询当前事件、实时数据或知识库外的事实时使用此工具。

    Args:
        query: 搜索关键词。

    Returns:
        包含标题、摘要和URL的搜索结果。
    """
    # 优先使用 Tavily（如果配置了 API key）
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            response = client.search(query, max_results=5)
            results = response.get("results", [])
            if not results:
                return "未找到相关搜索结果。"
            lines = ["Source: Tavily\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "无标题")
                snippet = r.get("content", "无摘要")
                url = r.get("url", "")
                lines.append(f"{i}. {title}\n   摘要: {snippet}\n   URL: {url}")
            return "\n\n".join(lines)
        except Exception:
            pass  # fallback to DuckDuckGo

    # DuckDuckGo（免费，无需 API key）
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "未找到相关搜索结果。"
        lines = ["Source: DuckDuckGo\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            snippet = r.get("body", "无摘要")
            url = r.get("href", "")
            lines.append(f"{i}. {title}\n   摘要: {snippet}\n   URL: {url}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"网络搜索不可用，请稍后再试。错误: {e}"


@tool
def web_fetch(url: str) -> str:
    """从指定URL获取并提取可读内容。用于读取文章、文档或任何网页内容。

    Args:
        url: 完整的URL地址（如 https://example.com/article）。

    Returns:
        页面提取的纯文本内容。
    """
    # 优先使用 trafilatura（最佳文本提取）
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                return text[:8000]
    except Exception:
        pass  # fallback to BeautifulSoup

    # 回退到 BeautifulSoup
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # 移除 script 和 style 标签
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if text:
            return text[:8000]
        return "页面内容为空，无法提取文本。"
    except Exception as e:
        return f"无法获取页面内容: {e}"


# ─── 工具列表（供 create_react_agent 使用）────────────────

TOOLS = [search_knowledge, get_current_time, calculate, web_search, web_fetch]
