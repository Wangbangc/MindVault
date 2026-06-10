"""
统一检索器 — 融合知识库和记忆的搜索结果

通过 RRF (Reciprocal Rank Fusion) 将 KB chunks 和 memories
合并为单一排序列表，供 agent 系统提示词使用。
"""

from typing import Optional


class UnifiedRetriever:
    """组合 KnowledgeBase 和 MemoryManager，提供统一检索接口。"""

    def __init__(self, knowledge_base, memory_manager):
        self._kb = knowledge_base
        self._mm = memory_manager

    def search(
        self,
        query: str,
        top_k: int = 5,
        include_kb: bool = True,
        include_memory: bool = True,
    ) -> list[dict]:
        """搜索两个来源并返回统一排序的结果列表。

        每条结果包含:
            - content: str
            - source: "kb" | "memory"
            - score: float (RRF 融合分数)
            - metadata: dict (来源特定信息)
        """
        kb_results: list[dict] = []
        mem_results: list[dict] = []

        # 搜索知识库
        if include_kb:
            try:
                raw = self._kb.search(query, top_k=top_k * 3, hybrid=True)
                for i, r in enumerate(raw):
                    kb_results.append({
                        "id": f"kb:{r.get('source', 'unknown')}:{i}",
                        "content": r["text"],
                        "source": "kb",
                        "score": r.get("score", 0.0),
                        "metadata": {"file": r.get("source", "unknown")},
                    })
            except Exception as e:
                print(f"KB 搜索异常: {e}")

        # 搜索记忆
        if include_memory:
            try:
                raw = self._mm.search_memories(query, top_k=top_k * 3, hybrid=True)
                for r in raw:
                    mem_results.append({
                        "id": f"mem:{r['id']}",
                        "content": r["content"],
                        "source": "memory",
                        "score": r.get("score", 0.0),
                        "metadata": {"category": r.get("category", "general")},
                    })
            except Exception as e:
                print(f"记忆搜索异常: {e}")

        # 如果只有一个来源有结果，直接返回
        if not kb_results:
            return mem_results[:top_k]
        if not mem_results:
            return kb_results[:top_k]

        # RRF 融合
        return self._rrf_merge(kb_results, mem_results, top_k)

    def _rrf_merge(self, kb_results: list[dict], mem_results: list[dict], top_k: int, k: int = 60) -> list[dict]:
        """RRF 融合两个来源的结果。"""
        # 按 score 降序排列并分配 rank
        kb_sorted = sorted(kb_results, key=lambda x: -x["score"])
        mem_sorted = sorted(mem_results, key=lambda x: -x["score"])

        fused: dict[str, tuple[dict, float]] = {}

        for rank, r in enumerate(kb_sorted):
            rrf_score = 1.0 / (k + rank + 1)
            fused[r["id"]] = (r, fused.get(r["id"], (r, 0))[1] + rrf_score)

        for rank, r in enumerate(mem_sorted):
            rrf_score = 1.0 / (k + rank + 1)
            if r["id"] in fused:
                existing, prev_score = fused[r["id"]]
                fused[r["id"]] = (existing, prev_score + rrf_score)
            else:
                fused[r["id"]] = (r, rrf_score)

        # 按 RRF 分数降序排列
        ranked = sorted(fused.values(), key=lambda x: -x[1])

        results = []
        for item, score in ranked[:top_k]:
            result = dict(item)
            result["score"] = round(score, 6)
            results.append(result)

        return results

    def format_for_prompt(self, results: list[dict]) -> str:
        """将统一结果格式化为系统提示词的一段上下文。

        如果结果为空，返回空字符串。
        """
        if not results:
            return ""

        lines = ["相关上下文（来自知识库和个人记忆）："]
        for i, r in enumerate(results, 1):
            tag = "记忆" if r["source"] == "memory" else "知识"
            meta = r.get("metadata", {})
            if r["source"] == "kb":
                label = f"[{tag}] {meta.get('file', '未知')}"
            else:
                label = f"[{tag}] {meta.get('category', 'general')}"
            content = r["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"{i}. {label} — {content}")

        return "\n".join(lines)
