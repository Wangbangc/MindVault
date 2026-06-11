"""
分层记忆系统
- 情景记忆 (Episodic): 会话摘要，JSON 文件存储（不变）
- 记忆管理器 (MemoryManager): 跨会话持久知识，JSON + 内存嵌入
  - 语义搜索（embedding + keyword 混合检索）
  - 衰减归档（自动遗忘不活跃记忆）
  - CRUD（增删改查、分类浏览）
"""

import os
import re
import uuid
import math
import numpy as np
from datetime import datetime, timezone
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from storage import atomic_write_json, load_json


# ==================== 情景记忆（不变）====================

class EpisodicMemory:
    """会话级别的摘要记忆，存储在 JSON 文件中"""

    def __init__(self, episodes_dir: str = "./episodes"):
        self.episodes_dir = episodes_dir
        os.makedirs(episodes_dir, exist_ok=True)

    def generate_summary(self, messages: list, llm: ChatOpenAI) -> str:
        """从对话历史中生成会话摘要"""
        recent = messages[-20:]
        conversation = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in recent
        ])
        prompt = f"""基于以下对话，写一段简要的会话摘要，包含：
1. 讨论了哪些主题
2. 用户做了什么决定或请求
3. 是否有未完成的任务
用中文，不超过150字。

对话：
{conversation}
"""
        return llm.invoke(prompt).content

    def save(self, thread_id: str, summary: str):
        """保存会话摘要"""
        path = os.path.join(self.episodes_dir, f"{thread_id}.json")
        data = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
        }
        atomic_write_json(path, data)

    def load(self, thread_id: str) -> Optional[dict]:
        """加载会话摘要"""
        path = os.path.join(self.episodes_dir, f"{thread_id}.json")
        return load_json(path, None)


# ==================== 记忆嵌入器 ====================

class MemoryEmbedder:
    """封装 SentenceTransformer 用于记忆的语义嵌入。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self._model_name = model_name
        self._available = False
        self._dimension = 384
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            self._available = True
        except Exception as e:
            print(f"MemoryEmbedder 加载失败，将回退到纯关键词搜索: {e}")
            self._model = None

    @property
    def available(self) -> bool:
        return self._available

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        """批量嵌入文档文本（不加指令前缀）。"""
        if not self._available or not texts:
            return np.array([])
        return self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        """嵌入查询文本（加 bge 指令前缀）。"""
        if not self._available:
            return np.array([])
        prefix = "为这个句子生成表示以用于检索相关文档："
        return self._model.encode(
            [prefix + query], show_progress_bar=False, normalize_embeddings=True
        )[0]


# ==================== 记忆管理器 ====================

class MemoryManager:
    """跨会话持久记忆管理器。

    存储: memories.json (活跃) + memories_archive.json (归档)
    搜索: embedding cosine + keyword Jaccard, RRF 融合
    衰减: 0.95^days_since_access + access_count bonus

    访问统计（access_count/last_accessed_at）为软状态：仅用于衰减排序，
    按阈值延迟落盘以消除搜索路径的写放大，进程退出前未刷盘的少量统计
    丢失不影响记忆内容。如需确保最终一致，可显式调用 flush_access_stats()。
    多进程（多个 MCP 客户端各持实例）下访问统计仍可能相互覆盖——这属于
    既有的并发写问题，需文件锁 / SQLite 另行解决，不在本机制范围内。
    """

    # 记忆分类的单一事实源（single source of truth）。
    # MCP server、UI 编辑、add_memory 均以此为准，避免出现某一侧能存入、
    # 另一侧无法识别的“幽灵分类”（曾导致 UI 编辑时 .index() 抛 ValueError）。
    FACT_TYPES = [
        "general",
        "preference",
        "fact",
        "plan",
        "relationship",
        "personal",
        "work",
        "code",
        "project",
    ]
    DEFAULT_CATEGORY = "general"

    @classmethod
    def category_index(cls, category: str) -> int:
        """返回分类在 FACT_TYPES 中的下标；未知分类回退到默认分类的下标。

        供 UI 下拉框反查使用，对历史数据 / 外部写入的未知分类做容错，
        避免 list.index() 抛 ValueError 导致整页崩溃。
        """
        try:
            return cls.FACT_TYPES.index(category)
        except ValueError:
            return cls.FACT_TYPES.index(cls.DEFAULT_CATEGORY)

    def __init__(
        self,
        data_dir: str = "./memory_data",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        llm_model: str = "gpt-4o-mini",
        decay_threshold: float = 0.15,
    ):
        self._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._memories_path = os.path.join(data_dir, "memories.json")
        self._archive_path = os.path.join(data_dir, "memories_archive.json")
        self._decay_threshold = decay_threshold

        # LLM for fact extraction
        self.llm = ChatOpenAI(model=llm_model, temperature=0)

        # Embedder
        self._embedder = MemoryEmbedder(embedding_model)

        # In-memory store
        self._memories: list[dict] = []
        self._memory_vectors: Optional[np.ndarray] = None  # (N, dim)
        self._archive: dict = {"archived_at": None, "memories": []}

        # 检索访问统计的延迟落盘计数：search_memories 每次命中会 bump
        # access_count/last_accessed_at，但不必每次都全量重写 memories.json。
        # 累计到阈值才刷盘，把“每次搜索一次全量写”降为“每 N 次搜索一次”。
        # 注意：必须在下方 run_decay_sweep() 之前初始化——该方法会调用
        # _save_memories()，而后者读写 _pending_access_writes。
        self._pending_access_writes = 0
        self._access_flush_threshold = 20

        # Load data
        self._load_memories()
        self._load_archive()

        # Build embedding index
        self._rebuild_embeddings()

        # Run decay sweep on startup
        self.run_decay_sweep()

    # ─── Memory Embedding & Search ─────────────────────────────

    def _rebuild_embeddings(self):
        """为所有活跃记忆计算嵌入并缓存。"""
        if not self._embedder.available or not self._memories:
            self._memory_vectors = None
            return
        texts = [m["content"] for m in self._memories]
        self._memory_vectors = self._embedder.embed(texts)

    def _add_embedding(self, content: str) -> Optional[np.ndarray]:
        """为单条记忆计算嵌入并追加到缓存。"""
        if not self._embedder.available:
            return None
        vec = self._embedder.embed([content])[0]
        if self._memory_vectors is not None:
            self._memory_vectors = np.vstack([self._memory_vectors, vec.reshape(1, -1)])
        else:
            self._memory_vectors = vec.reshape(1, -1)
        return vec

    def search_memories(self, query: str, top_k: int = 5, hybrid: bool = True) -> list[dict]:
        """搜索记忆，支持混合（embedding + keyword）检索。

        Returns: [{id, content, category, score, source:"memory"}, ...]
        """
        if not self._memories:
            return []

        if not query or not query.strip():
            # 空查询返回最近的记忆
            sorted_mems = sorted(self._memories, key=lambda m: m.get("created_at", ""), reverse=True)
            return [self._format_memory(m, 0.0) for m in sorted_mems[:top_k]]

        # Vector search
        vec_results: dict[int, float] = {}
        if self._embedder.available and self._memory_vectors is not None:
            q_vec = self._embedder.embed_query(query)
            if q_vec.size > 0:
                sims = np.dot(self._memory_vectors, q_vec)
                top_indices = np.argsort(sims)[::-1][:top_k * 3]
                for idx in top_indices:
                    if sims[idx] > 0.2:
                        vec_results[int(idx)] = float(sims[idx])

        # Keyword search (Jaccard)
        kw_results: dict[int, float] = {}
        query_words = set(self._tokenize(query))
        for i, mem in enumerate(self._memories):
            mem_words = set(self._tokenize(mem["content"]))
            if not mem_words:
                continue
            jaccard = len(query_words & mem_words) / max(len(query_words | mem_words), 1)
            if jaccard > 0:
                kw_results[i] = jaccard

        if not hybrid or not vec_results:
            # Pure keyword fallback
            if kw_results:
                sorted_kw = sorted(kw_results.items(), key=lambda x: -x[1])
                return [self._format_memory(self._memories[i], score) for i, score in sorted_kw[:top_k]]
            return []

        # RRF fusion
        K_RRF = 60
        all_indices = set(vec_results.keys()) | set(kw_results.keys())

        def _rank_dict(d: dict) -> dict[int, int]:
            sorted_items = sorted(d.items(), key=lambda x: -x[1])
            return {item[0]: rank + 1 for rank, item in enumerate(sorted_items)}

        vec_ranks = _rank_dict(vec_results)
        kw_ranks = _rank_dict(kw_results)

        fused: list[tuple[int, float]] = []
        for idx in all_indices:
            v_score = 1.0 / (K_RRF + vec_ranks.get(idx, 999))
            k_score = 1.0 / (K_RRF + kw_ranks.get(idx, 999))
            fused.append((idx, v_score + k_score))

        fused.sort(key=lambda x: -x[1])

        # Update access stats for returned memories
        results = []
        for idx, score in fused[:top_k]:
            mem = self._memories[idx]
            mem["last_accessed_at"] = datetime.now(timezone.utc).isoformat()
            mem["access_count"] = mem.get("access_count", 0) + 1
            results.append(self._format_memory(mem, score))

        # 访问统计已在内存更新；按阈值延迟落盘，避免每次搜索都全量重写。
        self._mark_access_dirty()
        return results

    def _format_memory(self, mem: dict, score: float) -> dict:
        return {
            "id": mem["id"],
            "content": mem["content"],
            "category": mem.get("category", "general"),
            "score": round(score, 4),
            "source": "memory",
        }

    def _tokenize(self, text: str) -> list[str]:
        """中英文分词。"""
        return re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text.lower())

    # ─── Memory CRUD ───────────────────────────────────────────

    def add_memory(self, content: str, category: str = "general") -> str:
        """添加一条记忆，自动去重。返回 memory_id。"""
        content = content.strip()
        if not content:
            return ""

        # 归一化分类：未知分类回退到默认，保证 FACT_TYPES 是唯一事实源，
        # 不让外部调用（如 MCP）写入 UI 无法识别的幽灵分类。
        if category not in self.FACT_TYPES:
            category = self.DEFAULT_CATEGORY

        # 去重检查（embedding 相似度 > 0.85 视为重复）
        if self._embedder.available and self._memory_vectors is not None and len(self._memories) > 0:
            q_vec = self._embedder.embed_query(content)
            if q_vec.size > 0:
                sims = np.dot(self._memory_vectors, q_vec)
                max_idx = int(np.argmax(sims))
                if sims[max_idx] > 0.85:
                    existing = self._memories[max_idx]
                    if existing.get("category") == category:
                        # 保留更长版本
                        if len(content) > len(existing["content"]):
                            existing["content"] = content
                            existing["last_accessed_at"] = datetime.now(timezone.utc).isoformat()
                            self._rebuild_embeddings()
                            self._save_memories()
                        return existing["id"]

        now = datetime.now(timezone.utc).isoformat()
        memory = {
            "id": str(uuid.uuid4()),
            "content": content,
            "category": category,
            "created_at": now,
            "last_accessed_at": now,
            "access_count": 0,
            "decay_score": 1.0,
        }
        self._memories.append(memory)
        self._add_embedding(content)
        self._save_memories()
        return memory["id"]

    def delete_memory(self, memory_id: str) -> bool:
        """删除一条记忆。"""
        for i, mem in enumerate(self._memories):
            if mem["id"] == memory_id:
                self._memories.pop(i)
                self._rebuild_embeddings()
                self._save_memories()
                return True
        return False

    def update_memory(self, memory_id: str, content: str = None, category: str = None) -> bool:
        """编辑记忆的内容和/或分类。内容变更时重新计算嵌入。"""
        for mem in self._memories:
            if mem["id"] == memory_id:
                changed = False
                if content is not None and content.strip() != mem["content"]:
                    mem["content"] = content.strip()
                    mem["last_accessed_at"] = datetime.now(timezone.utc).isoformat()
                    changed = True
                if category is not None and category != mem.get("category"):
                    mem["category"] = category
                    changed = True
                if changed:
                    self._rebuild_embeddings()
                    self._save_memories()
                return changed
        return False

    def get_memory_detail(self, memory_id: str) -> Optional[dict]:
        """返回单条记忆的完整详情。"""
        for mem in self._memories:
            if mem["id"] == memory_id:
                return dict(mem)
        return None

    def list_by_category(
        self,
        category: Optional[str] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict]:
        """列出记忆，可按分类过滤。"""
        items = self._memories if category is None else [
            m for m in self._memories if m.get("category") == category
        ]
        reverse = sort_order == "desc"
        items = sorted(items, key=lambda m: m.get(sort_by, ""), reverse=reverse)
        return items[offset: offset + limit]

    # ─── Decay & Archive ───────────────────────────────────────

    def run_decay_sweep(self, threshold: float = None) -> dict:
        """评估所有活跃记忆，将低分记忆归档。

        Returns: {archived_count, active_count}
        """
        if threshold is None:
            threshold = self._decay_threshold

        now = datetime.now(timezone.utc)
        to_archive = []
        remaining = []

        for mem in self._memories:
            score = self._compute_decay_score(mem, now)
            mem["decay_score"] = round(score, 4)
            if score < threshold:
                to_archive.append(mem)
            else:
                remaining.append(mem)

        if to_archive:
            archive_time = now.isoformat()
            for mem in to_archive:
                mem["archived_at"] = archive_time
                mem["decay_score_at_archive"] = mem["decay_score"]
            self._archive["archived_at"] = archive_time
            self._archive["memories"].extend(to_archive)
            self._memories = remaining
            self._rebuild_embeddings()
            self._save_memories()
            self._save_archive()

        return {
            "archived_count": len(to_archive),
            "active_count": len(self._memories),
        }

    def _compute_decay_score(self, mem: dict, now: datetime = None) -> float:
        """计算衰减分数。"""
        if now is None:
            now = datetime.now(timezone.utc)
        last_accessed = mem.get("last_accessed_at", mem.get("created_at", now.isoformat()))
        try:
            last_dt = datetime.fromisoformat(last_accessed)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            days = max((now - last_dt).days, 0)
        except (ValueError, TypeError):
            days = 0

        base_score = 1.0
        access_bonus = min(mem.get("access_count", 0) * 0.02, 0.3)
        score = base_score * (0.95 ** days) + access_bonus
        return max(0.0, min(score, 1.0))

    def restore_from_archive(self, memory_id: str) -> bool:
        """从归档恢复一条记忆到活跃状态。"""
        for i, mem in enumerate(self._archive["memories"]):
            if mem["id"] == memory_id:
                restored = self._archive["memories"].pop(i)
                restored["decay_score"] = 1.0
                restored["last_accessed_at"] = datetime.now(timezone.utc).isoformat()
                restored.pop("archived_at", None)
                restored.pop("decay_score_at_archive", None)
                self._memories.append(restored)
                self._rebuild_embeddings()
                self._save_memories()
                self._save_archive()
                return True
        return False

    def search_archive(self, query: str, top_k: int = 5) -> list[dict]:
        """关键词搜索归档记忆。"""
        if not self._archive["memories"]:
            return []
        query_words = set(self._tokenize(query))
        scored = []
        for mem in self._archive["memories"]:
            mem_words = set(self._tokenize(mem["content"]))
            jaccard = len(query_words & mem_words) / max(len(query_words | mem_words), 1)
            if jaccard > 0:
                scored.append((mem, jaccard))
        scored.sort(key=lambda x: -x[1])
        return [
            {
                "id": m["id"],
                "content": m["content"],
                "category": m.get("category", "general"),
                "archived_at": m.get("archived_at", ""),
                "score": round(s, 4),
            }
            for m, s in scored[:top_k]
        ]

    def get_memory_stats(self) -> dict:
        """返回记忆统计信息。"""
        active = len(self._memories)
        archived = len(self._archive["memories"])
        by_category: dict[str, int] = {}
        high = medium = low = 0
        for mem in self._memories:
            cat = mem.get("category", "general")
            by_category[cat] = by_category.get(cat, 0) + 1
            score = mem.get("decay_score", 1.0)
            if score > 0.8:
                high += 1
            elif score > 0.3:
                medium += 1
            else:
                low += 1
        return {
            "total": active + archived,
            "active": active,
            "archived": archived,
            "by_category": by_category,
            "decay_summary": {
                "high_score": high,
                "medium_score": medium,
                "low_score": low,
            },
        }

    # ─── Backward Compatibility ────────────────────────────────

    def extract_facts(self, conversation: str) -> list[str]:
        """从对话中提取新的事实、偏好、计划。"""
        prompt = f"""从以下最新对话中提取关于用户的新事实、偏好、计划或关系。
每条事实用简洁的陈述句，用换行分隔。如果没有新事实，回答"无"。
注意：仅提取长期有价值的信息，忽略临时闲聊。

对话：
{conversation}
"""
        response = self.llm.invoke(prompt).content
        if "无" in response:
            return []
        return [line.strip("- ").strip() for line in response.split("\n") if line.strip() and line.strip() != "无"]

    def retrieve(self, query: str, k: int = 3) -> str:
        """检索与查询相关的记忆，返回拼接好的上下文字符串。"""
        results = self.search_memories(query, top_k=k, hybrid=True)
        if not results:
            return ""
        lines = [f"- {r['content']}" for r in results]
        return "关于你的一些信息：\n" + "\n".join(lines)

    def extract_and_store(self, user_msg: str, ai_msg: str):
        """便捷方法：从一轮对话中提取并存储新记忆。"""
        conversation = f"User: {user_msg}\nAI: {ai_msg}"
        facts = self.extract_facts(conversation)
        for fact in facts:
            self.add_memory(fact)

    # ─── Persistence ───────────────────────────────────────────

    def _load_memories(self):
        self._memories = load_json(self._memories_path, [])

    def _save_memories(self):
        os.makedirs(self._data_dir, exist_ok=True)
        atomic_write_json(self._memories_path, self._memories)
        # 任何一次全量落盘都已把内存中累积的访问统计写出，重置脏计数。
        self._pending_access_writes = 0

    def _mark_access_dirty(self):
        """记录一次访问统计变更；累计到阈值才真正落盘。

        access_count/last_accessed_at 只用于衰减排序，是“软状态”，
        即使进程退出前丢失最后几次未刷盘的统计也不影响记忆内容本身，
        因此可安全地按阈值批量落盘以消除搜索路径的写放大。
        """
        self._pending_access_writes += 1
        if self._pending_access_writes >= self._access_flush_threshold:
            self._save_memories()

    def flush_access_stats(self):
        """强制落盘尚未持久化的访问统计（如有）。

        供应用在关闭 / 归档等时机显式调用，确保软状态最终一致。
        """
        if self._pending_access_writes > 0:
            self._save_memories()
    def _load_archive(self):
        self._archive = load_json(self._archive_path, {"archived_at": None, "memories": []})

    def _save_archive(self):
        os.makedirs(self._data_dir, exist_ok=True)
        atomic_write_json(self._archive_path, self._archive)
