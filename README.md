# MindVault — 个人知识库助手

基于 Streamlit + LangGraph + DeepSeek 的个人知识库助手，采用 ReAct（Reasoning + Acting）架构，实现了三层记忆系统 + 文档 RAG + 混合检索 + 多工具自主调用 + MCP Server 对外暴露。Embedding 使用本地模型，零 API 成本。

## 功能特性

- **三层记忆系统**
  - **工作记忆** — LangGraph `MemorySaver` 自动管理当前会话上下文
  - **情景记忆** — 会话结束时 LLM 自动生成摘要，下次会话自动加载
  - **语义记忆** — 从对话中提取事实/偏好/计划，跨会话持久化，支持衰减归档
- **ReAct Agent** — 基于 LangGraph `create_react_agent`，LLM 自主决定何时调用工具，支持流式输出
- **5 个内置工具**
  - `search_knowledge` — 搜索已导入的文档知识库
  - `get_current_time` — 获取当前北京时间（UTC+8）
  - `calculate` — AST 安全计算数学表达式（无 exec）
  - `web_search` — DuckDuckGo 搜索，可选 Tavily 回退
  - `web_fetch` — trafilatura + BeautifulSoup4 网页内容提取（8000 字符上限）
- **文档知识库（RAG）** — 支持 `.md` / `.txt` / `.pdf` / `.json`，自动分块、向量化、SHA-256 去重
- **混合检索** — BM25（中英文分词）+ 向量相似度，RRF 融合排序（k=60）
- **统一检索器** — `UnifiedRetriever` 将知识库与记忆结果融合，注入系统提示词
- **记忆管理仪表盘** — 查看/搜索/编辑/删除语义记忆，分类过滤，衰减统计，归档与恢复
- **滑动窗口上下文** — 默认保留最近 20 轮对话，控制 token 成本
- **自动去重** — 语义记忆相似度 > 0.85 时自动去重，保留更优版本
- **记忆衰减** — `score = 1.0 * 0.95^天数 + min(访问次数 * 0.02, 0.3)`，低于 0.15 自动归档
- **MCP Server** — 通过 stdio 协议向外部 AI 客户端（Claude Code、Claude Desktop 等）暴露 8 工具 + 4 资源 + 2 提示词
- **防死循环** — 最多 10 轮工具调用（递归上限 21），超出自动终止

## 项目结构

```
knowledge_agent/
├── main.py                 # Streamlit 主界面（聊天 + 知识库管理 + 记忆仪表盘）
├── agent.py                # ReAct Agent 定义，图构建，记忆注入
├── memory.py               # EpisodicMemory + MemoryManager（语义记忆 + 衰减 + 归档）
├── knowledge_base.py       # 文档 RAG 引擎：加载、分块、Embedding、混合检索
├── retriever.py            # UnifiedRetriever：知识库 + 记忆 RRF 融合检索
├── context_manager.py      # 滑动窗口上下文截断（默认 20 轮）
├── tools.py                # 5 个 LangChain @tool 定义
├── mcp_server.py           # MCP Server（8 工具 / 4 资源 / 2 提示词）
├── requirements.txt        # Python 依赖
├── .env                    # 环境变量配置（API Key 等）
├── .env.example            # 环境变量模板
├── .gitignore
│
├── data/                   # 上传的文档存放目录
├── episodes/               # 情景记忆 JSON 文件（每用户一个）
├── memory_data/            # 语义记忆存储（memories.json + memories_archive.json）
├── chroma_db/              # 知识库 ChromaDB 向量库 + kb_manifest.json
│
├── docs/
│   ├── upgrade-roadmap.md  # 开发路线图（5 阶段）
│   └── specs/              # 各阶段详细实现规格
│
└── tests/
    └── test_mcp_server.py  # MCP Server 测试（15 个异步用例）
```

## 快速启动

### 1. 创建虚拟环境并激活

```bash
cd c:/Users/74788/Desktop/bs/knowledge_agent
python -m venv .venv

# Git Bash
source .venv/Scripts/activate
# CMD
.venv\Scripts\activate.bat
# PowerShell
.venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# LLM（默认 DeepSeek，也支持 OpenAI 等兼容接口）
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

> Embedding 使用本地模型 `BAAI/bge-small-zh-v1.5`（384 维），首次启动自动下载，无需配置。
> 如果想用 OpenAI 作为 LLM，将 `OPENAI_BASE_URL` 改为 `https://api.openai.com/v1`，`OPENAI_MODEL` 改为 `gpt-4o-mini` 即可。

### 4. 启动应用

```bash
streamlit run main.py
```

浏览器会自动打开 `http://localhost:8501`。

### 5.（可选）启动 MCP Server

MCP Server 作为独立进程运行，通过 stdio 与外部 AI 客户端通信，与 Streamlit 应用共享同一份数据：

```bash
python mcp_server.py
```

在 Claude Desktop 或其他 MCP 客户端的配置中添加即可使用。



## 使用说明

### 聊天

1. 在聊天框输入问题，Agent 会自主决定是否调用工具（知识库检索、网页搜索、数学计算等）
2. 支持流式输出，实时显示 Agent 回答
3. 侧边栏「记忆透镜」可实时查看当前上下文中的记忆和知识库状态

### 知识库管理

1. 切换到「知识库」标签页，上传 `.md` / `.txt` / `.pdf` / `.json` 文件（支持多选）
2. 文档自动分块（500 字符 / 50 字符重叠）、向量化并存入 ChromaDB
3. 支持文档列表查看、单文档删除、分块预览、一键清空
4. 提问时自动通过 BM25 + 向量混合检索相关段落

### 记忆管理

1. 切换到「记忆仪表盘」标签页
2. 查看所有语义记忆，支持按分类过滤、关键词搜索
3. 可编辑、删除单条记忆，查看衰减分数和访问统计
4. 归档记忆可恢复，支持一键清理低分记忆

### MCP 工具列表

| 工具 | 说明 |
|------|------|
| `search_all` | 统一搜索知识库 + 记忆 |
| `search_knowledge` | 仅搜索文档知识库 |
| `search_memory` | 仅搜索语义记忆 |
| `list_documents` | 列出已导入的文档 |
| `get_document` | 获取指定文档的分块内容 |
| `get_stats` | 获取知识库和记忆统计 |
| `add_memory` | 添加语义记忆 |
| `add_to_knowledge` | 将文本添加到知识库 |

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端 UI | Streamlit |
| Agent 框架 | LangGraph（ReAct 架构） |
| LLM | DeepSeek Chat / OpenAI GPT-4o-mini（可配置） |
| Embedding | BAAI/bge-small-zh-v1.5（本地运行，384 维，免费） |
| 向量数据库 | ChromaDB（持久化，余弦相似度） |
| 关键词检索 | rank_bm25（BM25Okapi） |
| PDF 解析 | PyMuPDF |
| 网页搜索 | DuckDuckGo + 可选 Tavily |
| 网页抓取 | trafilatura + BeautifulSoup4 |
| MCP 协议 | Anthropic MCP Python SDK（stdio 传输） |
| 环境管理 | python-dotenv |

## 架构

```
用户输入 → Streamlit UI (main.py)
    │
    ▼
UnifiedRetriever.search() → BM25 + 向量 RRF 融合（知识库 + 语义记忆）
    │
    ▼
系统提示词注入（上下文 + 知识库状态 + Agent 指令）
    │
    ▼
LangGraph ReAct Agent (agent.py) → LLM 推理循环
    │
    ├── search_knowledge → KnowledgeBase (ChromaDB + BM25)
    ├── get_current_time
    ├── calculate (AST 安全计算)
    ├── web_search → DuckDuckGo / Tavily
    └── web_fetch → trafilatura / BeautifulSoup
    │
    ▼
流式输出 → Streamlit 前端
    │
    ▼
对话后处理：MemoryManager.extract_and_store() → LLM 事实提取 → memories.json
```

## 开发进度

| 阶段 | 状态 | 内容 |
|------|------|------|
| Phase 1 | ✅ 完成 | ReAct Agent、3 工具、记忆注入、Streamlit UI |
| Phase 2 | ✅ 完成 | web_search、web_fetch、流式输出、上下文管理 |
| Phase 3 | ✅ 完成 | 文档 CRUD、混合 BM25+Vector 检索、统一 Embedding |
| Phase 4 | ✅ 完成 | 记忆 Embedding、衰减归档、记忆仪表盘、统一检索 |
| Phase 5 | ✅ 完成 | MCP Server（8 工具 / 4 资源 / 2 提示词） |

## 后续规划

- Docker 部署
- 外部数据源导入（浏览器书签、微信文章、Notion、Obsidian）

