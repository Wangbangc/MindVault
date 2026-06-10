"""MindVault MCP Server — exposes knowledge base and memory as MCP tools/resources/prompts."""

import os
import sys
import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()

from mcp.server import Server, NotificationOptions
import mcp.server.stdio
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    Prompt,
    PromptMessage,
    GetPromptResult,
)

from knowledge_base import KnowledgeBase
from memory import MemoryManager
from retriever import UnifiedRetriever

# ─── Logging (stderr only — stdout is the MCP transport) ─────────

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mindvault-mcp")

# ─── Data paths (must match agent.py / main.py) ─────────────────

_ROOT = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(_ROOT, "chroma_db")
DATA_DIR = os.path.join(_ROOT, "data")
MEMORY_DIR = os.path.join(_ROOT, "memory_data")

# ─── Initialize components ──────────────────────────────────────


def initialize():
    """Initialize KnowledgeBase, MemoryManager, UnifiedRetriever."""
    try:
        kb = KnowledgeBase(persist_directory=CHROMA_DIR)
        logger.info("KnowledgeBase initialized: %s", CHROMA_DIR)
    except Exception as e:
        logger.critical("Failed to initialize KnowledgeBase: %s", e)
        raise

    try:
        mm = MemoryManager(
            data_dir=MEMORY_DIR,
            llm_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
        logger.info("MemoryManager initialized: %s", MEMORY_DIR)
    except Exception as e:
        logger.critical("Failed to initialize MemoryManager: %s", e)
        raise

    retriever = UnifiedRetriever(kb, mm)
    return kb, mm, retriever


kb, mm, retriever = initialize()
server = Server("mindvault")

# ─── Tool definitions ───────────────────────────────────────────

TOOLS = [
    Tool(
        name="search_all",
        description=(
            "Search both knowledge base and personal memory for relevant information. "
            "Use this as the primary search when you need context about the user's "
            "knowledge, research, or personal information. Returns ranked results with source labels."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — natural language, topic, or keywords",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 15)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_knowledge",
        description=(
            "Search only the knowledge base (imported documents, research notes, PDFs). "
            "Use when the user asks about specific documents, research topics, or imported materials. "
            "Does NOT search personal memories."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {
                    "type": "integer",
                    "description": "Max results (default: 5, max: 15)",
                    "default": 5,
                },
                "hybrid": {
                    "type": "boolean",
                    "description": "Use BM25+Vector hybrid search (default: true)",
                    "default": True,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_memory",
        description=(
            "Search only personal memories (user preferences, life details, recurring facts). "
            "Use when the user asks about personal information, habits, or past conversations. "
            "Does NOT search the knowledge base."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {
                    "type": "integer",
                    "description": "Max results (default: 5, max: 15)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="list_documents",
        description=(
            "List all documents in the knowledge base. Returns file names, chunk counts, "
            "and import times. Use to discover what the user has imported."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "detail": {
                    "type": "string",
                    "enum": ["summary", "full"],
                    "description": (
                        "summary = just names and counts; "
                        "full = includes import timestamp and file path (default: summary)"
                    ),
                    "default": "summary",
                }
            },
        },
    ),
    Tool(
        name="get_document",
        description=(
            "Retrieve the full content or chunk listing of a specific document by its "
            "source filename. Use list_documents first to see available documents."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source filename (from list_documents output)",
                },
                "mode": {
                    "type": "string",
                    "enum": ["chunks", "full"],
                    "description": "chunks = list all chunk previews; full = concatenated full text (default: chunks)",
                    "default": "chunks",
                },
            },
            "required": ["source"],
        },
    ),
    Tool(
        name="get_stats",
        description=(
            "Return statistics about the MindVault knowledge system: "
            "document count, chunk count, memory count, archive count, category breakdowns."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="add_memory",
        description=(
            "Save a new personal memory. Use when the user shares a personal fact, "
            "preference, or recurring detail that should be remembered across all AI tools. "
            "Memories are searchable by all connected tools."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "The memory content to save. Be specific and self-contained "
                        "so it's retrievable later."
                    ),
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Memory category (default: general). One of: "
                        + ", ".join(MemoryManager.FACT_TYPES)
                    ),
                    "enum": list(MemoryManager.FACT_TYPES),
                    "default": "general",
                },
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="add_to_knowledge",
        description=(
            "Import a short text into the knowledge base. Use for ad-hoc notes, "
            "quick research snippets, or temporary reference material. "
            "For full documents, use the MindVault UI's document upload instead."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text content to import"},
                "source": {
                    "type": "string",
                    "description": "Source label (e.g. 'chat-note.md', 'idea-20260609.txt')",
                },
            },
            "required": ["text", "source"],
        },
    ),
]


# ─── Tool handlers ──────────────────────────────────────────────


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "search_all":
            return _handle_search_all(arguments)
        elif name == "search_knowledge":
            return _handle_search_knowledge(arguments)
        elif name == "search_memory":
            return _handle_search_memory(arguments)
        elif name == "list_documents":
            return _handle_list_documents(arguments)
        elif name == "get_document":
            return _handle_get_document(arguments)
        elif name == "get_stats":
            return _handle_get_stats()
        elif name == "add_memory":
            return _handle_add_memory(arguments)
        elif name == "add_to_knowledge":
            return _handle_add_to_knowledge(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error("Tool %s failed: %s", name, e)
        return [TextContent(type="text", text=f"Error: {e}")]


def _handle_search_all(args: dict) -> list[TextContent]:
    query = (args.get("query") or "").strip()
    if not query:
        return [TextContent(type="text", text="Please provide a search query.")]
    if len(query) > 500:
        logger.warning("Truncating query from %d to 500 chars", len(query))
        query = query[:500]
    top_k = min(int(args.get("top_k", 5)), 15)
    results = retriever.search(query, top_k=top_k)
    if not results:
        return [TextContent(type="text", text="No relevant information found in knowledge base or memory.")]
    formatted = retriever.format_for_prompt(results)
    return [TextContent(type="text", text=formatted)]


def _handle_search_knowledge(args: dict) -> list[TextContent]:
    query = (args.get("query") or "").strip()
    if not query:
        return [TextContent(type="text", text="Please provide a search query.")]
    if len(query) > 500:
        logger.warning("Truncating query from %d to 500 chars", len(query))
        query = query[:500]
    top_k = min(int(args.get("top_k", 5)), 15)
    hybrid = args.get("hybrid", True)
    results = kb.search(query, top_k=top_k, hybrid=hybrid)
    if not results:
        return [TextContent(type="text", text="No relevant documents found.")]
    lines = []
    for i, r in enumerate(results, 1):
        preview = r["text"][:200] + ("..." if len(r["text"]) > 200 else "")
        lines.append(f"{i}. [{r['source']}] {preview}")
    return [TextContent(type="text", text="\n".join(lines))]


def _handle_search_memory(args: dict) -> list[TextContent]:
    query = (args.get("query") or "").strip()
    if not query:
        return [TextContent(type="text", text="Please provide a search query.")]
    if len(query) > 500:
        logger.warning("Truncating query from %d to 500 chars", len(query))
        query = query[:500]
    top_k = min(int(args.get("top_k", 5)), 15)
    results = mm.search_memories(query, top_k=top_k)
    if not results:
        return [TextContent(type="text", text="No relevant memories found.")]
    lines = []
    for i, r in enumerate(results, 1):
        preview = r["content"][:200] + ("..." if len(r["content"]) > 200 else "")
        lines.append(f"{i}. [{r['category']}] {preview}")
    return [TextContent(type="text", text="\n".join(lines))]


def _handle_list_documents(args: dict) -> list[TextContent]:
    docs = kb.list_documents()
    if not docs:
        return [TextContent(type="text", text="Knowledge base is empty. Import documents via the MindVault UI or the add_to_knowledge tool.")]
    detail = args.get("detail", "summary")
    lines = [f"Documents in knowledge base ({len(docs)} total):"]
    for i, d in enumerate(docs, 1):
        if detail == "full":
            lines.append(f"{i}. {d['source']} — {d['chunk_count']} chunks (imported: {d['imported_at']}, path: {d.get('file_path', '')})")
        else:
            lines.append(f"{i}. {d['source']} — {d['chunk_count']} chunks")
    return [TextContent(type="text", text="\n".join(lines))]


def _handle_get_document(args: dict) -> list[TextContent]:
    source = (args.get("source") or "").strip()
    if not source:
        return [TextContent(type="text", text="Please provide a source filename.")]
    mode = args.get("mode", "chunks")
    info = kb.get_document_info(source)
    if info is None:
        return [TextContent(type="text", text=f"Document '{source}' not found in knowledge base. Use list_documents to see available documents.")]
    chunks = info.get("chunks", [])
    if not chunks:
        return [TextContent(type="text", text=f"Document '{source}' exists but has no chunks. It may be empty or corrupted.")]
    if mode == "full":
        full_text = "\n\n---\n\n".join(c.get("text", "") for c in chunks)
        return [TextContent(type="text", text=full_text)]
    # mode == "chunks"
    lines = [f"Document: {source} ({info['total_chunks']} chunks)", ""]
    for c in chunks:
        preview = c.get("text_preview", c.get("text", "")[:100])
        lines.append(f"  Chunk {c.get('index', '?')}: {preview}")
    return [TextContent(type="text", text="\n".join(lines))]


def _handle_get_stats() -> list[TextContent]:
    kb_doc_count = len(kb.list_documents())
    kb_chunk_count = kb.count()
    mem_stats = mm.get_memory_stats()
    lines = [
        "MindVault Statistics:",
        f"Knowledge Base: {kb_doc_count} documents, {kb_chunk_count} chunks",
        f"Memory: {mem_stats.get('active', 0)} active, {mem_stats.get('archived', 0)} archived",
    ]
    by_cat = mem_stats.get("by_category", {})
    if by_cat:
        cat_str = ", ".join(f"{k} ({v})" for k, v in by_cat.items())
        lines.append(f"Categories: {cat_str}")
    return [TextContent(type="text", text="\n".join(lines))]


def _handle_add_memory(args: dict) -> list[TextContent]:
    content = (args.get("content") or "").strip()
    if not content:
        return [TextContent(type="text", text="Error: Content cannot be empty.")]
    if len(content) > 2000:
        logger.warning("Truncating memory content from %d to 2000 chars", len(content))
        content = content[:2000]
    category = args.get("category", "general")
    try:
        memory_id = mm.add_memory(content, category)
        if not memory_id:
            return [TextContent(type="text", text="Error: Failed to save memory (empty result).")]
        preview = content[:80] + ("..." if len(content) > 80 else "")
        return [TextContent(type="text", text=f"Memory saved: {preview} (category: {category})")]
    except Exception as e:
        return [TextContent(type="text", text=f"Failed to save memory: {e}")]


def _handle_add_to_knowledge(args: dict) -> list[TextContent]:
    text = (args.get("text") or "").strip()
    source = (args.get("source") or "").strip()
    if not text:
        return [TextContent(type="text", text="Error: Text content cannot be empty.")]
    if not source:
        return [TextContent(type="text", text="Error: Source label cannot be empty.")]
    if len(text) > 100_000:
        return [TextContent(type="text", text="Content exceeds 100,000 character limit. Use the MindVault UI's file upload for larger documents.")]
    try:
        chunk_count = kb.add_document(text, source)
        return [TextContent(type="text", text=f"Knowledge added: {source} ({chunk_count} chunks)")]
    except ValueError as e:
        return [TextContent(type="text", text=f"Error: {e}")]
    except Exception as e:
        logger.error("add_to_knowledge failed: %s", e)
        return [TextContent(type="text", text=f"Failed to add knowledge: {e}")]


# ─── Resource definitions ───────────────────────────────────────


@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    return [
        Resource(
            uri="memory://recent",
            name="Recent Memories",
            description="10 most recent active memories",
            mimeType="text/plain",
        ),
        Resource(
            uri="memory://stats",
            name="Memory Statistics",
            description="Memory count, categories, archive stats",
            mimeType="text/plain",
        ),
        Resource(
            uri="knowledge://stats",
            name="Knowledge Base Statistics",
            description="Document count, chunk count, import timestamps",
            mimeType="text/plain",
        ),
        Resource(
            uri="knowledge://documents",
            name="Document List",
            description="Full list of documents in the knowledge base",
            mimeType="text/plain",
        ),
    ]


@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    uri_str = str(uri)

    if uri_str == "memory://recent":
        memories = mm.list_by_category(limit=10)
        if not memories:
            return "No memories yet."
        lines = ["Recent memories (top 10):", ""]
        for m in memories:
            preview = m["content"][:80] + ("..." if len(m["content"]) > 80 else "")
            lines.append(f"- [{m['category']}] {preview}")
        return "\n".join(lines)

    if uri_str == "memory://stats":
        stats = mm.get_memory_stats()
        return (
            f"Memory Statistics:\n"
            f"Active: {stats.get('active', 0)}\n"
            f"Archived: {stats.get('archived', 0)}\n"
            f"Categories: {stats.get('by_category', {})}"
        )

    if uri_str == "knowledge://stats":
        docs = kb.list_documents()
        return (
            f"Knowledge Base Statistics:\n"
            f"Documents: {len(docs)}\n"
            f"Chunks: {kb.count()}"
        )

    if uri_str == "knowledge://documents":
        docs = kb.list_documents()
        if not docs:
            return "Knowledge base is empty."
        lines = ["Documents:", ""]
        for d in docs:
            lines.append(f"- {d['source']} ({d['chunk_count']} chunks, imported {d['imported_at']})")
        return "\n".join(lines)

    raise ValueError(f"Unknown resource: {uri_str}")


# ─── Prompt definitions ─────────────────────────────────────────


@server.list_prompts()
async def handle_list_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="search-first",
            description="Guidance: always search MindVault before answering user questions about knowledge or personal information",
        ),
        Prompt(
            name="save-memories",
            description="Guidance: when user shares personal information, save it to MindVault memory",
        ),
    ]


@server.get_prompt()
async def handle_get_prompt(name: str, arguments: dict | None = None) -> GetPromptResult:
    if name == "search-first":
        return GetPromptResult(
            description="Search-first approach",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            "When the user asks a question, ALWAYS start by searching MindVault's knowledge base and memory "
                            "before answering. The user has accumulated personal knowledge, research notes, and preferences "
                            "in MindVault.\n\n"
                            "Steps:\n"
                            "1. Call search_all(query=<user's question>) to check for relevant context\n"
                            "2. If the results contain useful information, incorporate it into your answer\n"
                            "3. If no relevant results found, answer based on your own knowledge and prompt the user to "
                            "consider adding the information to MindVault\n\n"
                            "This ensures your answer is personalized and grounded in the user's own knowledge."
                        ),
                    ),
                )
            ],
        )

    if name == "save-memories":
        return GetPromptResult(
            description="Save memories proactively",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            "When the user shares personal information about themselves — preferences, habits, "
                            "life events, recurring details, project context, or technical setup — save it to MindVault "
                            "using the add_memory tool.\n\n"
                            "Categories to use:\n"
                            "- personal: life details, preferences, relationships, pets\n"
                            "- work: job info, tools, workflows, technical setup\n"
                            "- code: code conventions, project structure, dependency choices\n"
                            "- project: ongoing projects, milestones, plans\n"
                            "- general: everything else\n\n"
                            "Be specific in the content so the memory is retrievable later. "
                            "Do NOT save obvious ephemeral chat content or one-off greetings."
                        ),
                    ),
                )
            ],
        )

    raise ValueError(f"Unknown prompt: {name}")


# ─── Main entrypoint ────────────────────────────────────────────


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(
                notification_options=NotificationOptions(),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
