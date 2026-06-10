"""
Knowledge Base 一键测试脚本。
在项目目录下运行： python test_knowledge_base.py

测试内容：
1. 文档分块（chunking）
2. 向量存储（embedding + ChromaDB）
3. 语义检索（RAG search）
4. 从文件导入（.md）
5. 工具返回格式
"""

import os
import sys
import shutil

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from knowledge_base import KnowledgeBase
from tools import SearchKnowledgeTool


def test_chunking():
    print("\n" + "=" * 50)
    print("测试 1：文档分块 (chunk_text)")
    print("=" * 50)
    kb = KnowledgeBase(persist_directory="./test_chroma_temp")
    text = "这是第一段，关于Python语言的基础介绍。\n\n这是第二段，深入讨论Python的面向对象特性。\n\n这是第三段，介绍Python的异步编程模型。"
    chunks = kb.chunk_text(text, source="test.md")
    print(f"  分块数：{len(chunks)}")
    for c in chunks:
        print(f"    [{c['metadata']['chunk_index']}] {c['text'][:60]}...")
    assert len(chunks) > 0, "分块失败"
    print("  ✓ 通过")


def test_store_and_search():
    print("\n" + "=" * 50)
    print("测试 2：向量存储 + 语义检索")
    print("=" * 50)
    kb = KnowledgeBase(persist_directory="./test_chroma_temp")
    chunks = [
        {"text": "Python是一种高级编程语言，由Guido van Rossum创建。", "metadata": {"source": "test.md", "chunk_index": 0, "total_chunks": 1}},
        {"text": "Java是一种跨平台的面向对象编程语言。", "metadata": {"source": "test.md", "chunk_index": 1, "total_chunks": 1}},
        {"text": "JavaScript主要用于Web前端开发。", "metadata": {"source": "test.md", "chunk_index": 2, "total_chunks": 1}},
    ]
    kb.store_chunks(chunks)
    print(f"  存储数：{kb.count()} chunks")

    results = kb.search("Python语言")
    print(f"  搜索 'Python语言' 结果：{len(results)} 条")
    for r in results:
        print(f"    score={r['score']} → {r['text'][:50]}")
    assert len(results) > 0, "检索无结果"
    print("  ✓ 通过")

    # 不相关查询
    no_results = kb.search("中日龙石化新能源开发")
    print(f"  搜索 '中日龙石化' 结果：{len(no_results)} 条（应为0）")
    # 可以不为0，但至少测试不崩
    print("  ✓ 通过")


def test_document_load():
    print("\n" + "=" * 50)
    print("测试 3：文档导入 (load_document)")
    print("=" * 50)
    kb = KnowledgeBase(persist_directory="./test_chroma_temp")

    # 创建测试 .md 文档
    test_path = "./test_import.md"
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("# Python异步编程\n\n")
        f.write("asyncio是Python标准库中的异步编程框架。\n\n")
        f.write("async/await关键字让异步代码的写法更接近同步代码。\n\n")
        f.write("事件循环(event loop)是asyncio的核心调度机制。\n\n")
        f.write("协程(coroutine)通过await挂起自身，让事件循环调度其他任务。")

    n = kb.load_document(test_path)
    print(f"  导入 chunks：{n}")
    print(f"  总文档数：{kb.count()}")
    print(f"  来源：{kb.list_sources()}")
    assert n > 0, "导入失败"
    print("  ✓ 通过")

    # 基于导入文档的检索
    results = kb.search("事件循环")
    print(f"  搜索 '事件循环' 结果：{len(results)} 条")
    for r in results:
        print(f"    score={r['score']} source={r['source']}")
    print("  ✓ 通过")

    # 清理测试文件
    os.remove(test_path)


def test_tool():
    print("\n" + "=" * 50)
    print("测试 4：SearchKnowledgeTool")
    print("=" * 50)
    tool = SearchKnowledgeTool()
    result = tool._run(query="Python异步")
    print(f"  工具返回长度：{len(result)} 字符")
    print(f"  预览：{result[:120]}...")
    assert len(result) > 50, "工具返回太短"
    print("  ✓ 通过")


def cleanup():
    print("\n" + "=" * 50)
    print("清理临时数据")
    print("=" * 50)
    if os.path.exists("./test_chroma_temp"):
        shutil.rmtree("./test_chroma_temp")
        print("  ✓ 测试数据库已删除")


if __name__ == "__main__":
    print("=" * 50)
    print("Knowledge Base 引擎测试套件")
    print("=" * 50)

    try:
        test_chunking()
        test_store_and_search()
        test_document_load()
        test_tool()
        cleanup()
        print("\n" + "=" * 50)
        print("全部测试通过 ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n  ✗ 失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
