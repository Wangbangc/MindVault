"""chunk_text overlap 行为单测。

不加载 embedding 模型 / ChromaDB——只通过 __new__ 绕过 __init__，
直接给实例挂上 chunk_size / chunk_overlap，测纯切分逻辑。
"""

from knowledge_base import KnowledgeBase


def _make_kb(chunk_size=60, chunk_overlap=15):
    kb = KnowledgeBase.__new__(KnowledgeBase)  # 跳过重量级 __init__
    kb.chunk_size = chunk_size
    kb.chunk_overlap = chunk_overlap
    return kb


def test_consecutive_chunks_actually_overlap():
    """相邻 chunk 之间必须有真实重叠——上一块的尾部出现在下一块开头。"""
    kb = _make_kb(chunk_size=60, chunk_overlap=15)
    # 三个段落，每个都接近 chunk_size，强制产生多个 chunk
    paras = [
        "第一段内容用于验证分块重叠行为是否真正生效需要足够长度",
        "第二段内容同样需要足够的长度来触发缓冲区刷新和重叠逻辑",
        "第三段内容继续填充以保证至少切出三个独立的分块结果集合",
    ]
    text = "\n\n".join(paras)
    chunks = [c["text"] for c in kb.chunk_text(text, source="t.md")]

    assert len(chunks) >= 2, f"应切出多个 chunk，实际 {len(chunks)}"

    # 至少一对相邻 chunk 存在重叠：前块某个尾部子串出现在后块开头区域
    found_overlap = False
    for prev, nxt in zip(chunks, chunks[1:]):
        tail = prev[-kb.chunk_overlap:]
        # 取尾部一段（去掉可能的空白），看是否前缀式出现在下一块
        probe = tail.strip()[:6]
        if probe and probe in nxt[: kb.chunk_overlap + 10]:
            found_overlap = True
            break
    assert found_overlap, (
        "相邻 chunk 之间没有任何重叠——overlap 逻辑失效。\n"
        + "\n".join(f"[{i}] {c!r}" for i, c in enumerate(chunks))
    )


def test_no_tiny_garbage_fragment_chunks():
    """超长段落分裂时，不得把 overlap 尾巴当成独立的小碎片 chunk 入列。"""
    kb = _make_kb(chunk_size=50, chunk_overlap=15)
    # 第一段填满 buffer，第二段是一个远超 chunk_size 的超长段落（触发 _split_long_text）
    first = "起始段落用来先把缓冲区填到接近上限位置"
    long_para = "超长段落" + "。这是一句需要被进一步切分的较长句子内容" * 6
    text = first + "\n\n" + long_para
    chunks = [c["text"] for c in kb.chunk_text(text, source="t.md")]

    # 不应存在「长度 <= overlap 且是纯 overlap 尾巴」的垃圾碎片
    garbage = [c for c in chunks if len(c.strip()) <= kb.chunk_overlap]
    assert not garbage, (
        f"出现疑似垃圾碎片 chunk（长度 <= overlap={kb.chunk_overlap}）：{garbage}\n"
        + "\n".join(f"[{i}] {c!r}" for i, c in enumerate(chunks))
    )


def test_chunk_indices_are_contiguous():
    """chunk_index 必须是 0..N-1 连续，total_chunks 一致。"""
    kb = _make_kb(chunk_size=40, chunk_overlap=10)
    text = "\n\n".join(f"第{i}段落内容填充用于切分测试需要一定长度" for i in range(5))
    result = kb.chunk_text(text, source="t.md")
    idxs = [c["metadata"]["chunk_index"] for c in result]
    assert idxs == list(range(len(result))), f"chunk_index 不连续: {idxs}"
    assert all(c["metadata"]["total_chunks"] == len(result) for c in result)


def test_short_text_single_chunk_unchanged():
    """短文本仍是单块，不受 overlap 改动影响（回归保护）。"""
    kb = _make_kb(chunk_size=500, chunk_overlap=50)
    chunks = kb.chunk_text("一小段短文本。", source="t.md")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "一小段短文本。"


def test_no_empty_chunks():
    """任何分块结果都不应包含空字符串 chunk。"""
    kb = _make_kb(chunk_size=45, chunk_overlap=12)
    text = "\n\n".join(f"段落{i}内容用于触发多次缓冲刷新逻辑保证覆盖" for i in range(6))
    chunks = [c["text"] for c in kb.chunk_text(text, source="t.md")]
    assert all(c.strip() for c in chunks), f"存在空 chunk: {chunks}"
