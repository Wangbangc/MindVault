import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from agent import agent_graph, memory_manager, episodic_memory, llm, knowledge_base, context_mgr
from memory import MemoryManager

# ==================== 页面全局配置 ====================
st.set_page_config(
    page_title="MindVault | 三层记忆知识库助手", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入轻量级 CSS 优化视觉间距
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #e9ecef; }
    </style>
""", unsafe_allow_html=True)


# ==================== 会话状态初始化 ====================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "default_user"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "episode_loaded" not in st.session_state:
    st.session_state.episode_loaded = False


# ==================== 加载情景记忆 ====================
def load_episode_context():
    if st.session_state.episode_loaded:
        return
    episode = episodic_memory.load(st.session_state.thread_id)
    if episode:
        st.session_state.messages.append({
            "role": "system",
            "content": f"[上次会话回忆] {episode['summary']}",
        })
    st.session_state.episode_loaded = True

load_episode_context()


# ==================== 🛠️ 侧边栏：仅保留全局控制 ====================
with st.sidebar:
    st.markdown("## 🧠 MindVault AI")
    st.caption("三层记忆体架构 · Central Control System")
    st.divider()
    
    st.session_state.thread_id = st.text_input("👤 当前用户 ID", st.session_state.thread_id)
    
    st.divider()
    st.subheader("会话生命周期")
    
    if st.button("💾 结束本次对话并归档摘要", use_container_width=True, type="primary"):
        lc_messages = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_messages.append(AIMessage(content=m["content"]))
        if lc_messages:
            summary = episodic_memory.generate_summary(lc_messages, llm)
            episodic_memory.save(st.session_state.thread_id, summary)
            st.success(f"会话摘要已固化存入情景记忆！")
            st.info(f"摘要预览: {summary[:50]}...")
        # 会话结束，落盘本次累积但未刷盘的检索访问统计
        memory_manager.flush_access_stats()

    if st.button("🗑️ 清空当前对话流", use_container_width=True):
        st.session_state.messages = []
        context_mgr.reset()
        st.rerun()


# ==================== 🏛️ 主界面：多维工作区切换 ====================
tab_chat, tab_kb, tab_mem = st.tabs(["💬 智能对话交互", "📚 知识联邦中心", "🧠 记忆矩阵控制台"])

# ─── 1. 💬 智能对话交互 Tab ───
with tab_chat:
    # 采用双栏分屏：左侧聊天，右侧透视记忆状态
    col_chat_flow, col_memory_lens = st.columns([3, 1], gap="medium")
    
    with col_chat_flow:
        st.caption("🤖 AI Assistant Chatflow")
        
        # 渲染历史消息
        for msg in st.session_state.messages:
            if msg["role"] == "system":
                # 用特殊的提示盒展示上次会话回忆，不破坏聊天流
                st.info(msg["content"], icon="⏳")
                continue
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # 对话输入与处理
        if prompt := st.chat_input("输入你的问题，激发三层记忆..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            context_mgr.add_turn("user", prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""

                lc_messages = []
                for m in st.session_state.messages:
                    if m["role"] == "user":
                        lc_messages.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        lc_messages.append(AIMessage(content=m["content"]))

                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                for event in agent_graph.stream(
                    {"messages": lc_messages},
                    config=config,
                    stream_mode="messages",
                ):
                    msg = event[0] if isinstance(event, tuple) else event
                    if hasattr(msg, "content") and isinstance(msg, AIMessage) and not msg.tool_calls:
                        if msg.content:
                            full_response = msg.content
                            placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            context_mgr.add_turn("assistant", full_response)

            try:
                memory_manager.extract_and_store(prompt, full_response)
            except Exception as e:
                print(f"语义记忆提取失败: {e}")
                
    with col_memory_lens:
        st.subheader("💡 记忆透视镜")
        st.caption("实时观测当前上下文激发的内核状态")
        
        # 实时动态看板占位（可根据你的 agent_graph 实际检索状态进行联动）
        with st.container(border=True):
            st.markdown("**📍 当前工作流上下文**")
            st.json({"thread_id": st.session_state.thread_id, "active_context": "已就绪"})
            
        with st.container(border=True):
            st.markdown("**🧠 语义/情景记忆检索激活**")
            st.caption("当有对话产生时，此处可输出被 Agent 召回的最相关的 Fact 或 上次会话摘要。")


# ─── 2. 📚 知识联邦中心 Tab ───
with tab_kb:
    st.subheader("📚 知识库文档联邦")
    st.caption("上传本地文档，将其碎片化向量化并合并入长效语义空间")
    st.divider()
    
    col_upload, col_list = st.columns([2, 3], gap="large")
    
    with col_upload:
        with st.container(border=True):
            st.markdown("### 📤 导入新长效资产")
            uploaded_files = st.file_uploader(
                "拖拽或点击上传",
                type=["md", "txt", "pdf", "json"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                has_changes = False
                for uf in uploaded_files:
                    save_path = os.path.join("data", uf.name)
                    with open(save_path, "wb") as f:
                        f.write(uf.getbuffer())
                    try:
                        count = knowledge_base.load_document(save_path)
                        if count > 0:
                            st.success(f"✅ {uf.name}（+{count} 分块）")
                            has_changes = True
                        else:
                            st.warning(f"🔁 {uf.name} 内容无变化已跳过")
                    except Exception as e:
                        st.error(f"❌ {uf.name} 失败: {e}")
                if has_changes:
                    st.rerun()
                    
            st.divider()
            if st.button("🔥 危险操作：清空完整知识库", type="secondary", use_container_width=True):
                knowledge_base.clear()
                st.success("整个长效知识库已被洗刷一空")
                st.rerun()

    with col_list:
        st.markdown("### 🗂️ 已托管知识库快照")
        kb_count = knowledge_base.count()
        if kb_count > 0:
            docs = knowledge_base.list_documents()
            st.info(f"💡 目前系统长效空间共托管 **{kb_count}** 个知识分块，源自 **{len(docs)}** 份独立文档。")
            
            for doc in docs:
                with st.expander(f"📄 {doc['source']} ({doc['chunk_count']} chunks)", expanded=False):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.caption(f"文档路径: {doc['source']}")
                    with c2:
                        if st.button("🗑️ 删除", key=f"del_{doc['source']}", type="secondary"):
                            if knowledge_base.delete_document(doc['source']):
                                st.success("删除成功")
                                st.rerun()
                    
                    # 预览切片
                    doc_info = knowledge_base.get_document_info(doc['source'])
                    if doc_info and doc_info["chunks"]:
                        st.markdown("**前 5 个切片预览：**")
                        for chunk in doc_info["chunks"][:5]:
                            st.code(chunk['text_preview'], language="text")
                        if doc_info["total_chunks"] > 5:
                            st.caption(f" *... 还有 {doc_info['total_chunks'] - 5} 个分段未显示*")
        else:
            st.info("知識库空空如也，请在左侧上传文件。")


# ─── 3. 🧠 记忆矩阵控制台 Tab ───
with tab_mem:
    st.subheader("🧠 语义碎片与记忆衰减管理")
    st.caption("AI 在对话中自动沉淀的 Fact（事实记忆）控制台，支持按权重衰减及手动归档")
    st.divider()
    
    stats = memory_manager.get_memory_stats()
    decay = stats["decay_summary"]
    
    # 核心指标可视化看板
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("🟢 活跃记忆", stats['active'])
    m_col2.metric("📦 归档记忆", stats['archived'])
    m_col3.metric("🔥 高留存率", decay['high_score'])
    m_col4.metric("⏳ 中权重", decay['medium_score'])
    m_col5.metric("❄️ 低权重", decay['low_score'])
    
    st.divider()
    
    # 操作与搜索排版
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        mem_search = st.text_input("🔍 检索实时语义碎片", placeholder="输入事实、实体或关键词...")
    with ctrl_col2:
        categories = list(stats["by_category"].keys()) if stats["by_category"] else ["general"]
        cat_options = ["全部"] + sorted(categories)
        selected_cat = st.selectbox("📂 实体分类过滤", cat_options)
        filter_cat = None if selected_cat == "全部" else selected_cat

    # 获取并清洗数据
    if mem_search:
        mems = memory_manager.search_memories(mem_search, top_k=25, hybrid=True)
        display_mems = []
        for m in mems:
            detail = memory_manager.get_memory_detail(m["id"])
            if detail and (filter_cat is None or detail.get("category") == filter_cat):
                display_mems.append(detail)
    else:
        display_mems = memory_manager.list_by_category(category=filter_cat, limit=25)

    # 展现记忆列表
    if not display_mems:
        st.info("没有找到匹配的主动记忆实体。")
    else:
        for mem in display_mems:
            with st.container(border=True):
                # 顶层元数据
                created = mem.get("created_at", "")[:10]
                meta_str = f"🏷️ **{mem.get('category', 'general')}** | 📅 创建: {created} | ⚡ 记忆留存率: {mem.get('decay_score', 1.0):.2f} | 📈 历史唤醒: {mem.get('access_count', 0)} 次"
                st.markdown(meta_str)
                
                # 记忆体核心内容
                st.write(mem["content"])
                
                # 交互按钮
                b1, b2, _ = st.columns([1, 1, 8])
                with b1:
                    if st.button("✏️ 编辑", key=f"edit_{mem['id']}"):
                        st.session_state["editing_memory_id"] = mem["id"]
                with b2:
                    if st.button("🗑️ 擦除", key=f"delmem_{mem['id']}", type="secondary"):
                        memory_manager.delete_memory(mem["id"])
                        st.rerun()
                        
                # 内联编辑展开
                if st.session_state.get("editing_memory_id") == mem["id"]:
                    with st.form(key=f"form_{mem['id']}"):
                        new_content = st.text_area("重写记忆事实", value=mem["content"])
                        new_cat = st.selectbox("修正分类", MemoryManager.FACT_TYPES, index=MemoryManager.FACT_TYPES.index(mem.get("category", "general")))
                        f_save, f_cancel = st.columns(2)
                        if f_save.form_submit_button("保存变更"):
                            memory_manager.update_memory(mem["id"], content=new_content, category=new_cat)
                            st.session_state.pop("editing_memory_id", None)
                            st.rerun()
                        if f_cancel.form_submit_button("取消"):
                            st.session_state.pop("editing_memory_id", None)
                            st.rerun()

    # 底部归档操作区
    st.divider()
    st.subheader("📦 历史记忆压缩与归档箱")
    arc_btn, src_btn = st.columns(2)
    
    if arc_btn.button("🧹 一键执行记忆权重衰减与旧记忆冷归档", use_container_width=True):
        result = memory_manager.run_decay_sweep()
        if result["archived_count"] > 0:
            st.success(f"成功将 {result['archived_count']} 条低频低留存记忆移入冷归档。")
        else:
            st.info("当前所有实体记忆活跃度良好，未触发冷归档阈值。")
        st.rerun()
        
    if src_btn.button("🔍 开启/关闭 墓碑归档区检索", use_container_width=True):
        st.session_state["show_archive_search"] = not st.session_state.get("show_archive_search", False)
        st.rerun()

    if st.session_state.get("show_archive_search"):
        archive_query = st.text_input("输入关键词，打捞已冷凝的归档记忆...", key="archive_query")
        if archive_query:
            archived = memory_manager.search_archive(archive_query, top_k=10)
            if archived:
                for a in archived:
                    with st.chat_message("system"):
                        st.markdown(f"**[已冷冻]** {a['content']}")
                        st.caption(f"归档时间: {a.get('archived_at', '')[:10]}")
                        if st.button("🔄 唤醒并恢复至活体状态", key=f"restore_{a['id']}"):
                            memory_manager.restore_from_archive(a["id"])
                            st.success("已成功重新激活该实体记忆！")
                            st.rerun()
            else:
                st.caption("冷归档库中未发现匹配碎片。")