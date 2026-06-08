"""
农业知识 RAG 问答系统 — Gradio Web 界面
支持: HyDE + Metadata过滤 + Re-rank 一键切换
"""
import gradio as gr
from query import rag_query, rag_query_hyde, rag_query_filtered, list_categories, get_stats
from config import EMBEDDING_MODEL, EMBEDDING_PROVIDER, CHUNKING_STRATEGY


def query_rag(query: str, mode: str, category: str, pest_type: str,
              use_rerank: bool, use_hyde: bool) -> tuple[str, str]:
    """
    统一查询入口
    mode: "auto" | "basic" | "hyde"
    """
    if not query.strip():
        return "", "请输入问题"

    if mode == "basic":
        answer = rag_query(query, use_rerank=use_rerank, show_context=False)
        info = f"模式: 基础RAG | Re-rank: {use_rerank}"
    elif mode == "hyde":
        answer = rag_query_hyde(query, use_rerank=use_rerank, show_context=False)
        info = f"模式: HyDE | Re-rank: {use_rerank}"
    else:  # auto — 使用最完整的管线: HyDE + Metadata过滤 + Re-rank
        answer = rag_query_filtered(
            query,
            category=category if category != "全部" else None,
            pest_type=pest_type if pest_type != "全部" else None,
            use_rerank=use_rerank,
            use_hyde=use_hyde,
            show_context=False,
        )
        parts = []
        if use_hyde:
            parts.append("HyDE")
        if category != "全部" or pest_type != "全部":
            parts.append(f"过滤(category={category}, type={pest_type})")
        if use_rerank:
            parts.append("Re-rank")
        info = "模式: " + (" + ".join(parts) if parts else "基础检索")

    return answer, info


# ====================== UI 布局 ======================

with gr.Blocks(title="农业知识 RAG 问答系统") as demo:
    gr.Markdown("""
    # 🌾 农业知识 RAG 问答系统
    ### 基于 HyDE + Re-ranking + Metadata 过滤的专业农业病虫害智能问答
    """)

    # 系统状态栏
    stats = get_stats()
    cats = list_categories()
    status_text = (
        f"**向量库**: {stats['chunk_count']} 条文档 | "
        f"**Embedding**: {EMBEDDING_PROVIDER}/{EMBEDDING_MODEL} | "
        f"**切分**: {CHUNKING_STRATEGY}"
    )
    gr.Markdown(status_text)

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="输入你的问题",
                placeholder="例如：水稻稻瘟病怎么防治？小麦赤霉病用什么农药？",
                lines=2,
            )

            with gr.Row():
                submit_btn = gr.Button("查询", variant="primary", size="lg")
                clear_btn = gr.Button("清空", size="lg")

            answer_output = gr.Textbox(
                label="回答",
                lines=15,
                max_lines=25,
            )
            debug_output = gr.Textbox(label="运行时信息", lines=1, visible=False)

        with gr.Column(scale=1):
            gr.Markdown("### 检索选项")

            mode_radio = gr.Radio(
                choices=[
                    ("智能模式 (自动)", "auto"),
                    ("基础 RAG", "basic"),
                    ("HyDE RAG", "hyde"),
                ],
                value="auto",
                label="查询模式",
            )

            gr.Markdown("### 过滤条件 (仅智能模式)")

            category_dropdown = gr.Dropdown(
                choices=["全部"] + list(cats.keys()),
                value="全部",
                label="作物类别",
            )

            pest_type_dropdown = gr.Dropdown(
                choices=["全部", "病害", "虫害", "草害", "综合方案"],
                value="全部",
                label="病虫害类型",
            )

            gr.Markdown("### 高级开关")

            rerank_checkbox = gr.Checkbox(value=True, label="Re-rank 精排")
            hyde_checkbox = gr.Checkbox(value=True, label="HyDE 假设文档")

    # 事件绑定
    submit_btn.click(
        fn=query_rag,
        inputs=[query_input, mode_radio, category_dropdown,
                pest_type_dropdown, rerank_checkbox, hyde_checkbox],
        outputs=[answer_output, debug_output],
    )

    query_input.submit(
        fn=query_rag,
        inputs=[query_input, mode_radio, category_dropdown,
                pest_type_dropdown, rerank_checkbox, hyde_checkbox],
        outputs=[answer_output, debug_output],
    )

    clear_btn.click(
        fn=lambda: ("", ""),
        outputs=[query_input, answer_output],
    )

    gr.Markdown("""
    ---
    **技术栈**: Bi-Encoder (智谱 Embedding-3) → HNSW 索引 → Metadata 过滤 →
    Cross-Encoder (BGE-Reranker-v2-m3) → HyDE 假设文档 → DeepSeek 生成
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
