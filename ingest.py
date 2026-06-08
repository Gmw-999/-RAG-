"""
RAG 数据入库主流程
支持双数据源: PDF(政策文档) + JSON(病虫害知识条目)
PDF → 文本提取 → 切分 → 向量化 → ChromaDB 存储
JSON → 字段拼接 → 切分(超长时) → 向量化 → ChromaDB 存储
"""
import json
from pathlib import Path
from parse_pdf import parse_all_pdfs
from chunker import chunk_text
from embedder import embed_texts
from vector_store import get_collection, add_chunks, get_stats
from config import CHUNKING_STRATEGY, EMBEDDING_MODEL


def load_json_knowledge(filepath: str = "knowledge_base.json") -> tuple[list[str], list[dict]]:
    """从 LLM 生成的 JSON 加载知识条目，返回 (文本列表, 元数据列表)
    每条元数据包含: category(作物), type(病害/虫害/草害), title(名称)
    """
    path = Path(__file__).parent / filepath
    if not path.exists():
        print(f"[JSON] {filepath} 不存在，跳过")
        return [], []

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs, metas = [], []
    for item in data:
        parts = []
        if item.get("title"):
            parts.append(f"【{item['title']}】")
        for field in ["summary", "symptoms", "cause", "prevention", "treatment"]:
            if item.get(field):
                parts.append(item[field])
        text = "\n".join(parts)
        if len(text) >= 100:
            docs.append(text)
            metas.append({
                "category": item.get("category", "未知"),
                "type": item.get("type", "未知"),
                "title": item.get("title", "未知"),
            })
    return docs, metas


def ingest(strategy: str = None, model: str = None, source: str = "all",
           reset: bool = True):
    """
    执行完整入库流程

    Args:
        strategy: 切分策略 (fixed_size / semantic / recursive)
        model: embedding 模型名
        source: 数据源 (pdf / json / all)
        reset: 是否清空旧数据重新入库
    """
    strategy = strategy or CHUNKING_STRATEGY
    model = model or EMBEDDING_MODEL

    # 清空旧数据
    if reset:
        collection = get_collection(reset=True)
        print(f"[重置] 已清空旧数据")

    print("=" * 60)
    print(f"RAG 数据入库")
    print(f"  数据源:   {source}")
    print(f"  切分策略: {strategy}")
    print(f"  向量模型: {model}")
    print("=" * 60)

    # ========== 加载数据 ==========
    all_chunks = []
    metadatas = []

    # 数据源 1: PDF
    if source in ("pdf", "all"):
        print("\n[PDF] 解析 PDF 文档...")
        docs = parse_all_pdfs()
        all_text = "\n\n".join(docs.values())
        print(f"  PDF 总字符数: {len(all_text)}")
        pdf_chunks = chunk_text(all_text, strategy)
        print(f"  切分块数: {len(pdf_chunks)}")
        all_chunks.extend(pdf_chunks)
        metadatas.extend([{"source": "pdf", "strategy": strategy,
                           "category": "通用", "type": "综合方案", "title": ""}
                          for _ in pdf_chunks])

    # 数据源 2: JSON
    if source in ("json", "all"):
        print("\n[JSON] 加载知识条目...")
        json_docs, json_metas = load_json_knowledge()
        print(f"  JSON 条目数: {len(json_docs)}")

        json_chunks, chunk_metas = [], []
        for doc, meta in zip(json_docs, json_metas):
            if len(doc) > 800:  # 长文本递归切
                sub_chunks = chunk_text(doc, strategy)
                json_chunks.extend(sub_chunks)
                chunk_metas.extend([{**meta, "strategy": strategy} for _ in sub_chunks])
            else:
                json_chunks.append(doc)
                chunk_metas.append({**meta, "strategy": strategy})

        print(f"  切分块数: {len(json_chunks)}")
        all_chunks.extend(json_chunks)
        metadatas.extend(chunk_metas)

    print(f"\n[总计] 待入库: {len(all_chunks)} 块")

    if not all_chunks:
        print("无数据可入库")
        return

    # ========== 向量化 ==========
    print(f"\n[向量化] 模型: {model}")
    embeddings = embed_texts(all_chunks, model)
    print(f"  向量维度: {len(embeddings[0])}")

    # ========== 入库 ==========
    print(f"\n[入库] ChromaDB ...")
    count = add_chunks(all_chunks, embeddings, metadatas)
    print(f"  写入 {count} 条")

    # ========== 统计 ==========
    stats = get_stats()
    print(f"\n{'=' * 60}")
    print(f"完成! 集合: {stats['collection']}, 总文档数: {stats['chunk_count']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    ingest(source="all")
