"""
RAG 查询接口
支持两种模式:
  普通:  用户提问 → 向量检索(粗排) → Re-rank(精排) → DeepSeek 回答
  HyDE:  用户提问 → LLM生成假设答案 → 向量化假答案 → 检索 → Re-rank → DeepSeek 回答
"""
import requests
from embedder import embed_single
from vector_store import search, get_stats
from reranker import get_reranker
from hyde import generate_hypothetical_doc
from config import (DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL,
                    RETRIEVAL_TOP_K, RERANK_TOP_K)


def build_prompt(query: str, contexts: list[str]) -> str:
    """将检索结果拼接成 LLM prompt"""
    ctx_text = ""
    for i, ctx in enumerate(contexts):
        ctx_text += f"\n【参考资料 {i+1}】\n{ctx[:800]}\n"

    return f"""你是专业的农业植保专家。请严格根据以下参考资料回答用户问题。
如果参考资料中没有相关信息，请如实说"参考资料中未找到相关信息"。
回答要专业、具体、可操作。

{ctx_text}

用户问题：{query}

请回答："""


def call_deepseek(prompt: str, temperature: float = 0.3,
                  max_tokens: int = 1024) -> str:
    """调用 DeepSeek API"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _retrieve_and_rerank(query: str, top_k: int, use_rerank: bool,
                         show_context: bool, where: dict = None
                         ) -> tuple[list[str], list[dict], list]:
    """检索 + 可选精排，返回 (contexts, raw_results, reranked)"""
    retrieve_k = RETRIEVAL_TOP_K if use_rerank else top_k
    results = search(embed_single(query), retrieve_k, where=where)

    if show_context:
        print(f"\n--- 粗排结果 (top-{retrieve_k}) ---")
        for i, r in enumerate(results):
            print(f"  [{i+1}] distance={r['distance']:.4f} | {r['text'][:80]}...")

    reranked = []
    if use_rerank:
        reranker = get_reranker()
        docs = [r["text"] for r in results]
        reranked = reranker.rerank(query, docs, top_k=top_k)

        if show_context:
            print(f"\n--- 精排结果 (top-{top_k}) ---")
            for rank, (idx, doc, score) in enumerate(reranked):
                print(f"  [{rank+1}] score={score:.4f} (原排名{idx+1}) | {doc[:80]}...")

        contexts = [doc for _, doc, _ in reranked]
    else:
        contexts = [r["text"] for r in results[:top_k]]

    return contexts, results, reranked


def rag_query(query: str, use_rerank: bool = True,
              top_k: int = None, show_context: bool = True) -> str:
    """普通 RAG 查询：直接用 query 向量检索"""
    if top_k is None:
        top_k = RERANK_TOP_K if use_rerank else RETRIEVAL_TOP_K

    contexts, _, _ = _retrieve_and_rerank(query, top_k, use_rerank, show_context)

    prompt = build_prompt(query, contexts)

    if show_context:
        print(f"\n--- LLM 回答 ---")

    return call_deepseek(prompt)


def rag_query_hyde(query: str, use_rerank: bool = True,
                   top_k: int = None, show_context: bool = True) -> str:
    """
    HyDE RAG 查询:
      1. LLM 生成假设答案 (hypothetical doc)
      2. 用假设答案向量去检索 (而不是用原始 query)
      3. Re-rank + LLM 最终回答

    关键: 假设答案可以编错，它只是检索用的"探针"
    """
    if top_k is None:
        top_k = RERANK_TOP_K if use_rerank else RETRIEVAL_TOP_K

    # Step 1: LLM 生成假设答案
    print(f"\n--- HyDE: 生成假设答案 ---")
    hypo_doc = generate_hypothetical_doc(query)
    print(f"  假设答案 ({len(hypo_doc)}字): {hypo_doc[:150]}...")

    # Step 2: 用假设答案向量化 + 检索 (而不是原始 query)
    #         但 rerank 时还是用原始 query (因为 CrossEncoder 不用向量)
    print(f"  检索使用: 假设答案向量")
    contexts, _, _ = _retrieve_and_rerank(hypo_doc, top_k, use_rerank, show_context)

    # Step 3: 用原始 query + 检索到的真实文档生成最终答案
    prompt = build_prompt(query, contexts)

    if show_context:
        print(f"\n--- LLM 回答 ---")

    return call_deepseek(prompt)


def rag_query_filtered(query: str, category: str = None, pest_type: str = None,
                       use_rerank: bool = True, use_hyde: bool = True,
                       top_k: int = None, show_context: bool = True) -> str:
    """
    带元数据过滤的 RAG 查询：限定作物/病虫害类型后再检索

    Args:
        category:  作物类别，如 "水稻"、"小麦"、"玉米"、"蔬菜"
        pest_type: 病虫害类型，如 "病害"、"虫害"、"草害"
        use_hyde:  是否用 HyDE 假答案检索
    """
    if top_k is None:
        top_k = RERANK_TOP_K if use_rerank else RETRIEVAL_TOP_K

    # 构建 ChromaDB where 条件
    conditions = []
    if category:
        conditions.append({"category": category})
    if pest_type:
        conditions.append({"type": pest_type})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    filter_desc = f"category={category or '不限'}, type={pest_type or '不限'}"
    print(f"\n[过滤] {filter_desc} (where: {where})")

    # HyDE 模式: 先用 LLM 编假答案
    search_query = query
    if use_hyde:
        print(f"\n--- HyDE: 生成假设答案 ---")
        search_query = generate_hypothetical_doc(query)
        print(f"  假设答案 ({len(search_query)}字): {search_query[:150]}...")

    contexts, _, _ = _retrieve_and_rerank(search_query, top_k, use_rerank,
                                           show_context, where=where)

    prompt = build_prompt(query, contexts)

    if show_context:
        print(f"\n--- LLM 回答 ---")

    return call_deepseek(prompt)


def list_categories() -> dict:
    """列出向量库中各分类的文档数"""
    from vector_store import get_collection
    collection = get_collection()
    result = collection.get(include=["metadatas"])
    cats = {}
    for m in result["metadatas"]:
        c = m.get("category", "未知")
        cats[c] = cats.get(c, 0) + 1
    return dict(sorted(cats.items(), key=lambda x: -x[1]))


if __name__ == "__main__":
    stats = get_stats()
    print(f"向量库状态: {stats['chunk_count']} 条文档\n")

    # 看看有哪些分类
    cats = list_categories()
    print(f"分类分布: {cats}\n")

    # 测试: 限定"水稻"+"病害"的过滤查询
    q = "水稻稻瘟病怎么防治？"
    print(f"{'='*60}")
    print(f"[Query] {q}")
    print(f"{'='*60}")
    ans = rag_query_filtered(q, category="水稻", pest_type="病害")
    print(ans)
