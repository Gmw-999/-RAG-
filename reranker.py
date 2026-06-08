"""
Step 5: Re-ranking (精排)
Bi-Encoder 粗筛 top-10 → Cross-Encoder 精排 → top-3

原理:
  Bi-Encoder: query 和 doc 分别编码，互不相见 → 速度快但粗糙
  Cross-Encoder: query 和 doc 拼在一起编码 → 慢但精准

模型: BAAI/bge-reranker-v2-m3 (支持中文，0.57B 参数)
      从 ModelScope 下载到本地，路径见 config.py
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 强制离线，不从 HF 下载

import numpy as np
from sentence_transformers import CrossEncoder

_reranker_model = None


class Reranker:
    def __init__(self, model_name: str):
        global _reranker_model
        if _reranker_model is None:
            print(f"[Reranker] 加载模型: {model_name}")
            _reranker_model = CrossEncoder(model_name)
        self.model = _reranker_model

    def rerank(self, query: str, documents: list[str],
               top_k: int = 3) -> list[tuple[int, str, float]]:
        """
        Args:
            query:     用户问题
            documents: 粗排返回的文档列表
            top_k:     返回前几条
        Returns:
            [(原始索引, 文档文本, 相关性分数), ...]  按分数降序
        """
        # 1. 构造 (query, doc) 配对，doc 截断到 1000 字防止 OOM
        pairs = [(query, doc[:1000]) for doc in documents]

        # 2. Cross-Encoder 一次性给所有配对打分
        scores = self.model.predict(pairs)

        # 3. 按分数降序，取 top_k
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()

        ranked = sorted(
            enumerate(zip(documents, scores)),
            key=lambda x: x[1][1],
            reverse=True
        )[:top_k]

        return [(idx, doc, score) for idx, (doc, score) in ranked]


# 全局单例
_reranker_instance = None


def get_reranker(model_name: str = None) -> Reranker:
    global _reranker_instance
    if _reranker_instance is None:
        from config import RERANKER_MODEL
        model_name = model_name or RERANKER_MODEL
        _reranker_instance = Reranker(model_name)
    return _reranker_instance


if __name__ == "__main__":
    # 使用 config 中的本地路径
    from config import RERANKER_MODEL
    r = Reranker(RERANKER_MODEL)
    query = "水稻稻瘟病怎么防治？"
    docs = [
        "稻瘟病是由稻瘟病菌引起的真菌病害，主要危害水稻叶片和穗部。",
        "小麦赤霉病是一种危害小麦的重要病害，主要发生在抽穗扬花期。",
        "水稻稻瘟病的防治方法：选用抗病品种、合理施肥、适时喷施三环唑。",
    ]
    results = r.rerank(query, docs, top_k=3)
    print(f"查询: {query}\n")
    for rank, (idx, doc, score) in enumerate(results):
        print(f"[{rank + 1}] score={score:.4f} | idx={idx} | {doc[:60]}...")
