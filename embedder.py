"""
Step 3: 向量化 (Embedding)
支持三种 Provider:
  - local:  本地 Sentence-Transformers 模型 (离线免费)
  - doubao: 火山引擎豆包 Embedding API (已下架)
  - zhipu:  智谱 Embedding-3 API (在线，中文效果好)

Provider 和模型名在 config.py 里切换
"""
import os
import time
import requests
from sentence_transformers import SentenceTransformer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

_model = None


# ====================== 智谱 Embedding API ======================

def _zhipu_embed(texts: list[str], api_key: str, model: str) -> list[list[float]]:
    """调用智谱 Embedding-3 API"""
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    all_embeddings = []
    for i in range(0, len(texts), 40):
        batch = texts[i:i + 40]
        payload = {"model": model, "input": batch}

        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"智谱 API 错误 {resp.status_code}: {resp.text}")

        data = resp.json()
        for item in data["data"]:
            all_embeddings.append(item["embedding"])

        if i + 40 < len(texts):
            time.sleep(0.2)

    return all_embeddings


# ====================== 豆包 Embedding API (已下架，保留) ======================

def _doubao_embed(texts: list[str], api_key: str, model: str) -> list[list[float]]:
    """调用豆包 Embedding API"""
    url = "https://ark.cn-beijing.volces.com/api/v3/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    all_embeddings = []
    for i in range(0, len(texts), 40):
        batch = texts[i:i + 40]
        payload = {"model": model, "input": batch, "encoding_format": "float"}
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"豆包 API 错误 {resp.status_code}: {resp.text}")
        for item in resp.json()["data"]:
            all_embeddings.append(item["embedding"])
        if i + 40 < len(texts):
            time.sleep(0.3)

    return all_embeddings


# ====================== 统一接口 ======================

def embed_texts(texts: list[str], model_name: str = None,
                provider: str = None) -> list[list[float]]:
    from config import EMBEDDING_MODEL, EMBEDDING_PROVIDER

    provider = provider or EMBEDDING_PROVIDER
    model_name = model_name or EMBEDDING_MODEL

    if provider == "zhipu":
        from config import ZHIPU_EMBEDDING_API_KEY
        print(f"[Embedder] 智谱 API: {model_name}")
        return _zhipu_embed(texts, ZHIPU_EMBEDDING_API_KEY, model_name)

    elif provider == "doubao":
        from config import DOUBAO_EMBEDDING_API_KEY
        print(f"[Embedder] 豆包 API: {model_name}")
        return _doubao_embed(texts, DOUBAO_EMBEDDING_API_KEY, model_name)

    elif provider == "local":
        global _model
        if _model is None:
            print(f"[Embedder] 加载本地模型: {model_name}")
            _model = SentenceTransformer(model_name)
        print(f"[Embedder] 本地编码 {len(texts)} 条...")
        embeddings = _model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        return embeddings.tolist()

    else:
        raise ValueError(f"未知 Provider: {provider}")


def embed_single(text: str, model_name: str = None,
                 provider: str = None) -> list[float]:
    return embed_texts([text], model_name, provider)[0]


if __name__ == "__main__":
    samples = ["水稻稻瘟病是由梨孢霉引起的真菌病害。"]
    vec = embed_single(samples[0])
    print(f"向量维度: {len(vec)}")
    print(f"前 10 值: {vec[:10]}")
