"""
Step 6: HyDE (Hypothetical Document Embedding)
让 LLM 先"编"一个假设答案，用这个假答案去检索

原理:
  用户 query 太短/口语化 → 和知识库文档不在同一语义空间 → 检索效果差
  LLM 生成假设答案 → 风格和知识库文档接近 → 检索更准

关键: 假设答案可以编错，它只是"探针"，最终答案从真实文档生成
"""
import requests
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL


def generate_hypothetical_doc(query: str) -> str:
    """
    用 LLM 生成一个假设的理想答案
    这个答案可能包含事实错误，没关系——它只用于检索
    """
    prompt = f"""你是一个农业知识助手。请根据你的知识，为以下问题写一段详细的答案。
即使你不确定，也请尽量提供专业、具体的回答。字数在200-400字之间。

问题：{query}

请直接回答，不要加"根据我的知识"之类的开头："""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,  # 稍微高点，让生成的内容更丰富
        "max_tokens": 600,
    }
    resp = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    q = "水稻稻瘟病怎么防治？"
    hypo = generate_hypothetical_doc(q)
    print(f"Query: {q}\n")
    print(f"假设答案 ({len(hypo)}字):")
    print(hypo)
