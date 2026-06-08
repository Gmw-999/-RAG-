"""
RAG 管线统一配置
所有可调参数集中在这里，方便做对比实验
密钥统一从 .env 文件读取，不硬编码
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ====================== 路径 ======================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# 加载 .env 文件中的密钥
load_dotenv(BASE_DIR / ".env")

def _env(key: str, default: str = "") -> str:
    """读取环境变量"""
    return os.getenv(key, default)

# ====================== PDF 数据源 ======================
PDF_FILES = [
    "2025粮食作物防控方案.pdf",
    "2025油料经济作物防控方案.pdf",
    "2024油料经济作物防控方案.pdf",
    "2023园艺作物防控方案.pdf",
]

# ====================== 文本切分配置 ======================
# 可选策略: "fixed_size" | "semantic" | "recursive"
CHUNKING_STRATEGY = "semantic"

# Fixed-size 参数
FIXED_CHUNK_SIZE = 500        # 每块字符数
FIXED_CHUNK_OVERLAP = 100     # 块间重叠字符数

# Semantic 参数
SEMANTIC_SEPARATORS = "\n\n"  # 按段落切分
SEMANTIC_MIN_CHUNK = 200

# Recursive 参数
RECURSIVE_CHUNK_SIZE = 500
RECURSIVE_CHUNK_OVERLAP = 100
RECURSIVE_SEPARATORS = ["\n\n", "\n", "。", "；", "，", " "]

# ====================== Embedding 模型配置 ======================
# Provider: "zhipu" (智谱API) | "local" (离线) | "doubao" (已下架)
EMBEDDING_PROVIDER = "zhipu"

# zhipu 模式模型名: "embedding-3"
# local 模式模型名: "paraphrase-multilingual-MiniLM-L12-v2" | "BAAI/bge-large-zh-v1.5"
EMBEDDING_MODEL = "embedding-3"

# 智谱 Embedding API Key (从 .env 读取)
ZHIPU_EMBEDDING_API_KEY = _env("ZHIPU_EMBEDDING_API_KEY")

# 豆包 Embedding API Key (已下架，保留备用)
DOUBAO_EMBEDDING_API_KEY = _env("DOUBAO_EMBEDDING_API_KEY")

# ====================== ChromaDB 配置 ======================
CHROMA_COLLECTION = "agri_knowledge"
CHROMA_TOP_K = 5              # 检索返回条数（不带 rerank 时用）

# ====================== Re-ranking 配置 ======================
# Cross-Encoder 模型路径 (支持: 本地路径 / HuggingFace 模型名 / 环境变量)
RERANKER_MODEL = _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RETRIEVAL_TOP_K = 10          # 粗排取几条给 reranker
RERANK_TOP_K = 3              # 精排后保留几条

# ====================== HyDE 配置 ======================
# True: 开启 HyDE (LLM生成假设答案 → 向量化假答案 → 检索)
# False: 关闭 (直接用 query 检索)
HYDE_ENABLED = True
HYDE_TEMPERATURE = 0.7        # 假设答案生成的创造度

# ====================== LLM 配置（DeepSeek） ======================
DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = _env("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL = _env("DEEPSEEK_MODEL", "deepseek-chat")
