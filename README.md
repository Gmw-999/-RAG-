# 🌾 农业知识 RAG 问答系统

基于 **HyDE + Re-ranking + Metadata 过滤** 的农业病虫害专业智能问答系统。以农业农村部防控方案和结构化病虫害知识库为数据源，集成多种高级 RAG 技术，构建面向垂直领域的工业级检索增强生成管线。

## 架构概览

```
用户提问
    │
    ▼
┌─ HyDE (假设文档嵌入) ────────────────┐
│  DeepSeek 生成假设答案 (~500字)       │
│  解决短问句语义稀疏问题               │
└──────────────────────────────────────┘
    │
    ▼
┌─ Metadata 过滤 ──────────────────────┐
│  ChromaDB where: {category, type}    │
│  搜索空间缩小 93%                     │
└──────────────────────────────────────┘
    │
    ▼
┌─ Bi-Encoder 粗排 ────────────────────┐
│  智谱 Embedding-3 (2048维)           │
│  HNSW 索引, O(log n) 检索, top-10    │
└──────────────────────────────────────┘
    │
    ▼
┌─ Cross-Encoder 精排 ─────────────────┐
│  BGE-Reranker-v2-m3 (0.57B)          │
│  top-3 精准筛选                       │
└──────────────────────────────────────┘
    │
    ▼
┌─ LLM 生成最终答案 ───────────────────┐
│  DeepSeek + 3条精准上下文 → 专业回答  │
└──────────────────────────────────────┘
```

## 项目结构

```
rag-project/
├── config.py              # 统一配置中心 (密钥从 .env 读取)
├── ingest.py              # 数据入库主流程 (PDF + JSON → ChromaDB)
├── query.py               # RAG 查询接口 (3种模式)
├── app.py                 # Gradio Web 界面
├── chunker.py             # 文本切分 (fixed/semantic/recursive)
├── embedder.py            # 向量化 (智谱API / 本地 / 豆包)
├── reranker.py            # Cross-Encoder 精排 (BGE-Reranker)
├── hyde.py                # HyDE 假设文档生成
├── vector_store.py        # ChromaDB 向量存储 + 检索
├── parse_pdf.py           # PDF 文本提取 (PyMuPDF)
├── generate_knowledge.py  # LLM 批量生成病虫害知识条目
├── knowledge_base.json    # 325条结构化病虫害知识
├── data/                  # 7份 PDF 防控方案文档
├── .env.example           # 环境变量模板
└── .gitignore
```

## 快速开始

### 环境要求

- Python 3.9+
- Git (用于安装可选依赖)

### 1. 安装依赖

```bash
pip install chromadb sentence-transformers pymupdf requests python-dotenv gradio
```

### 2. 配置密钥

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key：
#   ZHIPU_EMBEDDING_API_KEY  — 智谱 Embedding API (https://open.bigmodel.cn)
#   DEEPSEEK_API_KEY         — DeepSeek API (https://platform.deepseek.com)
#   RERANKER_MODEL           — Reranker 模型路径 (见下文)
```

### 3. 下载 Reranker 模型 (可选，默认需联网)

如果你在国内，建议用 ModelScope 下载本地模型：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3', cache_dir='E:/python/huggingface_models')"
```

然后在 `.env` 中设置 `RERANKER_MODEL=E:/python/huggingface_models/BAAI/bge-reranker-v2-m3`

### 4. 构建向量库

```bash
python ingest.py
```

输出示例：
```
[PDF] 解析 PDF 文档...  → 107 块
[JSON] 加载知识条目... → 325 块
[总计] 待入库: 432 块
完成! 总文档数: 432
```

### 5. 启动 Web 界面

```bash
python app.py
```

浏览器打开 `http://localhost:7860`

## 使用方式

### Gradio Web 界面

| 模式 | 说明 |
|------|------|
| **智能模式 (推荐)** | HyDE + Metadata 过滤 + Re-rank 全开 |
| 基础 RAG | query 直接检索 + 可选 Re-rank |
| HyDE RAG | 假设文档检索 + 可选 Re-rank |

过滤条件（仅智能模式）：
- **作物类别**: 水稻、小麦、玉米、蔬菜...
- **病虫害类型**: 病害、虫害、草害

高级开关：
- **Re-rank 精排**: 开启 Cross-Encoder 重排序
- **HyDE 假设文档**: 开启 LLM 生成假设答案检索

### Python API

```python
from query import rag_query, rag_query_hyde, rag_query_filtered

# 基础 RAG
answer = rag_query("水稻稻瘟病怎么防治？", use_rerank=True)

# HyDE RAG (假设文档检索)
answer = rag_query_hyde("小麦赤霉病用什么农药？")

# 智能模式 (HyDE + 过滤 + Re-rank)
answer = rag_query_filtered(
    "玉米草地贪夜蛾的防控措施",
    category="玉米",
    pest_type="虫害",
    use_rerank=True,
    use_hyde=True,
)
```

## 核心模块说明

### 文本切分 (`chunker.py`)

支持 3 种策略，通过 `config.py` 的 `CHUNKING_STRATEGY` 切换：

| 策略 | 配置值 | 特点 | 入库块数 |
|------|--------|------|----------|
| 语义分块 | `semantic` | 按段落边界，单块话题纯净 | 432 |
| 递归分块 | `recursive` | 层级分隔符，原文完整性最优 | 546 |
| 固定分块 | `fixed_size` | 固定窗口+重叠，简单可靠 | 538 |

### Embedding 模型 (`embedder.py`)

| Provider | 配置值 | 模型 | 维度 | 需要 |
|----------|--------|------|------|------|
| 智谱 API | `zhipu` | embedding-3 | 2048 | API Key |
| 本地 | `local` | bge-large-zh-v1.5 | 1024 | GPU/内存 |
| 豆包 (已下架) | `doubao` | - | - | - |

### RAG 查询模式 (`query.py`)

| 模式 | 函数 | 流程 |
|------|------|------|
| 基础 | `rag_query()` | query向量 → 检索 → (Re-rank) → LLM |
| HyDE | `rag_query_hyde()` | LLM编假答案 → 假答案向量检索 → (Re-rank) → LLM真答案 |
| 过滤+HyDE | `rag_query_filtered()` | HyDE → Metadata过滤 → 检索 → Re-rank → LLM |

## 实验结论摘要

详细实验数据见 `农业知识 RAG 问答系统.docx`

### 关键技术指标

| 指标 | 基础 RAG | +Rerank | +HyDE+过滤 |
|------|----------|---------|------------|
| Top1 余弦距离 | 0.8334 | 0.8334 | **0.3816** |
| 搜索空间 | 432 条 | 432 条 | **31 条** |
| 检索命中率 | 100% | 100% | **100%** |
| 干扰文档 | 较多 | 较少 | **极少** |

### 三大策略对比

| 切分策略 | Top1 距离 | 块数 | 综合评价 |
|----------|-----------|------|----------|
| 语义分块 | **0.3816** | 432 | 话题纯净，过滤后匹配最佳 |
| 递归分块 | 0.8334 | 546 | 原文完整性最优 |
| 固定分块 | 0.8334 | 538 | 简单可靠，但干扰较多 |

### 结论

**语义分块 + 智谱 Embedding-3 + HyDE + Re-rank + Metadata 过滤** 的终极架构，实现了检索距离降低 54%、搜索空间缩小 93%、答案零幻觉的工业级效果。

## 技术栈

| 组件 | 技术 |
|------|------|
| PDF 解析 | PyMuPDF (fitz) |
| 文本切分 | 自定义 3 种策略 |
| Embedding | 智谱 Embedding-3 API (2048d) |
| 向量库 | ChromaDB (HNSW 索引) |
| Re-ranking | BGE-Reranker-v2-m3 (Cross-Encoder) |
| HyDE | DeepSeek 生成假设文档 |
| LLM | DeepSeek API |
| 前端 | Gradio 6.x |

## License

MIT
