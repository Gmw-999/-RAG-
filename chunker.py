"""
Step 2: 文本切分 (Chunking)
支持 3 种策略，方便做对比实验:
  - fixed_size:   固定窗口大小 + 重叠
  - semantic:     按段落边界切分，保持语义完整
  - recursive:    先用大分隔符(段落)再逐级用小分隔符切
"""
import re
from config import (
    CHUNKING_STRATEGY,
    FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP,
    SEMANTIC_MIN_CHUNK,
    RECURSIVE_CHUNK_SIZE, RECURSIVE_CHUNK_OVERLAP, RECURSIVE_SEPARATORS,
)


def chunk_fixed_size(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """固定窗口切分：每块 chunk_size 字，块间重叠 overlap 字"""
    chunk_size = chunk_size or FIXED_CHUNK_SIZE
    overlap = overlap or FIXED_CHUNK_OVERLAP
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) >= 50:  # 过滤太短的块
            chunks.append(chunk)
    return chunks


def chunk_semantic(text: str, min_chunk: int = None) -> list[str]:
    """语义切分：按自然段落(\n\n)边界切分"""
    min_chunk = min_chunk or SEMANTIC_MIN_CHUNK
    raw = text.split("\n\n")
    chunks = []
    for para in raw:
        para = para.strip()
        if len(para) >= min_chunk:
            chunks.append(para)
    return chunks


def chunk_recursive(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
    separators: list[str] = None
) -> list[str]:
    """
    递归切分：先用大分隔符(段落)切，超出 chunk_size 的再用次级分隔符切
    这是 LangChain RecursiveCharacterTextSplitter 的思路
    """
    chunk_size = chunk_size or RECURSIVE_CHUNK_SIZE
    overlap = overlap or RECURSIVE_CHUNK_OVERLAP
    separators = separators or RECURSIVE_SEPARATORS

    def _split(text: str, seps: list[str]) -> list[str]:
        """递归切分核心"""
        if not seps:
            # 最后一级：按固定长度切
            result = []
            for i in range(0, len(text), chunk_size - overlap):
                piece = text[i:i + chunk_size].strip()
                if len(piece) >= 50:
                    result.append(piece)
            return result

        sep = seps[0]
        if sep not in text:
            return _split(text, seps[1:])

        parts = text.split(sep)
        results = []
        current = ""
        for part in parts:
            if len(current) + len(part) + len(sep) <= chunk_size:
                current += (sep if current else "") + part
            else:
                if len(current.strip()) >= 50:
                    results.append(current.strip())
                # 当前块单独超长，用次级分隔符继续切
                if len(part) > chunk_size:
                    results.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part
        if len(current.strip()) >= 50:
            results.append(current.strip())
        return results

    return _split(text, separators)


def chunk_text(text: str, strategy: str = None) -> list[str]:
    """统一入口：按指定策略切分文本"""
    strategy = strategy or CHUNKING_STRATEGY
    if strategy == "fixed_size":
        return chunk_fixed_size(text)
    elif strategy == "semantic":
        return chunk_semantic(text)
    elif strategy == "recursive":
        return chunk_recursive(text)
    else:
        raise ValueError(f"未知切分策略: {strategy}")


if __name__ == "__main__":
    from parse_pdf import parse_all_pdfs

    docs = parse_all_pdfs()
    all_text = "\n\n".join(docs.values())

    for strategy in ["fixed_size", "semantic", "recursive"]:
        chunks = chunk_text(all_text, strategy)
        lens = [len(c) for c in chunks]
        print(f"\n[{strategy}] 块数: {len(chunks)}, "
              f"平均: {sum(lens)//len(lens)} 字符, "
              f"最小: {min(lens)}, 最大: {max(lens)}")
        print(f"  第1块前100字: {chunks[0][:100]}...")
