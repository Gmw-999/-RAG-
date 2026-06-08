"""
Step 1: PDF 文本提取
输入: PDF 文件路径
输出: 纯文本字符串
"""
import fitz
from pathlib import Path
from config import DATA_DIR, PDF_FILES


def parse_pdf(filepath: str) -> str:
    """解析单个 PDF 文件，返回完整文本"""
    doc = fitz.open(filepath)
    pages = []
    for i in range(doc.page_count):
        text = doc[i].get_text()
        if text.strip():
            pages.append(text.strip())
    doc.close()
    return "\n\n".join(pages)


def parse_all_pdfs() -> dict:
    """解析 data/ 下所有配置的 PDF，返回 {文件名: 文本}"""
    docs = {}
    for name in PDF_FILES:
        path = DATA_DIR / name
        if not path.exists():
            print(f"[跳过] {name} 不存在")
            continue
        print(f"[解析] {name} ...", end=" ")
        text = parse_pdf(str(path))
        docs[name] = text
        print(f"{len(text)} 字符")
    return docs


if __name__ == "__main__":
    docs = parse_all_pdfs()
    total = sum(len(t) for t in docs.values())
    print(f"\n共 {len(docs)} 个 PDF，总计 {total} 字符")
    if docs:
        first_text = list(docs.values())[0]
        print(f"\n=== 第一篇预览 (前 400 字) ===\n{first_text[:400]}")
