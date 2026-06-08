"""
Step 4: ChromaDB 向量存储 + 检索
"""
import chromadb
from chromadb.config import Settings
from config import CHROMA_DIR, CHROMA_COLLECTION, CHROMA_TOP_K


def get_collection(reset: bool = False):
    """获取或创建 ChromaDB collection"""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass

    # 如果集合存在就返回，否则创建
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        collection = client.create_collection(CHROMA_COLLECTION)

    return collection


def add_chunks(chunks: list[str], embeddings: list[list[float]],
               metadatas: list[dict] = None):
    """批量添加向量到 ChromaDB"""
    collection = get_collection()
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # 如果元数据为空，给空字典
    if metadatas is None:
        metadatas = [{}] * len(chunks)

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def search(query_embedding: list[float], top_k: int = None,
            where: dict = None) -> list[dict]:
    """向量检索，返回 top-k 结果。可选 metadata 过滤"""
    top_k = top_k or CHROMA_TOP_K
    collection = get_collection()
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "distances", "metadatas"],
    }
    if where:
        kwargs["where"] = where  # ChromaDB 原生 metadata 过滤

    results = collection.query(**kwargs)
    items = []
    for i in range(len(results["ids"][0])):
        items.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
        })
    return items


def get_stats() -> dict:
    """获取向量库统计信息"""
    collection = get_collection()
    return {
        "collection": CHROMA_COLLECTION,
        "chunk_count": collection.count(),
        "storage_path": str(CHROMA_DIR),
    }


if __name__ == "__main__":
    stats = get_stats()
    print(f"集合: {stats['collection']}")
    print(f"文档数: {stats['chunk_count']}")
    print(f"存储路径: {stats['storage_path']}")
