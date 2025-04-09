from services.faiss_vector_store import FAISSVectorStore, Document

import numpy as np
import os
import shutil


def test_faiss_vector_store():
    # 创建一个临时目录用于测试
    test_dir = "./data/index/test_vector_store"
    os.makedirs(test_dir, exist_ok=True)

    # 测试数据
    dimension = 3072
    docs = [
        Document(text="这是第一个测试文档", metadata={"source": "test"}),
        Document(text="这是第二个测试文档", metadata={"source": "test"}),
        Document(text="这是完全不同主题的文档", metadata={"source": "other"})
    ]
    embeddings = np.random.rand(len(docs), dimension).astype('float32')

    # 创建向量存储
    vector_store = FAISSVectorStore(dimension=dimension)
    vector_store.add(docs, embeddings)

    # 测试搜索
    query_embedding = np.random.rand(dimension).astype('float32')
    results = vector_store.search(query_embedding, top_k=2)
    assert len(results) == 2

    # 测试保存和加载
    vector_store.save(test_dir)
    loaded_store = FAISSVectorStore.load(test_dir)
    assert len(loaded_store.documents) == len(docs)
    assert loaded_store.dimension == dimension

    # 测试删除
    deleted_count = loaded_store.delete([docs[0].doc_id])
    assert deleted_count == 1
    assert len(loaded_store.documents) == len(docs) - 1

    # 测试获取文档
    doc = loaded_store.get_document_by_id(docs[1].doc_id)
    assert doc is not None
    assert doc.text == docs[1].text

    # 测试获取不存在的文档
    doc = loaded_store.get_document_by_id("nonexistent_id")
    assert doc is None

    # 测试统计信息
    stats = loaded_store.get_stats()
    assert stats["document_count"] == len(docs) - 1
    assert stats["dimension"] == dimension

    # 清理临时目录
    shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_faiss_vector_store()
