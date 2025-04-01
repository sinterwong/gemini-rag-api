import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
from models.document import Document


class FAISSVectorStore:
    """
    基于FAISS的向量存储，用于存储文档嵌入并执行相似度搜索

    这个类处理所有向量存储操作，包括添加、删除、搜索和持久化，
    完全独立于嵌入生成的方式。
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        metric_type: str = "l2",
        nlist: int = 100  # 仅用于IVF索引
    ):
        """
        初始化FAISS向量存储

        params:
            dimension: 嵌入向量的维度
            index_type: FAISS索引类型 ('flat', 'ivfflat', 或 'hnsw')
            metric_type: 距离度量类型 ('l2' 或 'ip' [内积])
            nlist: IVF索引的聚类数量（仅用于'ivfflat'索引类型）
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type

        if metric_type == "l2":
            faiss_metric = faiss.METRIC_L2
        elif metric_type == "ip":
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"不支持的度量类型: {metric_type}. 可用选项: 'l2', 'ip'")

        if index_type == "flat":
            if metric_type == "l2":
                self.index = faiss.IndexFlatL2(dimension)
            else:  # inner product
                self.index = faiss.IndexFlatIP(dimension)

        elif index_type == "ivfflat":
            # IVF索引需要一个量化器和训练
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, dimension, nlist, faiss_metric)
            self.index_needs_training = True

        elif index_type == "hnsw":
            # HNSW索引对于大规模数据集有很好的性能
            self.index = faiss.IndexHNSWFlat(
                dimension, 32, faiss_metric)  # 32是M参数

        else:
            raise ValueError(
                f"不支持的索引类型: {index_type}. 可用选项: 'flat', 'ivfflat', 'hnsw'")

        # 存储文档和ID
        self.documents: List[Document] = []
        self.doc_ids: List[str] = []

        # 索引状态
        self.is_trained = index_type == "flat" or index_type == "hnsw"  # flat和hnsw不需要训练

    def train(self, embeddings: np.ndarray):
        """
        训练索引（仅对需要训练的索引类型如IVF有效）

        params:
            embeddings: 用于训练的嵌入向量数组
        """
        if not self.is_trained and hasattr(self.index, 'train'):
            if embeddings.shape[0] < 100:  # 需要足够的样本来训练
                print("警告: 训练样本数量较少，可能影响索引质量")

            self.index.train(embeddings.astype('float32'))
            self.is_trained = True
            print(f"已使用{embeddings.shape[0]}个样本训练索引")

    def add(self, documents: List[Document], embeddings: np.ndarray) -> List[str]:
        """
        向索引中添加文档和嵌入

        params:
            documents: 要添加的文档对象列表
            embeddings: 对应的嵌入向量数组，shape应为(len(documents), dimension)

        return:
            添加的文档ID列表
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"文档数量 ({len(documents)}) 必须匹配嵌入数量 ({embeddings.shape[0]})")

        if not self.is_trained and hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.train(embeddings)

        self.index.add(embeddings.astype('float32'))

        doc_ids = [doc.doc_id for doc in documents]
        self.documents.extend(documents)
        self.doc_ids.extend(doc_ids)

        return doc_ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        搜索最相似的文档

        params:
            query_embedding: 查询的嵌入向量
            top_k: 返回结果数量

        return:
            (Document, 相似度分数)元组列表
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(
            query_embedding.astype('float32'), top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = self.documents[idx]
                distance = distances[0][i]

                # 对于inner product，转换为相似度分数（距离越大越好）
                score = distance if self.metric_type == "ip" else 1.0 / \
                    (1.0 + distance)

                results.append((doc, float(score)))

        return results

    def delete(self, doc_ids: List[str]) -> int:
        """
        从索引中删除文档

        注意: FAISS不直接支持删除，此实现通过重建索引实现

        params:
            doc_ids: 要删除的文档ID列表

        return:
            成功删除的文档数量
        """
        # 当前FAISS不直接支持元素删除，需要重建索引
        # 对于生产环境，应考虑使用标记-压缩策略或更高级的方法

        # 创建ID集合以便快速查找
        id_set = set(doc_ids)

        # 获取要保留的文档和索引
        new_documents = []
        keep_indices = []

        for i, doc_id in enumerate(self.doc_ids):
            if doc_id not in id_set:
                new_documents.append(self.documents[i])
                keep_indices.append(i)

        # 如果没有要删除的内容，立即返回
        if len(keep_indices) == len(self.documents):
            return 0

        # 重置索引
        deleted_count = len(self.documents) - len(new_documents)

        # 如果删除所有文档，直接重置
        if len(new_documents) == 0:
            self.__init__(self.dimension, self.index_type, self.metric_type)
            return deleted_count

        # 重建索引（在实际应用中可能需要重新获取嵌入）
        # 这里需要访问原始索引中的向量，但大多数FAISS索引不支持直接访问
        # 因此，这个功能在许多情况下需要外部存储嵌入或重新计算

        # 考虑到这个限制，这里仅提供一个警告
        self.documents = new_documents
        self.doc_ids = [doc.doc_id for doc in new_documents]

        print(f"警告: 已从文档存储中移除{deleted_count}个文档，但FAISS索引未更新。")
        print("要完全更新索引，请保存当前文档然后用其嵌入重建索引。")

        return deleted_count

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        通过ID获取文档

        params:
            doc_id: 要获取的文档ID

        return:
            找到的文档或None
        """
        for i, id_val in enumerate(self.doc_ids):
            if id_val == doc_id:
                return self.documents[i]
        return None

    def save(self, directory: str) -> None:
        """
        保存向量存储到指定目录

        params:
            directory: 保存目录路径
        """
        os.makedirs(directory, exist_ok=True)

        # 保存FAISS索引
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))

        # 保存文档和配置
        docs_data = [doc.to_dict() for doc in self.documents]
        config = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "doc_ids": self.doc_ids,
            "documents": docs_data,
            "is_trained": self.is_trained
        }

        with open(os.path.join(directory, "vector_store_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str) -> 'FAISSVectorStore':
        """
        从目录加载向量存储

        params:
            directory: 加载目录路径

        return:
            加载的FAISSVectorStore实例
        """
        # 加载配置
        config_path = os.path.join(directory, "vector_store_config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 创建实例
        vector_store = cls(
            dimension=config["dimension"],
            index_type=config["index_type"],
            metric_type=config["metric_type"]
        )

        # 加载文档
        vector_store.documents = [Document.from_dict(
            doc_dict) for doc_dict in config["documents"]]
        vector_store.doc_ids = config["doc_ids"]
        vector_store.is_trained = config["is_trained"]

        # 加载FAISS索引
        index_path = os.path.join(directory, "index.faiss")
        if os.path.exists(index_path):
            vector_store.index = faiss.read_index(index_path)
        else:
            raise FileNotFoundError(f"FAISS索引文件不存在: {index_path}")

        return vector_store

    def get_stats(self) -> Dict[str, Any]:
        """
        获取向量存储统计信息

        return:
            包含统计信息的字典
        """
        return {
            "document_count": len(self.documents),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "is_trained": self.is_trained
        }


# 使用示例
def example_usage():
    # 假设我们已经有嵌入和文档
    dimension = 3072  # Gemini嵌入维度
    vector_store = FAISSVectorStore(dimension=dimension)

    # 模拟文档和嵌入
    docs = [
        Document(text="这是第一个测试文档", metadata={"source": "test"}),
        Document(text="这是第二个测试文档", metadata={"source": "test"}),
        Document(text="这是完全不同主题的文档", metadata={"source": "other"})
    ]

    # 创建随机嵌入（在实际应用中，这些将由嵌入模型生成）
    embeddings = np.random.rand(len(docs), dimension).astype('float32')

    # 添加到向量存储
    vector_store.add(docs, embeddings)

    # 搜索
    query_embedding = np.random.rand(dimension).astype('float32')
    results = vector_store.search(query_embedding, top_k=2)

    for doc, score in results:
        print(f"文档 '{doc.text}' (ID: {doc.doc_id}), 分数: {score:.4f}")

    # 保存和加载
    vector_store.save("./vector_store_example")
    loaded_store = FAISSVectorStore.load("./vector_store_example")
    print(f"加载的向量存储包含 {len(loaded_store.documents)} 个文档")


if __name__ == "__main__":
    example_usage()
