import logging
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from gemini_client import GeminiClient
from faiss_vector_store import FAISSVectorStore, Document


class RAGService:
    """
    检索增强生成(RAG)服务

    将Gemini API和FAISS向量存储结合，提供完整的RAG功能。
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        vector_store: Optional[FAISSVectorStore] = None,
        chunk_size: int = 4096,
        chunk_overlap: int = 512,
        max_workers: int = 4
    ):
        """
        初始化RAG服务

        params:
            gemini_client: 初始化好的GeminiClient实例
            vector_store: 可选的FAISSVectorStore实例，如果不提供将自动创建
            chunk_size: 文档分块大小
            chunk_overlap: 文档分块重叠大小
            max_workers: 用于并行处理的最大线程数
        """
        self.gemini_client = gemini_client

        if vector_store is None:
            dimension = gemini_client.get_embedding_dimension()
            self.vector_store = FAISSVectorStore(dimension=dimension)
        else:
            self.vector_store = vector_store

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_workers = max_workers

    def _chunk_text(self, text: str) -> List[str]:
        """
        将文本分割为重叠的块

        params:
            text: 要分块的文本

        return:
            文本块列表
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            if end < len(text):
                last_break = max(
                    text.rfind(". ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("! ", start, end),
                    text.rfind(".\n", start, end),
                    text.rfind("\n\n", start, end)
                )

                if last_break != -1 and last_break > start + self.chunk_size // 2:
                    end = last_break + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = start + self.chunk_size - self.chunk_overlap

        return chunks

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        task_type: str = "retrieval_document"
    ) -> List[str]:
        """
        添加单个文档到RAG系统

        params:
            text: 文档文本
            metadata: 可选的文档元数据
            task_type: 嵌入任务类型

        return:
            添加的文档块ID列表
        """
        return self.add_documents([doc_id], [text], [metadata] if metadata else [{}], task_type)

    def add_documents(
        self,
        doc_ids: List[str],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        task_type: str = "retrieval_document"
    ) -> List[str]:
        """
        添加多个文档到RAG系统

        params:
            doc_ids: 文档ID列表
            texts: 文档文本列表
            metadatas: 可选的元数据字典列表
            task_type: 嵌入任务类型

        return:
            添加的文档块ID列表
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        if len(texts) != len(metadatas):
            raise ValueError("文档和元数据数量必须匹配")

        all_chunks = []
        all_embeddings = []
        all_docs = []

        for i, (doc_id, text, metadata) in enumerate(zip(doc_ids, texts, metadatas)):
            chunks = self._chunk_text(text)

            for j, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    "source_index": i,
                    "chunk_index": j,
                    "chunk_count": len(chunks),
                    "source_text": text[:200] + "..." if len(text) > 200 else text
                })

                doc = Document(doc_id=doc_id, text=chunk,
                               metadata=chunk_metadata)
                all_chunks.append(chunk)
                all_docs.append(doc)

        def get_embedding_safe(text):
            try:
                return self.gemini_client.embed_text(text, task_type=task_type)
            except Exception as e:
                print(f"获取嵌入时出错: {str(e)}")
                dimension = self.vector_store.dimension
                return np.zeros(dimension, dtype=np.float32)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            all_embeddings = list(executor.map(get_embedding_safe, all_chunks))

        all_embeddings = np.array(all_embeddings, dtype=np.float32)

        doc_ids = self.vector_store.add(all_docs, all_embeddings)

        return doc_ids

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        task_type: str = "retrieval_query"
    ) -> List[Tuple[Document, float]]:
        """
        查询相关文档

        params:
            query_text: 查询文本
            top_k: 返回结果数量
            task_type: 嵌入任务类型

        return:
            (Document, 相似度分数)元组列表
        """
        query_embedding = self.gemini_client.embed_text(
            query_text, task_type=task_type)

        results = self.vector_store.search(query_embedding, top_k=top_k)

        return results

    def generate(
        self,
        query_text: str,
        top_k: int = 5,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        include_sources: bool = True,
        prompt_template: Optional[str] = None,
        snippet_length: int = 50
    ) -> Dict[str, Any]:
        """
        生成对查询的回答

        params:
            query_text: 用户查询
            top_k: 检索文档数量
            temperature: 生成温度
            max_output_tokens: 生成的最大令牌数
            include_sources: 是否在响应中包含源文档
            prompt_template: 可选的自定义提示模板
            snippet_length: 相关文档展示的文本片段长度

        return:
            包含生成回答和源文档的字典
        """
        retrieval_results = self.query(query_text, top_k=top_k)

        # 如果啥也没找到，给个明确的提示
        if not retrieval_results:
            return {
                "summary": "抱歉，根据您提供的查询，未能找到足够的相关信息来生成摘要。",
                "retrieved_documents": [],
            }

        context_parts = []
        formatted_documents_for_output = []

        for i, (doc, score) in enumerate(retrieval_results):
            context_parts.append(f"文档 {i+1}:\n{doc.text}")

            # 如果需要包含来源信息，现在就处理好，提取需要的信息
            if include_sources:
                # 从 text 创建片段
                snippet = doc.text[:snippet_length] + \
                    ('...' if len(doc.text) > snippet_length else '')

                formatted_documents_for_output.append({
                    "snippet": snippet,
                    "metadata": doc.metadata,
                })

        context = "\n\n".join(context_parts)

        if prompt_template is None:
            # 设置默认的prompt
            prompt = f"""
            请仔细阅读以下提供的 {len(retrieval_results)} 篇文档，并根据用户的查询，生成一个简洁、连贯、信息准确的摘要。
            摘要应概括文档中与查询最相关的核心内容。
            请不要添加文档中未提及的信息或进行过度推断。

            --- 开始提供的文档 ---
            {context}
            --- 结束提供的文档 ---

            用户的查询: {query_text}

            请生成摘要:
            """
        else:
            prompt = prompt_template.replace(
                "{context}", context).replace("{query}", query_text)
            logging.info("Using custom prompt template for summarization.")

        logging.info(f"Generating summary for query: '{query_text[:50]}...'")
        try:
            summary_text = self.gemini_client.generate_content(
                prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
            logging.info("Summary generated successfully.")
        except Exception as e:
            logging.error(
                f"Error generating summary from Gemini: {e}", exc_info=True)
            summary_text = "抱歉，在为您生成摘要时遇到了问题。"

        result = {
            "summary": summary_text,
            "retrieved_documents": formatted_documents_for_output if include_sources else [],
        }

        return result

    def save(self, directory: str) -> None:
        """
        保存RAG系统状态

        params:
            directory: 保存目录
        """
        os.makedirs(directory, exist_ok=True)

        vector_store_dir = os.path.join(directory, "vector_store")
        self.vector_store.save(vector_store_dir)

        config = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_workers": self.max_workers
        }

    @classmethod
    def load(cls, directory: str, gemini_client: GeminiClient) -> 'RAGService':
        """
        加载RAG系统状态

        params:
            directory: 加载目录
            gemini_client: 初始化好的GeminiClient实例

        return:
            加载的RAGService实例
        """
        # 加载向量存储
        vector_store_dir = os.path.join(directory, "vector_store")
        vector_store = FAISSVectorStore.load(vector_store_dir)

        rag_service = cls(
            gemini_client=gemini_client,
            vector_store=vector_store
        )

        return rag_service
