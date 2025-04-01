from services.gemini_client import GeminiClient
from services.rag_service import RAGService


def test_rag_service():
    api_key = "YOUR_API_KEY"
    gemini_client = GeminiClient(api_key=api_key)

    rag_service = RAGService(gemini_client=gemini_client)

    documents = [
        """
        检索增强生成（RAG）是一种将大型语言模型与外部知识源结合的AI技术。
        它通过从文档语料库检索相关信息，然后将这些信息作为上下文提供给语言模型，
        从而增强模型的响应能力。RAG可以减少幻觉，提高事实准确性，并使模型能够
        回答关于其训练数据之外信息的问题。
        """,
        """
        FAISS（Facebook AI Similarity Search）是Facebook Research开发的相似性搜索库。
        它专为高效搜索和聚类大规模向量集设计，特别适用于需要快速搜索相似性的应用，
        如相似图像搜索、推荐系统和文本语义搜索。FAISS提供了多种索引类型，包括精确和
        近似搜索方法，能够处理数十亿个向量。
        """
    ]

    rag_service.add_documents(documents)

    results = rag_service.query("什么是FAISS？", top_k=1)
    print(f"检索结果: {results[0][0].text[:100]}...")

    generation = rag_service.generate("解释RAG和FAISS如何结合使用")
    print("\n生成回答:")
    print(generation["answer"])


if __name__ == "__main__":
    test_rag_service()
