from flask import current_app, g
import os

from gemini_client import GeminiClient
from faiss_vector_store import FAISSVectorStore, Document
from rag_service import RAGService


def get_rag_service():
    """
    获取或初始化RAG服务的单例实例

    该函数使用Flask的g对象在请求周期内缓存服务实例，
    以避免重复初始化。
    """
    if 'rag_service' not in g:
        # 获取配置
        api_key = current_app.config['GEMINI_API_KEY']
        data_dir = current_app.config['DATA_DIR']
        vector_store_dir = os.path.join(data_dir, 'vector_store')

        # 自定义模型映射（如果配置中有）
        models = current_app.config.get('GEMINI_MODELS', {
            "embedding": "gemini-embedding-exp-03-07",
            "chat": "gemini-2.0-flash",
        })

        # 初始化Gemini客户端
        gemini_client = GeminiClient(
            api_key=api_key,
            models=models,
            requests_per_minute=current_app.config.get(
                'REQUESTS_PER_MINUTE', 60)
        )

        # 检查是否有现有向量存储
        if os.path.exists(vector_store_dir):
            try:
                # 尝试加载现有向量存储
                vector_store = FAISSVectorStore.load(vector_store_dir)
                current_app.logger.info(
                    f"已加载向量存储，包含{len(vector_store.documents)}个文档")
            except Exception as e:
                current_app.logger.error(f"加载向量存储失败: {str(e)}")
                # 如果加载失败，创建新的
                dimension = gemini_client.get_embedding_dimension()
                vector_store = FAISSVectorStore(
                    dimension=dimension,
                    index_type=current_app.config.get(
                        'FAISS_INDEX_TYPE', 'flat')
                )
        else:
            # 创建新的向量存储
            dimension = gemini_client.get_embedding_dimension()
            vector_store = FAISSVectorStore(
                dimension=dimension,
                index_type=current_app.config.get('FAISS_INDEX_TYPE', 'flat')
            )

        # 初始化RAG服务
        g.rag_service = RAGService(
            gemini_client=gemini_client,
            vector_store=vector_store,
            chunk_size=current_app.config.get('CHUNK_SIZE', 500),
            chunk_overlap=current_app.config.get('CHUNK_OVERLAP', 100),
            max_workers=current_app.config.get('MAX_WORKERS', 4)
        )

        # 注册请求结束时的保存函数
        @current_app.teardown_appcontext
        def save_vector_store(exception=None):
            # 如果存在RAG服务，则保存向量存储
            if hasattr(g, 'rag_service'):
                try:
                    # 保存向量存储
                    os.makedirs(vector_store_dir, exist_ok=True)
                    g.rag_service.vector_store.save(vector_store_dir)
                    current_app.logger.info("向量存储已保存")
                except Exception as e:
                    current_app.logger.error(f"保存向量存储时出错: {str(e)}")

    return g.rag_service
