import os
import logging
from flask import Flask, request, jsonify
from services.gemini_client import GeminiClient
from services.faiss_vector_store import FAISSVectorStore, Document
from services.rag_service import RAGService

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

VECTOR_STORE_PATH = os.environ.get(
    "VECTOR_STORE_PATH", "./data/index/rag_vector_store")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

gemini_client: GeminiClient = None
vector_store: FAISSVectorStore = None
rag_service: RAGService = None

app = Flask(__name__)


def initialize_rag_service():
    """
    初始化 Gemini 客户端、向量存储和 RAG 服务。
    这是关键，确保服务启动时这些都准备好了。
    """
    global gemini_client, vector_store, rag_service

    logging.info("Initializing RAG service...")

    if not GEMINI_API_KEY:
        logging.error("FATAL: GEMINI_API_KEY environment variable not set.")
        raise ValueError("Missing GEMINI_API_KEY environment variable.")

    try:
        logging.info("Initializing Gemini Client...")
        gemini_client = GeminiClient(api_key=GEMINI_API_KEY)
        if not gemini_client.is_available():
            logging.warning(
                "Gemini API might not be available. Check API key or network.")
        logging.info("Gemini Client initialized.")

        logging.info(
            f"Looking for existing vector store at: {VECTOR_STORE_PATH}")
        vector_store_config_path = os.path.join(
            VECTOR_STORE_PATH, "vector_store_config.json")

        if os.path.exists(vector_store_config_path):
            try:
                logging.info("Loading existing FAISSVectorStore...")
                vector_store = FAISSVectorStore.load(VECTOR_STORE_PATH)
                logging.info(
                    f"FAISSVectorStore loaded successfully. Stats: {vector_store.get_stats()}")
            except Exception as e:
                logging.error(
                    f"Failed to load vector store from {VECTOR_STORE_PATH}: {e}", exc_info=True)
                logging.info("Creating a new FAISSVectorStore instead.")
                dimension = gemini_client.get_embedding_dimension()
                vector_store = FAISSVectorStore(dimension=dimension)
        else:
            logging.info("No existing vector store found. Creating a new one.")
            dimension = gemini_client.get_embedding_dimension()
            vector_store = FAISSVectorStore(dimension=dimension)
            logging.info(
                f"New FAISSVectorStore created with dimension {dimension}.")
            os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

        logging.info("Initializing RAGService...")
        rag_service = RAGService(
            gemini_client=gemini_client, vector_store=vector_store)
        logging.info("RAGService initialized.")
        logging.info("RAG service initialization complete.")

    except Exception as e:
        logging.error(
            f"FATAL: Failed to initialize RAG service: {e}", exc_info=True)
        raise RuntimeError(f"RAG Service initialization failed: {e}") from e


@app.route('/health', methods=['GET'])
def health_check():
    """
    简单的健康检查端点。
    看看服务是不是活着，向量库状态如何。
    """
    if rag_service and vector_store:
        stats = vector_store.get_stats()
        return jsonify({
            "status": "ok",
            "message": "RAG service is running.",
            "vector_store_stats": stats
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": "RAG service is not initialized."
        }), 503  # Service Unavailable


@app.route('/documents', methods=['POST'])
def add_documents_route():
    """
    添加新文档到向量存储。
    请求体应该是 JSON 格式: {"documents": [{"text": "...", "metadata": {...}}, ...]}
    """
    if not rag_service:
        return jsonify({"error": "RAG service not initialized"}), 503

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if not data or 'documents' not in data or not isinstance(data['documents'], list):
        return jsonify({"error": "Invalid request format. Expected JSON: {'documents': [{'text': '...', 'metadata': {...}}]}"}), 400

    try:
        doc_ids = [doc.get('doc_id') for doc in data['documents']]
        texts = [doc.get('text') for doc in data['documents']]
        metadatas = [doc.get('metadata', {})
                     for doc in data['documents']]

        if not all(texts):
            return jsonify({"error": "All documents must have a non-empty 'text' field"}), 400

        logging.info(f"Received request to add {len(texts)} documents.")
        doc_ids = rag_service.add_documents(doc_ids, texts, metadatas)
        logging.info(f"Successfully added {len(doc_ids)} document chunks.")

        try:
            vector_store.save(VECTOR_STORE_PATH)
            logging.info(f"Vector store saved to {VECTOR_STORE_PATH}")
        except Exception as e:
            logging.error(
                f"Failed to save vector store after adding documents: {e}", exc_info=True)

        return jsonify({
            "message": f"Successfully added {len(texts)} documents ({len(doc_ids)} chunks).",
            "chunk_doc_ids": doc_ids
        }), 201

    except ValueError as ve:
        logging.warning(f"Value error during document addition: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Error adding documents: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {e}"}), 500


@app.route('/generate', methods=['POST'])
def generate_answer_route():
    """
    根据用户查询，执行 RAG 流程并生成答案。
    请求体应该是 JSON 格式: {"query_text": "...", "top_k": 5, "include_sources": true} (top_k, include_sources 可选)
    """
    if not rag_service:
        return jsonify({"error": "RAG service not initialized"}), 503

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query_text = data.get('query_text')
    if not query_text:
        return jsonify({"error": "Missing 'query_text' in request body"}), 400

    # 获取可选参数，给个默认值
    top_k = data.get('top_k', 5)
    include_sources = data.get('include_sources', True)
    temperature = data.get('temperature', 0.5)

    try:
        logging.info(
            f"Received query: '{query_text[:50]}...' with top_k={top_k}")
        result = rag_service.generate(
            query_text=query_text,
            top_k=top_k,
            include_sources=include_sources,
            temperature=temperature
        )
        logging.info(f"Generated answer for query: '{query_text[:50]}...'")
        return jsonify(result), 200  # OK

    except Exception as e:
        logging.error(
            f"Error generating answer for query '{query_text[:50]}...': {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {e}"}), 500


@app.route('/query', methods=['POST'])
def query_documents_route():
    """
    仅执行相似度搜索，不调用 LLM 生成。
    请求体 JSON: {"query_text": "...", "top_k": 5}
    """
    if not rag_service:
        return jsonify({"error": "RAG service not initialized"}), 503

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query_text = data.get('query_text')
    if not query_text:
        return jsonify({"error": "Missing 'query_text'"}), 400

    top_k = data.get('top_k', 5)

    try:
        logging.info(
            f"Received similarity query: '{query_text[:50]}...' with top_k={top_k}")
        results = rag_service.query(query_text=query_text, top_k=top_k)
        logging.info(f"Found {len(results)} relevant documents for query.")

        serializable_results = [
            {"document": doc.to_dict(), "score": score} for doc, score in results
        ]

        return jsonify({"results": serializable_results}), 200

    except Exception as e:
        logging.error(
            f"Error querying documents for '{query_text[:50]}...': {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {e}"}), 500


@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Not Found", "message": "The requested URL was not found on the server."}), 404


@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Server Error: {error}", exc_info=True)
    return jsonify({"error": "Internal Server Error", "message": "An internal error occurred."}), 500


if __name__ == '__main__':
    try:
        initialize_rag_service()
        logging.info("Starting Flask development server...")
        app.run(host='0.0.0.0', port=9797, debug=False)

    except Exception as e:
        logging.critical(f"Application failed to start: {e}", exc_info=True)
        exit(1)
