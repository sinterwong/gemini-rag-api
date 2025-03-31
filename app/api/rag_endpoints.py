from flask import request, jsonify, current_app
from app.api import rag_bp
from app.services.rag_service import get_rag_service


@rag_bp.route('/query', methods=['POST'])
def query():
    """
    查询相关文档

    ---
    请求示例:
    {
        "query": "什么是RAG？",
        "top_k": 5
    }
    """
    data = request.json

    if not data or 'query' not in data:
        return jsonify({'error': '缺少查询参数'}), 400

    query_text = data['query']
    top_k = data.get('top_k', 5)

    try:
        # 获取RAG服务
        rag_service = get_rag_service()

        # 查询文档
        results = rag_service.query(query_text, top_k=top_k)

        # 准备响应
        response = {
            'results': [
                {
                    'text': doc.text,
                    'score': float(score),
                    'metadata': doc.metadata,
                    'doc_id': doc.doc_id
                }
                for doc, score in results
            ]
        }

        return jsonify(response)

    except Exception as e:
        current_app.logger.error(f"查询时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500


@rag_bp.route('/generate', methods=['POST'])
def generate():
    """
    生成回答

    ---
    请求示例:
    {
        "query": "什么是RAG？",
        "top_k": 5,
        "temperature": 0.2,
        "max_tokens": 1024,
        "include_sources": true
    }
    """
    data = request.json

    if not data or 'query' not in data:
        return jsonify({'error': '缺少查询参数'}), 400

    query_text = data['query']
    top_k = data.get('top_k', 5)
    temperature = data.get('temperature', 0.2)
    max_tokens = data.get('max_tokens', 1024)
    include_sources = data.get('include_sources', True)

    try:
        # 获取RAG服务
        rag_service = get_rag_service()

        # 生成回答
        result = rag_service.generate(
            query_text,
            top_k=top_k,
            temperature=temperature,
            max_output_tokens=max_tokens,
            include_sources=include_sources
        )

        return jsonify(result)

    except Exception as e:
        current_app.logger.error(f"生成回答时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500


@rag_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    try:
        # 获取RAG服务
        rag_service = get_rag_service()

        # 获取向量存储统计
        stats = rag_service.vector_store.get_stats()

        return jsonify({
            'status': 'ok',
            'vector_store': stats
        })

    except Exception as e:
        current_app.logger.error(f"健康检查时出错: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
