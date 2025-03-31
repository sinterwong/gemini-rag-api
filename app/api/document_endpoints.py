from flask import request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import uuid
from app.api import document_bp
from app.services.rag_service import get_rag_service


@document_bp.route('', methods=['POST'])
def add_document():
    """
    添加文档

    支持两种方式：
    1. JSON格式的文本文档
    2. 上传的文件

    ---
    JSON文档示例:
    {
        "text": "这是文档内容",
        "metadata": {
            "source": "用户提交",
            "author": "张三"
        }
    }
    """
    # 检查是否为文件上传
    if 'file' in request.files:
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400

        if file:
            # 安全地获取文件名并创建唯一路径
            filename = secure_filename(file.filename)
            file_id = str(uuid.uuid4())
            upload_folder = current_app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)

            file_path = os.path.join(upload_folder, f"{file_id}_{filename}")
            file.save(file_path)

            # 读取文件内容
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    # 尝试其他编码
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                except Exception as e:
                    return jsonify({'error': f'无法读取文件: {str(e)}'}), 400

            # 准备元数据
            metadata = {
                'source': 'file_upload',
                'filename': filename,
                'file_id': file_id
            }

            # 处理表单中的其他字段作为元数据
            for key, value in request.form.items():
                if key != 'file':
                    metadata[key] = value

    # JSON提交的文档
    else:
        data = request.json

        if not data or 'text' not in data:
            return jsonify({'error': '缺少文档文本'}), 400

        content = data['text']
        metadata = data.get('metadata', {})

    try:
        # 获取RAG服务
        rag_service = get_rag_service()

        # 添加文档
        doc_ids = rag_service.add_document(content, metadata)

        return jsonify({
            'message': '文档添加成功',
            'doc_ids': doc_ids
        })

    except Exception as e:
        current_app.logger.error(f"添加文档时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500


@document_bp.route('', methods=['GET'])
def list_documents():
    """
    列出所有文档

    可选params:
    - limit: 返回的最大文档数
    - offset: 起始偏移量
    """
    limit = request.args.get('limit', default=100, type=int)
    offset = request.args.get('offset', default=0, type=int)

    try:
        # 获取RAG服务
        rag_service = get_rag_service()

        # 获取文档列表
        documents = rag_service.vector_store.documents[offset:offset+limit]

        # 准备响应
        response = {
            'documents': [
                {
                    'doc_id': doc.doc_id,
                    # 截断文本以减小响应大小
                    'text': doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
                    'metadata': doc.metadata
                }
                for doc in documents
            ],
            'total': len(rag_service.vector_store.documents),
            'limit': limit,
            'offset': offset
        }

        return jsonify(response)

    except Exception as e:
        current_app.logger.error(f"列出文档时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500


@document_bp.route('/<doc_id>', methods=['GET'])
def get_document(doc_id):
    """获取单个文档"""
    try:
        # 获取RAG服务
        rag_service = get_rag_service()

        # 获取文档
        doc = rag_service.vector_store.get_document_by_id(doc_id)

        if doc is None:
            return jsonify({'error': '文档不存在'}), 404

        return jsonify({
            'doc_id': doc.doc_id,
            'text': doc.text,
            'metadata': doc.metadata
        })

    except Exception as e:
        current_app.logger.error(f"获取文档时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500


@document_bp.route('/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """删除文档"""
    try:
        # 获取RAG服务
        rag_service = get_rag_service()

        # 删除文档
        deleted = rag_service.vector_store.delete([doc_id])

        if deleted == 0:
            return jsonify({'error': '文档不存在'}), 404

        return jsonify({
            'message': '文档删除成功',
            'deleted_count': deleted
        })

    except Exception as e:
        current_app.logger.error(f"删除文档时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500
