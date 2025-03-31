from flask import Blueprint

rag_bp = Blueprint('rag', __name__, url_prefix='/api/rag')
document_bp = Blueprint('document', __name__, url_prefix='/api/documents')


def init_api(app):
    """初始化API蓝图"""
    from app.api import rag_endpoints, document_endpoints

    app.register_blueprint(rag_bp)
    app.register_blueprint(document_bp)
