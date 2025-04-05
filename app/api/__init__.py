from flask import Blueprint

rag_bp = Blueprint('rag', __name__, url_prefix='/api/rag')
document_bp = Blueprint('document', __name__, url_prefix='/api/documents')


def init_api(app):
    pass
