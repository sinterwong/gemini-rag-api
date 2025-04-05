from flask import request, jsonify, current_app
from app.api import rag_bp
from app.services.rag_service import get_rag_service


@rag_bp.route('/query', methods=['POST'])
def query():
    pass


@rag_bp.route('/generate', methods=['POST'])
def generate():
    pass


@rag_bp.route('/health', methods=['GET'])
def health_check():
    pass
