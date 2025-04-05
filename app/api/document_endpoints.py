from flask import request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import uuid
from app.api import document_bp
from app.services.rag_service import get_rag_service


@document_bp.route('', methods=['POST'])
def add_document():
    pass


@document_bp.route('', methods=['GET'])
def list_documents():
    pass


@document_bp.route('/<doc_id>', methods=['GET'])
def get_document(doc_id):
    pass


@document_bp.route('/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    pass
