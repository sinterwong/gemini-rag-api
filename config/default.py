from config.default import Config
import os
from datetime import timedelta


class Config:
    """默认配置"""

    # 应用设置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    DEBUG = False

    # Gemini API设置
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    GEMINI_MODELS = {
        "embedding": "gemini-embedding-exp-03-07",
        "chat": "gemini-2.0-flash",
    }
    REQUESTS_PER_MINUTE = 60

    # RAG设置
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    MAX_WORKERS = 4

    # FAISS设置
    FAISS_INDEX_TYPE = 'flat'  # 'flat', 'ivfflat', 或 'hnsw'

    # 存储设置
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB最大上传大小
