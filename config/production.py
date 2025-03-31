import os
from config.default import Config


class Config(Config):
    """生产环境配置"""

    # 生产环境必须设置SECRET_KEY
    SECRET_KEY = os.environ.get('SECRET_KEY')

    # 更保守的请求限制，避免API超额
    REQUESTS_PER_MINUTE = 50

    # 使用更高效的FAISS索引
    FAISS_INDEX_TYPE = 'ivfflat'
