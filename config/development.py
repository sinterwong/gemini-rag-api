from config.default import Config


class Config(Config):
    """开发环境配置"""
    DEBUG = True

    REQUESTS_PER_MINUTE = 120
