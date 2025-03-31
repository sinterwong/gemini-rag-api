from flask import Flask
from flask_cors import CORS
import os
import logging
from logging.handlers import RotatingFileHandler


def create_app(config_name=None):
    """创建Flask应用实例"""
    app = Flask(__name__)

    # 启用CORS
    CORS(app)

    # 配置日志
    if not os.path.exists('logs'):
        os.mkdir('logs')

    file_handler = RotatingFileHandler(
        'logs/rag_api.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('RAG API启动')

    # 加载配置
    if config_name == 'production':
        app.config.from_object('config.production.Config')
    elif config_name == 'development':
        app.config.from_object('config.development.Config')
    else:
        app.config.from_object('config.default.Config')

    # 从环境变量加载配置
    app.config.from_prefixed_env()

    # 初始化服务和注册路由
    from app.services import init_services
    init_services(app)

    from app.api import init_api
    init_api(app)

    return app
