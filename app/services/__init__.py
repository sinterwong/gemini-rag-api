from flask import Flask


def init_services(app):
    """初始化服务"""
    # 确保数据目录存在
    import os
    os.makedirs(app.config.get('DATA_DIR', 'data'), exist_ok=True)

    # 初始化上传目录
    os.makedirs(app.config.get('UPLOAD_FOLDER', 'data/uploads'), exist_ok=True)
