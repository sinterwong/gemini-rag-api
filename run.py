import os
from app import create_app
from flask.cli import FlaskGroup
import click
from faiss_vector_store import FAISSVectorStore


# 根据环境变量创建应用实例
app = create_app(os.getenv('FLASK_ENV', 'development'))

cli = FlaskGroup(app)


@cli.command('rebuild-index')
@click.option('--confirm', is_flag=True, help='确认重建索引')
def rebuild_index(confirm):
    """重建FAISS索引（警告：这将删除现有索引）"""
    if not confirm:
        click.echo('警告：这将删除现有索引并重建。请使用--confirm标志确认此操作。')
        return

    from app.services.rag_service import get_rag_service

    with app.app_context():
        # 获取RAG服务
        rag_service = get_rag_service()

        # 保存当前文档
        documents = rag_service.vector_store.documents

        # 重新初始化向量存储
        dimension = rag_service.vector_store.dimension
        index_type = app.config.get('FAISS_INDEX_TYPE', 'flat')
        rag_service.vector_store = FAISSVectorStore(
            dimension=dimension, index_type=index_type)

        # 保存空索引（清除旧索引）
        data_dir = app.config['DATA_DIR']
        vector_store_dir = os.path.join(data_dir, 'vector_store')
        os.makedirs(vector_store_dir, exist_ok=True)
        rag_service.vector_store.save(vector_store_dir)

        # 重新添加文档
        click.echo(f"正在重新添加{len(documents)}个文档...")

        # 按批次处理文档
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]

            texts = [doc.text for doc in batch]
            metadatas = [doc.metadata for doc in batch]

            # 添加文档
            rag_service.add_documents(texts, metadatas)

            click.echo(
                f"已处理 {min(i+batch_size, len(documents))}/{len(documents)} 个文档")

        click.echo("索引重建完成！")


if __name__ == '__main__':
    cli()
