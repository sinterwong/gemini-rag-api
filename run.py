import os
from app import create_app
from flask.cli import FlaskGroup
import click
from services.faiss_vector_store import FAISSVectorStore


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
        rag_service = get_rag_service()

        documents = rag_service.vector_store.documents

        dimension = rag_service.vector_store.dimension
        index_type = app.config.get('FAISS_INDEX_TYPE', 'flat')
        rag_service.vector_store = FAISSVectorStore(
            dimension=dimension, index_type=index_type)

        data_dir = app.config['DATA_DIR']
        vector_store_dir = os.path.join(data_dir, 'vector_store')
        os.makedirs(vector_store_dir, exist_ok=True)
        rag_service.vector_store.save(vector_store_dir)

        click.echo(f"正在重新添加{len(documents)}个文档...")

        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]

            texts = [doc.text for doc in batch]
            metadatas = [doc.metadata for doc in batch]

            rag_service.add_documents(texts, metadatas)

            click.echo(
                f"已处理 {min(i+batch_size, len(documents))}/{len(documents)} 个文档")

        click.echo("索引重建完成！")


if __name__ == '__main__':
    cli()
