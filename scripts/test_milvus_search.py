
from app.rag.embedding import EmbeddingModel
from app.rag.vector_store import MilvusVectorStore


def main():
    # 1. 加载Embedding模型
    model = EmbeddingModel(device="cpu")

    # 2. 连接已有Milvus数据库
    store = MilvusVectorStore()

    if not store.client.has_collection(
        collection_name=store.collection_name
    ):
        raise RuntimeError(
            "Collection不存在，请先运行build_index.py"
        )

    # 3. 输入检索问题
    query = "车联网中如何通过服务缓存和任务卸载优化计算资源？"

    print(f"\n用户问题：{query}")

    # 4. 问题向量化
    query_vector = model.embed_query(query)

    # 5. Top-K检索
    results = store.search(
        query_vector=query_vector,
        top_k=5
    )

    # 6. 输出检索结果
    print(f"\n检索到 {len(results)} 个Chunk")

    for rank, result in enumerate(results, start=1):
        entity = result["entity"]

        print("\n" + "=" * 60)
        print(f"排名：Top {rank}")
        print(f"相似度：{result['distance']:.4f}")
        print(f"来源：{entity.get('source')}")
        print(f"页码：{entity.get('page_number')}")
        print(f"Chunk ID：{result['id']}")
        print(f"正文：\n{entity.get('text', '')[:500]}")


if __name__ == "__main__":
    main()
