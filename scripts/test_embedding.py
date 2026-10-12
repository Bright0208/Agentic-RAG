import numpy as np

from app.rag.embedding import EmbeddingModel


def main():
    model = EmbeddingModel(device="cpu")

    texts = [
        "车联网中的任务卸载可以降低车辆端计算负载。",
        "深度强化学习可以优化边缘计算资源调度。",
        "今天广州的天气非常晴朗。"
    ]

    query = "如何优化车联网中的计算任务卸载？"

    # 文档Embedding
    doc_vectors = model.embed_documents(texts)

    # 查询Embedding
    query_vector = model.embed_query(query)

    print("文档向量形状:", doc_vectors.shape)
    print("查询向量形状:", query_vector.shape)

    # 归一化向量的点积等于余弦相似度
    scores = np.dot(doc_vectors, query_vector)

    print("\n相似度排名:")

    for index in np.argsort(scores)[::-1]:
        print(f"{scores[index]:.4f} | {texts[index]}")


if __name__ == "__main__":
    main()