
import time

from app.rag.rag_chain import RAGChain


def main():
    rag = RAGChain(top_k=5)

    question = (
        "论文采用什么算法联合优化"
        "车联网中的服务缓存和任务卸载？"
    )

    print("用户问题：", question)
    print("\n开始RAG问答...")

    start = time.perf_counter()

    result = rag.ask(question)

    elapsed = time.perf_counter() - start

    print("\n========== RAG回答 ==========")
    print(result["answer"])

    print("\n========== 回答引用 ==========")

    for source in result["cited_sources"]:
        print(
            f"[{source['id']}] "
            f"{source['filename']} "
            f"第{source['page']}页 "
            f"Chunk={source['chunk_id']}"
        )

    print("\n========== 检索证据 ==========")

    for source in result["sources"]:
        print(
            f"[{source['id']}] "
            f"相似度={source['score']:.4f} "
            f"页码={source['page']}"
        )

    print("\n无效引用：", result["invalid_citations"])
    print(f"总耗时：{elapsed:.2f}秒")


if __name__ == "__main__":
    main()
