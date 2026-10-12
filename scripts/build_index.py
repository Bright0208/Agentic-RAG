
from pathlib import Path

from app.rag.document_loader import load_pdf
from app.rag.text_splitter import split_documents
from app.rag.embedding import EmbeddingModel
from app.rag.vector_store import MilvusVectorStore


def count_tokens(model, text):
    """计算Embedding模型实际接收的Token数量。"""
    return len(
        model.model.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False
        )["input_ids"]
    )


def main():
    pdf_path = Path("data/documents/test.pdf")

    # 1. 加载PDF
    documents = load_pdf(str(pdf_path))
    print(f"PDF页数：{len(documents)}")

    # 2. 使用原来的参数切分
    chunks = split_documents(
        documents,
        chunk_size=800,
        chunk_overlap=120
    )
    print(f"原始Chunk数量：{len(chunks)}")

    # 3. 加载Embedding模型
    model = EmbeddingModel(device="cpu")
    max_tokens = model.model.max_seq_length

    if max_tokens is None:
        raise RuntimeError("无法确定Embedding模型的Token限制")

    def longest_chunk(items):
        return max(
            count_tokens(model, chunk.page_content)
            for chunk in items
        )

    max_length = longest_chunk(chunks)

    print(f"模型最大Token数：{max_tokens}")
    print(f"原始Chunk最大Token数：{max_length}")

    # 4. 避免超出模型长度导致截断
    if max_length > max_tokens:
        print("发现超长Chunk，改用较小粒度重新切分")

        chunks = split_documents(
            documents,
            chunk_size=350,
            chunk_overlap=50
        )

        max_length = longest_chunk(chunks)

        print(f"重新切分后Chunk数：{len(chunks)}")
        print(f"重新切分后最大Token数：{max_length}")

        if max_length > max_tokens:
            raise RuntimeError(
                "仍有Chunk超出模型Token限制，请进一步减小chunk_size"
            )

    if not chunks:
        raise RuntimeError("没有可入库的Chunk")

    # 5. 生成Embedding
    texts = [chunk.page_content for chunk in chunks]

    print("开始生成Embedding...")

    vectors = model.embed_documents(texts)

    print(f"Embedding形状：{vectors.shape}")

    # 6. 存储到Milvus
    store = MilvusVectorStore()

    # 当前为本地重建测试，会清除旧Collection
    store.create_collection(reset=True)
    store.insert_chunks(chunks, vectors)

    print("Milvus知识库构建完成！")


if __name__ == "__main__":
    main()
