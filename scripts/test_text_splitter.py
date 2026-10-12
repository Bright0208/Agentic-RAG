from collections import Counter
from pathlib import Path

from app.rag.document_loader import load_pdf
from app.rag.text_splitter import split_documents


def main():
    pdf_path = Path("data/documents/test.pdf")

    # 1. 解析PDF
    documents = load_pdf(str(pdf_path))

    # 2. 切分文本
    chunks = split_documents(
        documents,
        chunk_size=800,
        chunk_overlap=120
    )

    print(f"PDF页数：{len(documents)}")
    print(f"Chunk总数：{len(chunks)}")

    if not chunks:
        print("没有生成Chunk，请检查PDF是否包含可提取文本")
        return

    # 3. Chunk长度统计
    lengths = [len(chunk.page_content) for chunk in chunks]

    print(f"最短Chunk：{min(lengths)}字符")
    print(f"最长Chunk：{max(lengths)}字符")
    print(f"平均Chunk：{sum(lengths) / len(lengths):.2f}字符")

    # 4. 每页Chunk统计
    page_counts = Counter(
        chunk.metadata["page_number"]
        for chunk in chunks
    )

    print("\n各页Chunk数量：")
    for page, count in sorted(page_counts.items()):
        print(f"第{page}页：{count}个")

    # 5. 输出前3个Chunk
    for chunk in chunks[:3]:
        print("\n" + "=" * 60)
        print(f"Chunk ID：{chunk.metadata['chunk_id']}")
        print(f"来源：{chunk.metadata.get('source')}")
        print(f"页码：{chunk.metadata['page_number']}")
        print(f"起始位置：{chunk.metadata.get('start_index')}")
        print(f"长度：{len(chunk.page_content)}")
        print(f"内容：\n{chunk.page_content[:500]}")


if __name__ == "__main__":
    main()