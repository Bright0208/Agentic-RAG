from pathlib import Path

from app.rag.document_loader import load_pdf


def main():
    pdf_path = Path("data/documents/test.pdf")

    documents = load_pdf(str(pdf_path))

    print(f"成功解析PDF，共 {len(documents)} 页")

    for i, doc in enumerate(documents[:3]):
        print(f"\n{'=' * 50}")
        print(f"第 {i + 1} 个Document")
        print(f"元数据: {doc.metadata}")
        print(f"文本长度: {len(doc.page_content)}")
        print(f"文本内容预览:\n{doc.page_content[:500]}")


if __name__ == "__main__":
    main()