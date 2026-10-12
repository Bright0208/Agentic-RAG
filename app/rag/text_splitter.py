from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> list[Document]:
    """
    将PDF Document切分为多个Chunk。

    Args:
        documents: PDF解析得到的文档列表
        chunk_size: 每个Chunk最大字符数
        chunk_overlap: 相邻Chunk的重叠字符数

    Returns:
        带有来源和页码信息的Chunk列表
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap必须小于chunk_size且不能为负数")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            ". ",
            " ",
            ""
        ],
        add_start_index=True
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["page_number"] = chunk.metadata.get("page", 0) + 1

    return chunks