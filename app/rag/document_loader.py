from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    """
    解析PDF文档，按页返回Document列表。
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF文件不存在: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("只支持PDF文件")

    loader = PyPDFLoader(str(path), mode="page")

    documents = loader.load()

    return documents