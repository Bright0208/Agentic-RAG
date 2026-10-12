
import re

from app.core.llm import get_llm
from app.rag.embedding import EmbeddingModel
from app.rag.vector_store import MilvusVectorStore
from app.rag.prompt import RAG_PROMPT


class RAGChain:

    def __init__(self, top_k=5):
        print("[1/3] 正在加载Embedding模型...", flush=True)
        self.embedding = EmbeddingModel(device="cpu")

        print("[2/3] 正在连接Milvus...", flush=True)
        self.store = MilvusVectorStore()

        print("[3/3] 正在初始化Ollama客户端...", flush=True)
        self.llm = get_llm()

        self.top_k = top_k
        print("RAG初始化完成！", flush=True)

    def retrieve(self, question: str):
        """将问题向量化并进行Milvus检索。"""

        query_vector = self.embedding.embed_query(question)

        return self.store.search(
            query_vector=query_vector,
            top_k=self.top_k
        )

    @staticmethod
    def build_context(results):
        """构建带来源编号的检索上下文。"""

        contexts = []
        sources = []

        for index, result in enumerate(results, start=1):
            entity = result["entity"]

            source_id = f"S{index}"
            filename = entity.get("source", "unknown")
            page = entity.get("page_number", 0)
            text = entity.get("text", "")

            contexts.append(
                f"[{source_id}]\n"
                f"文件：{filename}\n"
                f"页码：{page}\n"
                f"正文：\n{text}"
            )

            sources.append({
                "id": source_id,
                "filename": filename,
                "page": page,
                "chunk_id": result["id"],
                "score": float(result["distance"]),
                "text": text
            })

        return "\n\n---\n\n".join(contexts), sources

    def ask(self, question: str):
        """执行检索、上下文构建与答案生成。"""

        if not question.strip():
            raise ValueError("问题不能为空")

        print("[1/4] 正在检索Milvus...", flush=True)
        # 1. 从Milvus检索论文片段
        results = self.retrieve(question)

        print(f"[2/4] 检索完成：{len(results)}条", flush=True)

        if not results:
            return {
                "question": question,
                "answer": "知识库中没有检索到相关内容。",
                "sources": [],
                "cited_sources": [],
                "invalid_citations": []
            }

        # 2. 构建Prompt所需上下文
        context, sources = self.build_context(results)

        messages = RAG_PROMPT.format_messages(
            context=context,
            question=question
        )

        print("[3/4] 正在调用Ollama生成回答...", flush=True)
        # 3. 调用Ollama
        response = self.llm.invoke(messages,think=False)
        answer = str(response.content)

        print("[4/4] Ollama回答完成！", flush=True)
        # 4. 检查引用编号
        referenced_ids = set(
            re.findall(r"\[S\d+\]", answer)
        )

        valid_ids = {
            f"[{source['id']}]"
            for source in sources
        }

        invalid_citations = sorted(
            referenced_ids - valid_ids
        )

        # 删除没有对应检索证据的引用
        for citation in invalid_citations:
            answer = clean_response(str(response.content))

        cited_sources = [
            source for source in sources
            if f"[{source['id']}]" in answer
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "cited_sources": cited_sources,
            "invalid_citations": invalid_citations
        }
def clean_response(text: str) -> str:
    """移除完整的Qwen思考标签区段。"""
    import re

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    )

    return text.strip()