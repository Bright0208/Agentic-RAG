import time
from contextlib import asynccontextmanager

from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse
from app.rag.rag_chain import RAGChain


router = APIRouter(prefix="/api", tags=["RAG问答"])

rag_chain = None


def get_rag_chain():
    global rag_chain

    if rag_chain is None:
        rag_chain = RAGChain(top_k=5)

    return rag_chain


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    start = time.perf_counter()

    try:
        rag = get_rag_chain()
        result = rag.ask(request.question)

        elapsed = time.perf_counter() - start

        return ChatResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[
                {
                    "id": source["id"],
                    "filename": source["filename"],
                    "page": source["page"],
                    "chunk_id": source["chunk_id"],
                    "score": source["score"]
                }
                for source in result["cited_sources"]
            ],
            elapsed_seconds=round(elapsed, 2)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="RAG问答服务调用失败"
        ) from e