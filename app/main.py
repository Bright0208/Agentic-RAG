from fastapi import FastAPI

from app.api.chat import router as chat_router


app = FastAPI(
    title="Agentic RAG Knowledge Base",
    description="基于LangGraph与Milvus的智能知识库系统",
    version="0.2.0"
)


@app.get("/")
def index():
    return {"message": "Agentic RAG is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(chat_router)