from fastapi import FastAPI

app = FastAPI(
    title="Agentic RAG Knowledge Base",
    version="0.1.0",
    description="基于LangGraph与Milvus的智能知识库系统"
)


@app.get("/")
def index():
    return {
        "message": "Agentic RAG is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }