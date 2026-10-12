from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户问题"
    )


class SourceItem(BaseModel):
    id: str
    filename: str
    page: int
    chunk_id: int
    score: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    elapsed_seconds: float