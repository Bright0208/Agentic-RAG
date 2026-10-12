
import os

from langchain_ollama import ChatOllama


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b-instruct"
)


def get_llm() -> ChatOllama:
    """
    创建Ollama聊天模型。

    temperature=0:
        降低随机性，便于后续RAG实验比较。

    num_predict=512:
        限制单次生成的最大Token数。
    """

    # llm = ChatOllama(
    #     model=OLLAMA_MODEL,
    #     base_url=OLLAMA_BASE_URL,
    #     temperature=0,
    #     num_predict=512,
    #     reasoning=False
    # )
    # llm = ChatOllama(
    #     model=OLLAMA_MODEL,
    #     base_url=OLLAMA_BASE_URL,
    #     temperature=0,
    #     num_predict=1024,
    #     reasoning=False
    # )
    return ChatOllama(
        model="qwen3:4b-instruct",
        base_url="http://127.0.0.1:11434",
        temperature=0,
        num_predict=1024,
        reasoning=False
    )