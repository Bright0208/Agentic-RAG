
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_llm


def main():

    # 1. 加载Ollama模型
    llm = get_llm()

    # 2. 构建消息
    messages = [
        SystemMessage(
            content=(
                "/no_think\n"
                "你是一名人工智能技术助手。"
                "直接回答问题，不要输出思考过程。"
                "请使用准确、简洁的中文。"
            )
        ),
        HumanMessage(
            content="请解释什么是RAG，它与直接调用LLM有什么区别？"
        )
    ]

    print("开始调用Ollama...")
    start_time = time.perf_counter()

    # 3. 调用LLM
    response = llm.invoke(messages)

    elapsed = time.perf_counter() - start_time

    # 4. 输出结果
    print("\n========== LLM回答 ==========")
    print(response.content)

    print("\n========== 运行信息 ==========")
    print(f"本次调用耗时：{elapsed:.2f}秒")
    print(f"模型：{response.response_metadata.get('model_name')}")
    print(f"Token统计：{response.usage_metadata}")


if __name__ == "__main__":
    main()
