# test gemini client

import numpy as np
from unittest.mock import MagicMock
from services.gemini_client import GeminiClient


def test_gemini_client():
    api_key = "YOUR_API_KEY"
    client = GeminiClient(api_key=api_key)

    text = "这是一个测试文本，用于生成嵌入向量。"
    embedding = client.embed_text(text, task_type="semantic_similarity")
    print(f"嵌入维度: {embedding.shape[0]}")

    texts = [
        "第一个测试文本。",
        "第二个测试文本，与第一个有些不同。",
        "第三个测试文本，完全不同的主题。"
    ]
    embeddings = client.embed_batch(texts)
    print(f"批量嵌入shape: {embeddings.shape}")

    prompt = "简要解释人工智能是什么？"
    response = client.generate_content(
        prompt, temperature=0.5, max_output_tokens=4096)
    print(f"生成响应:\n{response}")


if __name__ == "__main__":
    test_gemini_client()
