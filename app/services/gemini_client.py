import time
import numpy as np
from typing import List, Dict, Any, Optional, Union
from google import genai
from google.genai import types


class GeminiClient:
    """
    Gemini API客户端封装类

    封装了Gemini API的主要功能，包括文本嵌入和生成，
    同时实现了速率限制和异常处理。
    """

    DEFAULT_MODELS = {
        "embedding": "gemini-embedding-exp-03-07",  # 嵌入模型
        "chat": "gemini-2.0-flash",                # 文本生成模型
    }

    TASK_TYPES = {
        "semantic_similarity": "SEMANTIC_SIMILARITY",
        "retrieval_document": "RETRIEVAL_DOCUMENT",
        "retrieval_query": "RETRIEVAL_QUERY",
        "classification": "CLASSIFICATION",
    }

    def __init__(
        self,
        api_key: str,
        models: Optional[Dict[str, str]] = None,
        requests_per_minute: int = 60
    ):
        """
        初始化Gemini客户端

        params:
            api_key: Gemini API密钥
            models: 自定义模型映射，默认使用DEFAULT_MODELS
            requests_per_minute: 每分钟最大API请求数，用于速率限制
        """
        self.client = genai.Client(api_key=api_key)
        self.models = models or self.DEFAULT_MODELS

        # 速率限制参数
        self.requests_per_minute = requests_per_minute
        self.last_request_time = 0

    def _rate_limit(self):
        """实现基本速率限制，避免超过API配额"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / self.requests_per_minute

        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def embed_text(
        self,
        text: str,
        task_type: str = "semantic_similarity",
        model: Optional[str] = None
    ) -> np.ndarray:
        """
        获取文本的嵌入向量

        params:
            text: 要嵌入的文本
            task_type: 嵌入任务类型，可选值见TASK_TYPES
            model: 要使用的嵌入模型，默认使用self.models["embedding"]

        return:
            numpy数组形式的嵌入向量

        异常:
            ValueError: 如果task_type无效
            RuntimeError: 如果API调用失败
        """
        self._rate_limit()

        # 验证任务类型
        if task_type not in self.TASK_TYPES:
            valid_types = ", ".join(self.TASK_TYPES.keys())
            raise ValueError(f"无效的task_type: {task_type}. 有效值: {valid_types}")

        actual_task_type = self.TASK_TYPES[task_type]
        embedding_model = model or self.models["embedding"]

        try:
            result = self.client.models.embed_content(
                model=embedding_model,
                contents=text,
                config=types.EmbedContentConfig(task_type=actual_task_type)
            )

            embedding = np.array(result.embeddings[0].values, dtype=np.float32)
            return embedding

        except Exception as e:
            raise RuntimeError(f"嵌入生成失败: {str(e)}") from e

    def embed_batch(
        self,
        texts: List[str],
        task_type: str = "semantic_similarity",
        model: Optional[str] = None,
        batch_size: int = 10
    ) -> np.ndarray:
        """
        批量获取文本嵌入

        params:
            texts: 要嵌入的文本列表
            task_type: 嵌入任务类型
            model: 要使用的嵌入模型
            batch_size: 每批处理的文本数量

        return:
            numpy数组，每行是一个文本的嵌入
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            for text in batch:
                try:
                    embedding = self.embed_text(text, task_type, model)
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"警告: 嵌入文本时出错: {str(e)}")
                    # 对失败的嵌入使用零向量
                    if embeddings:  # 确保至少有一个成功的嵌入来确定维度
                        embeddings.append(np.zeros_like(embeddings[0]))
                    else:
                        # 如果第一个嵌入就失败了，我们需要明确维度
                        # 使用Gemini API的默认维度3072
                        embeddings.append(np.zeros(3072, dtype=np.float32))

        return np.array(embeddings, dtype=np.float32)

    def generate_content(
        self,
        prompt: Union[str, List[str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> str:
        """
        生成文本内容

        params:
            prompt: 提示文本或提示列表
            model: 要使用的生成模型，默认使用self.models["chat"]
            temperature: 生成温度，控制随机性
            max_output_tokens: 生成的最大令牌数
            safety_settings: 安全设置选项
            generation_config: 其他生成配置

        return:
            生成的文本内容
        """
        self._rate_limit()

        generation_model = model or self.models["chat"]

        contents = prompt if isinstance(prompt, list) else [prompt]

        # 暂时没用到
        config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }

        try:
            response = self.client.models.generate_content(
                model=generation_model,
                contents=contents,
                config=config
            )

            return response.text

        except Exception as e:
            raise RuntimeError(f"内容生成失败: {str(e)}") from e

    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """
        获取指定嵌入模型的维度

        params:
            model: 嵌入模型名称，默认使用self.models["embedding"]

        return:
            嵌入向量的维度
        """
        embedding_model = model or self.models["embedding"]

        # 生成一个短文本的嵌入来确定维度
        sample_text = "Dimension test."
        embedding = self.embed_text(sample_text, model=embedding_model)

        return embedding.shape[0]

    def is_available(self) -> bool:
        """
        检查API是否可用

        return:
            是否可用
        """
        try:
            _ = self.client.models.get_model(self.models["chat"])
            return True
        except Exception:
            return False
