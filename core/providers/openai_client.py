"""OpenAI LLM 客户端实现

使用 openai SDK (AsyncOpenAI) 调用 OpenAI API。
通过 LLMClientFactory.register("openai", OpenAIClient) 自动注册。
"""

from typing import AsyncIterator, Any

from openai import AsyncOpenAI

from core.llm_client import (
    CompletionResponse,
    LLMClient,
    LLMClientFactory,
    Message,
)


class OpenAIClient(LLMClient):
    """OpenAI API 客户端

    支持 OpenAI 官方 API 及兼容接口（如 Azure OpenAI、第三方代理）。

    配置项:
        api_key: OpenAI API Key（必填）
        model: 模型名称，默认 "gpt-4"
        base_url: 自定义 API 地址（可选，用于代理或兼容接口）
        organization: OpenAI 组织 ID（可选）
        timeout: 请求超时秒数，默认 60
    """

    def __init__(self, config: dict):
        super().__init__(config)

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if config.get("organization"):
            client_kwargs["organization"] = config["organization"]
        if config.get("timeout"):
            client_kwargs["timeout"] = config["timeout"]

        self._client = AsyncOpenAI(**client_kwargs)

    async def complete(
        self,
        messages: list[Message | dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        thinking_effort: str | None = None,  # 思考强度（low/medium/high）
        **kwargs,
    ) -> CompletionResponse:
        """调用 OpenAI Chat Completion API

        Args:
            messages: 消息列表
            temperature: 采样温度 (0.0 - 2.0)
            max_tokens: 最大生成 token 数
            thinking_effort: 思考强度（low/medium/high），映射到 reasoning_effort
            **kwargs: 传递给 OpenAI API 的额外参数
                （如 top_p, frequency_penalty, presence_penalty, stop 等）

        Returns:
            CompletionResponse: 包含内容、用量和模型信息
        """
        normalized = self.normalize_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": normalized,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens

        # 处理思考强度（映射到 OpenAI 的 reasoning_effort）
        if thinking_effort:
            # OpenAI o1 系列模型支持 reasoning_effort
            # 其他模型忽略此参数
            effort_map = {
                "low": "low",
                "medium": "medium",
                "high": "high",
            }
            if effort := effort_map.get(thinking_effort.lower()):
                request_kwargs["reasoning_effort"] = effort

        # 合并额外参数
        request_kwargs.update(kwargs)

        # 移除不支持的参数
        request_kwargs.pop("thinking_effort", None)

        response = await self._client.chat.completions.create(**request_kwargs)

        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        # 处理 MiMo 等模型的 reasoning_content 字段
        # 如果 content 为空但 reasoning_content 有值，使用 reasoning_content
        content = choice.message.content or ""
        if not content and hasattr(choice.message, 'reasoning_content'):
            reasoning = getattr(choice.message, 'reasoning_content', None)
            if reasoning:
                content = reasoning

        return CompletionResponse(
            content=content,
            usage=usage,
            model=response.model,
            raw_response=response,
        )

    async def stream(
        self,
        messages: list[Message | dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式调用 OpenAI Chat Completion API

        Yields:
            str: 每个 chunk 的文本增量
        """
        normalized = self.normalize_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": normalized,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        request_kwargs.update(kwargs)

        response_stream = await self._client.chat.completions.create(
            **request_kwargs
        )

        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# 自动注册到工厂
LLMClientFactory.register("openai", OpenAIClient)

# 注册 OpenAI 兼容接口的别名
# 这些提供商使用与 OpenAI 相同的 API 协议，只需不同的 base_url
LLMClientFactory.register("mimo", OpenAIClient)
LLMClientFactory.register("deepseek", OpenAIClient)
