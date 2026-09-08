"""LLM 客户端模块 - 提供统一的 LLM 调用接口"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Any, TYPE_CHECKING, Callable, Awaitable
import json

if TYPE_CHECKING:
    from core.llm_config import LLMConfig, ModelTier, TierFallbackNotice


@dataclass
class Message:
    """对话消息"""
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class CompletionResponse:
    """LLM 完成响应"""
    content: str
    usage: dict = field(default_factory=dict)
    model: str = ""
    raw_response: Any = None
    tier_notice: "TierFallbackNotice | None" = None  # 层级降级提示


class LLMClient(ABC):
    """LLM 客户端抽象基类

    支持两种使用模式:
    1. 直接模式: 通过 LLMClientFactory.create() 创建，使用固定模型
    2. 层级模式: 通过 TieredLLMClient 包装，按 fast/balanced/flagship 层级路由
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = config.get("model", "gpt-4")
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url")

    @abstractmethod
    async def complete(
        self,
        messages: list[Message | dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> CompletionResponse:
        """同步完成请求

        Args:
            messages: 消息列表，支持 Message 对象或 dict
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            **kwargs: 额外参数

        Returns:
            CompletionResponse: 完成响应
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[Message | dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """流式完成请求

        Args:
            messages: 消息列表
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            **kwargs: 额外参数

        Yields:
            str: 流式输出的文本片段
        """
        pass

    def normalize_messages(self, messages: list[Message | dict]) -> list[dict]:
        """将消息统一转换为 dict 格式"""
        result = []
        for msg in messages:
            if isinstance(msg, Message):
                result.append({"role": msg.role, "content": msg.content})
            else:
                result.append(msg)
        return result

    async def complete_json(
        self,
        messages: list[Message | dict],
        temperature: float = 0.3,
        max_retries: int = 2,
        thinking_effort: str | None = None,
        **kwargs
    ) -> dict | list:
        """完成请求并解析 JSON 响应

        自动从响应中提取 JSON 部分并解析，失败时重试。

        Args:
            messages: 消息列表
            temperature: 采样温度（建议较低值以获得稳定输出）
            max_retries: 解析失败时的最大重试次数
            thinking_effort: 思考强度（low/medium/high）
            **kwargs: 额外参数

        Returns:
            解析后的 JSON 对象（dict 或 list）

        Raises:
            ValueError: JSON 解析失败且超过重试次数
        """
        last_error = None
        for attempt in range(max_retries + 1):
            response = await self.complete(
                messages, temperature=temperature, thinking_effort=thinking_effort, **kwargs
            )
            try:
                return self._extract_json(response.content)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt < max_retries:
                    # 追加一条消息要求修正格式
                    messages = list(messages) + [
                        {"role": "assistant", "content": response.content},
                        {
                            "role": "user",
                            "content": "输出格式有误，请仅返回合法的 JSON，不要包含其他内容。"
                        },
                    ]
        raise ValueError(
            f"JSON 解析失败（重试 {max_retries} 次后）: {last_error}"
        )

    @staticmethod
    def _extract_json(text: str) -> dict | list:
        """从文本中提取并解析 JSON

        支持以下格式：
        - 纯 JSON 文本
        - 被 ```json ... ``` 包裹的 JSON
        - 文本中嵌入的 JSON 对象或数组
        """
        text = text.strip()

        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 代码块
        if "```json" in text:
            start = text.index("```json") + len("```json")
            end = text.index("```", start)
            return json.loads(text[start:end].strip())

        # 尝试提取 ``` ... ``` 代码块
        if "```" in text:
            start = text.index("```") + len("```")
            end = text.index("```", start)
            return json.loads(text[start:end].strip())

        # 尝试查找第一个 { 或 [ 到最后一个 } 或 ]
        for open_char, close_char in [("{", "}"), ("[", "]")]:
            first = text.find(open_char)
            last = text.rfind(close_char)
            if first != -1 and last != -1 and last > first:
                try:
                    return json.loads(text[first:last + 1])
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"无法从响应中提取 JSON: {text[:200]}...")


class LLMClientFactory:
    """LLM 客户端工厂

    使用 register 注册新的提供商实现，使用 create 创建客户端实例。

    示例:
        LLMClientFactory.register("openai", OpenAIClient)
        client = LLMClientFactory.create("openai", {"api_key": "sk-..."})
    """

    _clients: dict[str, type[LLMClient]] = {}

    @classmethod
    def register(cls, provider: str, client_class: type[LLMClient]):
        """注册 LLM 提供商

        provider 名称不区分大小写，统一以小写存储。

        Args:
            provider: 提供商标识（如 "openai"、"anthropic"）
            client_class: LLMClient 子类
        """
        cls._clients[provider.lower()] = client_class

    @classmethod
    def create(cls, provider: str, config: dict) -> LLMClient:
        """创建 LLM 客户端实例

        Args:
            provider: 提供商标识
            config: 配置字典，包含 api_key、model 等

        Returns:
            LLMClient 实例

        Raises:
            ValueError: 未知的提供商
        """
        provider_key = provider.lower()
        if provider_key not in cls._clients:
            available = ", ".join(cls._clients.keys()) or "(无)"
            raise ValueError(
                f"未知的 LLM 提供商: '{provider}'。"
                f"已注册的提供商: {available}。"
                f"请先调用 LLMClientFactory.register() 注册。"
            )
        return cls._clients[provider_key](config)

    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的提供商"""
        return list(cls._clients.keys())


# 回调类型：当发生层级降级时通知调用方
FallbackCallback = Callable[["TierFallbackNotice"], Awaitable[bool]]


class TieredLLMClient:
    """多层级 LLM 客户端

    包装 LLMConfig，根据工具层指定的推荐层级自动路由到对应模型。
    当推荐层级未配置时，通过 fallback_callback 回调通知调用方，
    由调用方决定是否继续使用降级/升级的模型。

    使用示例:
        config = LLMConfig.from_file("llm_config.json")
        client = TieredLLMClient(config)

        # 工具层只需指定推荐层级
        response = await client.complete(
            messages=[...],
            tier=ModelTier.FLAGSHIP,  # 推荐旗舰模型
        )

        # 如果 flagship 未配置，response.tier_notice 会包含降级信息
        if response.tier_notice:
            print(response.tier_notice.message)
    """

    def __init__(
        self,
        llm_config: "LLMConfig",
        fallback_callback: FallbackCallback | None = None,
    ):
        """初始化多层级客户端

        Args:
            llm_config: 多层级模型配置
            fallback_callback: 层级降级回调函数。
                签名为 async (notice: TierFallbackNotice) -> bool。
                返回 True 表示继续使用降级模型，False 表示中止。
                为 None 时自动接受降级（静默继续）。
        """
        self._llm_config = llm_config
        self._fallback_callback = fallback_callback
        self._clients: dict[str, LLMClient] = {}  # 缓存已创建的客户端

    def _get_client(self, tier_config: "TierModelConfig") -> LLMClient:
        """根据层级配置获取或创建 LLM 客户端（带缓存）"""
        cache_key = f"{tier_config.provider}:{tier_config.model}"
        if cache_key not in self._clients:
            self._clients[cache_key] = LLMClientFactory.create(
                tier_config.provider,
                tier_config.to_client_config(),
            )
        return self._clients[cache_key]

    async def _resolve(
        self,
        tier: "ModelTier",
    ) -> tuple[LLMClient, "TierFallbackNotice | None"]:
        """解析层级，返回客户端和降级提示

        如果有降级提示且设置了回调，等待回调决定是否继续。

        Raises:
            ValueError: 所有层级均未配置，或用户拒绝降级
        """
        from core.llm_config import ModelTier  # 延迟导入避免循环依赖

        tier_config, notice = self._llm_config.resolve_tier(tier)

        if notice and self._fallback_callback:
            accepted = await self._fallback_callback(notice)
            if not accepted:
                raise ValueError(
                    f"用户取消了降级调用。"
                    f"推荐层级: {notice.recommended_label}，"
                    f"实际层级: {notice.actual_label}"
                )

        client = self._get_client(tier_config)
        return client, notice

    async def complete(
        self,
        messages: list[Message | dict],
        tier: "ModelTier" = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> CompletionResponse:
        """按层级调用 LLM 完成请求

        Args:
            messages: 消息列表
            tier: 推荐模型层级。为 None 时使用 balanced。
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            **kwargs: 额外参数

        Returns:
            CompletionResponse: 包含内容、用量、模型信息和层级降级提示
        """
        from core.llm_config import ModelTier

        if tier is None:
            tier = ModelTier.BALANCED

        client, notice = await self._resolve(tier)
        response = await client.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        response.tier_notice = notice
        return response

    async def stream(
        self,
        messages: list[Message | dict],
        tier: "ModelTier" = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """按层级调用 LLM 流式请求

        Args:
            messages: 消息列表
            tier: 推荐模型层级。为 None 时使用 balanced。
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            **kwargs: 额外参数

        Yields:
            str: 流式输出的文本片段
        """
        from core.llm_config import ModelTier

        if tier is None:
            tier = ModelTier.BALANCED

        client, notice = await self._resolve(tier)

        # 如果有降级提示，先输出提示信息
        if notice:
            yield f"[提示] {notice.message}\n\n"

        async for chunk in client.stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            yield chunk

    async def complete_json(
        self,
        messages: list[Message | dict],
        tier: "ModelTier" = None,
        temperature: float = 0.3,
        max_retries: int = 2,
        thinking_effort: str | None = None,
        **kwargs,
    ) -> dict | list:
        """按层级调用 LLM 并解析 JSON 响应

        Args:
            messages: 消息列表
            tier: 推荐模型层级。为 None 时使用 balanced。
            temperature: 采样温度
            max_retries: JSON 解析失败时的最大重试次数
            thinking_effort: 思考强度（low/medium/high）
            **kwargs: 额外参数

        Returns:
            解析后的 JSON 对象
        """
        from core.llm_config import ModelTier

        if tier is None:
            tier = ModelTier.BALANCED

        client, notice = await self._resolve(tier)
        return await client.complete_json(
            messages,
            temperature=temperature,
            max_retries=max_retries,
            thinking_effort=thinking_effort,
            **kwargs,
        )

    @property
    def llm_config(self) -> "LLMConfig":
        """获取当前的 LLM 配置"""
        return self._llm_config
