"""Stub LLM 客户端模块 - 提供测试替身，摆脱真实 LLM 依赖

包含三种替身:
- StubLLMClient: 按预设返回内容（顺序回放 / 指纹映射 / 默认响应）
- RecordedLLMClient: 从磁带（内存 list 或 JSONL 文件）回放，未命中时抛异常
- RecordingLLMClient: 包装真实客户端，将请求/响应录制为磁带

用法:
    from core.stub_llm import StubLLMClient, RecordedLLMClient, RecordingLLMClient

    # 直接使用
    client = StubLLMClient({"responses": ["hello", "world"]})
    resp = await client.complete([{"role": "user", "content": "hi"}])

    # 通过工厂
    client = LLMClientFactory.create("stub", {"responses": ['{"key": "value"}']})
    result = await client.complete_json([{"role": "user", "content": "give json"}])
"""

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import AsyncIterator, Any

from core.llm_client import (
    CompletionResponse,
    LLMClient,
    LLMClientFactory,
    Message,
)


def _estimate_tokens(text: str) -> int:
    """简单字符数估算 token 数（~4 字符 ≈ 1 token）"""
    return max(1, len(text) // 4)


def _synthesize_usage(messages: list[dict], content: str) -> dict:
    """根据输入消息和输出内容合成 usage 字典"""
    prompt_text = "".join(m.get("content", "") for m in messages)
    prompt_tokens = _estimate_tokens(prompt_text)
    completion_tokens = _estimate_tokens(content)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _compute_fingerprint(messages: list[dict]) -> str:
    """计算 messages 的指纹（基于最后一条 user 消息内容的 SHA-256 前 16 位）"""
    last_user_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_content = msg.get("content", "")
            break
    return hashlib.sha256(last_user_content.encode("utf-8")).hexdigest()[:16]


def _normalize_response(raw: Any, messages: list[dict], model: str) -> CompletionResponse:
    """将多种格式的预设响应统一为 CompletionResponse"""
    if isinstance(raw, CompletionResponse):
        if not raw.usage:
            return dataclasses.replace(raw, usage=_synthesize_usage(messages, raw.content))
        return raw
    if isinstance(raw, dict):
        content = raw.get("content", "")
        usage = raw.get("usage") or _synthesize_usage(messages, content)
        return CompletionResponse(
            content=content,
            usage=usage,
            model=raw.get("model", model),
            raw_response=raw.get("raw_response"),
        )
    # 纯字符串
    content = str(raw)
    return CompletionResponse(
        content=content,
        usage=_synthesize_usage(messages, content),
        model=model,
    )


class StubLLMClient(LLMClient):
    """按预设返回内容的 Stub LLM 客户端

    支持两种喂法:
    (a) 顺序回放: config["responses"] = ["resp1", "resp2", ...]
    (b) 指纹映射: config["response_map"] = {"<fingerprint>": "resp", ...}
        指纹 = 最后一条 user 消息内容的 SHA-256 前 16 位

    未命中时返回 config["default_response"]，若也未设置则返回 "[stub] no response configured"。

    usage 自动合成（字符数/4 估算 token），保证下游 token 统计不崩。

    Config keys:
        responses: list[str | dict | CompletionResponse]  顺序回放列表
        response_map: dict[str, str | dict | CompletionResponse]  指纹→响应映射
        default_response: str | dict | CompletionResponse  默认响应
        model: str  模型名（默认 "stub-model"）
    """

    def __init__(self, config: dict):
        if "model" not in config:
            config = {**config, "model": "stub-model"}
        super().__init__(config)
        self._responses: list = list(config.get("responses", []))
        self._response_map: dict = dict(config.get("response_map", {}))
        self._default_response = config.get("default_response", "[stub] no response configured")
        self._call_index: int = 0
        self.call_history: list[dict] = []  # 记录所有调用，便于断言

    def _resolve_response(self, messages: list[dict]) -> Any:
        """按优先级解析预设响应: 顺序列表 > 指纹映射 > 默认"""
        # (a) 顺序回放
        if self._responses:
            idx = self._call_index
            self._call_index += 1
            if idx < len(self._responses):
                return self._responses[idx]
            # 列表用尽后 fall through 到指纹/默认

        # (b) 指纹映射
        fingerprint = _compute_fingerprint(messages)
        if fingerprint in self._response_map:
            return self._response_map[fingerprint]

        # (c) 默认响应
        return self._default_response

    async def complete(
        self,
        messages: list["Message | dict"],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> CompletionResponse:
        norm = self.normalize_messages(messages)
        self.call_history.append({"messages": norm, "temperature": temperature, "kwargs": kwargs})
        raw = self._resolve_response(norm)
        return _normalize_response(raw, norm, self.model)

    async def stream(
        self,
        messages: list["Message | dict"],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        response = await self.complete(messages, temperature, max_tokens, **kwargs)
        # 按 20 字符分块模拟流式输出
        content = response.content
        chunk_size = 20
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]


class TapeMissError(Exception):
    """磁带未命中时抛出，防止测试静默走真实网络"""
    pass


class RecordedLLMClient(LLMClient):
    """从磁带回放的 LLM 客户端（未命中时抛异常）

    磁带来源:
    - config["tape"]: list[dict]  内存中的磁带条目
    - config["tape_path"]: str | Path  JSONL 文件路径（每行一个 JSON 对象）

    磁带条目格式:
    {
        "fingerprint": "<sha256_16>",   # 可选，若无则从 request 计算
        "request": [{"role": ..., "content": ...}, ...],
        "response": {"content": "...", "usage": {...}, "model": "..."}
    }

    匹配逻辑: 按 messages 指纹查找，支持顺序消费同一指纹的多条记录。
    未命中时抛出 TapeMissError。

    Config keys:
        tape: list[dict]  内存磁带
        tape_path: str  JSONL 文件路径
        model: str  模型名（默认 "recorded-model"）
    """

    def __init__(self, config: dict):
        if "model" not in config:
            config = {**config, "model": "recorded-model"}
        super().__init__(config)
        self._tape: list[dict] = []
        self._load_tape(config)
        # 构建指纹索引: fingerprint -> list of response entries (按序消费)
        self._index: dict[str, list[dict]] = {}
        self._cursors: dict[str, int] = {}
        self._build_index()
        self.call_history: list[dict] = []

    def _load_tape(self, config: dict):
        """从内存或 JSONL 文件加载磁带"""
        if "tape" in config:
            self._tape = list(config["tape"])
        elif "tape_path" in config:
            path = Path(config["tape_path"])
            if not path.exists():
                raise FileNotFoundError(f"磁带文件不存在: {path}")
            self._tape = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._tape.append(json.loads(line))

    def _build_index(self):
        """构建指纹→响应列表的索引"""
        for entry in self._tape:
            fp = entry.get("fingerprint")
            if not fp and "request" in entry:
                fp = _compute_fingerprint(entry["request"])
            if not fp:
                continue
            self._index.setdefault(fp, []).append(entry)
            self._cursors.setdefault(fp, 0)

    def _lookup(self, messages: list[dict]) -> dict:
        """按指纹查找磁带条目"""
        fp = _compute_fingerprint(messages)
        if fp not in self._index:
            raise TapeMissError(
                f"磁带未命中: fingerprint={fp}, "
                f"最后一条 user 消息='{self._last_user_content(messages)[:100]}'. "
                f"已有指纹: {list(self._index.keys())}"
            )
        entries = self._index[fp]
        cursor = self._cursors[fp]
        if cursor >= len(entries):
            # 回绕到第一条（支持重复调用）
            cursor = 0
        self._cursors[fp] = cursor + 1
        return entries[cursor]

    @staticmethod
    def _last_user_content(messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    async def complete(
        self,
        messages: list["Message | dict"],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> CompletionResponse:
        norm = self.normalize_messages(messages)
        self.call_history.append({"messages": norm, "temperature": temperature, "kwargs": kwargs})
        entry = self._lookup(norm)
        resp_data = entry.get("response", {})
        content = resp_data.get("content", "") if isinstance(resp_data, dict) else str(resp_data)
        usage = resp_data.get("usage") if isinstance(resp_data, dict) else None
        if not usage:
            usage = _synthesize_usage(norm, content)
        model = resp_data.get("model", self.model) if isinstance(resp_data, dict) else self.model
        return CompletionResponse(content=content, usage=usage, model=model)

    async def stream(
        self,
        messages: list["Message | dict"],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        response = await self.complete(messages, temperature, max_tokens, **kwargs)
        content = response.content
        chunk_size = 20
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]


class RecordingLLMClient(LLMClient):
    """录制型 LLM 客户端 - 包装真实客户端，将请求/响应录制为磁带

    用于第三层测试生成磁带，第二层测试复用磁带（通过 RecordedLLMClient）。

    构造:
        real_client = LLMClientFactory.create("openai", {...})
        recorder = RecordingLLMClient(real_client, tape_path="output/tape.jsonl")

    或直接实例化（不通过工厂）:
        recorder = RecordingLLMClient(wrapped_client=real_client, config={"model": "..."})

    Config keys (当通过工厂创建时):
        wrapped_client: LLMClient 实例（必须通过直接实例化传入）
        tape_path: str  输出 JSONL 文件路径（可选，默认 "output/recording_tape.jsonl"）
        model: str  模型名
    """

    def __init__(self, config: dict, wrapped_client: "LLMClient | None" = None):
        # 支持两种构造方式:
        # 1. RecordingLLMClient(config, wrapped_client=real)
        # 2. RecordingLLMClient(config) 其中 config["wrapped_client"] = real
        if wrapped_client is None:
            wrapped_client = config.get("wrapped_client")
        if wrapped_client is None:
            raise ValueError(
                "RecordingLLMClient 需要一个被包装的真实客户端。"
                "请通过 wrapped_client 参数或 config['wrapped_client'] 传入。"
            )
        super().__init__(config)
        self._wrapped: LLMClient = wrapped_client
        self._tape_path: Path = Path(config.get("tape_path", "output/recording_tape.jsonl"))
        self._tape: list[dict] = []

    @property
    def tape(self) -> list[dict]:
        """获取已录制的磁带条目"""
        return list(self._tape)

    def save_tape(self, path: "str | Path | None" = None):
        """将磁带保存为 JSONL 文件"""
        target = Path(path) if path else self._tape_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            for entry in self._tape:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _record(self, messages: list[dict], response: CompletionResponse):
        """录制一条请求/响应"""
        entry = {
            "fingerprint": _compute_fingerprint(messages),
            "request": messages,
            "response": {
                "content": response.content,
                "usage": response.usage,
                "model": response.model,
            },
        }
        self._tape.append(entry)

    async def complete(
        self,
        messages: list["Message | dict"],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> CompletionResponse:
        norm = self.normalize_messages(messages)
        response = await self._wrapped.complete(norm, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self._record(norm, response)
        return response

    async def stream(
        self,
        messages: list["Message | dict"],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        norm = self.normalize_messages(messages)
        chunks = []
        async for chunk in self._wrapped.stream(norm, temperature=temperature, max_tokens=max_tokens, **kwargs):
            chunks.append(chunk)
            yield chunk
        # 录制完整流式响应
        full_content = "".join(chunks)
        fake_response = CompletionResponse(
            content=full_content,
            usage=_synthesize_usage(norm, full_content),
            model=self._wrapped.model,
        )
        self._record(norm, fake_response)


# ---- 注册到工厂（公开 API，不使用 "mock" 名称） ----
LLMClientFactory.register("stub", StubLLMClient)
LLMClientFactory.register("recorded", RecordedLLMClient)
