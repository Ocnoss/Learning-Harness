"""测试多层级模型配置系统

验证 LLMConfig 的层级解析、降级提示生成和 TieredLLMClient 的路由逻辑。
不需要真实 API 调用，使用 mock 客户端验证。

运行方式:
    D:\miniconda\python.exe test_llm_config.py
"""

import asyncio
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.llm_config import (
    LLMConfig,
    ModelTier,
    TierModelConfig,
    TierFallbackNotice,
    TIER_LABELS,
)
from core.llm_client import (
    CompletionResponse,
    LLMClient,
    LLMClientFactory,
    TieredLLMClient,
    Message,
)


# ---- Mock LLM 客户端（不调用真实 API） ----

class MockLLMClient(LLMClient):
    """模拟 LLM 客户端，用于测试"""

    def __init__(self, config: dict):
        super().__init__(config)

    async def complete(self, messages, temperature=0.7, max_tokens=None, **kwargs):
        return CompletionResponse(
            content=f"[Mock:{self.model}] 回复内容",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            model=self.model,
        )

    async def stream(self, messages, temperature=0.7, max_tokens=None, **kwargs):
        yield f"[Mock:{self.model}] "
        yield "流式输出"

    async def complete_json(self, messages, temperature=0.3, max_retries=2, **kwargs):
        return {"mock": True, "model": self.model}


# 注册 mock 提供商
LLMClientFactory.register("mock", MockLLMClient)


# ---- 测试用例 ----

def test_tier_labels():
    """测试层级标签"""
    print("=" * 50)
    print("测试 1: 层级标签")
    print("=" * 50)
    assert TIER_LABELS[ModelTier.FAST] == "快速模型"
    assert TIER_LABELS[ModelTier.BALANCED] == "均衡模型"
    assert TIER_LABELS[ModelTier.FLAGSHIP] == "旗舰模型"
    print("[PASS] 层级标签正确")
    return True


def test_tier_configured():
    """测试层级配置检查"""
    print()
    print("=" * 50)
    print("测试 2: 层级配置检查")
    print("=" * 50)

    config = LLMConfig(
        fast=TierModelConfig(provider="mock", model="fast-model"),
        balanced=TierModelConfig(provider="mock", model="balanced-model"),
        flagship=TierModelConfig(),  # 未配置
    )

    assert config.is_tier_configured(ModelTier.FAST)
    assert config.is_tier_configured(ModelTier.BALANCED)
    assert not config.is_tier_configured(ModelTier.FLAGSHIP)

    configured = config.get_configured_tiers()
    assert ModelTier.FAST in configured
    assert ModelTier.BALANCED in configured
    assert ModelTier.FLAGSHIP not in configured

    print(f"已配置层级: {[TIER_LABELS[t] for t in configured]}")
    print("[PASS] 层级配置检查正确")
    return True


def test_resolve_exact_match():
    """测试精确匹配的层级解析"""
    print()
    print("=" * 50)
    print("测试 3: 精确匹配层级解析")
    print("=" * 50)

    config = LLMConfig(
        fast=TierModelConfig(provider="mock", model="fast-model"),
        balanced=TierModelConfig(provider="mock", model="balanced-model"),
        flagship=TierModelConfig(provider="mock", model="flagship-model"),
    )

    # 精确匹配，无降级提示
    tier_config, notice = config.resolve_tier(ModelTier.BALANCED)
    assert tier_config.model == "balanced-model"
    assert notice is None

    print("精确匹配 balanced -> balanced-model (无提示)")
    print("[PASS] 精确匹配正确")
    return True


def test_resolve_fallback_to_weaker():
    """测试降级到更弱模型"""
    print()
    print("=" * 50)
    print("测试 4: 降级到更弱模型（flagship 未配置）")
    print("=" * 50)

    config = LLMConfig(
        fast=TierModelConfig(provider="mock", model="fast-model"),
        balanced=TierModelConfig(provider="mock", model="balanced-model"),
        flagship=TierModelConfig(),  # 未配置
    )

    tier_config, notice = config.resolve_tier(ModelTier.FLAGSHIP)
    assert tier_config.model == "balanced-model"
    assert notice is not None
    assert notice.severity == "warning"
    assert notice.recommended_tier == ModelTier.FLAGSHIP
    assert notice.actual_tier == ModelTier.BALANCED
    assert "旗舰模型" in notice.message
    assert "均衡模型" in notice.message
    assert "效果不佳" in notice.message

    print(f"降级提示: {notice.message}")
    print(f"严重程度: {notice.severity}")
    print("[PASS] 降级到更弱模型正确")
    return True


def test_resolve_fallback_to_stronger():
    """测试升级到更强模型"""
    print()
    print("=" * 50)
    print("测试 5: 升级到更强模型（fast 未配置）")
    print("=" * 50)

    config = LLMConfig(
        fast=TierModelConfig(),  # 未配置
        balanced=TierModelConfig(provider="mock", model="balanced-model"),
        flagship=TierModelConfig(provider="mock", model="flagship-model"),
    )

    tier_config, notice = config.resolve_tier(ModelTier.FAST)
    assert tier_config.model == "balanced-model"
    assert notice is not None
    assert notice.severity == "notice"
    assert notice.recommended_tier == ModelTier.FAST
    assert notice.actual_tier == ModelTier.BALANCED
    assert "快速模型" in notice.message
    assert "均衡模型" in notice.message
    assert "消耗" in notice.message

    print(f"升级提示: {notice.message}")
    print(f"严重程度: {notice.severity}")
    print("[PASS] 升级到更强模型正确")
    return True


def test_resolve_only_one_configured():
    """测试只配置一个层级"""
    print()
    print("=" * 50)
    print("测试 6: 只配置一个层级")
    print("=" * 50)

    config = LLMConfig(
        fast=TierModelConfig(provider="mock", model="fast-model"),
        balanced=TierModelConfig(),
        flagship=TierModelConfig(),
    )

    # flagship -> 降级到 fast
    tier_config, notice = config.resolve_tier(ModelTier.FLAGSHIP)
    assert tier_config.model == "fast-model"
    assert notice is not None
    assert notice.severity == "warning"

    print(f"flagship -> fast: {notice.message}")

    # balanced -> 降级到 fast
    tier_config, notice = config.resolve_tier(ModelTier.BALANCED)
    assert tier_config.model == "fast-model"
    assert notice is not None
    assert notice.severity == "warning"

    print(f"balanced -> fast: {notice.message}")

    # fast -> 精确匹配
    tier_config, notice = config.resolve_tier(ModelTier.FAST)
    assert tier_config.model == "fast-model"
    assert notice is None

    print("fast -> fast (精确匹配，无提示)")
    print("[PASS] 单层级配置正确")
    return True


def test_resolve_none_configured():
    """测试所有层级都未配置"""
    print()
    print("=" * 50)
    print("测试 7: 所有层级都未配置")
    print("=" * 50)

    config = LLMConfig()

    try:
        config.resolve_tier(ModelTier.BALANCED)
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        print(f"正确抛出异常: {e}")
        print("[PASS] 全空配置正确抛出异常")
        return True


def test_config_serialization():
    """测试配置序列化和反序列化"""
    print()
    print("=" * 50)
    print("测试 8: 配置序列化/反序列化")
    print("=" * 50)

    original = LLMConfig(
        fast=TierModelConfig(
            provider="mock",
            model="fast-model",
        ),
        balanced=TierModelConfig(
            provider="mock",
            model="balanced-model",
        ),
        flagship=TierModelConfig(),  # 留空
    )

    # 序列化
    data = original.to_dict()
    print(f"序列化结果: {data}")

    # 反序列化
    restored = LLMConfig.from_dict(data)
    assert restored.fast.model == "fast-model"
    assert restored.balanced.model == "balanced-model"
    assert not restored.flagship.is_configured()

    # 验证 to_client_config 不包含 temperature/max_tokens
    client_cfg = restored.fast.to_client_config()
    assert "temperature" not in client_cfg
    assert "max_tokens" not in client_cfg
    assert client_cfg["model"] == "fast-model"
    assert client_cfg["api_key"] == ""
    print(f"to_client_config 结果: {client_cfg}")

    print("[PASS] 序列化/反序列化正确")
    return True


def test_config_file():
    """测试配置文件加载"""
    print()
    print("=" * 50)
    print("测试 9: 配置文件加载")
    print("=" * 50)

    config_path = Path(__file__).parent / "llm_config.json"
    config = LLMConfig.from_file(config_path)

    print(f"fast: {config.fast.provider}/{config.fast.model}")
    print(f"balanced: {config.balanced.provider}/{config.balanced.model}")
    print(f"flagship: {config.flagship.provider}/{config.flagship.model}")

    assert config.fast.is_configured()
    assert config.balanced.is_configured()
    assert config.flagship.is_configured()

    print("[PASS] 配置文件加载正确")
    return True


async def test_tiered_client():
    """测试 TieredLLMClient 路由"""
    print()
    print("=" * 50)
    print("测试 10: TieredLLMClient 层级路由")
    print("=" * 50)

    config = LLMConfig(
        fast=TierModelConfig(provider="mock", model="fast-model"),
        balanced=TierModelConfig(provider="mock", model="balanced-model"),
        flagship=TierModelConfig(),  # 未配置
    )

    client = TieredLLMClient(config)

    # 精确匹配
    resp = await client.complete(
        [{"role": "user", "content": "test"}],
        tier=ModelTier.FAST,
    )
    assert "fast-model" in resp.content
    assert resp.tier_notice is None
    print(f"fast -> {resp.content} (无降级)")

    # 默认层级（balanced）
    resp = await client.complete(
        [{"role": "user", "content": "test"}],
    )
    assert "balanced-model" in resp.content
    assert resp.tier_notice is None
    print(f"default -> {resp.content} (无降级)")

    # 降级（flagship 未配置，降级到 balanced）
    resp = await client.complete(
        [{"role": "user", "content": "test"}],
        tier=ModelTier.FLAGSHIP,
    )
    assert "balanced-model" in resp.content
    assert resp.tier_notice is not None
    assert resp.tier_notice.severity == "warning"
    print(f"flagship -> {resp.content}")
    print(f"  降级提示: {resp.tier_notice.message}")

    # complete_json
    result = await client.complete_json(
        [{"role": "user", "content": "test"}],
        tier=ModelTier.FAST,
    )
    assert result["model"] == "fast-model"
    print(f"complete_json fast -> {result}")

    print("[PASS] TieredLLMClient 层级路由正确")
    return True


async def test_tiered_client_callback():
    """测试 TieredLLMClient 降级回调"""
    print()
    print("=" * 50)
    print("测试 11: TieredLLMClient 降级回调")
    print("=" * 50)

    config = LLMConfig(
        fast=TierModelConfig(provider="mock", model="fast-model"),
        balanced=TierModelConfig(),
        flagship=TierModelConfig(),
    )

    # 回调拒绝降级
    async def reject_callback(notice: TierFallbackNotice) -> bool:
        print(f"  回调收到提示: {notice.message}")
        print(f"  回调决定: 拒绝")
        return False

    client = TieredLLMClient(config, fallback_callback=reject_callback)

    try:
        await client.complete(
            [{"role": "user", "content": "test"}],
            tier=ModelTier.FLAGSHIP,
        )
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        print(f"  正确抛出异常: {e}")

    # 回调接受降级
    async def accept_callback(notice: TierFallbackNotice) -> bool:
        print(f"  回调收到提示: {notice.message}")
        print(f"  回调决定: 接受")
        return True

    client2 = TieredLLMClient(config, fallback_callback=accept_callback)
    resp = await client2.complete(
        [{"role": "user", "content": "test"}],
        tier=ModelTier.FLAGSHIP,
    )
    assert "fast-model" in resp.content
    print(f"  降级执行成功: {resp.content}")

    print("[PASS] 降级回调正确")
    return True


async def test_tiered_client_stream():
    """测试 TieredLLMClient 流式输出"""
    print()
    print("=" * 50)
    print("测试 12: TieredLLMClient 流式输出")
    print("=" * 50)

    config = LLMConfig(
        fast=TierModelConfig(provider="mock", model="fast-model"),
        balanced=TierModelConfig(),
        flagship=TierModelConfig(),
    )

    client = TieredLLMClient(config)

    # 无降级的流式输出
    chunks = []
    async for chunk in client.stream(
        [{"role": "user", "content": "test"}],
        tier=ModelTier.FAST,
    ):
        chunks.append(chunk)
    full = "".join(chunks)
    assert "fast-model" in full
    print(f"fast 流式输出: {full}")

    # 有降级的流式输出（应包含提示前缀）
    chunks = []
    async for chunk in client.stream(
        [{"role": "user", "content": "test"}],
        tier=ModelTier.FLAGSHIP,
    ):
        chunks.append(chunk)
    full = "".join(chunks)
    assert "[提示]" in full
    assert "fast-model" in full
    print(f"flagship 流式输出: {full}")

    print("[PASS] 流式输出正确")
    return True


async def main():
    print("多层级模型配置系统测试")
    print()

    results = []

    # 同步测试
    results.append(("层级标签", test_tier_labels()))
    results.append(("层级配置检查", test_tier_configured()))
    results.append(("精确匹配", test_resolve_exact_match()))
    results.append(("降级到更弱模型", test_resolve_fallback_to_weaker()))
    results.append(("升级到更强模型", test_resolve_fallback_to_stronger()))
    results.append(("单层级配置", test_resolve_only_one_configured()))
    results.append(("全空配置", test_resolve_none_configured()))
    results.append(("序列化/反序列化", test_config_serialization()))
    results.append(("配置文件加载", test_config_file()))

    # 异步测试
    results.append(("TieredLLMClient 路由", await test_tiered_client()))
    results.append(("TieredLLMClient 回调", await test_tiered_client_callback()))
    results.append(("TieredLLMClient 流式", await test_tiered_client_stream()))

    # 汇总
    print()
    print("=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(r[1] for r in results)
    print()
    print("全部通过!" if all_passed else "存在失败项!")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
