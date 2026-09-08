"""真实 API 测试 - 使用 llm_config.json 中的配置调用 LLM

运行方式:
    D:\miniconda\python.exe test_llm_live.py
"""

import asyncio
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 导入 providers 触发自动注册
from core.providers.openai_client import OpenAIClient  # noqa: F401
from core.llm_client import TieredLLMClient, LLMClientFactory
from core.llm_config import LLMConfig, ModelTier, TIER_LABELS


async def test_tier(client: TieredLLMClient, tier: ModelTier, label: str):
    """测试单个层级的 complete 调用"""
    print(f"\n{'=' * 50}")
    print(f"测试 {label} ({tier.value})")
    print(f"{'=' * 50}")

    try:
        response = await client.complete(
            messages=[
                {"role": "user", "content": "用一句话回答：1+1等于几？"},
            ],
            tier=tier,
            temperature=0.1,
            max_tokens=50,
        )

        print(f"模型: {response.model}")
        print(f"回复: {response.content}")
        print(f"用量: {response.usage}")

        if response.tier_notice:
            print(f"降级提示: {response.tier_notice.message}")
            print(f"严重程度: {response.tier_notice.severity}")
        else:
            print("层级: 精确匹配（无降级）")

        return True

    except Exception as e:
        print(f"[错误] {type(e).__name__}: {e}")
        return False


async def test_json(client: TieredLLMClient, tier: ModelTier, label: str):
    """测试 complete_json 调用"""
    print(f"\n{'=' * 50}")
    print(f"测试 {label} complete_json")
    print(f"{'=' * 50}")

    try:
        result = await client.complete_json(
            messages=[
                {
                    "role": "system",
                    "content": "你是一个数据提取助手。仅返回 JSON 对象，不要包含其他内容。",
                },
                {
                    "role": "user",
                    "content": '提取信息：{"name": "测试", "value": 42}',
                },
            ],
            tier=tier,
            temperature=0.1,
        )

        print(f"解析结果: {result}")
        print(f"类型: {type(result).__name__}")
        return True

    except Exception as e:
        print(f"[错误] {type(e).__name__}: {e}")
        return False


async def main():
    print("LLM Config 真实 API 测试")
    print()

    # 加载配置
    config_path = Path(__file__).parent / "llm_config.json"
    config = LLMConfig.from_file(config_path)

    # 打印配置摘要
    print("当前配置:")
    for tier in ModelTier:
        tc = config.get_tier_config(tier)
        status = f"{tc.provider}/{tc.model}" if tc.is_configured() else "(未配置)"
        print(f"  {TIER_LABELS[tier]}: {status}")

    print(f"\n已注册提供商: {LLMClientFactory.list_providers()}")

    # 创建 TieredLLMClient
    client = TieredLLMClient(config)

    results = []

    # 逐层级测试
    for tier in ModelTier:
        label = TIER_LABELS[tier]
        ok = await test_tier(client, tier, label)
        results.append((f"{label} complete", ok))

    # 测试 complete_json（使用 fast 层级）
    ok = await test_json(client, ModelTier.FAST, "fast")
    results.append(("fast complete_json", ok))

    # 汇总
    print(f"\n{'=' * 50}")
    print("测试结果汇总")
    print(f"{'=' * 50}")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(r[1] for r in results)
    print(f"\n{'全部通过!' if all_passed else '存在失败项!'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
