"""Tool-Agent 架构测试

测试核心组件：
1. LLM 配置（thinking_effort）
2. 工具注册与发现
3. 交互模式
4. Tool-Agent 循环
"""

import asyncio
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 导入 providers 触发自动注册
from core.providers.openai_client import OpenAIClient  # noqa: F401

from core import (
    LLMConfig,
    ModelTier,
    ThinkingEffort,
    TieredLLMClient,
    ToolContext,
    ToolRegistry,
    ToolAgent,
    InteractionMode,
    InteractionHandler,
    ToolManifest,
)


def test_llm_config():
    """测试 LLM 配置"""
    print("\n=== 测试 LLM 配置 ===")

    config = LLMConfig.from_file("llm_config.json")

    # 检查配置
    fast_config = config.get_tier_config(ModelTier.FAST)
    balanced_config = config.get_tier_config(ModelTier.BALANCED)
    flagship_config = config.get_tier_config(ModelTier.FLAGSHIP)

    print(f"fast: {fast_config.provider}/{fast_config.model}")
    print(f"balanced: {balanced_config.provider}/{balanced_config.model}")
    print(f"flagship: {flagship_config.provider}/{flagship_config.model}")

    # 验证 thinking_effort 不在配置层
    assert not hasattr(fast_config, 'thinking_effort') or fast_config.thinking_effort is None
    assert not hasattr(balanced_config, 'thinking_effort') or balanced_config.thinking_effort is None
    assert not hasattr(flagship_config, 'thinking_effort') or flagship_config.thinking_effort is None

    print("[PASS] LLM 配置测试通过")
    return True


def test_tool_registry():
    """测试工具注册与发现"""
    print("\n=== 测试工具注册与发现 ===")

    # 创建上下文
    config = LLMConfig.from_file("llm_config.json")
    llm = TieredLLMClient(config)
    context = ToolContext(llm_client=llm)

    # 发现工具
    registry = ToolRegistry(context)
    loaded = registry.discover_tools("tools/")

    print(f"发现 {len(loaded)} 个工具: {', '.join(loaded)}")

    # 检查工具
    assert "resource-indexer" in registry
    assert "resource-classifier" in registry
    assert "question-splitter" in registry
    assert "tool-develop" in registry

    # 获取工具清单
    manifest = registry.get_manifest("resource-indexer")
    assert manifest is not None
    assert manifest.recommended_tier == "fast"
    assert manifest.recommended_thinking == "low"

    print(f"resource-indexer manifest: tier={manifest.recommended_tier}, thinking={manifest.recommended_thinking}")

    # 生成 LLM 描述
    description = registry.get_tools_description_for_llm()
    assert "resource-indexer" in description
    assert "resource-classifier" in description

    print("[PASS] 工具注册与发现测试通过")
    return True


def test_interaction_handler():
    """测试交互模式"""
    print("\n=== 测试交互模式 ===")

    # AUTO 模式
    auto_handler = InteractionHandler(mode=InteractionMode.AUTO)
    assert auto_handler.should_auto_proceed("normal") == True
    assert auto_handler.should_auto_proceed("high") == False

    # YOLO 模式
    yolo_handler = InteractionHandler(mode=InteractionMode.YOLO)
    assert yolo_handler.should_auto_proceed("normal") == True
    assert yolo_handler.should_auto_proceed("high") == True

    # ASK 模式
    ask_handler = InteractionHandler(mode=InteractionMode.ASK_WHEN_NEEDED)
    assert ask_handler.should_auto_proceed("normal") == False
    assert ask_handler.should_auto_proceed("high") == False

    print("[PASS] 交互模式测试通过")
    return True


async def test_tool_agent_creation():
    """测试 Tool-Agent 创建"""
    print("\n=== 测试 Tool-Agent 创建 ===")

    from core import create_agent

    # 测试默认创建（不指定 thinking）
    agent = await create_agent(
        config_path="llm_config.json",
        tools_dir="tools/",
        interaction_mode=InteractionMode.ASK_WHEN_NEEDED,
        max_iterations=5,
    )

    assert agent is not None
    assert len(agent.registry) > 0
    assert agent.default_thinking is None  # 默认由工具决定

    print(f"Agent 创建成功，加载 {len(agent.registry)} 个工具")
    print(f"默认思考强度: {agent.default_thinking} (None 表示由工具决定)")

    # 测试指定 thinking
    agent_with_thinking = await create_agent(
        config_path="llm_config.json",
        tools_dir="tools/",
        interaction_mode=InteractionMode.ASK_WHEN_NEEDED,
        max_iterations=5,
        default_thinking=ThinkingEffort.HIGH,
    )

    assert agent_with_thinking.default_thinking == ThinkingEffort.HIGH
    print(f"指定思考强度: {agent_with_thinking.default_thinking}")

    print("[PASS] Tool-Agent 创建测试通过")
    return True


def test_tool_manifest():
    """测试工具清单"""
    print("\n=== 测试工具清单 ===")

    config = LLMConfig.from_file("llm_config.json")
    llm = TieredLLMClient(config)
    context = ToolContext(llm_client=llm)

    registry = ToolRegistry(context)
    registry.discover_tools("tools/")

    # 检查各工具的 manifest
    for name in ["resource-indexer", "resource-classifier", "question-splitter", "tool-develop"]:
        manifest = registry.get_manifest(name)
        assert manifest is not None, f"{name} manifest 不存在"
        assert manifest.description_for_llm, f"{name} 缺少描述"
        print(f"  {name}: tier={manifest.recommended_tier}, thinking={manifest.recommended_thinking}")

    print("[PASS] 工具清单测试通过")
    return True


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("Tool-Agent 架构测试")
    print("=" * 60)

    results = []

    # 同步测试
    results.append(("LLM 配置", test_llm_config()))
    results.append(("工具注册与发现", test_tool_registry()))
    results.append(("交互模式", test_interaction_handler()))
    results.append(("工具清单", test_tool_manifest()))

    # 异步测试
    results.append(("Tool-Agent 创建", await test_tool_agent_creation()))

    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  [{status}] {name}")

    all_passed = all(r[1] for r in results)
    print(f"\n{'全部通过!' if all_passed else '存在失败项!'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
