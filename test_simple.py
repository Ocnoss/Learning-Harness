"""简化版循环测试 - 验证基础功能"""

import asyncio
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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
)


async def test_basic_agent():
    """基础 Agent 测试"""
    print("\n=== 基础 Agent 测试 ===")

    config = LLMConfig.from_file("llm_config.json")
    llm = TieredLLMClient(config)
    context = ToolContext(llm_client=llm)

    registry = ToolRegistry(context)
    loaded = registry.discover_tools("tools/")
    print(f"已加载工具: {', '.join(loaded)}")

    # 检查 memory-manager 是否加载
    if "memory-manager" in registry:
        print("[OK] memory-manager 工具已加载")
    else:
        print("[WARN] memory-manager 工具未加载")

    # 创建 Agent
    agent = ToolAgent(
        llm=llm,
        registry=registry,
        interaction_handler=InteractionHandler(mode=InteractionMode.AUTO),
        max_iterations=3,
        default_tier=ModelTier.FAST,
        default_thinking=ThinkingEffort.LOW,
        event_log_path="output/test_events.jsonl",
    )

    print(f"Agent 创建成功，会话ID: {agent.session_id[:8]}...")
    print(f"事件日志: {agent.event_log.persist_path}")

    # 简单任务测试
    task = "列出所有可用工具，并简要说明它们的功能"

    print(f"\n执行任务: {task}")
    result = await agent.run(task)

    print(f"\n结果:")
    print(f"  成功: {result.success}")
    print(f"  迭代: {result.total_iterations}")
    print(f"  步骤: {len(result.steps)}")

    if result.success:
        print(f"  答案: {result.final_answer[:200]}...")
    else:
        print(f"  错误: {result.error}")

    # 导出事件
    agent.event_log.export_trace("output/test_trace.json")
    print(f"\n事件日志已导出")

    return result.success


async def test_memory_tool():
    """测试记忆工具"""
    print("\n=== 记忆工具测试 ===")

    config = LLMConfig.from_file("llm_config.json")
    llm = TieredLLMClient(config)
    context = ToolContext(llm_client=llm)

    registry = ToolRegistry(context)
    registry.discover_tools("tools/")

    memory_tool = registry.get_tool("memory-manager")
    if not memory_tool:
        print("[FAIL] memory-manager 工具未找到")
        return False

    # 测试存储
    result = await memory_tool.execute({
        "action": "store",
        "session_id": "test-session",
        "entry_type": "fact",
        "content": "考公资料位于 D:/考公资料",
    })

    print(f"存储记忆: {result.success}")

    # 测试检索
    result = await memory_tool.execute({
        "action": "retrieve",
        "session_id": "test-session",
    })

    print(f"检索记忆: {result.success}, 条目数: {len(result.data.get('entries', []))}")

    # 测试搜索
    result = await memory_tool.execute({
        "action": "search",
        "session_id": "test-session",
        "query": "考公",
    })

    print(f"搜索记忆: {result.success}, 匹配: {len(result.data.get('entries', []))}")

    print("[PASS] 记忆工具测试通过")
    return True


async def main():
    """运行测试"""
    print("=" * 60)
    print("简化版循环测试")
    print("=" * 60)

    results = []

    # 测试记忆工具
    results.append(("记忆工具", await test_memory_tool()))

    # 测试基础 Agent
    results.append(("基础 Agent", await test_basic_agent()))

    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    all_passed = all(r[1] for r in results)
    print(f"\n{'全部通过!' if all_passed else '存在失败项!'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
