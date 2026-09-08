"""考公资料处理循环测试

测试 Tool-Agent 完成具体学习任务的能力：
1. 索引考公资料
2. 分类考公资料
3. 分题出题库
4. 讲义整理

注意：此测试通过 LLM 调用工具完成，而非直接执行
"""

import asyncio
import json
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
    EventLog,
    EventType,
)


# 考公资料路径
KGZL_PATH = "D:/考公资料"
OUTPUT_DIR = Path("d:/Learning plugins/output/test_cycles")


async def test_cycle_1_index_and_classify():
    """循环测试1：索引并分类考公资料"""
    print("\n" + "=" * 70)
    print("循环测试 1: 索引并分类考公资料")
    print("=" * 70)

    # 创建事件日志
    event_log_path = OUTPUT_DIR / "cycle1_events.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 创建 Agent
    config = LLMConfig.from_file("llm_config.json")
    llm = TieredLLMClient(config)
    context = ToolContext(llm_client=llm)

    registry = ToolRegistry(context)
    loaded = registry.discover_tools("tools/")
    print(f"已加载工具: {', '.join(loaded)}")

    # 创建 Agent（使用 fast 模型，低思考强度）
    agent = ToolAgent(
        llm=llm,
        registry=registry,
        interaction_handler=InteractionHandler(mode=InteractionMode.AUTO),
        max_iterations=10,
        default_tier=ModelTier.FAST,
        default_thinking=ThinkingEffort.LOW,
        event_log_path=event_log_path,
    )

    # 执行任务
    task = f"""请完成以下任务：
1. 使用 resource-indexer 索引文件夹 {KGZL_PATH}
2. 基于索引结果，使用 resource-classifier 对资料进行分类
3. 总结分类结果

请按顺序执行，并报告每一步的结果。"""

    result = await agent.run(task)

    # 输出结果
    print("\n执行结果:")
    print(f"  成功: {result.success}")
    print(f"  迭代次数: {result.total_iterations}")
    print(f"  步骤数: {len(result.steps)}")

    if result.success:
        print(f"\n最终答案:\n{result.final_answer[:500]}...")
    else:
        print(f"\n错误: {result.error}")

    # 导出事件日志
    agent.event_log.export_trace(OUTPUT_DIR / "cycle1_trace.json")
    print(f"\n事件日志已导出: {event_log_path}")

    return result


async def test_cycle_2_question_split():
    """循环测试2：分题出题库"""
    print("\n" + "=" * 70)
    print("循环测试 2: 分题出题库")
    print("=" * 70)

    # 创建事件日志
    event_log_path = OUTPUT_DIR / "cycle2_events.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 创建 Agent
    config = LLMConfig.from_file("llm_config.json")
    llm = TieredLLMClient(config)
    context = ToolContext(llm_client=llm)

    registry = ToolRegistry(context)
    registry.discover_tools("tools/")

    # 创建 Agent（使用 balanced 模型，中思考强度）
    agent = ToolAgent(
        llm=llm,
        registry=registry,
        interaction_handler=InteractionHandler(mode=InteractionMode.AUTO),
        max_iterations=15,
        default_tier=ModelTier.BALANCED,
        default_thinking=ThinkingEffort.MEDIUM,
        event_log_path=event_log_path,
    )

    # 执行任务
    task = f"""请完成以下任务：
1. 使用 resource-indexer 索引文件夹 {KGZL_PATH}，获取文件列表
2. 从索引结果中识别出题集类 PDF 文件
3. 选择一个题集 PDF，使用 question-splitter 提取题目
4. 报告提取的题目数量和类型

注意：如果找不到题集 PDF，请说明原因并给出建议。"""

    result = await agent.run(task)

    # 输出结果
    print("\n执行结果:")
    print(f"  成功: {result.success}")
    print(f"  迭代次数: {result.total_iterations}")
    print(f"  步骤数: {len(result.steps)}")

    if result.success:
        print(f"\n最终答案:\n{result.final_answer[:500]}...")
    else:
        print(f"\n错误: {result.error}")

    # 导出事件日志
    agent.event_log.export_trace(OUTPUT_DIR / "cycle2_trace.json")
    print(f"\n事件日志已导出: {event_log_path}")

    return result


async def test_cycle_3_model_comparison():
    """循环测试3：不同模型对比"""
    print("\n" + "=" * 70)
    print("循环测试 3: 不同模型对比")
    print("=" * 70)

    # 创建事件日志
    event_log_path = OUTPUT_DIR / "cycle3_events.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 创建 Agent
    config = LLMConfig.from_file("llm_config.json")
    llm = TieredLLMClient(config)
    context = ToolContext(llm_client=llm)

    registry = ToolRegistry(context)
    registry.discover_tools("tools/")

    # 测试任务
    task = f"使用 resource-indexer 索引文件夹 {KGZL_PATH}，并生成简要摘要"

    # 对比不同模型
    models_to_test = [
        (ModelTier.FAST, ThinkingEffort.LOW, "fast+low"),
        (ModelTier.BALANCED, ThinkingEffort.MEDIUM, "balanced+medium"),
    ]

    results = {}

    for tier, thinking, label in models_to_test:
        print(f"\n测试模型: {label}")

        agent = ToolAgent(
            llm=llm,
            registry=registry,
            interaction_handler=InteractionHandler(mode=InteractionMode.AUTO),
            max_iterations=5,
            default_tier=tier,
            default_thinking=thinking,
            event_log_path=OUTPUT_DIR / f"cycle3_{label.replace('+', '_')}_events.jsonl",
        )

        result = await agent.run(task)
        results[label] = {
            "success": result.success,
            "iterations": result.total_iterations,
            "steps": len(result.steps),
        }

        print(f"  结果: 成功={result.success}, 迭代={result.total_iterations}")

    # 输出对比
    print("\n模型对比结果:")
    for label, stats in results.items():
        print(f"  {label}: {stats}")

    return results


async def main():
    """运行所有循环测试"""
    print("=" * 70)
    print("考公资料处理循环测试")
    print("=" * 70)
    print(f"资料路径: {KGZL_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")

    # 检查资料路径
    if not Path(KGZL_PATH).exists():
        print(f"\n错误: 资料路径不存在: {KGZL_PATH}")
        print("请修改 KGZL_PATH 为实际路径")
        return 1

    results = []

    # 运行测试
    try:
        result1 = await test_cycle_1_index_and_classify()
        results.append(("索引并分类", result1.success))
    except Exception as e:
        print(f"测试1失败: {e}")
        results.append(("索引并分类", False))

    try:
        result2 = await test_cycle_2_question_split()
        results.append(("分题出题库", result2.success))
    except Exception as e:
        print(f"测试2失败: {e}")
        results.append(("分题出题库", False))

    try:
        result3 = await test_cycle_3_model_comparison()
        results.append(("模型对比", True))
    except Exception as e:
        print(f"测试3失败: {e}")
        results.append(("模型对比", False))

    # 汇总
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    all_passed = all(r[1] for r in results)
    print(f"\n{'全部通过!' if all_passed else '存在失败项!'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
