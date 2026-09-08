"""考公资料实际任务循环测试

通过 LLM 调用工具完成具体学习任务：
1. 索引考公资料（获取具体路径）
2. 分类考公资料
3. 分题出题库
4. 讲义整理

注意：此测试通过 LLM 自主调用工具完成，而非直接执行
"""

import asyncio
import json
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


# 考公资料路径
KGZL_PATH = "D:/考公资料"
OUTPUT_DIR = Path("d:/Learning plugins/output/kgzl_cycles")


async def run_cycle_test(
    test_name: str,
    task: str,
    tier: ModelTier = ModelTier.FAST,
    thinking: ThinkingEffort = ThinkingEffort.LOW,
    max_iterations: int = 10,
):
    """运行单个循环测试"""
    print(f"\n{'=' * 70}")
    print(f"循环测试: {test_name}")
    print(f"模型: {tier.value} / 思考强度: {thinking.value}")
    print(f"{'=' * 70}")

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 创建 Agent
    config = LLMConfig.from_file("llm_config.json")
    llm = TieredLLMClient(config)
    context = ToolContext(llm_client=llm)

    registry = ToolRegistry(context)
    loaded = registry.discover_tools("tools/")
    print(f"已加载工具: {', '.join(loaded)}")

    # 创建 Agent
    agent = ToolAgent(
        llm=llm,
        registry=registry,
        interaction_handler=InteractionHandler(mode=InteractionMode.AUTO),
        max_iterations=max_iterations,
        default_tier=tier,
        default_thinking=thinking,
        event_log_path=OUTPUT_DIR / f"{test_name}_events.jsonl",
    )

    print(f"会话ID: {agent.session_id[:8]}...")
    print(f"\n任务: {task}")
    print("-" * 70)

    # 执行任务
    result = await agent.run(task)

    # 输出结果
    print("\n" + "-" * 70)
    print("执行结果:")
    print(f"  成功: {result.success}")
    print(f"  迭代次数: {result.total_iterations}")
    print(f"  步骤数: {len(result.steps)}")

    if result.success:
        print(f"\n最终答案:")
        print(result.final_answer[:1000])
        if len(result.final_answer) > 1000:
            print("... ( truncated)")
    else:
        print(f"\n错误: {result.error}")

    # 导出事件日志
    trace_path = OUTPUT_DIR / f"{test_name}_trace.json"
    agent.event_log.export_trace(trace_path)
    print(f"\n事件日志: {trace_path}")

    # 分析事件
    stats = agent.event_log.get_stats()
    print(f"\n事件统计:")
    print(f"  总事件: {stats['total_events']}")
    print(f"  工具调用: {stats['tool_calls']}")
    print(f"  LLM 调用: {stats['llm_calls']}")
    print(f"  错误: {stats['errors']}")
    print(f"  总耗时: {stats['total_duration_ms']:.0f}ms")

    return result


async def test_1_index_with_paths():
    """测试1：索引考公资料并输出具体路径"""
    task = f"""请使用 resource-indexer 工具索引文件夹 {KGZL_PATH}。

要求：
1. 获取完整的文件列表和文件夹结构
2. 确保输出包含具体文件路径（file_paths）和基础路径（base_path）
3. 生成索引文档并保存到 output/kgzl_cycles/ 目录
4. 报告文件总数和主要分类

请执行并报告结果。"""

    return await run_cycle_test(
        "test1_index",
        task,
        tier=ModelTier.FAST,
        thinking=ThinkingEffort.LOW,
        max_iterations=5,
    )


async def test_2_classify():
    """测试2：分类考公资料"""
    task = f"""请完成以下任务：

1. 使用 resource-indexer 索引文件夹 {KGZL_PATH}，获取文件列表
2. 使用 resource-classifier 对文件进行分类（教材/讲义/笔记/题集/题解/其他）
3. 统计各类别的文件数量
4. 找出题集类文件，列出它们的路径

请按顺序执行，并报告分类结果。"""

    return await run_cycle_test(
        "test2_classify",
        task,
        tier=ModelTier.FAST,
        thinking=ThinkingEffort.LOW,
        max_iterations=8,
    )


async def test_3_question_split():
    """测试3：分题出题库"""
    task = f"""请完成以下任务：

1. 使用 resource-indexer 索引文件夹 {KGZL_PATH}
2. 从索引结果中找到题集类 PDF 文件（如包含"真题"、"题库"、"刷题"等关键词的文件）
3. 选择一个题集 PDF，使用 question-splitter 提取题目
4. 将提取的题目保存到 output/kgzl_cycles/questions.json
5. 报告提取的题目数量和类型分布

如果找不到题集 PDF，请说明原因并给出建议。"""

    return await run_cycle_test(
        "test3_questions",
        task,
        tier=ModelTier.BALANCED,
        thinking=ThinkingEffort.MEDIUM,
        max_iterations=10,
    )


async def test_4_lecture_notes():
    """测试4：讲义整理"""
    task = f"""请完成以下任务：

1. 使用 resource-indexer 索引文件夹 {KGZL_PATH}
2. 识别出讲义类文件（如包含"讲义"、"课件"、"视频"等关键词的文件）
3. 使用 resource-classifier 确认它们的分类
4. 为讲义类文件生成学习建议和使用顺序
5. 将结果保存到 output/kgzl_cycles/lecture_notes.md

请执行并报告结果。"""

    return await run_cycle_test(
        "test4_lectures",
        task,
        tier=ModelTier.BALANCED,
        thinking=ThinkingEffort.MEDIUM,
        max_iterations=10,
    )


async def test_5_model_comparison():
    """测试5：不同模型对比"""
    print("\n" + "=" * 70)
    print("循环测试 5: 不同模型对比")
    print("=" * 70)

    task = f"使用 resource-indexer 索引文件夹 {KGZL_PATH}，并生成简要摘要（100字以内）"

    models = [
        (ModelTier.FAST, ThinkingEffort.LOW, "fast+low"),
        (ModelTier.BALANCED, ThinkingEffort.MEDIUM, "balanced+medium"),
    ]

    results = {}

    for tier, thinking, label in models:
        print(f"\n测试模型: {label}")

        config = LLMConfig.from_file("llm_config.json")
        llm = TieredLLMClient(config)
        context = ToolContext(llm_client=llm)

        registry = ToolRegistry(context)
        registry.discover_tools("tools/")

        agent = ToolAgent(
            llm=llm,
            registry=registry,
            interaction_handler=InteractionHandler(mode=InteractionMode.AUTO),
            max_iterations=5,
            default_tier=tier,
            default_thinking=thinking,
            event_log_path=OUTPUT_DIR / f"test5_{label.replace('+', '_')}_events.jsonl",
        )

        result = await agent.run(task)
        results[label] = {
            "success": result.success,
            "iterations": result.total_iterations,
            "steps": len(result.steps),
            "answer_length": len(result.final_answer) if result.success else 0,
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
    print("考公资料实际任务循环测试")
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
    tests = [
        ("索引考公资料", test_1_index_with_paths),
        ("分类考公资料", test_2_classify),
        ("分题出题库", test_3_question_split),
        ("讲义整理", test_4_lecture_notes),
    ]

    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result.success))
        except Exception as e:
            print(f"\n[ERROR] {name} 测试异常: {e}")
            results.append((name, False))

    # 模型对比测试
    try:
        await test_5_model_comparison()
        results.append(("模型对比", True))
    except Exception as e:
        print(f"\n[ERROR] 模型对比测试异常: {e}")
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
    print(f"\n输出目录: {OUTPUT_DIR}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
