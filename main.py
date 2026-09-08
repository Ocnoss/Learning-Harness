"""Learning Plugins - Tool-Agent CLI 启动入口

独立运行的 Tool-Agent，使用配置的 LLM 自主调用工具完成任务。

使用方式：
    python main.py "帮我整理 D:/学习资料 文件夹"
    python main.py --mode auto "分类 D:/资料 中的所有文件"
    python main.py --mode yolo "提取 D:/题库/数学.pdf 的题目"
    python main.py --interactive  # 交互模式
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 确保能导入 core 模块
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
    create_agent,
)


def print_banner():
    """打印启动横幅"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           Learning Plugins - Tool-Agent CLI                   ║
║                                                               ║
║   LLM 驱动的智能工具调用 Agent                                 ║
║   支持 auto / yolo / ask-when-needed 交互模式                  ║
╚═══════════════════════════════════════════════════════════════╝
""")


def print_tools(registry: ToolRegistry):
    """打印已加载的工具"""
    print("\n已加载工具:")
    print("-" * 60)
    for name, manifest in sorted(registry.get_all_manifests().items()):
        tier = manifest.recommended_tier
        thinking = manifest.recommended_thinking
        print(f"  [{tier}/{thinking}] {name}")
        print(f"      {manifest.description_for_llm[:60]}...")
    print("-" * 60)


async def run_task(agent: ToolAgent, task: str):
    """执行单个任务"""
    print(f"\n执行任务: {task}")
    print("=" * 60)

    result = await agent.run(task)

    print("\n" + "=" * 60)
    if result.success:
        print("✓ 任务完成")
        print(f"\n最终答案:\n{result.final_answer}")
    else:
        print("✗ 任务失败")
        print(f"错误: {result.error}")

    print(f"\n执行统计:")
    print(f"  迭代次数: {result.total_iterations}")
    print(f"  步骤数: {len(result.steps)}")

    return result


async def interactive_mode(agent: ToolAgent):
    """交互模式"""
    print("\n进入交互模式 (输入 'quit' 或 'exit' 退出)")
    print("-" * 60)

    while True:
        try:
            task = input("\n请输入任务: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出交互模式")
            break

        if not task:
            continue

        if task.lower() in ("quit", "exit", "q"):
            print("退出交互模式")
            break

        await run_task(agent, task)


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Learning Plugins Tool-Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py "帮我整理 D:/学习资料"
  python main.py --mode auto "分类 D:/资料"
  python main.py --interactive
        """
    )

    parser.add_argument(
        "task",
        nargs="?",
        help="要执行的任务描述"
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["auto", "yolo", "ask"],
        default="ask",
        help="交互模式: auto(自动) / yolo(全自动) / ask(需要时询问)"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="进入交互模式"
    )

    parser.add_argument(
        "--config", "-c",
        default="llm_config.json",
        help="LLM 配置文件路径 (默认: llm_config.json)"
    )

    parser.add_argument(
        "--tools", "-t",
        default="tools/",
        help="工具目录路径 (默认: tools/)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="最大迭代次数 (默认: 20)"
    )

    parser.add_argument(
        "--thinking", "-th",
        choices=["low", "medium", "high"],
        default=None,
        help="思考强度: low(快速) / medium(平衡) / high(深度)。不指定则由工具决定"
    )

    parser.add_argument(
        "--tier", "-tr",
        choices=["fast", "balanced", "flagship"],
        default="balanced",
        help="模型层级: fast(快速) / balanced(均衡) / flagship(旗舰)"
    )

    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="列出所有可用工具"
    )

    args = parser.parse_args()

    # 打印横幅
    print_banner()

    # 检查配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {args.config}")
        print("请创建 llm_config.json 或指定正确的配置文件路径")
        return 1

    # 检查工具目录
    tools_dir = Path(args.tools)
    if not tools_dir.exists():
        print(f"错误: 工具目录不存在: {args.tools}")
        return 1

    # 解析交互模式
    mode_map = {
        "auto": InteractionMode.AUTO,
        "yolo": InteractionMode.YOLO,
        "ask": InteractionMode.ASK_WHEN_NEEDED,
    }
    interaction_mode = mode_map[args.mode]

    print(f"配置: {args.config}")
    print(f"工具目录: {args.tools}")
    print(f"交互模式: {args.mode}")
    print(f"模型层级: {args.tier}")
    print(f"思考强度: {args.thinking or '由工具决定'}")
    print(f"最大迭代: {args.max_iterations}")

    # 解析模型层级和思考强度
    tier_map = {
        "fast": ModelTier.FAST,
        "balanced": ModelTier.BALANCED,
        "flagship": ModelTier.FLAGSHIP,
    }
    thinking_map = {
        "low": ThinkingEffort.LOW,
        "medium": ThinkingEffort.MEDIUM,
        "high": ThinkingEffort.HIGH,
    }
    default_tier = tier_map[args.tier]
    default_thinking = thinking_map.get(args.thinking) if args.thinking else None

    try:
        # 创建 Agent
        agent = await create_agent(
            config_path=config_path,
            tools_dir=tools_dir,
            interaction_mode=interaction_mode,
            max_iterations=args.max_iterations,
            default_tier=default_tier,
            default_thinking=default_thinking,
        )

        # 列出工具
        if args.list_tools:
            print_tools(agent.registry)
            return 0

        # 打印已加载工具
        print_tools(agent.registry)

        # 交互模式
        if args.interactive:
            await interactive_mode(agent)
            return 0

        # 单任务模式
        if not args.task:
            print("\n错误: 请提供任务描述或使用 --interactive 进入交互模式")
            parser.print_help()
            return 1

        result = await run_task(agent, args.task)
        return 0 if result.success else 1

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
