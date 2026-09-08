"""文件操作工具测试

测试所有新开发的文件操作工具：
1. file-reader
2. file-writer
3. file-editor
4. glob-search
5. grep-search
6. web-search
"""

import asyncio
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core import (
    LLMConfig,
    ModelTier,
    ThinkingEffort,
    TieredLLMClient,
    ToolContext,
    ToolRegistry,
)


async def test_file_operations():
    """测试文件操作工具"""
    print("=" * 70)
    print("文件操作工具测试")
    print("=" * 70)

    # 创建工具注册表
    config = LLMConfig.from_file("llm_config.json")
    llm = TieredLLMClient(config)
    context = ToolContext(llm_client=llm)

    registry = ToolRegistry(context)
    loaded = registry.discover_tools("tools/")
    print(f"已加载工具: {', '.join(loaded)}")

    # 测试文件路径
    test_dir = Path("d:/Learning plugins/output/test_file_ops")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "test.txt"

    results = []

    # 测试 1: file-writer
    print("\n测试 1: file-writer")
    writer = registry.get_tool("file-writer")
    if writer:
        result = await writer.execute({
            "file_path": str(test_file),
            "content": "Hello, World!\nThis is a test file.\nLine 3",
            "mode": "create",
        })
        print(f"  结果: {'[PASS]' if result.success else '[FAIL]'}")
        print(f"  摘要: {result.summary_for_llm}")
        results.append(("file-writer", result.success))
    else:
        print("  [FAIL] 工具未加载")
        results.append(("file-writer", False))

    # 测试 2: file-reader
    print("\n测试 2: file-reader")
    reader = registry.get_tool("file-reader")
    if reader:
        result = await reader.execute({
            "file_path": str(test_file),
            "limit": 10,
        })
        print(f"  结果: {'[PASS]' if result.success else '[FAIL]'}")
        if result.success:
            print(f"  内容: {result.data['content'][:50]}...")
        results.append(("file-reader", result.success))
    else:
        print("  [FAIL] 工具未加载")
        results.append(("file-reader", False))

    # 测试 3: file-editor
    print("\n测试 3: file-editor")
    editor = registry.get_tool("file-editor")
    if editor:
        result = await editor.execute({
            "file_path": str(test_file),
            "old_string": "Hello, World!",
            "new_string": "Hello, Python!",
        })
        print(f"  结果: {'[PASS]' if result.success else '[FAIL]'}")
        print(f"  摘要: {result.summary_for_llm}")
        results.append(("file-editor", result.success))
    else:
        print("  [FAIL] 工具未加载")
        results.append(("file-editor", False))

    # 测试 4: glob-search
    print("\n测试 4: glob-search")
    glob = registry.get_tool("glob-search")
    if glob:
        result = await glob.execute({
            "pattern": "*.txt",
            "base_path": str(test_dir),
        })
        print(f"  结果: {'[PASS]' if result.success else '[FAIL]'}")
        if result.success:
            print(f"  找到 {result.data['count']} 个文件")
        results.append(("glob-search", result.success))
    else:
        print("  [FAIL] 工具未加载")
        results.append(("glob-search", False))

    # 测试 5: grep-search
    print("\n测试 5: grep-search")
    grep = registry.get_tool("grep-search")
    if grep:
        result = await grep.execute({
            "pattern": "Python",
            "base_path": str(test_dir),
        })
        print(f"  结果: {'[PASS]' if result.success else '[FAIL]'}")
        if result.success:
            print(f"  找到 {result.data['count']} 处匹配")
        results.append(("grep-search", result.success))
    else:
        print("  [FAIL] 工具未加载")
        results.append(("grep-search", False))

    # 测试 6: web-search
    print("\n测试 6: web-search")
    web = registry.get_tool("web-search")
    if web:
        result = await web.execute({
            "query": "Python asyncio",
            "max_results": 3,
        })
        print(f"  结果: {'[PASS]' if result.success else '[FAIL]'}")
        if result.success:
            print(f"  找到 {result.data['count']} 条结果")
        results.append(("web-search", result.success))
    else:
        print("  [FAIL] 工具未加载")
        results.append(("web-search", False))

    # 汇总
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    all_passed = all(r[1] for r in results)
    print(f"\n{'全部通过!' if all_passed else '存在失败项!'}")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(test_file_operations())
    sys.exit(0 if success else 1)
