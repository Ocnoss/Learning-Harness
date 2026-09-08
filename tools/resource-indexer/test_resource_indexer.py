"""测试 resource-indexer 工具（LLM 二次确认版本）

运行方式:
    D:\miniconda\python.exe tools/resource-indexer/test_resource_indexer.py
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 导入 providers 触发自动注册
from core.providers.openai_client import OpenAIClient  # noqa: F401
from core.llm_client import TieredLLMClient
from core.llm_config import LLMConfig
from core.base_tool import ToolContext

# 动态导入 resource-indexer 模块（文件夹名含连字符）
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "resource_indexer",
    Path(__file__).parent / "main.py"
)
_resource_indexer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_resource_indexer)
create_tool = _resource_indexer.create_tool


async def test_normal_folder():
    """测试正常学习资料文件夹（LLM 判断为学习资料，直接生成文档）"""
    print("=" * 50)
    print("测试 1: 正常学习资料文件夹")
    print("=" * 50)

    test_folder = Path(__file__).parent.parent.parent / "tools"

    config = LLMConfig.from_file(Path(__file__).parent.parent.parent / "llm_config.json")
    llm = TieredLLMClient(config)
    context = ToolContext(llm_client=llm)
    tool = create_tool(context)

    result = await tool.execute({
        "folder_path": str(test_folder),
        "max_depth": 2,
        "include_extensions": [".py", ".md", ".json"],
    })

    print(f"成功: {result.success}")
    print(f"文件数: {result.data['file_count']}")
    print(f"需要确认: {result.data['needs_confirmation']}")

    judgment = result.data.get("llm_judgment")
    if judgment:
        print(f"LLM 判断: is_learning={judgment['is_learning_material']}, "
              f"confidence={judgment['confidence']}")
        print(f"理由: {judgment['reason']}")

    if result.data["needs_confirmation"]:
        print(f"确认信息: {result.data['confirmation_message']}")
    else:
        doc = result.data["index_document"]
        print(f"文档长度: {len(doc)} 字符")
        print(f"文档预览:\n{doc[:500]}...")

    return result.success


async def test_suspicious_folder():
    """测试可疑文件夹（LLM 判断为非学习资料，触发二次确认）"""
    print()
    print("=" * 50)
    print("测试 2: 可疑文件夹（LLM 二次确认）")
    print("=" * 50)

    test_folder = Path("C:/Windows/System32")

    if not test_folder.exists():
        print("跳过: 系统目录不存在")
        return True

    config = LLMConfig.from_file(Path(__file__).parent.parent.parent / "llm_config.json")
    llm = TieredLLMClient(config)
    context = ToolContext(llm_client=llm)
    tool = create_tool(context)

    result = await tool.execute({
        "folder_path": str(test_folder),
        "max_depth": 1,
        "include_extensions": [".exe", ".dll", ".sys"],
    })

    print(f"成功: {result.success}")
    print(f"文件数: {result.data['file_count']}")
    print(f"需要确认: {result.data['needs_confirmation']}")

    judgment = result.data.get("llm_judgment")
    if judgment:
        print(f"LLM 判断: is_learning={judgment['is_learning_material']}, "
              f"confidence={judgment['confidence']}")
        print(f"理由: {judgment['reason']}")

    if result.data["needs_confirmation"]:
        print(f"确认信息:\n{result.data['confirmation_message']}")
        print("\n[PASS] LLM 正确触发了二次确认")
        return True
    else:
        print("[FAIL] 未触发二次确认")
        return False


async def test_skip_confirmation():
    """测试跳过二次确认（用户已确认）"""
    print()
    print("=" * 50)
    print("测试 3: 跳过二次确认")
    print("=" * 50)

    test_folder = Path(__file__).parent.parent.parent

    config = LLMConfig.from_file(Path(__file__).parent.parent.parent / "llm_config.json")
    llm = TieredLLMClient(config)
    context = ToolContext(llm_client=llm)
    tool = create_tool(context)

    result = await tool.execute({
        "folder_path": str(test_folder),
        "max_depth": 1,
        "include_extensions": [".py", ".md", ".json"],
        "skip_confirmation": True,  # 跳过确认
    })

    print(f"成功: {result.success}")
    print(f"文件数: {result.data['file_count']}")
    print(f"需要确认: {result.data['needs_confirmation']}")

    if not result.data["needs_confirmation"]:
        doc = result.data["index_document"]
        print(f"文档长度: {len(doc)} 字符")
        print(f"文档预览:\n{doc[:500]}...")
        print("\n[PASS] 跳过确认直接生成文档")
        return True
    else:
        print("[FAIL] 不应触发二次确认")
        return False


async def test_user_context():
    """测试用户先验（跳过 LLM 判断）"""
    print()
    print("=" * 50)
    print("测试 4: 用户先验（跳过 LLM 判断）")
    print("=" * 50)

    test_folder = Path(__file__).parent.parent.parent / "core"

    config = LLMConfig.from_file(Path(__file__).parent.parent.parent / "llm_config.json")
    llm = TieredLLMClient(config)
    context = ToolContext(llm_client=llm)
    tool = create_tool(context)

    result = await tool.execute({
        "folder_path": str(test_folder),
        "max_depth": 2,
        "include_extensions": [".py"],
        "user_context": "这是 Python 学习项目的核心代码",
    })

    print(f"成功: {result.success}")
    print(f"文件数: {result.data['file_count']}")
    print(f"需要确认: {result.data['needs_confirmation']}")

    judgment = result.data.get("llm_judgment")
    if judgment:
        print(f"LLM 判断: {judgment}")
    else:
        print("LLM 判断: 未执行（用户已提供先验）")

    if not result.data["needs_confirmation"]:
        doc = result.data["index_document"]
        print(f"文档长度: {len(doc)} 字符")
        print("\n[PASS] 用户先验跳过了 LLM 判断")
        return True
    else:
        print("[FAIL] 不应触发二次确认")
        return False


async def main():
    print("resource-indexer 工具测试（LLM 二次确认版本）")
    print()

    results = []

    try:
        results.append(("正常文件夹", await test_normal_folder()))
    except Exception as e:
        print(f"[错误] {e}")
        import traceback
        traceback.print_exc()
        results.append(("正常文件夹", False))

    try:
        results.append(("可疑文件夹", await test_suspicious_folder()))
    except Exception as e:
        print(f"[错误] {e}")
        import traceback
        traceback.print_exc()
        results.append(("可疑文件夹", False))

    try:
        results.append(("跳过确认", await test_skip_confirmation()))
    except Exception as e:
        print(f"[错误] {e}")
        import traceback
        traceback.print_exc()
        results.append(("跳过确认", False))

    try:
        results.append(("用户先验", await test_user_context()))
    except Exception as e:
        print(f"[错误] {e}")
        import traceback
        traceback.print_exc()
        results.append(("用户先验", False))

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
