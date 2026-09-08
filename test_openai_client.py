"""测试 OpenAI 客户端的 complete() 和 complete_json() 方法

运行方式:
    python test_openai_client.py

需要设置环境变量:
    LLM_API_KEY - OpenAI API Key
    LLM_BASE_URL - (可选) 自定义 API 地址
    LLM_MODEL - (可选) 模型名称，默认 gpt-4
"""

import asyncio
import os
import sys
from pathlib import Path

# 确保能导入 core 模块
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 导入 providers 模块以触发自动注册
from core.providers.openai_client import OpenAIClient  # noqa: F401
from core.llm_client import LLMClientFactory, Message


def get_config() -> dict:
    """从环境变量构建配置"""
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        print("错误: 请设置环境变量 LLM_API_KEY")
        print("  PowerShell: $env:LLM_API_KEY = 'sk-...'")
        sys.exit(1)

    config = {"api_key": api_key}

    base_url = os.environ.get("LLM_BASE_URL")
    if base_url:
        config["base_url"] = base_url

    model = os.environ.get("LLM_MODEL")
    if model:
        config["model"] = model

    return config


async def test_complete():
    """测试 complete() 方法"""
    print("=" * 50)
    print("测试 1: complete() - 基本对话")
    print("=" * 50)

    config = get_config()
    client = LLMClientFactory.create("openai", config)
    print(f"提供商: openai")
    print(f"模型: {client.model}")
    print()

    # 测试 dict 格式消息
    response = await client.complete(
        messages=[
            {"role": "system", "content": "你是一个简洁的助手，用一句话回答。"},
            {"role": "user", "content": "什么是机器学习？"},
        ],
        temperature=0.7,
        max_tokens=100,
    )

    print(f"回复: {response.content}")
    print(f"模型: {response.model}")
    print(f"用量: {response.usage}")
    print()

    # 测试 Message 对象格式
    response2 = await client.complete(
        messages=[
            Message(role="user", content="用一句话解释什么是 Python。"),
        ],
        temperature=0.5,
        max_tokens=100,
    )

    print(f"回复 (Message 对象): {response2.content}")
    print()

    print("[PASS] complete() 测试通过")
    return True


async def test_complete_json():
    """测试 complete_json() 方法"""
    print()
    print("=" * 50)
    print("测试 2: complete_json() - JSON 结构化输出")
    print("=" * 50)

    config = get_config()
    client = LLMClientFactory.create("openai", config)

    result = await client.complete_json(
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个数据提取助手。请从用户输入中提取信息，"
                    "以 JSON 格式返回。仅返回 JSON，不要包含其他内容。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请提取以下文本的关键信息：\n"
                    "小明今年15岁，是一名初中生，"
                    "他喜欢数学和编程，梦想是成为软件工程师。"
                ),
            },
        ],
        temperature=0.1,
    )

    print(f"解析结果: {result}")
    print(f"类型: {type(result).__name__}")

    assert isinstance(result, (dict, list)), "返回值应为 dict 或 list"
    print()

    # 测试返回数组
    result_list = await client.complete_json(
        messages=[
            {
                "role": "system",
                "content": "你是一个列表生成助手。仅返回 JSON 数组，不要包含其他内容。",
            },
            {
                "role": "user",
                "content": "请列出 3 种编程语言，每种包含 name 和 year 字段。",
            },
        ],
        temperature=0.1,
    )

    print(f"数组结果: {result_list}")
    assert isinstance(result_list, list), "返回值应为 list"
    print()

    print("[PASS] complete_json() 测试通过")
    return True


async def test_stream():
    """测试 stream() 方法"""
    print()
    print("=" * 50)
    print("测试 3: stream() - 流式输出")
    print("=" * 50)

    config = get_config()
    client = LLMClientFactory.create("openai", config)

    print("流式回复: ", end="", flush=True)
    full_text = ""
    async for chunk in client.stream(
        messages=[
            {"role": "user", "content": "用一句话解释什么是深度学习。"},
        ],
        temperature=0.5,
        max_tokens=100,
    ):
        print(chunk, end="", flush=True)
        full_text += chunk

    print()
    print()

    assert len(full_text) > 0, "流式输出不应为空"
    print("[PASS] stream() 测试通过")
    return True


async def test_factory():
    """测试工厂注册"""
    print()
    print("=" * 50)
    print("测试 4: LLMClientFactory - 提供商注册")
    print("=" * 50)

    providers = LLMClientFactory.list_providers()
    print(f"已注册提供商: {providers}")

    assert "openai" in providers, "openai 应已注册"
    print("[PASS] 工厂注册测试通过")
    return True


async def main():
    print("OpenAI 客户端测试")
    print()

    results = []

    # 工厂注册测试（不需要 API 调用）
    results.append(("工厂注册", await test_factory()))

    # 需要 API 调用的测试
    try:
        results.append(("complete()", await test_complete()))
    except Exception as e:
        print(f"[FAIL] complete() 测试失败: {e}")
        results.append(("complete()", False))

    try:
        results.append(("complete_json()", await test_complete_json()))
    except Exception as e:
        print(f"[FAIL] complete_json() 测试失败: {e}")
        results.append(("complete_json()", False))

    try:
        results.append(("stream()", await test_stream()))
    except Exception as e:
        print(f"[FAIL] stream() 测试失败: {e}")
        results.append(("stream()", False))

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
