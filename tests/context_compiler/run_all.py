"""三层测试统一入口（契约 §7 架构验证）—— 串跑第一/二/三层并汇总。

用法:
    D:\\miniconda\\python.exe tests\\context_compiler\\run_all.py

行为:
    - 第一层：import test_layer1_invariants 并调用其 main（不变量 §7.1）；
    - 第二层：黄金场景 §7.2（全程内存 store，默认无 LLM/网络）；
    - 第三层：headless A/B §6.1（stub 模式必须过；live 无 key 自动跳过）；
    - 磁带冒烟：RecordingLLMClient/RecordedLLMClient 录制-回放（纯离线）；
    - 入口打印一次 live 门控状态（LH_LIVE_LLM / LLM_API_KEY），仅作提示，
      跳过逻辑与退出码完全由第三层内部决定；
    - 打印逐层与总汇总；退出码 0=全通过，1=存在失败。

各层 main() 均为 async，本入口在同一事件循环内依次 await——不嵌套 asyncio.run，
避免"运行中循环内再启循环"的 RuntimeError。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_REPO_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_layer1_invariants as layer1  # noqa: E402
import test_layer2_scenarios as layer2  # noqa: E402
import test_layer3_ab_eval as layer3  # noqa: E402
import test_stub_tape as tape_smoke  # noqa: E402

_LAYERS = [
    ("第一层 不变量 (§7.1)", layer1),
    ("第二层 黄金场景 (§7.2)", layer2),
    ("第三层 headless A/B (§6.1)", layer3),
    ("磁带录制-回放冒烟 (offline)", tape_smoke),
]


def _print_live_gate() -> None:
    """入口提示 live 门控状态（不改变跳过逻辑与退出码）。"""
    live_flag = bool(os.environ.get("LH_LIVE_LLM"))
    api_key = bool(os.environ.get("LLM_API_KEY"))
    enabled = live_flag and api_key
    print("# live gate: LH_LIVE_LLM=%s, LLM_API_KEY=%s -> layer3 live %s"
          % ("set" if live_flag else "unset",
             "set" if api_key else "unset",
             "ENABLED" if enabled else "SKIP (stub only)"))


async def main() -> int:
    print("#" * 60)
    print("# LH 上下文编译器 —— 三层测试统一入口")
    print("# 纯确定性：无 wall-clock / random / 网络（第三层 live 除外，默认跳过）")
    print("#" * 60)
    _print_live_gate()

    results: list[tuple[str, bool]] = []
    for name, mod in _LAYERS:
        print("\n" + "=" * 60)
        print(f">>> 运行 {name}")
        print("=" * 60)
        try:
            code = await mod.main()
            results.append((name, int(code) == 0))
        except Exception as exc:  # noqa: BLE001 —— 任一层崩溃都计入失败
            print(f"[ERROR] {name} 运行异常: {type(exc).__name__}: {exc}")
            results.append((name, False))

    print("\n" + "#" * 60)
    print("# 测试总汇总（三层 + 磁带冒烟）")
    print("#" * 60)
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    all_passed = all(p for _, p in results)
    passed_n = sum(1 for _, p in results if p)
    print()
    print(f"结果：{passed_n}/{len(results)} 项通过 —— "
          + ("全部通过!" if all_passed else "存在失败项!"))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
