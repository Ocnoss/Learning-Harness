"""第三层 headless A/B 评估（契约 §6.1）—— 三臂对比 + rubric 裁判。

三臂（同一 goal，headless，无前端）:
    compiled —— 编译上下文：由 bundle.items 拼提示词（ContextBundle 无
                to_prompt_text，故在本测试内拼，绝不改 compiler.py）；
    full     —— 全量塞入：所有事件渲染文本；
    recent   —— 最近 N 轮：最近 N 条事件渲染文本。

每臂各经 LLM 客户端得答案，裁判按 rubric（相关性/事实性/教学适配/简洁性）打分。
核心指标断言（§6.1）:
    compiled.input_tokens < TOKEN_RATIO_THRESHOLD(0.3) × full.input_tokens
    且 judge_score(compiled) ≥ judge_score(full) − EPSILON(0.05)

门控:
    - 默认 stub 模式：StubLLMClient 顺序回放答案 + canned 裁判分（离线、确定性），
      精心设计使上述断言被**真实行使**（验证 harness 机制本身成立），必须过；
    - live 模式：设 LH_LIVE_LLM 且有 LLM_API_KEY 时用真实 flagship 跑并打印实际
      指标；无 key 自动跳过。

运行方式:
    D:\\miniconda\\python.exe tests\\context_compiler\\test_layer3_ab_eval.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_REPO_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.context import rebuild, compile, ContextQuery  # noqa: E402
from core.context.compiler import is_within_scope  # noqa: E402
from core.context.projection import render_text, derive_level  # noqa: E402
from core.stub_llm import StubLLMClient  # noqa: E402
from harness import fake_learner as fl  # noqa: E402
from harness.judge_rubric import (  # noqa: E402
    JudgeRubric,
    RubricScoreCache,
    judge_answer,
)

# ---------------------------------------------------------------- 评估常量
GOAL = "资料分析 速算 提升"
N_RECENT = 5
TOKEN_RATIO_THRESHOLD = 0.3   # 编译臂 input_tokens 必须 < 0.3 × 全量臂
EPSILON = 0.05                # judge_score(compiled) ≥ judge_score(full) − ε

# canned 答案（stub 模式；内容不影响 input_tokens，仅占位）
_ANSWER_COMPILED = "建议：聚焦资料分析速算限时训练，先复盘基期概念，再做 15 题限时。"
_ANSWER_FULL = "建议：浏览全部历史后，泛泛地多练资料分析，兼顾各模块。"
_ANSWER_RECENT = "建议：延续最近几轮的判断推理练习即可。"

# canned 裁判各维度分（0..1），聚合后：compiled≈0.910 > full≈0.750 > recent≈0.578
_DIMS_COMPILED = {"relevance": 0.95, "factuality": 0.90, "pedagogy": 0.90, "conciseness": 0.85}
_DIMS_FULL = {"relevance": 0.80, "factuality": 0.85, "pedagogy": 0.70, "conciseness": 0.50}
_DIMS_RECENT = {"relevance": 0.55, "factuality": 0.60, "pedagogy": 0.50, "conciseness": 0.70}


@dataclass
class ArmResult:
    name: str
    input_tokens: int
    score: float
    dims: dict
    answer: str


# ---------------------------------------------------------------- 三臂上下文
def _compiled_context(bundle) -> str:
    """编译臂上下文：由 bundle.items 拼提示词文本（测试内拼，不改 compiler）。"""
    return "\n".join(f"[L{i.level}] {i.text}" for i in bundle.items)


def _full_context(events) -> str:
    """全量臂上下文：所有事件按层级渲染文本拼接。"""
    ordered = sorted(events, key=lambda e: (e.timestamp, e.id))
    return "\n".join(render_text(e, derive_level(e)) for e in ordered)


def _recent_context(events, n: int) -> str:
    """最近 N 轮臂上下文：按 (timestamp, id) 取最后 n 条渲染文本。"""
    ordered = sorted(events, key=lambda e: (e.timestamp, e.id))
    return "\n".join(render_text(e, derive_level(e)) for e in ordered[-n:])


def _arm_prompt(context: str, goal: str) -> str:
    return (f"以下是学习上下文：\n{context}\n\n"
            f"问题：针对学习目标「{goal}」，给出一条下一步教学建议。")


def _build_arms(projection, events) -> tuple[dict, object]:
    """构造三臂提示词，并返回 (arms, compiled_bundle)。"""
    # 预算必须按 compile 的作用域语义过滤（is_within_scope），否则若投影含
    # 无关 scope 的 L3，预算会偏大，compile 在第 3 步可能把命中 goal 的 L2
    # 塞进 bundle，违反"编译臂只含 L3"的断言。
    l3_total = sum(i.token_len for i in projection.items
                   if i.level == 3 and is_within_scope(i.scope, fl.XC))
    # 紧预算：只装得下 L3 断言（编译臂 = 抽象梯子顶层，最省 token）
    bundle = compile(projection, ContextQuery(
        scope=fl.XC, goal=GOAL, budget_tokens=l3_total,
        as_of=fl.AS_OF, allow_l1_deref=False))
    arms = {
        "compiled": _arm_prompt(_compiled_context(bundle), GOAL),
        "full": _arm_prompt(_full_context(events), GOAL),
        "recent": _arm_prompt(_recent_context(events, N_RECENT), GOAL),
    }
    return arms, bundle


# ---------------------------------------------------------------- stub 模式
async def test_ab_eval_stub():
    """stub 模式三臂 A/B（离线、确定性，必须过）。

    行使：token 预算优势断言、rubric 质量断言、bundle-hash 缓存命中。
    """
    print("\n=== 第三层 A/B（stub 模式，确定性） ===")
    events = fl.build_ab_events()
    projection = rebuild(events)
    arms, bundle = _build_arms(projection, events)

    # 编译臂确为"只到断言层"的紧预算包
    assert bundle.items and all(i.level == 3 for i in bundle.items), "编译臂应只含 L3"
    assert bundle.deepest_layer_reached == 3

    answer_client = StubLLMClient(
        {"responses": [_ANSWER_COMPILED, _ANSWER_FULL, _ANSWER_RECENT]})
    judge_client = StubLLMClient(
        {"responses": [json.dumps(_DIMS_COMPILED, ensure_ascii=False),
                       json.dumps(_DIMS_FULL, ensure_ascii=False),
                       json.dumps(_DIMS_RECENT, ensure_ascii=False)]})
    rubric = JudgeRubric()
    cache = RubricScoreCache()

    results: dict[str, ArmResult] = {}
    for name in ("compiled", "full", "recent"):
        resp = await answer_client.complete([{"role": "user", "content": arms[name]}])
        input_tokens = int(resp.usage.get("prompt_tokens", 0))
        # 缓存键：编译臂用 bundle.to_json()（bundle-hash），其余臂用其上下文串
        cache_repr = bundle.to_json() if name == "compiled" else arms[name]
        key = cache.key_for(cache_repr, rubric)
        score, dims = await judge_answer(judge_client, rubric, GOAL, resp.content,
                                         cache=cache, cache_key=key)
        results[name] = ArmResult(name, input_tokens, score, dims, resp.content)

        # bundle-hash 缓存：编译臂二次评分应命中缓存、不再调用裁判
        if name == "compiled":
            hist_before = len(judge_client.call_history)
            score2, _ = await judge_answer(judge_client, rubric, GOAL, resp.content,
                                           cache=cache, cache_key=key)
            assert score2 == score, "缓存命中分数应一致"
            assert len(judge_client.call_history) == hist_before, \
                "bundle-hash 缓存命中时不应重复调用裁判"
            assert cache.hits >= 1, "缓存应至少命中一次"

    c, f, r = results["compiled"], results["full"], results["recent"]

    # ---- 核心指标（§6.1）----
    token_ratio = c.input_tokens / f.input_tokens if f.input_tokens else float("inf")
    assert c.input_tokens < TOKEN_RATIO_THRESHOLD * f.input_tokens, (
        f"编译臂 input_tokens({c.input_tokens}) 未 < "
        f"{TOKEN_RATIO_THRESHOLD}*全量臂({f.input_tokens})，比值 {token_ratio:.4f}")
    assert c.score >= f.score - EPSILON, (
        f"judge_score(compiled)={c.score:.4f} < "
        f"judge_score(full)={f.score:.4f} - eps({EPSILON})")

    # ---- 合理性 sanity（非核心，帮助定位夹具退化）----
    assert f.input_tokens > c.input_tokens, "全量臂应比编译臂耗更多 input token"
    assert f.input_tokens >= r.input_tokens, "全量臂应不少于最近 N 轮臂"

    print(f"  臂        input_tokens  judge_score  维度分")
    for res in (c, f, r):
        print(f"  {res.name:<9} {res.input_tokens:>10}     {res.score:.4f}      "
              + json.dumps(res.dims, ensure_ascii=False))
    print(f"  -> token 比值 compiled/full = {token_ratio:.4f} "
          f"(阈值 < {TOKEN_RATIO_THRESHOLD})  [OK]")
    print(f"  -> judge_score: compiled {c.score:.4f} >= full {f.score:.4f} - eps {EPSILON}  [OK]")
    print(f"  -> bundle-hash 缓存命中 {cache.hits} 次（编译臂二次评分未再调裁判）")
    print("[PASS] stub 模式：编译臂 token 显著更省且质量不劣于全量臂；缓存生效")


# ---------------------------------------------------------------- live 模式
def _live_enabled() -> bool:
    return bool(os.environ.get("LH_LIVE_LLM")) and bool(os.environ.get("LLM_API_KEY"))


def _build_live_client():
    """用真实 flagship 构造客户端（仅在 live 门控通过时调用）。"""
    from core.llm_client import LLMClientFactory
    import core.providers.openai_client  # noqa: F401 —— 触发 "openai" 注册
    cfg = {
        "api_key": os.environ["LLM_API_KEY"],
        "model": os.environ.get("LH_LIVE_MODEL", "gpt-4o"),
    }
    base_url = os.environ.get("LLM_BASE_URL")
    if base_url:
        cfg["base_url"] = base_url
    # 服务商侧可能排队（reasoning 模型首 token 慢/并发限制），默认 60s 偏紧：
    # LH_LIVE_TIMEOUT（秒）可按服务商实测放宽
    timeout = os.environ.get("LH_LIVE_TIMEOUT")
    if timeout:
        cfg["timeout"] = float(timeout)
    provider = os.environ.get("LH_LIVE_PROVIDER", "openai")
    return LLMClientFactory.create(provider, cfg)


async def _live_call(fn):
    """live 调用的有界重试包装。

    服务商并发拥塞窗口的实测表现：请求排队后连接被网关切断
    （APIConnectionError，约 45s）或客户端超时（APITimeoutError）——
    均为瞬态，重试即可骑过拥塞窗口；鉴权/参数等非瞬态错误立即抛出。
    次数与间隔可用 LH_LIVE_RETRIES（默认 4）/ LH_LIVE_BACKOFF 秒（默认 15）调整。
    """
    import openai
    retries = int(os.environ.get("LH_LIVE_RETRIES", "4"))
    backoff = float(os.environ.get("LH_LIVE_BACKOFF", "15"))
    transient = (openai.APIConnectionError, openai.APITimeoutError,
                 openai.RateLimitError)
    for attempt in range(retries + 1):
        try:
            return await fn()
        except transient as exc:
            if attempt == retries:
                raise
            print(f"  [retry {attempt + 1}/{retries}] 瞬态 {type(exc).__name__}，"
                  f"{backoff:g}s 后重试")
            await asyncio.sleep(backoff)


async def test_ab_eval_live():
    """live 模式三臂 A/B：真实 flagship 作答 + 真实裁判，打印实际指标。

    门控：未设 LH_LIVE_LLM 或无 LLM_API_KEY → 跳过（stub 模式已覆盖机制验证）。
    """
    print("\n=== 第三层 A/B（live 模式） ===")
    if not _live_enabled():
        print("[SKIP] live 未启用（需环境变量 LH_LIVE_LLM 且 LLM_API_KEY）；"
              "stub 模式已确定性行使核心断言")
        return

    events = fl.build_ab_events()
    projection = rebuild(events)
    arms, bundle = _build_arms(projection, events)

    client = _build_live_client()
    rubric = JudgeRubric()
    cache = RubricScoreCache()   # live 同样接 bundle-hash 缓存（避免重复花钱调裁判）
    # 生成封顶：reasoning 模型不受限时倾向输出超长 markdown，实测在部分网关下
    # 生成过久触发空闲切断（服务端照计费、客户端拿不到响应）。LH_LIVE_MAX_TOKENS
    # （如 512）可显著缩短单请求时长；不设则保持模型默认（通用服务商行为不变）。
    max_tokens_env = os.environ.get("LH_LIVE_MAX_TOKENS")
    max_tokens = int(max_tokens_env) if max_tokens_env else None
    results: dict[str, ArmResult] = {}
    for name in ("compiled", "full", "recent"):
        resp = await _live_call(
            lambda: client.complete([{"role": "user", "content": arms[name]}],
                                    max_tokens=max_tokens))
        input_tokens = int((resp.usage or {}).get("prompt_tokens", 0))
        # 与 stub 分支同构：编译臂用 bundle.to_json()（bundle-hash），其余臂用其上下文串
        cache_repr = bundle.to_json() if name == "compiled" else arms[name]
        key = cache.key_for(cache_repr, rubric)
        score, dims = await _live_call(lambda: judge_answer(
            client, rubric, GOAL, resp.content, cache=cache, cache_key=key))
        results[name] = ArmResult(name, input_tokens, score, dims, resp.content)
    print(f"  -> live rubric cache: hits={cache.hits} misses={cache.misses}")

    c, f = results["compiled"], results["full"]
    token_ratio = c.input_tokens / f.input_tokens if f.input_tokens else float("inf")
    print("  live 实际指标：")
    for res in (c, f, results["recent"]):
        print(f"    {res.name:<9} input_tokens={res.input_tokens:>6}  "
              f"judge_score={res.score:.4f}")
    print(f"    token 比值 compiled/full = {token_ratio:.4f}（阈值 < {TOKEN_RATIO_THRESHOLD}）")
    print(f"    judge_score 差 compiled-full = {c.score - f.score:+.4f}（eps={EPSILON}）")

    # token 优势由提示词规模决定，真实模式下同样应成立
    assert c.input_tokens < TOKEN_RATIO_THRESHOLD * f.input_tokens, \
        f"live: 编译臂 token 未达 <{TOKEN_RATIO_THRESHOLD}* 全量臂（{token_ratio:.4f}）"
    # 裁判为真实模型，质量差仅打印告警、不硬失败（避免 live 抖动阻断）
    if c.score < f.score - EPSILON:
        print("  [WARN] live 裁判给出 compiled < full - eps，请人工复核（不阻断）")
    print("[PASS] live 模式跑通并打印真实指标")


# ---------------------------------------------------------------- runner

_TESTS = [
    ("第三层 A/B stub", test_ab_eval_stub),
    ("第三层 A/B live", test_ab_eval_live),
]


async def main() -> int:
    print("=" * 60)
    print("第三层 headless A/B 评估（契约 §6.1）—— 三臂对比 + rubric 裁判")
    print("=" * 60)

    results: list[tuple[str, bool]] = []
    for name, fn in _TESTS:
        try:
            await fn()
            results.append((name, True))
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}")
            results.append((name, False))
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
            results.append((name, False))

    print()
    print("=" * 60)
    print("第三层结果汇总")
    print("=" * 60)
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    all_passed = all(p for _, p in results)
    print()
    print("第三层全部通过!" if all_passed else "第三层存在失败项!")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
