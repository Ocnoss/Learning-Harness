"""第二层黄金场景测试（契约 §7.2）—— 五场景端到端，全程内存 store。

默认无 LLM / 无网络 / 无前端：所有场景把 §7.2 脚本灌入 InMemoryEventStore，
rebuild → compile，对编译产物做断言。仅"分类器升级"场景演示一次分类器插件
调用，用 StubLLMClient（离线确定性），其余场景绝不引入 LLM。

运行方式（两种等价）:
    D:\\miniconda\\python.exe tests\\context_compiler\\test_layer2_scenarios.py
    （未来装了 pytest + pytest-asyncio 后）pytest test_layer2_scenarios.py

场景 → 测试函数 → 断言映射:
    §7.2-1 中断回归   → test_scenario_interrupt_resume
    §7.2-2 跨课程 handoff → test_scenario_cross_course_handoff
    §7.2-3 资料生命周期 → test_scenario_resource_lifecycle
    §7.2-4 分类器升级  → test_scenario_classifier_upgrade
    §7.2-5 用户争议   → test_scenario_user_dispute
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_REPO_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.context import (  # noqa: E402
    InMemoryEventStore,
    rebuild,
    fold,
    compile,
    ContextQuery,
    NegationQuery,
)
from harness import fake_learner as fl  # noqa: E402

HUGE = 100_000


def _q(scope: str, **kwargs) -> ContextQuery:
    """默认全额预算、锚定 AS_OF 的查询构造器。"""
    defaults = dict(scope=scope, goal="", budget_tokens=HUGE, as_of=fl.AS_OF)
    defaults.update(kwargs)
    return ContextQuery(**defaults)


# ---------------------------------------------------------------- §7.2-1

async def test_scenario_interrupt_resume():
    """中断回归：学完第3章 → 18 天空档 → "继续吧"。

    断言：
      (a) 编译包含第3章 L3 断言；
      (b) query 携带否定查询时产出"近 18 天未复习第3章"缺失断言
          （provenance=context-compiler/negation，source_event_ids=()，level=3）；
      (c) 不带否定查询时无缺失断言（对照）；
      (d) 编译确定性（同 query 逐字节一致）。
    """
    print("\n=== §7.2-1 中断回归 ===")
    events = fl.build_interrupt_resume_events()
    store = fl.SimulatedLearner("interrupt_resume", events).feed()
    assert isinstance(store, InMemoryEventStore) and len(store) == len(events)
    projection = rebuild(store)

    # (a) 含第3章 L3 断言
    bundle = compile(projection, _q(fl.CH3, goal="第三章 继续", allow_l1_deref=True))
    l3_texts = [i.text for i in bundle.items if i.level == 3 and i.event_id]
    assert any("第三章已掌握" in t for t in l3_texts), f"缺少第3章 L3 断言: {l3_texts}"

    # (b) 否定查询产出"18 天未复习第3章"缺失断言
    neg = NegationQuery(type="act.reviewed", topic="第三章", within_days=18,
                        label="复习第三章")
    bn = compile(projection, _q(fl.CH3, goal="第三章 继续", allow_l1_deref=True,
                                negations=(neg,)))
    negs = [i for i in bn.items
            if i.provenance.get("source") == "context-compiler/negation"]
    assert len(negs) == 1, f"应产出 1 条缺失断言，实际 {len(negs)}"
    miss = negs[0]
    assert miss.source_event_ids == (), "缺失断言不应有出处事件"
    assert miss.level == 3, "缺失断言属断言层"
    assert "缺失断言" in miss.text and "18" in miss.text and "复习第三章" in miss.text, \
        f"缺失断言文本不符: {miss.text}"

    # (c) 对照：不带否定查询则无缺失断言
    assert not [i for i in bundle.items
                if i.provenance.get("source") == "context-compiler/negation"], \
        "无否定查询时不应出现缺失断言"

    # (d) 确定性
    assert compile(projection, _q(fl.CH3, goal="第三章 继续", allow_l1_deref=True,
                                  negations=(neg,))).to_json() == bn.to_json()

    print(f"[PASS] 第3章 L3 断言在包中；18 天未复习缺失断言正确产出；确定性一致")


# ---------------------------------------------------------------- §7.2-2

async def test_scenario_cross_course_handoff():
    """跨课程 handoff：/meta 发起 → /meta/xingce 接收。

    断言：
      (a) 接收方 bundle 含 handoff 包 L3（provenance source=handoff_packet）；
      (b) 接收方 bundle 不含 /meta 的 L1/L2 原文（作用域不泄漏）；
      (c) 接收方自己的 L1/L2 正常装入；
      (d) 对照：父域 /meta 编译能看到自己的 L1/L2。
    """
    print("\n=== §7.2-2 跨课程 handoff ===")
    events = fl.build_handoff_events()
    projection = rebuild(events)
    packet = fl.handoff_packet_from(events)

    bundle = compile(projection, _q(fl.XC, allow_l1_deref=True, handoff_packet=packet))

    # (a) handoff 包 L3
    ho = [i for i in bundle.items if i.provenance == {"source": "handoff_packet"}]
    assert len(ho) == 1, f"应有 1 条 handoff 包条目，实际 {len(ho)}"
    assert ho[0].level == 3, "handoff 包属断言层"
    assert "资料分析专项突破" in ho[0].text, f"handoff 文本不符: {ho[0].text}"

    # (b) 父域 /meta 的 L1/L2 原文不泄漏
    ids = {i.event_id for i in bundle.items}
    for item in bundle.items:
        assert item.scope != fl.META, f"父域原文泄漏进接收方包: {item.event_id}"
    assert not ({"s2-m1", "s2-m2"} & ids), f"父域条目泄漏: {({'s2-m1','s2-m2'} & ids)}"
    texts = " ".join(i.text for i in bundle.items)
    assert "导师周记" not in texts, "父域原文文本泄漏"

    # (c) 接收方自己的 L1/L2 装入
    assert "s2-x2" in ids, "接收方 L2 证据应在包中"
    assert "s2-x1" in ids, "接收方 L1 原文（deref）应在包中"

    # (d) 对照：父域编译能看到自己的原文
    pm = compile(projection, _q(fl.META, allow_l1_deref=True))
    mid = {i.event_id for i in pm.items}
    assert {"s2-m1", "s2-m2"} <= mid, "父域应能编译到自己的 L1/L2"

    print("[PASS] handoff 包以 L3 进接收方包；父域原文零泄漏；接收方自身条目正常")


# ---------------------------------------------------------------- §7.2-3

async def test_scenario_resource_lifecycle():
    """资料生命周期：导入 → 索引 → 匹配 → 删除 → 重编。

    断言：
      (a) 删除前：被删资源 content_ref 与匹配证据可见；
      (b) 删除后：content_ref 痕迹为 0、matcher provenance 痕迹为 0、
          依赖被删资源的派生断言经指针链一并清除；
      (c) 墓碑登记正确（content_ref 集合 + provenance 组合）；
      (d) 无关资料存活（删除精确，非全域清空）。
    """
    print("\n=== §7.2-3 资料生命周期 ===")
    events = fl.build_resource_lifecycle_events()
    del_ts = next(e.timestamp for e in events if e.id == "s3-del")

    # (a) 删除前：痕迹存在
    p_pre = rebuild([e for e in events if e.timestamp < del_ts])
    b_pre = compile(p_pre, _q(fl.XC, allow_l1_deref=True))
    assert any(i.content_ref == "res://xingce/drill-A.pdf" for i in b_pre.items), \
        "删除前应能看到资源 content_ref"
    pre_ids = {i.event_id for i in b_pre.items}
    assert {"s3-e1", "s3-e3"} <= pre_ids, "删除前应能看到导入与匹配证据"

    # 删除后（全量重放含墓碑）
    p_full = rebuild(events)
    # (c) 墓碑登记
    assert "res://xingce/drill-A.pdf" in p_full.tombstone_refs
    assert (("plugin", "resource-matcher"), ("version", "1.0")) in p_full.tombstone_provenance, \
        "matcher provenance 墓碑未登记"

    b_full = compile(p_full, _q(fl.XC, allow_l1_deref=True))
    # (b) content_ref 痕迹为 0
    assert not any(i.content_ref == "res://xingce/drill-A.pdf" for i in b_full.items), \
        "被删资源 content_ref 痕迹残留"
    # (b) matcher provenance 痕迹为 0
    for item in b_full.items:
        assert not (str(item.provenance.get("plugin")) == "resource-matcher"
                    and str(item.provenance.get("version")) == "1.0"), \
            f"matcher provenance 痕迹残留: {item.event_id}"
    full_ids = {i.event_id for i in b_full.items}
    assert not ({"s3-e1", "s3-e2", "s3-e3", "s3-e4"} & full_ids), \
        f"被删资源派生痕迹仍在包中: {({'s3-e1','s3-e2','s3-e3','s3-e4'} & full_ids)}"

    # (d) 无关资料存活
    assert {"s3-e0", "s3-e5"} <= full_ids, "无关资料被误删（删除应精确）"

    print("[PASS] content_ref + provenance 双路径删除无痕迹；指针链清除派生断言；无关资料存活")


# ---------------------------------------------------------------- §7.2-4

async def test_scenario_classifier_upgrade():
    """分类器升级：同批 L1，v1 与 v2 追加共存；登记 v1 provenance 墓碑后翻新。

    断言：
      (a) 墓碑前：v1 与 v2 断言并存（追加非替换）；
      (b) 墓碑后：新 bundle 用 v2 断言，旧 v1 断言/证据不出现，v1 provenance 零残留；
      (c) 共享 L1 原文保留（重放翻新不毁原文）；
      (d) 分类器插件调用（StubLLMClient，离线确定性）可产出可折叠的 L2 事件。
    """
    print("\n=== §7.2-4 分类器升级 ===")
    from core.stub_llm import StubLLMClient

    events = fl.build_classifier_upgrade_events()
    tomb_ts = next(e.timestamp for e in events if e.id == "s4-tomb")

    # (a) 墓碑前：v1 + v2 并存
    p_pre = rebuild([e for e in events if e.timestamp < tomb_ts])
    b_pre = compile(p_pre, _q(fl.XC, goal="资料分析"))
    pre_texts = " ".join(i.text for i in b_pre.items)
    assert "v1 评估：资料分析初步掌握" in pre_texts, "墓碑前应含 v1 断言"
    assert "v2 评估：资料分析熟练掌握" in pre_texts, "墓碑前应含 v2 断言（追加非替换）"

    # (b)(c) 墓碑后：翻新
    p_full = rebuild(events)
    b_full = compile(p_full, _q(fl.XC, goal="资料分析", allow_l1_deref=True))
    full_texts = " ".join(i.text for i in b_full.items)
    assert "v2 评估：资料分析熟练掌握" in full_texts, "翻新后应保留 v2 断言"
    assert "v1 评估" not in full_texts, "翻新后不应出现 v1 断言"
    for item in b_full.items:
        assert not (str(item.provenance.get("plugin")) == "classifier-quadruple"
                    and str(item.provenance.get("version")) == "1.0.0"), \
            f"v1 证据残留: {item.event_id}"
    full_ids = {i.event_id for i in b_full.items}
    assert not ({"s4-v1a", "s4-v1b"} & full_ids), "v1 派生条目未清除"
    assert "s4-l1" in full_ids, "共享 L1 原文应保留（翻新不毁原文）"

    # (d) 分类器插件调用（StubLLMClient 离线确定性）
    stub = StubLLMClient({"responses": [
        '{"act": "answered", "topics": ["资料分析"], '
        '"confidence": 0.9, "summary": "stub 分类：截位直算"}',
    ]})
    l1 = next(e for e in events if e.id == "s4-l1")
    produced = await fl.run_classifier_plugin(
        stub, raw_text=l1.payload["text"], scope=fl.XC, provenance=fl.CLF_V2,
        ref_id="s4-l1", event_id="s4-stub", offset_days=-1.0)
    assert produced.type == "act.answered", f"分类器应产 act.* 事件: {produced.type}"
    assert produced.topics == ("资料分析",)
    assert stub.call_history, "分类器插件应发生一次 stub 调用"
    # stub 产出可折叠进投影（离线、确定性）
    p_stub = fold(p_full, produced)
    assert "s4-stub" in p_stub.item_by_event_id
    assert p_stub.item_by_event_id["s4-stub"].level == 2

    print("[PASS] v1/v2 追加共存 → v1 墓碑翻新（v2 保留、v1 零残留、原文不毁）；stub 分类器可用")


# ---------------------------------------------------------------- §7.2-5

async def test_scenario_user_dispute():
    """用户争议：L3 断言 + 用户反驳事件。

    编译器只"呈现"，置信度演化属建模插件——本测试断言"存在 + 出处"，
    绝不断言置信度算术：
      (a) 原断言与用户反驳都被纳入 bundle；
      (b) 反驳由 user 发起，出处可审计（source_event_ids 指向被争议断言）；
      (c) 争议事件是 additive/非破坏性：原断言文本与"无争议"投影逐字节一致。
    """
    print("\n=== §7.2-5 用户争议 ===")
    events = fl.build_dispute_events()
    projection = rebuild(events)
    bundle = compile(projection, _q(fl.XC, goal="资料分析"))
    ids = {i.event_id for i in bundle.items}

    # (a) 原断言 + 反驳都在
    assert {"s5-a1", "s5-d1"} <= ids, f"断言与反驳应同时纳入: {ids}"

    # (b) 反驳出处可审计
    d = projection.item_by_event_id["s5-d1"]
    assert d.actor == "user", "反驳事件应由 user 发起"
    disp = next(i for i in bundle.items if i.event_id == "s5-d1")
    assert disp.level == 3, "争议记录属断言层"
    assert "s5-a1" in disp.source_event_ids, "反驳应指向被争议断言（可审计出处）"
    assert "用户反驳" in disp.text

    # (c) 非破坏性：原断言文本与"无争议"投影逐字节一致（不重算置信度）
    p_no = rebuild(fl.build_dispute_events_without_dispute())
    b_no = compile(p_no, _q(fl.XC, goal="资料分析"))
    a1_no = next(i for i in b_no.items if i.event_id == "s5-a1")
    a1_with = next(i for i in bundle.items if i.event_id == "s5-a1")
    assert a1_no.text == a1_with.text, "争议事件不应改写原断言（置信度演化属建模插件）"

    print("[PASS] 断言+反驳均纳入；反驳出处可审计；原断言非破坏性呈现（不涉置信度算术）")


# ---------------------------------------------------------------- runner

_TESTS = [
    ("§7.2-1 中断回归", test_scenario_interrupt_resume),
    ("§7.2-2 跨课程 handoff", test_scenario_cross_course_handoff),
    ("§7.2-3 资料生命周期", test_scenario_resource_lifecycle),
    ("§7.2-4 分类器升级", test_scenario_classifier_upgrade),
    ("§7.2-5 用户争议", test_scenario_user_dispute),
]


async def main() -> int:
    print("=" * 60)
    print("第二层黄金场景测试（契约 §7.2）—— 全程内存 store，默认无 LLM/网络/前端")
    print("=" * 60)

    results: list[tuple[str, bool]] = []
    for name, fn in _TESTS:
        try:
            await fn()
            results.append((name, True))
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}")
            results.append((name, False))
        except Exception as exc:  # noqa: BLE001 —— 场景异常也计入失败
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
            results.append((name, False))

    print()
    print("=" * 60)
    print("第二层结果汇总")
    print("=" * 60)
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    all_passed = all(p for _, p in results)
    print()
    print("第二层全部通过!" if all_passed else "第二层存在失败项!")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
