"""§7.3 模拟学习者（Fake Learner）—— 把 §7.2 五场景脚本化为确定性 LHEvent 流。

设计（对齐 fixtures/build_golden.py 的时间纪律，但完全独立、不修改它）:
    - "学习产品的 bug 大多和时间有关，而时间不花真钱就能加速"（§7.3）：
      用**相对固定锚点 AS_OF（day 0）的 offset_days** 描述事件，把数周学习史
      压缩进一个确定性序列——绝不读 wall-clock。
    - 每个场景一个 build_*_events() 构造器，返回按 (timestamp, id) 升序的
      list[LHEvent]；SimulatedLearner 负责把序列灌入 InMemoryEventStore。
    - 层级派生沿用内核规则（见 core/context/projection.derive_level）：
        type 以 "act." 开头 → L2；以 "assertion." 开头 → L3；
        resource.*/interaction.* → L1。本模块只负责**产生事件**，不预判层级。
    - run_classifier_plugin：当某场景需要"分类器插件"调用时用 StubLLMClient
      离线驱动（deterministic，无网络）；不需要 LLM 的场景绝不引入。

只 import core.context.envelope.LHEvent 与 core.stub_llm（测试替身），
绝不 import 也不修改任何既有 core 运行时逻辑之外的东西。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.context.envelope import LHEvent  # noqa: E402

# ---------------------------------------------------------------- 时间锚点
# 与 fixtures/build_golden.py 采用同一锚点约定（day 0），但此处独立定义，
# 保证 harness 自洽、不依赖夹具文件。
AS_OF = 1_760_000_000.0
DAY = 86400.0

# ---------------------------------------------------------------- 作用域
META = "/meta"                 # 导师父作用域
XC = "/meta/xingce"            # 行测课程作用域
CH3 = "/meta/xingce/ch3"       # 行测第三章（中断回归场景）

# ---------------------------------------------------------------- provenance
RECORDER = {"plugin": "dialog-recorder", "version": "1.0.0"}
MODELER = {"plugin": "learner-model-core", "version": "1.0.0"}
RES = {"plugin": "resource-indexer", "version": "1.0.0"}
MATCHER = {"plugin": "resource-matcher", "version": "1.0"}
CLF_V1 = {"plugin": "classifier-quadruple", "version": "1.0.0"}
CLF_V2 = {"plugin": "classifier-quadruple", "version": "2.0.0"}
META_AGENT = {"plugin": "meta-agent", "version": "1.0.0"}


def _ts(offset_days: float) -> float:
    """把相对 day 0 的偏移换算为绝对时间戳（确定性，无时钟）。"""
    return AS_OF + float(offset_days) * DAY


def _mk(offset_days: float, **spec) -> LHEvent:
    """构造一条 LHEvent：timestamp 由 offset_days 派生，其余字段透传信封。"""
    data = dict(spec)
    data["timestamp"] = _ts(offset_days)
    data.setdefault("schema_version", 1)
    return LHEvent.from_dict(data)


# ================================================================ §7.2-1
# 中断回归：学完第三章 → 18 天空档 → "我们继续吧"
#   - 最后一次"复习"发生在 day -19；第三章完成断言在 day -18；随后 18 天空档；
#     day 0 用户回归。否定查询 within_days=18 的窗口 [day-18, day0] 内无
#     reviewed 事件 → 产出"近 18 天未复习第三章"缺失断言。
def build_interrupt_resume_events() -> list[LHEvent]:
    return _sorted([
        _mk(-20.0, id="s1-e1", actor="user", type="interaction.answer", scope=CH3,
            session_ref="sess-ch3", content_ref="ct://ch3/s1-e1", provenance=RECORDER,
            payload={"text": "用户完成第三章 20 道真题，正确率 75%，用时 35 分钟。",
                     "topics": ["第三章"]}),
        _mk(-20.0, id="s1-e2", actor="classifier", type="act.answered", scope=CH3,
            session_ref="sess-ch3", provenance=CLF_V2,
            payload={"act": "answered", "topics": ["第三章"], "ref": "s1-e1",
                     "confidence": 0.75, "summary": "完成第三章 20 道真题作答"}),
        _mk(-19.0, id="s1-e3", actor="classifier", type="act.reviewed", scope=CH3,
            session_ref="sess-ch3", provenance=CLF_V2,
            payload={"act": "reviewed", "topics": ["第三章"], "ref": "s1-e1",
                     "confidence": 0.70, "summary": "复习第三章错题 5 道"}),
        _mk(-18.0, id="s1-e4", actor="modeler", type="assertion.raised", scope=CH3,
            session_ref="sess-ch3", provenance=MODELER,
            payload={"assertion": "第三章已掌握，可进入第四章", "topics": ["第三章"],
                     "confidence": "中", "evidence": ["s1-e2", "s1-e3"]}),
        # ---- 18 天空档：day -18 到 day 0 之间无任何事件 ----
        _mk(0.0, id="s1-e5", actor="user", type="interaction.resume", scope=CH3,
            session_ref="sess-ch3", content_ref="ct://ch3/s1-e5", provenance=RECORDER,
            payload={"text": "我们继续吧。", "topics": ["第三章"]}),
    ])


# ================================================================ §7.2-2
# 跨课程调度：导师 /meta 发起 handoff.initiated → /meta/xingce 接收入口语境。
def build_handoff_events() -> list[LHEvent]:
    return _sorted([
        # 导师侧（父作用域 /meta）的原文与证据——它们**不得**泄漏进接收方 bundle
        _mk(-4.0, id="s2-m1", actor="user", type="interaction.reflect", scope=META,
            session_ref="sess-meta", content_ref="ct://meta/s2-m1", provenance=RECORDER,
            payload={"text": "导师周记：行测整体进度落后，资料分析需专项突破。"}),
        _mk(-4.0, id="s2-m2", actor="classifier", type="act.planned", scope=META,
            session_ref="sess-meta", provenance=CLF_V2,
            payload={"act": "planned", "topics": ["调度"], "ref": "s2-m1",
                     "confidence": 0.6, "summary": "制定跨课程调度：重心转资料分析"}),
        # handoff 发起（父作用域）
        _mk(-3.0, id="s2-h1", actor="agent", type="handoff.initiated", scope=META,
            session_ref="sess-meta", provenance=META_AGENT,
            payload={"from": META, "to": XC, "reason": "资料分析进入专项突破期",
                     "packet": {"assertions": ["s2-m2"],
                                "suggested_start": "资料分析速算限时训练"},
                     "text": "交接：资料分析专项突破，起点为速算限时训练"}),
        # 接收方（子作用域 /meta/xingce）自己的原文与证据
        _mk(-3.0, id="s2-x1", actor="user", type="interaction.answer", scope=XC,
            session_ref="sess-xingce", content_ref="ct://xingce/s2-x1",
            provenance=RECORDER,
            payload={"text": "用户作答资料分析第 5 题，用截位直算完成。",
                     "topics": ["资料分析"]}),
        _mk(-2.0, id="s2-x2", actor="classifier", type="act.answered", scope=XC,
            session_ref="sess-xingce", provenance=CLF_V2,
            payload={"act": "answered", "topics": ["资料分析"], "ref": "s2-x1",
                     "confidence": 0.8, "summary": "完成资料分析第 5 题"}),
    ])


def handoff_packet_from(events: list[LHEvent], event_id: str = "s2-h1") -> dict:
    """从 handoff.initiated 事件派生接收方 query.handoff_packet（导师递条子）。"""
    ev = next(e for e in events if e.id == event_id)
    return {"text": ev.payload.get("text", ""), "source_event_ids": [ev.id]}


# ================================================================ §7.2-3
# 资料生命周期：导入 → 索引 → 因材匹配 → 删除 → 重放验证无痕迹。
def build_resource_lifecycle_events() -> list[LHEvent]:
    return _sorted([
        _mk(-10.0, id="s3-e1", actor="system", type="resource.imported", scope=XC,
            session_ref="sess-xingce", content_ref="res://xingce/drill-A.pdf",
            provenance=RES,
            payload={"text": "导入资料：资料分析刷题册 A", "topics": ["资料分析"]}),
        _mk(-9.0, id="s3-e2", actor="system", type="resource.indexed", scope=XC,
            session_ref="sess-xingce", content_ref="res://xingce/drill-A.pdf",
            provenance=RES,
            payload={"text": "索引完成：刷题册 A 覆盖 8 个知识点", "topics": ["资料分析"]}),
        _mk(-8.0, id="s3-e3", actor="classifier", type="act.matched", scope=XC,
            session_ref="sess-xingce", provenance=MATCHER,
            payload={"act": "matched", "topics": ["资料分析"], "ref": "s3-e1",
                     "confidence": 0.8, "summary": "因材匹配：将刷题册 A 推送给用户"}),
        # 断言依赖被删资源（evidence 同时指向匹配证据与资源原文）→ 删除后经
        # 指针链一并清除，符合 §5.4"派生状态重建不含该资料痕迹"。
        _mk(-7.0, id="s3-e4", actor="modeler", type="assertion.raised", scope=XC,
            session_ref="sess-xingce", provenance=MODELER,
            payload={"assertion": "刷题册 A 与用户当前水平匹配度高", "topics": ["资料分析"],
                     "confidence": "中", "evidence": ["s3-e3", "s3-e1"]}),
        # 无关资料（不同 topic、独立原文）——删除必须精确，不得全域清空
        _mk(-6.5, id="s3-e0", actor="user", type="interaction.answer", scope=XC,
            session_ref="sess-xingce", content_ref="ct://xingce/s3-e0",
            provenance=RECORDER,
            payload={"text": "用户完成一组判断推理定义判断题。", "topics": ["判断推理"]}),
        _mk(-6.0, id="s3-e5", actor="classifier", type="act.answered", scope=XC,
            session_ref="sess-xingce", provenance=CLF_V2,
            payload={"act": "answered", "topics": ["判断推理"], "ref": "s3-e0",
                     "confidence": 0.77, "summary": "完成一组判断推理题"}),
        # 墓碑：content_ref + provenance 双路径登记
        _mk(-5.0, id="s3-del", actor="user", type="resource.deleted", scope=XC,
            session_ref="sess-xingce", content_ref="res://xingce/drill-A.pdf",
            provenance=RES,
            payload={"deleted_refs": ["res://xingce/drill-A.pdf", "ct://xingce/s3-e1"],
                     "deleted_provenance": {"plugin": "resource-matcher", "version": "1.0"},
                     "reason": "刷题册 A 下架，清理其匹配痕迹"}),
    ])


# ================================================================ §7.2-4
# 分类器升级：同批 L1，先 v1 产 L2/L3，再 v2（追加非替换）；登记 v1 provenance
# 墓碑后，重放翻新——新 bundle 用 v2 断言，旧 v1 证据不出现，原文 L1 保留。
def build_classifier_upgrade_events() -> list[LHEvent]:
    return _sorted([
        # 共享 L1 原文（升级前后都在，重放翻新绝不毁原文）
        _mk(-10.0, id="s4-l1", actor="user", type="interaction.answer", scope=XC,
            session_ref="sess-xingce", content_ref="ct://xingce/s4-l1",
            provenance=RECORDER,
            payload={"text": "用户作答资料分析第 12 题，用截位直算 98 秒完成。",
                     "topics": ["资料分析"]}),
        # v1 分类器产出（L2 证据 + L3 断言）
        _mk(-9.0, id="s4-v1a", actor="classifier", type="act.answered", scope=XC,
            session_ref="sess-xingce", provenance=CLF_V1,
            payload={"act": "answered", "topics": ["资料分析"], "ref": "s4-l1",
                     "confidence": 0.60, "summary": "v1 评估：作答资料分析第 12 题"}),
        _mk(-9.0, id="s4-v1b", actor="classifier", type="assertion.raised", scope=XC,
            session_ref="sess-xingce", provenance=CLF_V1,
            payload={"assertion": "v1 评估：资料分析初步掌握", "topics": ["资料分析"],
                     "confidence": "低", "evidence": ["s4-v1a"]}),
        # v2 分类器产出（追加，不替换 v1 事件——append-only）
        _mk(-3.0, id="s4-v2a", actor="classifier", type="act.answered", scope=XC,
            session_ref="sess-xingce", provenance=CLF_V2,
            payload={"act": "answered", "topics": ["资料分析"], "ref": "s4-l1",
                     "confidence": 0.85, "summary": "v2 评估：截位直算完成第 12 题"}),
        _mk(-3.0, id="s4-v2b", actor="classifier", type="assertion.raised", scope=XC,
            session_ref="sess-xingce", provenance=CLF_V2,
            payload={"assertion": "v2 评估：资料分析熟练掌握", "topics": ["资料分析"],
                     "confidence": "高", "evidence": ["s4-v2a"]}),
        # v1 provenance 墓碑：清除全部 v1 派生（L2+L3），保留 v2 与原文
        _mk(-2.0, id="s4-tomb", actor="system", type="resource.deleted", scope=XC,
            session_ref="sess-xingce", provenance=RES,
            payload={"deleted_refs": [],
                     "deleted_provenance": {"plugin": "classifier-quadruple",
                                            "version": "1.0.0"},
                     "reason": "分类器升级：v1.0.0 产出作废，重放翻新"}),
    ])


# ================================================================ §7.2-5
# 用户争议：L3 断言 + 用户反驳事件；编译器只"呈现"，置信度演化属建模插件。
def build_dispute_events() -> list[LHEvent]:
    return _sorted([
        _mk(-6.0, id="s5-a1", actor="modeler", type="assertion.raised", scope=XC,
            session_ref="sess-xingce", provenance=MODELER,
            payload={"assertion": "资料分析速算稳定，正确率高", "topics": ["资料分析"],
                     "confidence": "高", "evidence": []}),
        _mk(-2.0, id="s5-d1", actor="user", type="assertion.disputed", scope=XC,
            session_ref="sess-xingce", provenance=RECORDER,
            payload={"text": "用户反驳：我其实用的是估算不是速算，正确率也没那么高。",
                     "target_assertion": "s5-a1", "ref": "s5-a1",
                     "topics": ["资料分析"]}),
    ])


def build_dispute_events_without_dispute() -> list[LHEvent]:
    """对照组：无争议事件（用于断言争议是 additive、非破坏性）。"""
    return [e for e in build_dispute_events() if e.id != "s5-d1"]


# ================================================================ §6.1 A/B 素材
# 一段"数周学习史"：大量 verbose L1 原文 + 中等 L2 证据 + 少量精炼 L3 断言。
# 用于第三层三臂对比：编译臂（紧预算只装 L3）vs 全量臂（所有事件文本）vs
# 最近 N 轮臂。verbose L1 保证全量臂 input_tokens 远大于编译臂。
_AB_TOPICS = ["资料分析", "判断推理", "数量关系", "言语理解"]


def build_ab_events(n_rounds: int = 12) -> list[LHEvent]:
    events: list[LHEvent] = []
    day = -30.0
    for k in range(1, n_rounds + 1):
        topic = _AB_TOPICS[(k - 1) % len(_AB_TOPICS)]
        l1_id = f"ab-l1-{k:02d}"
        l2_id = f"ab-l2-{k:02d}"
        # verbose L1 原文（模拟逐字对话转录，撑大全量臂体积）
        events.append(_mk(day, id=l1_id, actor="user", type="interaction.answer",
                          scope=XC, session_ref="sess-ab",
                          content_ref=f"ct://xingce/{l1_id}", provenance=RECORDER,
                          payload={"text": (
                              f"第 {k} 轮{topic}练习逐字转录：用户先读题约三十秒，"
                              f"随后口述解题思路，逐句分析材料中的关键数据与陷阱选项，"
                              f"中途自我纠正了两次，最终给出答案并解释了排除其它选项的理由，"
                              f"整段作答约用时三分钟，涵盖了本知识点的完整推理链条。"),
                              "topics": [topic]}))
        events.append(_mk(day, id=l2_id, actor="classifier", type="act.answered",
                          scope=XC, session_ref="sess-ab", provenance=CLF_V2,
                          payload={"act": "answered", "topics": [topic], "ref": l1_id,
                                   "confidence": round(0.55 + 0.03 * k, 2),
                                   "summary": f"{topic}第 {k} 轮作答证据摘要"}))
        day += 2.0
        # 每 4 轮沉淀一条精炼 L3 断言
        if k % 4 == 0:
            events.append(_mk(day, id=f"ab-l3-{k:02d}", actor="modeler",
                              type="assertion.raised", scope=XC, session_ref="sess-ab",
                              provenance=MODELER,
                              payload={"assertion": f"{topic}阶段掌握度上升",
                                       "topics": [topic], "confidence": "中",
                                       "evidence": [l2_id]}))
            day += 1.0
    return _sorted(events)


# ================================================================ 分类器插件（可选 LLM）
async def run_classifier_plugin(client, *, raw_text: str, scope: str, provenance: dict,
                                ref_id: str, event_id: str, offset_days: float,
                                topics: list[str] | None = None) -> LHEvent:
    """用（stub）LLM 客户端模拟一次"四元组分类器插件"调用，产出 L2 证据事件。

    client 需实现 complete_json（StubLLMClient 即可，离线确定性）。返回的
    LHEvent 由分类结果 + 传入信封字段组装；不读时钟、不联网。
    """
    resp = await client.complete_json([
        {"role": "system",
         "content": "你是四元组学习行为分类器，仅输出 JSON："
                    '{"act": str, "topics": [str], "confidence": float, "summary": str}'},
        {"role": "user", "content": raw_text},
    ])
    if isinstance(resp, list):  # 防御：complete_json 可能返回 list
        resp = resp[0] if resp else {}
    act = str(resp.get("act") or "answered")
    ev_topics = list(resp.get("topics") or topics or [])
    return _mk(offset_days, id=event_id, actor="classifier", type=f"act.{act}",
               scope=scope, session_ref="sess-xingce", provenance=provenance,
               payload={"act": act, "topics": ev_topics, "ref": ref_id,
                        "confidence": resp.get("confidence"),
                        "summary": resp.get("summary", "")})


# ================================================================ 模拟学习者
def _sorted(events: list[LHEvent]) -> list[LHEvent]:
    """按 (timestamp, id) 升序（与内核 rebuild 的确定性顺序一致）。"""
    return sorted(events, key=lambda e: (e.timestamp, e.id))


# 场景名 → 构造器（§7.2 五场景 + A/B 素材）
SCENARIOS = {
    "interrupt_resume": build_interrupt_resume_events,
    "handoff": build_handoff_events,
    "resource_lifecycle": build_resource_lifecycle_events,
    "classifier_upgrade": build_classifier_upgrade_events,
    "dispute": build_dispute_events,
    "ab_history": build_ab_events,
}


class SimulatedLearner:
    """§7.3 模拟学习者：把脚本化的数周事件"压缩灌入"内存事件存储。

    用法:
        learner = SimulatedLearner("interrupt_resume")
        store = learner.feed()            # 返回灌满事件的 InMemoryEventStore
        events = learner.events           # 确定性事件序列
    """

    def __init__(self, scenario: str, events: list[LHEvent] | None = None):
        if events is None:
            if scenario not in SCENARIOS:
                raise KeyError(f"未知场景: {scenario}；可选 {sorted(SCENARIOS)}")
            events = SCENARIOS[scenario]()
        self.scenario = scenario
        self.events = _sorted(list(events))

    def feed(self, store=None):
        """把事件按序灌入 InMemoryEventStore（append-only）；返回该 store。"""
        from core.context.store import InMemoryEventStore
        store = store if store is not None else InMemoryEventStore()
        for ev in self.events:
            store.append(ev)
        return store

    def __len__(self) -> int:
        return len(self.events)
