"""黄金事件日志夹具生成器 + 加载器（契约 §7.2/§7.3 的确定性数据底座）。

设计:
    - 事件用**相对时间戳 offset_days** 描述，加载时锚定固定 AS_OF（day 0），
      保证确定性与否定查询稳定（不依赖真实时钟）。
    - 覆盖面: ≥2 个 scope（父 /meta、子 /meta/xingce、无关 /meta/shenglun）、
      L1/L2/L3 各若干（L2 带 topics）、resource.deleted 墓碑（分别按
      content_ref 与 provenance 命中）、handoff 事件、schema_version 0/1 混合。

用法:
    python tests/context_compiler/fixtures/build_golden.py   # 重写 golden_log.jsonl
    from build_golden import load_golden, AS_OF               # 测试内加载
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.context.envelope import LHEvent  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_log.jsonl"

# 时间锚点：day 0。所有事件 timestamp = AS_OF + offset_days * 86400。
AS_OF = 1_760_000_000.0
DAY = 86400.0

# 分类器两版本：v1.0.0 的产出会被墓碑按 provenance 清除（模拟分类器升级/
# 资料删除后的痕迹清理），v2.0.0 的产出保留。
_CLF_V1 = {"plugin": "classifier-quadruple", "version": "1.0.0"}
_CLF_V2 = {"plugin": "classifier-quadruple", "version": "2.0.0"}
_MODELER = {"plugin": "learner-model-core", "version": "1.0.0"}
_RECORDER = {"plugin": "dialog-recorder", "version": "1.0.0"}
_RES = {"plugin": "resource-indexer", "version": "1.0.0"}
_AUTOIDX = {"plugin": "auto-indexer", "version": "0.9.0"}
_META_AGENT = {"plugin": "meta-agent", "version": "1.0.0"}

_XC = "/meta/xingce"
_SL = "/meta/shenglun"
_SX = "sess-xingce-01"
_SS = "sess-shenglun-01"
_SM = "sess-meta-01"

# (offset_days, 事件字典)。id 全局唯一；schema_version 0/1 混合。
_EVENT_SPECS: list[tuple[float, dict]] = [
    # ---- /meta/xingce：资料导入 + 早期交互（旧分类器 v1.0.0） ----
    (-30.0, dict(id="e-001", schema_version=0, actor="system", type="resource.imported",
                 scope=_XC, session_ref=_SX, content_ref="res://xingce/fenxi-1.pdf",
                 provenance=_RES,
                 payload={"text": "导入资料：行测资料分析真题集（旧版）"})),
    (-29.0, dict(id="e-002", schema_version=0, actor="user", type="interaction.answer",
                 scope=_XC, session_ref=_SX, content_ref="ct://xingce/e-002",
                 provenance=_RECORDER,
                 payload={"text": "用户作答：第3题选B， growth rate 估算用了直除法。",
                          "topics": ["资料分析"]})),
    (-29.0, dict(id="e-003", schema_version=0, actor="classifier", type="act.answered",
                 scope=_XC, session_ref=_SX, provenance=_CLF_V1,
                 payload={"act": "answered", "topics": ["资料分析"], "ref": "e-002",
                          "confidence": 0.62, "summary": "作答资料分析第3题（直除法）"})),
    (-25.0, dict(id="e-004", schema_version=0, actor="user", type="interaction.answer",
                 scope=_XC, session_ref=_SX, content_ref="ct://xingce/e-004",
                 provenance=_RECORDER,
                 payload={"text": "用户作答：第7题选C，比较增长率时看错了基期。",
                          "topics": ["资料分析"]})),
    (-25.0, dict(id="e-005", schema_version=0, actor="classifier", type="act.answered",
                 scope=_XC, session_ref=_SX, provenance=_AUTOIDX,
                 payload={"act": "answered", "topics": ["资料分析"], "ref": "e-004",
                          "confidence": 0.58, "summary": "作答第7题（基期看错）"})),
    (-24.0, dict(id="e-006", schema_version=0, actor="modeler", type="assertion.raised",
                 scope=_XC, session_ref=_SX, provenance=_MODELER,
                 payload={"assertion": "资料分析速算稳定性不足，基期概念易混",
                          "topics": ["资料分析"], "confidence": "中",
                          "evidence": ["e-003", "e-005"]})),
    # ---- 中断 10 天（§7.2-1 中断回归素材：-24 → -14 无事件） ----
    (-14.0, dict(id="e-007", schema_version=0, actor="user", type="interaction.resume",
                 scope=_XC, session_ref=_SX, content_ref="ct://xingce/e-007",
                 provenance=_RECORDER,
                 payload={"text": "用户：我们继续吧。", "topics": ["资料分析"]})),
    (-14.0, dict(id="e-008", schema_version=0, actor="system", type="resource.imported",
                 scope=_XC, session_ref=_SX, content_ref="res://xingce/fenxi-drill.pdf",
                 provenance=_RES,
                 payload={"text": "导入资料：资料分析专项刷题册",
                          "topics": ["资料分析"]})),
    (-13.0, dict(id="e-009", schema_version=0, actor="classifier", type="act.answered",
                 scope=_XC, session_ref=_SX, provenance=_CLF_V1,
                 payload={"act": "answered", "topics": ["资料分析"], "ref": "e-007",
                          "confidence": 0.71, "summary": "续学后完成一组速算题"})),
    (-12.0, dict(id="e-010", schema_version=1, actor="classifier", type="act.interpreted",
                 scope=_XC, session_ref=_SX, provenance=_CLF_V1,
                 payload={"act": "interpreted", "topics": ["资料分析"], "ref": "e-007",
                          "confidence": 0.66, "summary": "解释了基期与现期的区别",
                          "depth": "restructure", "spontaneity": "prompted"})),
    (-11.5, dict(id="e-011", schema_version=1, actor="modeler", type="assertion.raised",
                 scope=_XC, session_ref=_SX, provenance=_MODELER,
                 payload={"assertion": "基期概念重构中，需要更多 spontaneous 证据",
                          "topics": ["资料分析"], "confidence": "中",
                          "evidence": ["e-009", "e-010"]})),
    (-11.0, dict(id="e-012", schema_version=0, actor="agent", type="strategy.predicted",
                 scope=_XC, session_ref=_SX, provenance={"plugin": "strategy-pack", "version": "1.0.0"},
                 payload={"text": "预测：两周内对基期问题产出无提示 explain",
                          "target_assertion": "e-011", "window_days": 14})),
    # ---- 墓碑 1：按 content_ref 删除旧资料（连带 e-002 的 L1 原文） ----
    (-11.0, dict(id="e-013", schema_version=0, actor="user", type="resource.deleted",
                 scope=_XC, session_ref=_SX, content_ref="res://xingce/fenxi-1.pdf",
                 provenance=_RES,
                 payload={"deleted_refs": ["res://xingce/fenxi-1.pdf",
                                           "ct://xingce/e-002",
                                           "ct://xingce/e-004"],
                          "reason": "资料改版，删除旧版（连带其交互原文）"})),
    # ---- 新交互（新分类器 v2.0.0，产出不被墓碑清除） ----
    (-10.0, dict(id="e-014", schema_version=1, actor="user", type="interaction.answer",
                 scope=_XC, session_ref=_SX, content_ref="ct://xingce/e-014",
                 provenance=_RECORDER,
                 payload={"text": "用户作答：刷题册第12题，用截位直算 98 秒完成。",
                          "topics": ["资料分析"], "depth": "recall"})),
    (-10.0, dict(id="e-015", schema_version=1, actor="classifier", type="act.answered",
                 scope=_XC, session_ref=_SX, provenance=_CLF_V2,
                 payload={"act": "answered", "topics": ["资料分析"], "ref": "e-014",
                          "confidence": 0.83, "summary": "截位直算完成第12题",
                          "depth": "recall", "spontaneity": "prompted"})),
    (-9.0, dict(id="e-016", schema_version=1, actor="classifier", type="act.explained",
                 scope=_XC, session_ref=_SX, provenance=_CLF_V2,
                 payload={"act": "explained", "topics": ["判断推理"], "ref": "e-014",
                          "confidence": 0.77, "summary": "自发讲解了削弱型题目思路",
                          "depth": "transfer", "spontaneity": "spontaneous"})),
    (-5.0, dict(id="e-017", schema_version=1, actor="modeler", type="assertion.raised",
                 scope=_XC, session_ref=_SX, provenance=_MODELER,
                 payload={"assertion": "资料分析速算正确率上升；判断推理出现自发 explain",
                          "topics": ["资料分析", "判断推理"], "confidence": "高",
                          "evidence": ["e-015", "e-016"]})),
    # ---- 墓碑 2：按 provenance 清除旧分类器 v1.0.0 的全部派生 ----
    (-4.0, dict(id="e-019", schema_version=1, actor="system", type="resource.deleted",
                 scope=_XC, session_ref=_SX, provenance=_RES,
                 payload={"deleted_refs": [],
                          "deleted_provenance": {"plugin": "classifier-quadruple",
                                                 "version": "1.0.0"},
                          "reason": "分类器升级：v1.0.0 产出作废，重放翻新"})),
    # ---- /meta/shenglun：无关子作用域（专测作用域不泄漏） ----
    (-20.0, dict(id="e-020", schema_version=0, actor="user", type="interaction.answer",
                 scope=_SL, session_ref=_SS, content_ref="ct://shenglun/e-020",
                 provenance=_RECORDER,
                 payload={"text": "用户提交申论大作文提纲：乡村振兴主题。",
                          "topics": ["申论大作文"]})),
    (-20.0, dict(id="e-021", schema_version=0, actor="classifier", type="act.answered",
                 scope=_SL, session_ref=_SS, provenance=_CLF_V2,
                 payload={"act": "answered", "topics": ["申论大作文"], "ref": "e-020",
                          "confidence": 0.69, "summary": "提交大作文提纲"})),
    (-18.0, dict(id="e-022", schema_version=0, actor="modeler", type="assertion.raised",
                 scope=_SL, session_ref=_SS, provenance=_MODELER,
                 payload={"assertion": "申论大作文结构完整但论证素材偏少",
                          "topics": ["申论大作文"], "confidence": "中",
                          "evidence": ["e-021"]})),
    (-6.0, dict(id="e-023", schema_version=1, actor="user", type="interaction.answer",
                 scope=_SL, session_ref=_SS, content_ref="ct://shenglun/e-023",
                 provenance=_RECORDER,
                 payload={"text": "用户修改提纲：补充了三则案例素材。",
                          "topics": ["申论大作文"]})),
    # ---- /meta 父作用域：导师侧（handoff + 宏观断言） ----
    (-3.0, dict(id="e-024", schema_version=1, actor="agent", type="handoff.initiated",
                 scope="/meta", session_ref=_SM, provenance=_META_AGENT,
                 payload={"from": "/meta", "to": _XC,
                          "reason": "资料分析进入专项突破期",
                          "packet": {"assertions": ["e-017"],
                                     "suggested_start": "速算限时训练"},
                          "text": "交接：资料分析专项突破，起点为速算限时训练"})),
    (-2.0, dict(id="e-025", schema_version=1, actor="agent", type="assertion.raised",
                 scope="/meta", session_ref=_SM, provenance=_MODELER,
                 payload={"assertion": "整体学习节奏恢复，行测进入上行通道",
                          "confidence": "中", "evidence": ["e-017", "e-022"]})),
    (-1.0, dict(id="e-026", schema_version=0, actor="agent", type="session.planned",
                 scope="/meta", session_ref=_SM, provenance=_META_AGENT,
                 payload={"text": "计划：本周三次行测限时训练 + 一次申论提纲练习"})),
    # ---- 否定查询锚点：/meta/xingce 最后一次 asked 在 -20（7 天窗口外） ----
    (-20.0, dict(id="e-027", schema_version=0, actor="user", type="act.asked",
                 scope=_XC, session_ref=_SX, provenance=_CLF_V2,
                 payload={"act": "asked", "topics": ["数量关系"],
                          "summary": "问了行程问题的相遇模型", "confidence": 0.8})),
    (-2.0, dict(id="e-028", schema_version=1, actor="user", type="act.answered",
                 scope=_XC, session_ref=_SX, provenance=_CLF_V2,
                 payload={"act": "answered", "topics": ["资料分析"], "ref": "e-014",
                          "confidence": 0.85, "summary": "限时训练完成 15 题",
                          "depth": "recall", "spontaneity": "prompted"})),
    (-28.0, dict(id="e-029", schema_version=0, actor="system", type="session.started",
                 scope=_XC, session_ref=_SX, provenance=_RECORDER,
                 payload={"text": "会话开始：行测专项"})),
    (-8.0, dict(id="e-030", schema_version=1, actor="user", type="interaction.reflect",
                 scope="/meta", session_ref=_SM, content_ref="ct://meta/e-030",
                 provenance=_RECORDER,
                 payload={"text": "用户周反思：这周节奏比上周好。"})),
    (-7.0, dict(id="e-031", schema_version=0, actor="classifier", type="act.interpreted",
                 scope=_SL, session_ref=_SS, provenance=_CLF_V2,
                 payload={"act": "interpreted", "topics": ["申论大作文"], "ref": "e-023",
                          "confidence": 0.72, "summary": "解释了案例素材的选取标准"})),
    (-16.0, dict(id="e-032", schema_version=0, actor="user", type="interaction.answer",
                 scope=_XC, session_ref=_SX, content_ref="ct://xingce/e-032",
                 provenance=_RECORDER,
                 payload={"text": "用户作答：判断推理定义判断 5 题全对。",
                          "topics": ["判断推理"]})),
    (-15.0, dict(id="e-033", schema_version=0, actor="classifier", type="act.answered",
                 scope=_XC, session_ref=_SX, provenance=_CLF_V2,
                 payload={"act": "answered", "topics": ["判断推理"], "ref": "e-032",
                          "confidence": 0.8, "summary": "定义判断全对"})),
    (-1.5, dict(id="e-034", schema_version=1, actor="modeler", type="assertion.updated",
                 scope="/meta", session_ref=_SM, provenance=_MODELER,
                 payload={"assertion": "跨课程调度建议：本周重心放行测资料分析",
                          "confidence": "高", "evidence": ["e-025"]})),
    (-0.5, dict(id="e-035", schema_version=1, actor="system", type="resource.indexed",
                 scope=_XC, session_ref=_SX, content_ref="res://xingce/fenxi-drill.pdf",
                 provenance=_RES,
                 payload={"text": "刷题册索引完成：覆盖 12 个知识点",
                          "topics": ["资料分析"]})),
]


def build_events() -> list[LHEvent]:
    """按 (timestamp, id) 升序返回黄金事件列表（确定性，与文件顺序无关）。"""
    events = []
    for offset_days, spec in _EVENT_SPECS:
        data = dict(spec)
        data["timestamp"] = AS_OF + offset_days * DAY
        events.append(LHEvent.from_dict(data))
    events.sort(key=lambda e: (e.timestamp, e.id))
    return events


def write_golden(path: Path = GOLDEN_PATH) -> Path:
    """把黄金日志写为 JSONL（一行一事件，UTF-8，确定性键序）。"""
    lines = [
        json.dumps(e.to_dict(), ensure_ascii=False, sort_keys=True)
        for e in build_events()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_golden(path: Path | None = None) -> list[LHEvent]:
    """加载黄金事件（文件缺失时用 build_events() 兜底，保证测试可跑）。"""
    path = path or GOLDEN_PATH
    if not path.exists():
        return build_events()
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(LHEvent.from_dict(json.loads(line)))
    events.sort(key=lambda e: (e.timestamp, e.id))
    return events


if __name__ == "__main__":
    out = write_golden()
    events = load_golden()
    scopes = sorted({e.scope for e in events})
    versions = sorted({e.schema_version for e in events})
    print(f"已写入 {out}")
    print(f"事件数: {len(events)}")
    print(f"scope 覆盖: {scopes}")
    print(f"schema_version 覆盖: {versions}")
    print(f"墓碑事件: {[e.id for e in events if e.type == 'resource.deleted']}")
    print(f"handoff 事件: {[e.id for e in events if e.type == 'handoff.initiated']}")
