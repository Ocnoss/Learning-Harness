"""event_log_adapter —— 既有 EventLog 到 LHEvent 信封的单向只读桥接。

定位（见 envelope.py 顶部双轨说明）:
    core.event_log.Event 是 agent tracing 事件（event_id/event_type/data/...），
    不承载契约 §2.2 的完整信封。本适配器**只读**遍历既有 EventLog，把每个
    Event 映射为一个 LHEvent：能读到完整信封字段就用（Event.data 里若带有
    'scope'/'actor'/'provenance'/'content_ref'/'schema_version' 等键则采用），
    否则给最小默认（actor='system'、scope='/'、schema_version=0），
    provenance 注明来源 'core.event_log'。

    未来 Event 若演进为承载完整信封，本适配器可无痛下线。

防循环导入：`from core.event_log import EventLog` 放在函数体内延迟执行
（既有先例：core/llm_client.py 在 _resolve 内延迟导入 ModelTier）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.context.envelope import LHEvent

if TYPE_CHECKING:  # 仅类型检查期导入，运行期不触发
    from core.event_log import EventLog


def adapt_event_log(log: "EventLog") -> list[LHEvent]:
    """只读遍历既有 EventLog，映射为 LHEvent 列表（按日志内顺序）。

    不修改 log 的任何状态；不调用 record/end_event；只读取事件序列。
    """
    from core.event_log import EventLog  # noqa: F401  延迟导入防循环（运行期校验类型）

    result: list[LHEvent] = []
    for event in getattr(log, "_events", []):
        data: dict[str, Any] = dict(getattr(event, "data", {}) or {})
        event_type = getattr(event, "event_type", None)
        type_value = getattr(event_type, "value", str(event_type or ""))

        envelope_fields = {
            "actor", "scope", "provenance", "content_ref",
            "schema_version", "session_ref", "payload",
        }
        # data 中若带信封字段则采用，其余（含 topics）作为 payload 内容保留
        payload = dict(data.get("payload") or {})
        for key, value in data.items():
            if key not in envelope_fields and key != "payload":
                payload.setdefault(key, value)

        provenance = dict(data.get("provenance") or {})
        provenance.setdefault("adapter", "core.context.event_log_adapter")
        provenance.setdefault("source", "core.event_log")

        lh = LHEvent(
            id=str(getattr(event, "event_id", "")),
            timestamp=float(getattr(event, "timestamp", 0.0)),
            schema_version=int(data.get("schema_version", 0)),
            actor=str(data.get("actor", "system")),
            type=type_value,
            provenance=provenance,
            session_ref=str(
                data.get("session_ref") or getattr(event, "session_id", "") or ""
            ),
            scope=str(data.get("scope", "/")),
            content_ref=str(data.get("content_ref", "")),
            payload=payload,
        )
        result.append(lh)
    return result
