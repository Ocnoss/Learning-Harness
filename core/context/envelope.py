"""LHEvent —— 契约 §2.2 事件信封的第一等实现。

双轨说明（认知负担消解）:
    LHEvent = 契约 §2.2 信封的第一等实现;
    core.event_log.Event 是 agent tracing 事件;
    两者经 event_log_adapter 单向桥接（EventLog → list[LHEvent]，只读）;
    未来 Event 若演进为承载完整信封，适配器可无痛下线。

信封九字段（§2.2）: id / timestamp / schema_version / actor / type /
provenance / session_ref / scope / content_ref，外加 payload 承载内容
（大 payload 应进内容存储，事件里只留 content_ref 指针——§2.2）。

演进纪律（§2.2）:
    - 只做加法: 加字段可以，改语义、删字段不行;
    - from_dict 对缺失字段一律用默认值兜底，旧事件（schema_version=0，
      缺后续版本新增字段）永远可解析、可重放（§7.1-#7）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class LHEvent:
    """契约 §2.2 事件信封（frozen：事件不可变，append-only 的值对象）。

    字段顺序与 §2.2 信封表逐行对齐；payload 为第十个字段，承载内容本体。
    """

    id: str = ""                                # 全局唯一
    timestamp: float = 0.0                      # 事件发生时间（epoch 秒）
    schema_version: int = 0                     # 事件 schema 版本号
    actor: str = "system"                       # user / agent / system / 具体插件
    type: str = ""                              # 事件族 + 动词（xAPI 式）
    provenance: dict = field(default_factory=dict)   # 插件 id+版本、策略 id+版本
    session_ref: str = ""                       # 所属会话
    scope: str = "/"                            # 节点路径（课程树作用域）
    content_ref: str = ""                       # 内容指针（大 payload 不内嵌）
    payload: dict = field(default_factory=dict)  # 内容本体（additive 演进区）

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 可序列化字典（信封字段 + payload）。"""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LHEvent":
        """从字典反序列化。

        对缺失字段用默认值兜底（支持 additive 演进：旧事件缺新字段仍合法）；
        对未知多余字段忽略（前向兼容：新版本事件的多余字段不炸旧解析器）；
        provenance / payload 若为 None 或缺失，兜底为空 dict。
        """
        provenance = data.get("provenance") or {}
        payload = data.get("payload") or {}
        return cls(
            id=str(data.get("id", "")),
            timestamp=float(data.get("timestamp", 0.0)),
            schema_version=int(data.get("schema_version", 0)),
            actor=str(data.get("actor", "system")),
            type=str(data.get("type", "")),
            provenance=dict(provenance),
            session_ref=str(data.get("session_ref", "")),
            scope=str(data.get("scope", "/")),
            content_ref=str(data.get("content_ref", "")),
            payload=dict(payload),
        )

    # -- 便捷访问（派生属性，不写回事件） --

    @property
    def verb(self) -> str:
        """事件动词 = type 的最后一段（'resource.deleted' → 'deleted'）。

        xAPI 式 Actor-Verb-Object：type 为 '族.动词'。否定式查询（§2.3）
        的"同类事件"判定以 (scope, verb, topic) 为键。
        """
        return self.type.rsplit(".", 1)[-1] if self.type else ""

    @property
    def topics(self) -> tuple[str, ...]:
        """事件涉及的知识点（payload['topics']），无则空元组。

        §2.3 存储小要求：L2 事件须按 topic 可索引——topic 从 payload 读取，
        不进入信封字段（additive：旧事件无 topics 仍合法）。
        """
        raw = self.payload.get("topics") or ()
        if isinstance(raw, str):
            return (raw,)
        # additive 容错：payload["topics"] 为非可迭代标量（如 42 / True）时返回 ()，
        # 不抛 TypeError（旧/异常事件仍可解析重放，§7.1-#7）。
        if not isinstance(raw, (list, tuple, set)):
            return ()
        return tuple(str(t) for t in raw)
