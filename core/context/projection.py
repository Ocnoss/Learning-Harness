"""投影（Projection）—— 事件日志的派生只读视图（契约 §1.2 / §2.3）。

原则:
    - 原始的还是原始的，派生的永远可以重新派生（§2.3）：投影绝不写回事件，
      层级（L1/L2/L3）在此**派生**，不是事件固有属性。
    - 两条等价路径：rebuild(events) 全量 fold，fold(state, event) 增量。
      给定同一事件序列，两者产出逐字节相等的快照（§7.1-#1 的测试基础）。
    - fold 是纯函数：不修改入参，返回新 Projection。
    - 每条 item 的 token_len 在投影阶段用注入的估算器**预计算**，
      编译器只做整数加法比较（§6.1 预算分配的 O(1) 裁剪基础）。
    - 删除传播（§5.4/§7.1-#5）：resource.deleted 事件在投影中形成墓碑
      （content_ref 集合 + 匹配的 provenance 对），派生状态重建不含痕迹。
    - 否定式查询支撑（§2.3）：维护 (scope, verb, topic)→last_timestamp
      与 (scope, type)→last_timestamp 两级 last-seen 索引。

层级派生规则（MVP，确定性；词汇表约定属插件，这里只做最小机制）:
    1. payload["layer"] ∈ {1,2,3} 显式声明者优先（additive 字段，旧事件没有）;
    2. type 以 "act." 开头（分类器产出的四元组证据事件，对齐 §3.1
       ActClassified 形状），或同时带非空 topics + confidence/ref → L2;
    3. type 以 "assertion." 开头或 payload 带 assertion → L3;
    4. 其余 → L1（原子交互/原文，content_ref 指向内容存储；带 topics 的
       原始交互仍是 L1——topic 只是索引维度，不是层级判据）。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from core.context.envelope import LHEvent
from core.context.tokens import CharBasedEstimator, TokenEstimator

TOMBSTONE_TYPE = "resource.deleted"


class ProjectionItem:
    """投影中的一条派生条目（快照值对象；比较语义为全字段相等）。"""

    __slots__ = (
        "level", "event_id", "scope", "type", "actor", "timestamp",
        "topics", "text", "token_len", "content_ref", "source_event_ids",
        "provenance", "schema_version",
    )

    def __init__(
        self,
        level: int,
        event_id: str,
        scope: str,
        type: str,
        actor: str,
        timestamp: float,
        topics: tuple[str, ...],
        text: str,
        token_len: int,
        content_ref: str,
        source_event_ids: tuple[str, ...],
        provenance: dict,
        schema_version: int,
    ):
        self.level = level                    # 派生层级 1/2/3（不写回事件）
        self.event_id = event_id
        self.scope = scope
        self.type = type
        self.actor = actor
        self.timestamp = timestamp
        self.topics = topics
        self.text = text                      # 装入上下文包的文字形式
        self.token_len = token_len            # 预计算 token 长度
        self.content_ref = content_ref
        self.source_event_ids = source_event_ids  # 指针链（L2→L1、L3→L2 出处）
        self.provenance = dict(provenance)
        self.schema_version = schema_version

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ProjectionItem):
            return NotImplemented
        return self.canonical() == other.canonical()

    def __repr__(self) -> str:
        return (
            f"ProjectionItem(level={self.level}, event_id={self.event_id!r}, "
            f"scope={self.scope!r}, type={self.type!r}, token_len={self.token_len})"
        )

    def canonical(self) -> dict:
        """确定性字典形式（键序固定），用于等价比较与摘要。"""
        return {
            "level": self.level,
            "event_id": self.event_id,
            "scope": self.scope,
            "type": self.type,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "topics": list(self.topics),
            "text": self.text,
            "token_len": self.token_len,
            "content_ref": self.content_ref,
            "source_event_ids": list(self.source_event_ids),
            "provenance": dict(sorted(self.provenance.items())),
            "schema_version": self.schema_version,
        }


def derive_level(event: LHEvent) -> int:
    """从事件派生投影层级（1/2/3）——纯函数，见模块 docstring 规则。"""
    declared = event.payload.get("layer")
    # 守卫：True==1 / False==0，裸 `declared in (1,2,3)` 会把 payload["layer"]=True
    # 误判为 L1；仅接受真正的 int（排除 bool）。
    if isinstance(declared, int) and not isinstance(declared, bool) and declared in (1, 2, 3):
        return int(declared)
    etype = event.type
    if etype.startswith("act."):
        return 2
    if etype.startswith("assertion.") or "assertion" in event.payload:
        return 3
    if event.topics and ("confidence" in event.payload or "ref" in event.payload):
        return 2  # ActClassified 形状：分类器产出的证据事件
    return 1


def render_text(event: LHEvent, level: int) -> str:
    """把事件渲染为装入上下文包的文字（确定性模板，无时钟/随机）。"""
    payload = event.payload
    if level == 3:
        statement = payload.get("assertion") or payload.get("text") or payload.get("statement") or ""
        confidence = payload.get("confidence")
        topic_part = "、".join(event.topics)
        head = f"[断言/{event.scope}] {statement}"
        if topic_part:
            head += f"（知识点: {topic_part}）"
        if confidence is not None:
            head += f"（置信度: {confidence}）"
        return head
    if level == 2:
        act = payload.get("act") or event.verb
        topic_part = "、".join(event.topics)
        summary = payload.get("summary") or payload.get("text") or ""
        head = f"[证据/{event.scope}] {act}"
        if topic_part:
            head += f"（知识点: {topic_part}）"
        if summary:
            head += f": {summary}"
        return head
    # L1 原文/交互
    text = payload.get("text") or payload.get("content") or payload.get("transcript") or ""
    ref = f"（内容指针: {event.content_ref}）" if event.content_ref else ""
    return f"[原文/{event.scope}] {event.type}: {text}{ref}".rstrip()


class Projection:
    """事件日志的派生快照（不可变使用约定：只经 fold/rebuild 产生新实例）。

    公开只读视图:
        events            按 fold 顺序的全部事件（L1 指针解引用的取数来源）
        items             全部派生条目（按事件序）
        l2_by_topic       topic → [ProjectionItem]（§2.3 L2 按 topic 可索引）
        l3_by_scope       scope → [ProjectionItem]（断言按作用域索引）
        l1_items          L1 条目（按事件序）
        tombstone_refs    被删资源的 content_ref 集合（§5.4 删除不变量）
        tombstone_provenance  被删资源关联的 provenance 匹配集（元素为
                          键值对元组；item.provenance 包含全部键值对即命中）
        last_seen         (scope, verb, topic) → last_timestamp（否定查询 O(1)）
        last_seen_by_type (scope, type) → last_timestamp（粗粒度否定查询）
        item_by_event_id  event_id → ProjectionItem
    """

    def __init__(self, estimator: TokenEstimator | None = None):
        self.estimator: TokenEstimator = estimator or CharBasedEstimator()
        self.events: list[LHEvent] = []
        self.items: list[ProjectionItem] = []
        self.item_by_event_id: dict[str, ProjectionItem] = {}
        self.content_ref_by_event_id: dict[str, str] = {}
        self.l1_items: list[ProjectionItem] = []
        self.l2_by_topic: dict[str, list[ProjectionItem]] = {}
        self.l3_by_scope: dict[str, list[ProjectionItem]] = {}
        self.tombstone_refs: set[str] = set()
        self.tombstone_provenance: set[tuple[tuple[str, str], ...]] = set()
        self.last_seen: dict[tuple[str, str, str | None], float] = {}
        self.last_seen_by_type: dict[tuple[str, str], float] = {}

    # -- 纯函数复制（fold 不改入参的实现基础） --

    def _clone(self) -> "Projection":
        new = Projection(estimator=self.estimator)
        new.events = list(self.events)
        new.items = list(self.items)
        new.item_by_event_id = dict(self.item_by_event_id)
        new.content_ref_by_event_id = dict(self.content_ref_by_event_id)
        new.l1_items = list(self.l1_items)
        new.l2_by_topic = {k: list(v) for k, v in self.l2_by_topic.items()}
        new.l3_by_scope = {k: list(v) for k, v in self.l3_by_scope.items()}
        new.tombstone_refs = set(self.tombstone_refs)
        new.tombstone_provenance = set(self.tombstone_provenance)
        new.last_seen = dict(self.last_seen)
        new.last_seen_by_type = dict(self.last_seen_by_type)
        return new

    # -- 查询辅助 --

    def is_tombstoned(self, item: ProjectionItem) -> bool:
        """删除传播判定：content_ref / 指针链 / provenance 命中墓碑即算痕迹。

        provenance 命中 = 墓碑登记的键值对组合是 item.provenance 的子集
        （整体匹配，非单对匹配：避免 '有 plugin 键' 这种过宽误杀）。
        """
        if item.content_ref and item.content_ref in self.tombstone_refs:
            return True
        for ev_id in item.source_event_ids:
            # 语义澄清：tombstone_refs 是 content_ref 集合（见 fold 墓碑登记），
            # 这里用出处事件 id 直接探测属防御性匹配——当前夹具下事件 id 与
            # content_ref 命名空间不重叠故不触发；真正的指针链删除传播由下方
            # content_ref_by_event_id 穿透完成。保持既有匹配行为不变。
            if ev_id in self.tombstone_refs:
                return True
            # 指针链穿透：出处事件的内容指针被删 → 派生证据同属痕迹
            ref = self.content_ref_by_event_id.get(ev_id)
            if ref and ref in self.tombstone_refs:
                return True
        item_prov = {str(k): str(v) for k, v in item.provenance.items()}
        for required in self.tombstone_provenance:
            if all(item_prov.get(k) == v for k, v in required):
                return True
        return False

    def last_seen_of(
        self,
        scope: str,
        verb: str | None = None,
        type: str | None = None,
        topic: str | None = None,
    ) -> float | None:
        """投影侧否定查询点查：同类事件最后出现时间，未出现返回 None。"""
        if type is not None:
            ts = self.last_seen_by_type.get((scope, type))
            if topic is None:
                return ts
            # 带 topic 的 type 级查询退回 verb 索引
        if verb is None and type is not None:
            verb = type.rsplit(".", 1)[-1] if type else None
        if verb is None:
            return None
        return self.last_seen.get((scope, verb, topic))

    def canonical(self) -> dict:
        """确定性快照（JSON 可序列化），供 §7.1-#1 等价性断言逐字节比较。"""
        return {
            "events": [e.to_dict() for e in self.events],
            "items": [i.canonical() for i in self.items],
            "l2_by_topic": {
                t: [i.event_id for i in v] for t, v in sorted(self.l2_by_topic.items())
            },
            "l3_by_scope": {
                s: [i.event_id for i in v] for s, v in sorted(self.l3_by_scope.items())
            },
            "tombstone_refs": sorted(self.tombstone_refs),
            "tombstone_provenance": sorted(
                "{" + ", ".join(f"{k}={v}" for k, v in combo) + "}"
                for combo in self.tombstone_provenance
            ),
            "last_seen": {
                f"{s}|{v}|{t}": ts
                for (s, v, t), ts in sorted(
                    self.last_seen.items(), key=lambda kv: f"{kv[0][0]}|{kv[0][1]}|{kv[0][2]}"
                )
            },
            "last_seen_by_type": {
                f"{s}|{ty}": ts
                for (s, ty), ts in sorted(self.last_seen_by_type.items())
            },
        }

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Projection):
            return NotImplemented
        return self.canonical() == other.canonical()


def _make_item(event: LHEvent, level: int, estimator: TokenEstimator) -> ProjectionItem:
    text = render_text(event, level)
    payload = event.payload
    raw_refs = payload.get("ref") or payload.get("evidence") or payload.get("sources") or []
    if isinstance(raw_refs, str):
        source_refs: tuple[str, ...] = (raw_refs,)
    else:
        source_refs = tuple(str(r) for r in raw_refs)
    return ProjectionItem(
        level=level,
        event_id=event.id,
        scope=event.scope,
        type=event.type,
        actor=event.actor,
        timestamp=event.timestamp,
        topics=event.topics,
        text=text,
        token_len=estimator.count(text),
        content_ref=event.content_ref,
        source_event_ids=source_refs,
        provenance=event.provenance,
        schema_version=event.schema_version,
    )


def fold(state: Projection, event: LHEvent) -> Projection:
    """增量投影：纯函数，不修改入参 state，返回包含 event 的新 Projection。"""
    new = state._clone()
    new.events.append(event)

    # 重复 id 幂等（append-only 日志理论上无重复，防御性处理保证等价性）：
    # 早退前不写任何派生索引，使重复 id 对 content_ref_by_event_id 零副作用
    # （否则后到的同 id 事件会覆盖首条的 content_ref 映射，破坏 append-only 幂等）。
    if event.id in state.item_by_event_id:
        return new

    if event.content_ref:
        new.content_ref_by_event_id[event.id] = event.content_ref

    # last-seen 索引（否定查询支撑；max 保证与 fold 顺序无关的确定性）
    verb = event.verb
    ts = event.timestamp
    for topic_key in (*event.topics, None):
        key = (event.scope, verb, topic_key)
        if ts > new.last_seen.get(key, float("-inf")):
            new.last_seen[key] = ts
    type_key = (event.scope, event.type)
    if ts > new.last_seen_by_type.get(type_key, float("-inf")):
        new.last_seen_by_type[type_key] = ts

    # 墓碑事件：登记删除痕迹匹配键，不产出可装入条目
    if event.type == TOMBSTONE_TYPE:
        refs = event.payload.get("deleted_refs") or []
        if isinstance(refs, str):
            refs = [refs]
        for ref in refs:
            new.tombstone_refs.add(str(ref))
        if event.content_ref:
            new.tombstone_refs.add(event.content_ref)
        prov = event.payload.get("deleted_provenance") or {}
        if prov:
            combo = tuple(sorted(
                (str(k), str(v)) for k, v in dict(prov).items()
            ))
            new.tombstone_provenance.add(combo)
        return new

    level = derive_level(event)
    item = _make_item(event, level, new.estimator)
    new.items.append(item)
    new.item_by_event_id[event.id] = item
    if level == 1:
        new.l1_items.append(item)
    elif level == 2:
        for topic in event.topics:
            new.l2_by_topic.setdefault(topic, []).append(item)
        if not event.topics:
            new.l2_by_topic.setdefault("", []).append(item)
    else:  # level == 3
        new.l3_by_scope.setdefault(event.scope, []).append(item)
    return new


def rebuild(
    events: Sequence[LHEvent] | Iterable[LHEvent],
    estimator: TokenEstimator | None = None,
) -> Projection:
    """全量投影：从事件序列重放重建（§7.1-#1：两次重放结果一致）。

    与逐条 fold 完全等价：rebuild(events) == fold(fold(...empty..., e1), e2)...
    """
    state = Projection(estimator=estimator)
    for event in events:
        state = fold(state, event)
    return state
