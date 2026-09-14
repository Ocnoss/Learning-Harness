"""EventStore —— append-only 事件存储协议与内存实现（契约 §1.2 / §6.1-存储）。

设计要点:
    - append-only：唯一写入口是 append()，没有任何更新/删除方法
      （§7.1-#4 插件只能追加；"删除"以墓碑事件表达，见 projection/compiler）。
    - id → 事件对象 的直接映射（刻意不复现既有 EventLog 的 id→位置索引 +
      pop(0) 错位缺陷；既有代码保持不动，本实现独立正确）。
    - (scope, verb, topic) → last_timestamp 映射，使否定式查询（§2.3
      "自事件 X 后 N 天内无同类事件"）为 O(1) 点查。
    - 按 scope 前缀与 topic 建索引，满足 §2.3 存储小要求
      （"L2 事件须按 topic 可索引"）与作用域点查（§6.1-预算分配）。

本模块不 import 任何既有 core 模块，零循环依赖风险。
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from core.context.envelope import LHEvent


def _scope_prefixes(scope: str) -> list[str]:
    """返回 scope 的全部祖先前缀（含自身），用于前缀索引。

    '/meta/xingce' → ['/', '/meta', '/meta/xingce']
    """
    prefixes = ["/"]
    if scope in ("", "/"):
        return prefixes
    parts = [p for p in scope.split("/") if p]
    acc = ""
    for part in parts:
        acc += "/" + part
        prefixes.append(acc)
    return prefixes


@runtime_checkable
class EventStore(Protocol):
    """事件存储协议（内核冻结语法；物理实现可替换——文件/SQLite 是后续事）。"""

    def append(self, event: LHEvent) -> None:
        """追加事件（append-only；重复 id 幂等忽略）。"""
        ...

    def get(self, event_id: str) -> LHEvent | None:
        """按 id 点查，O(1)。"""
        ...

    def scan(
        self,
        scope_prefix: str | None = None,
        types: Iterable[str] | None = None,
        since: float | None = None,
        until: float | None = None,
        topic: str | None = None,
        limit: int | None = None,
    ) -> list[LHEvent]:
        """按 scope 前缀 / 事件类型集 / 时间窗 [since, until] / topic 过滤。

        返回按 (timestamp, id) 升序的事件列表；limit 截断结果数量。
        """
        ...

    def last_seen(
        self,
        scope: str | None = None,
        verb: str | None = None,
        type: str | None = None,
        topic: str | None = None,
    ) -> float | None:
        """否定式查询的 O(1) 点查：同类事件最后一次出现的时间戳。

        "同类"由 (scope, verb/type, topic) 三元组界定（None 分量不参与匹配，
        即回退到更粗的索引键）。从未出现过返回 None——"没发生"也是数据（§2.3）。
        """
        ...

    def snapshot(self) -> list[dict]:
        """全量快照（JSON 可序列化，按追加序），用于不变量对比。"""
        ...

    def __len__(self) -> int:
        ...


class InMemoryEventStore:
    """内存实现：MVP 默认存储（SQLite 等持久化实现明确推迟）。"""

    def __init__(self, events: Iterable[LHEvent] | None = None):
        self._by_id: dict[str, LHEvent] = {}          # id → 事件对象（非位置索引）
        self._order: list[str] = []                    # 追加序 id 列表
        self._by_scope: dict[str, list[str]] = {}      # scope 前缀 → id 列表
        self._by_type: dict[str, list[str]] = {}       # 事件 type → id 列表
        self._by_topic: dict[str, list[str]] = {}      # topic → id 列表
        # (scope, verb, topic) → last_timestamp；topic 分量 None 表示
        # "该 scope+verb 下任意 topic"的聚合键，支撑粗细两级否定查询。
        self._last_seen: dict[tuple[str, str, str | None], float] = {}
        if events:
            for event in events:
                self.append(event)

    # -- 写入（唯一入口） --

    def append(self, event: LHEvent) -> None:
        if event.id in self._by_id:
            return  # append-only 幂等：重复 id 忽略，不覆盖历史
        self._by_id[event.id] = event
        self._order.append(event.id)

        for prefix in _scope_prefixes(event.scope):
            self._by_scope.setdefault(prefix, []).append(event.id)
        self._by_type.setdefault(event.type, []).append(event.id)
        for topic in event.topics:
            self._by_topic.setdefault(topic, []).append(event.id)

        verb = event.verb
        ts = event.timestamp
        for topic_key in (*event.topics, None):
            key = (event.scope, verb, topic_key)
            if ts >= self._last_seen.get(key, float("-inf")):
                self._last_seen[key] = ts

    # -- 读取 --

    def get(self, event_id: str) -> LHEvent | None:
        return self._by_id.get(event_id)

    def scan(
        self,
        scope_prefix: str | None = None,
        types: Iterable[str] | None = None,
        since: float | None = None,
        until: float | None = None,
        topic: str | None = None,
        limit: int | None = None,
    ) -> list[LHEvent]:
        # 选取最小的候选 id 集合作为起点
        candidates: Iterable[str] = self._order
        if scope_prefix is not None and topic is not None:
            a = set(self._by_scope.get(scope_prefix, ()))
            b = set(self._by_topic.get(topic, ()))
            candidates = [i for i in self._order if i in a and i in b]
        elif scope_prefix is not None:
            candidates = self._by_scope.get(scope_prefix, [])
        elif topic is not None:
            candidates = self._by_topic.get(topic, [])

        type_set = set(types) if types is not None else None
        result: list[LHEvent] = []
        for event_id in candidates:
            event = self._by_id[event_id]
            if type_set is not None and event.type not in type_set:
                continue
            if since is not None and event.timestamp < since:
                continue
            if until is not None and event.timestamp > until:
                continue
            result.append(event)

        result.sort(key=lambda e: (e.timestamp, e.id))
        if limit is not None:
            result = result[:limit]
        return result

    def last_seen(
        self,
        scope: str | None = None,
        verb: str | None = None,
        type: str | None = None,
        topic: str | None = None,
    ) -> float | None:
        # type 优先；给 type 未给 verb 时按 type 末段推导 verb
        if verb is None and type is not None:
            verb = type.rsplit(".", 1)[-1] if type else None
        if verb is None:
            return None  # 无动词则"同类"无从界定
        if scope is not None:
            key = (scope, verb, topic)
            return self._last_seen.get(key)
        # scope 未指定：跨全部 scope 聚合（键数量级 = scope×verb×topic，仍是小表扫描）
        best: float | None = None
        for (s, v, t), ts in self._last_seen.items():
            if v != verb:
                continue
            # topic 指定时严格只匹配 t==topic：聚合键 t=None 混有该 scope+verb 下
            # 所有 topic 的更晚时间戳，误当命中会让 last_seen(topic=X) 返回别的
            # topic 的时间。仅当 topic is None 时才允许匹配任意 t（含聚合键）取 max。
            if topic is not None and t != topic:
                continue
            if best is None or ts > best:
                best = ts
        return best

    def snapshot(self) -> list[dict]:
        return [self._by_id[i].to_dict() for i in self._order]

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self):
        """按追加序迭代事件（供投影 rebuild 使用）。"""
        for event_id in self._order:
            yield self._by_id[event_id]
