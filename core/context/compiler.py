"""Context Compiler —— 按预算沿抽象梯子下潜的确定性编译器（契约 §6.1/§6.2）。

核心命题：上下文组装是**编译问题不是检索问题**（§6.1-2）。
compile(projection, query) -> ContextBundle 是确定性纯函数：
    - 禁用 time.time() / random / uuid4 / IO / 网络；时间只走 query.as_of；
    - 同一 (projection, query) 永远产出逐字节相同的 bundle（replay_digest 相同）。

算法（严格按 §6.2 + §7.1）:
    1. 作用域过滤（§7.1-#6）：本 scope 全层级可入；后代 scope **只允许 L3
       断言上行**，绝不引入其 L1/L2 原文；无关 scope 完全排除。
    2. 删除传播（§7.1-#5/§5.4）：item 的 content_ref / 指针链 / provenance
       命中墓碑集合即丢弃。
    3. 梯子下潜：预算内优先装 L3 断言（handoff 交接包最优先，§2.4"导师递
       条子"）；预算有余且命中 goal 的 topic 再展开 L2 证据；allow_l1_deref
       且仍有预算再沿指针取 L1 原文（被装入 L2 的出处指针优先）。
    4. 降级不截断（§7.1-#2）：单项超预算则**跳过该项**，绝不切半条；
       全部装不下则返回空 bundle + degraded=True。
    5. 否定查询（§2.3）：用投影 last-seen + query.as_of 判断窗口内是否
       "无同类事件"，产出缺失型断言项（什么都没发生也是数据）。
    6. 确定性：replay_digest = sha256(items 规范化 JSON)，同输入必同 digest。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from core.context.projection import Projection, ProjectionItem

L1 = 1
L2 = 2
L3 = 3


@dataclass(frozen=True)
class NegationQuery:
    """否定式查询（§2.3）："窗口内无同类事件"即产出缺失型断言。

    窗口二选一：within_days（相对 query.as_of 往前推）或显式 since/until。
    scope 缺省用 ContextQuery.scope；type 给全类型名（如 'act.answered'），
    verb 给动词短名（如 'answered'），二者给一即可（type 优先推导 verb）。
    """

    type: str = ""
    topic: str | None = None
    within_days: float | None = None
    since: float | None = None
    until: float | None = None
    scope: str | None = None
    label: str = ""  # 可选的人类可读标签（进缺失断言文本）


@dataclass(frozen=True)
class ContextQuery:
    """一次上下文编译请求（编译器的全部输入情境；时间锚点显式传入）。"""

    scope: str = "/"
    goal: str = ""
    budget_tokens: int = 0
    as_of: float = 0.0
    negations: tuple[NegationQuery, ...] = ()
    allow_l1_deref: bool = False
    handoff_packet: dict | None = None  # §2.4 交接上下文包（入口语境，最优先）


@dataclass
class ContextItem:
    """上下文包中的一个槽位（层级是投影派生的，不写回事件）。"""

    level: int                            # 1/2/3
    text: str                             # 装入包的文字（整条或不装，绝不截断）
    tokens: int                           # 预计算 token 长度
    source_event_ids: tuple[str, ...] = ()  # 指针链（可回溯到原文，§6.2）
    provenance: dict = field(default_factory=dict)
    scope: str = "/"
    event_id: str = ""
    content_ref: str = ""
    topics: tuple[str, ...] = ()

    def canonical(self) -> dict:
        return {
            "level": self.level,
            "text": self.text,
            "tokens": self.tokens,
            "source_event_ids": list(self.source_event_ids),
            "provenance": dict(sorted(self.provenance.items())),
            "scope": self.scope,
            "event_id": self.event_id,
            "content_ref": self.content_ref,
            "topics": list(self.topics),
        }


@dataclass
class ContextBundle:
    """编译产物：token 预算约束下的上下文包（§6.1-2）。"""

    items: list[ContextItem] = field(default_factory=list)
    total_tokens: int = 0
    budget_tokens: int = 0
    deepest_layer_reached: int = 0   # 3=只到断言层；1=下潜到了原文层；0=空包
    degraded: bool = False           # 有候选因预算被跳过 / 全超预算空包
    replay_digest: str = ""          # sha256(规范化 items)，同输入必同 digest

    def canonical(self) -> dict:
        return {
            "items": [i.canonical() for i in self.items],
            "total_tokens": self.total_tokens,
            "budget_tokens": self.budget_tokens,
            "deepest_layer_reached": self.deepest_layer_reached,
            "degraded": self.degraded,
        }

    def to_json(self) -> str:
        """确定性 JSON 序列化（键序固定），供逐字节比较。"""
        return json.dumps(self.canonical(), ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------- 作用域工具

def normalize_scope(scope: str) -> str:
    if not scope:
        return "/"
    if not scope.startswith("/"):
        scope = "/" + scope
    if len(scope) > 1 and scope.endswith("/"):
        scope = scope.rstrip("/")
    return scope


def is_within_scope(item_scope: str, query_scope: str) -> bool:
    """item_scope 是 query_scope 本身或其后代（前缀按路径段匹配）。"""
    item_scope = normalize_scope(item_scope)
    query_scope = normalize_scope(query_scope)
    if query_scope == "/":
        return True
    return item_scope == query_scope or item_scope.startswith(query_scope + "/")


def is_descendant_scope(item_scope: str, query_scope: str) -> bool:
    """item_scope 是 query_scope 的严格后代。"""
    return is_within_scope(item_scope, query_scope) and normalize_scope(
        item_scope
    ) != normalize_scope(query_scope)


# ---------------------------------------------------------------- goal 匹配

_TERM_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")


def _goal_terms(goal: str) -> list[str]:
    """从 goal 抽取确定性词项：连续汉字串 + ASCII 词。"""
    return _TERM_RE.findall(goal or "")


def _goal_score(item: ProjectionItem, goal: str) -> int:
    """L2 条目与 goal 的命中分：topic 子串命中 +2/词，text 命中 +1/词。"""
    terms = _goal_terms(goal)
    if not terms:
        return 0
    score = 0
    joined_topics = " ".join(item.topics)
    for term in terms:
        if joined_topics and term in joined_topics:
            score += 2
        elif term in item.text:
            score += 1
    return score


# ---------------------------------------------------------------- 编译器

def compile(projection: Projection, query: ContextQuery) -> ContextBundle:
    """确定性纯函数：把投影按预算编译为上下文包（算法见模块 docstring）。

    不读时钟/随机/IO；时间只来自 query.as_of；不修改 projection。
    """
    q_scope = normalize_scope(query.scope)
    budget = max(0, int(query.budget_tokens))

    # -- 候选收集（作用域过滤 §7.1-#6 + 删除传播 §7.1-#5 前置） --
    l3_own: list[ProjectionItem] = []
    l3_desc: list[ProjectionItem] = []
    for item in projection.items:
        if item.level != L3 or projection.is_tombstoned(item):
            continue
        # 本 scope 全入；后代 scope 只允许 L3 断言上行；无关 scope 完全排除。
        # item.scope 是事件原样、q_scope 已归一化，比较前先归一化 item.scope，
        # 否则无前导/尾随斜杠的 scope 会被静默判为“非本域”而丢弃
        # （is_descendant_scope 内部已归一化，无需重复）。
        if normalize_scope(item.scope) == q_scope:
            l3_own.append(item)
        elif is_descendant_scope(item.scope, q_scope):
            l3_desc.append(item)
    l2_own = [
        i for i in projection.items
        if i.level == L2
        and normalize_scope(i.scope) == q_scope
        and not projection.is_tombstoned(i)
    ]
    l1_own = [
        i for i in projection.items
        if i.level == L1
        and normalize_scope(i.scope) == q_scope
        and not projection.is_tombstoned(i)
    ]

    # 确定性排序：新事件优先（时间相同按 event_id 破平局）
    l3_own.sort(key=lambda i: (-i.timestamp, i.event_id))
    l3_desc.sort(key=lambda i: (-i.timestamp, i.event_id))
    l2_own.sort(key=lambda i: (-_goal_score(i, query.goal), -i.timestamp, i.event_id))
    l1_own.sort(key=lambda i: (-i.timestamp, i.event_id))

    # -- 装入循环（降级不截断 §7.1-#2） --
    packed: list[ContextItem] = []
    total = 0
    degraded = False

    def try_pack(ci: ContextItem) -> bool:
        nonlocal total, degraded
        if total + ci.tokens <= budget:
            packed.append(ci)
            total += ci.tokens
            return True
        degraded = True  # 跳过整项，绝不切半条
        return False

    def to_context_item(item: ProjectionItem) -> ContextItem:
        return ContextItem(
            level=item.level,
            text=item.text,
            tokens=item.token_len,
            source_event_ids=item.source_event_ids,
            provenance=dict(item.provenance),
            scope=item.scope,
            event_id=item.event_id,
            content_ref=item.content_ref,
            topics=item.topics,
        )

    # 0) handoff 交接包（§2.4：接收方的入口语境，最优先）
    if query.handoff_packet:
        packet = dict(query.handoff_packet)
        text = str(packet.get("text") or json.dumps(
            {k: v for k, v in sorted(packet.items()) if k != "text"},
            ensure_ascii=False, sort_keys=True,
        ))
        tokens = projection.estimator.count(text)
        try_pack(ContextItem(
            level=L3,
            text=text,
            tokens=tokens,
            source_event_ids=tuple(str(r) for r in (packet.get("source_event_ids") or ())),
            provenance={"source": "handoff_packet"},
            scope=q_scope,
            event_id="",
        ))

    # 1) L3 断言：本 scope 优先，后代 scope 断言上行其后
    for item in l3_own:
        try_pack(to_context_item(item))
    for item in l3_desc:
        try_pack(to_context_item(item))

    # 2) 否定查询产出的缺失型断言（§2.3）
    for neg in query.negations:
        neg_item = _eval_negation(projection, query, neg)
        if neg_item is not None:
            try_pack(neg_item)

    # 3) L2 证据：预算有余且命中 goal 的 topic 才展开（本 scope only）
    for item in l2_own:
        if query.goal and _goal_score(item, query.goal) <= 0:
            continue  # 与 goal 无关的 L2 不展开（编译不是检索：相关性是装入条件）
        try_pack(to_context_item(item))

    # 4) L1 原文：显式允许解引用且仍有预算；被装入 L2 的出处指针优先
    if query.allow_l1_deref:
        ref_set = _packed_source_refs(packed)
        referenced: list[ProjectionItem] = []
        unreferenced: list[ProjectionItem] = []
        for item in l1_own:
            # 仅按 ref_set（已装入 L2 的出处指针）判定优先级；原 packed_l2_ids 存的是
            # L2 的 event_id，与 L1 条目 item.event_id 恒不同，该 or 分支永不成立（已删）。
            if item.event_id in ref_set:
                referenced.append(item)
            else:
                unreferenced.append(item)
        for item in referenced + unreferenced:
            try_pack(to_context_item(item))

    # -- 收尾：deepest layer / digest / 空包降级 --
    candidates_exist = bool(
        query.handoff_packet or l3_own or l3_desc or l2_own or l1_own
    )
    if not packed and candidates_exist:
        degraded = True  # 全超预算 → 空 bundle + degraded（§7.1-#2）

    deepest = min((ci.level for ci in packed), default=0)
    # digest 覆盖 items + 收尾三态（total_tokens/deepest/degraded）：仅 items 时
    # “空包+降级”与“空包+无候选”同为空列表 → 同 digest 无法区分；纳入三态后二者可辨。
    digest = hashlib.sha256(
        json.dumps(
            {
                "items": [ci.canonical() for ci in packed],
                "total_tokens": total,
                "deepest_layer_reached": deepest,
                "degraded": degraded,
            },
            ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    return ContextBundle(
        items=packed,
        total_tokens=total,
        budget_tokens=budget,
        deepest_layer_reached=deepest,
        degraded=degraded,
        replay_digest=digest,
    )


def _packed_source_refs(packed: list[ContextItem]) -> set[str]:
    """已装入 L2 项的出处指针集合（§6.2 步骤 3：“被装入 L2 的出处指针优先”）。

    只取 L2：L3 evidence / 否定项 / handoff 的 source_event_ids 不参与 L1 解引用
    优先级判定（此前遍历全部已装入项，与“L2 出处指针优先”的注释语义不符）。
    """
    refs: set[str] = set()
    for ci in packed:
        if ci.level == L2:
            refs.update(ci.source_event_ids)
    return refs


def _eval_negation(
    projection: Projection, query: ContextQuery, neg: NegationQuery
) -> ContextItem | None:
    """判定一条否定查询；"窗口内无同类事件"成立则产出缺失型断言项。"""
    scope = normalize_scope(neg.scope if neg.scope is not None else query.scope)
    verb = neg.type.rsplit(".", 1)[-1] if neg.type else None
    if verb is None:
        return None

    # 窗口：since/until 显式优先，否则 as_of 往前 within_days
    if neg.until is not None:
        window_end = float(neg.until)
    else:
        window_end = float(query.as_of)
    if neg.since is not None:
        window_start = float(neg.since)
    elif neg.within_days is not None:
        window_start = window_end - float(neg.within_days) * 86400.0
    else:
        return None  # 无窗口定义则不可判定，不产出
    if window_start > window_end:
        return None

    last = projection.last_seen.get((scope, verb, neg.topic))
    if last is not None and window_start <= last <= window_end:
        return None  # 窗口内发生了同类事件 → 缺失断言不成立

    label = neg.label or (f"{neg.type}" + (f"/{neg.topic}" if neg.topic else ""))
    days = (window_end - window_start) / 86400.0
    text = (
        f"[缺失断言/{scope}] 近 {days:g} 天内无同类事件: {label}"
        f"（判定锚点 as_of={window_end:g}，最后出现="
        + (f"{last:g}" if last is not None else "从未")
        + "）"
    )
    return ContextItem(
        level=L3,
        text=text,
        tokens=projection.estimator.count(text),
        source_event_ids=(),  # 缺失没有出处事件——"什么都没发生也是数据"
        provenance={"source": "context-compiler/negation"},
        scope=scope,
        event_id="",
    )
