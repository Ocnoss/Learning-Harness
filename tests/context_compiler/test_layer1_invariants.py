"""第一层不变量测试 —— 契约 §7.1 八条不变量的可执行断言。

性质：纯函数测试。无 LLM、无网络、无原生依赖、无真实时钟
（时间全部来自黄金夹具的固定锚点 AS_OF 与 query.as_of）。

运行方式（两种等价）:
    D:\\miniconda\\python.exe tests\\context_compiler\\test_layer1_invariants.py
    pytest tests/context_compiler/test_layer1_invariants.py

不变量 → 测试函数映射:
    §7.1-#1 重放一致          → test_replay_determinism / test_compile_purity
    §7.1-#2 预算不超+不截断   → test_budget_never_exceeded
    §6.2   梯子下潜           → test_ladder_descent
    §7.1-#6 原始事件不跨域    → test_scope_no_leak
    §2.3   否定式查询         → test_negation_query
    §7.1-#5/§5.4 删除无痕迹   → test_delete_propagation
    §7.1-#7 additive 演进     → test_additive_schema_replay
    §7.1-#8 事实源不被改写    → test_store_not_mutated
    （附加）EventLog 单向桥接 → test_event_log_adapter_bridge
"""

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_FIXTURES = Path(__file__).resolve().parent / "fixtures"
if str(_FIXTURES) not in sys.path:
    sys.path.insert(0, str(_FIXTURES))

from core.context import (  # noqa: E402
    LHEvent,
    InMemoryEventStore,
    CharBasedEstimator,
    Projection,
    rebuild,
    fold,
    ContextQuery,
    ContextBundle,
    NegationQuery,
    compile,
    adapt_event_log,
)
from build_golden import load_golden, build_events, write_golden, AS_OF, DAY  # noqa: E402

XC = "/meta/xingce"
SL = "/meta/shenglun"
HUGE = 100_000


def _golden_projection() -> Projection:
    return rebuild(load_golden())


def _full_query(scope: str = XC, **kwargs) -> ContextQuery:
    defaults = dict(scope=scope, goal="", budget_tokens=HUGE, as_of=AS_OF,
                    allow_l1_deref=True)
    defaults.update(kwargs)
    return ContextQuery(**defaults)


# ---------------------------------------------------------------- §7.1-#1

def test_replay_determinism():
    """§7.1-#1: 重放一致——rebuild 与逐条 fold 等价；重放两次结果一致；
    fold 是纯函数（不改入参）；且与事件到达顺序无关（派生视图按内容归并）。"""
    print("\n=== test_replay_determinism (§7.1-#1 重放一致) ===")
    events = load_golden()
    assert events, "黄金日志为空"

    p_rebuild = rebuild(events)
    p_fold = Projection()
    for e in events:
        p_fold = fold(p_fold, e)
    assert p_rebuild.canonical() == p_fold.canonical(), "rebuild 与逐条 fold 不等价"

    # 重放两次逐字节一致
    assert rebuild(events).canonical() == p_rebuild.canonical(), "两次重放不一致"

    # fold 纯函数性：入参状态不被修改
    before = p_rebuild.canonical()
    _ = fold(p_rebuild, events[0])
    assert p_rebuild.canonical() == before, "fold 修改了入参 state"

    # 到达顺序无关（仅“集合意义”）：逆序重放得到同一条目集合、同 last_seen
    # （max 语义、与序无关）、同墓碑集合。注意——l2_by_topic / l3_by_scope 索引
    # 内的条目列表序、以及 events 列表序仍与到达序绑定，故此处对 items 用 sorted
    # 比较、且不直接断言这两个索引字典或 canonical() 全等。后续勿把本断言扩成
    # 逐字节 canonical 全等（那会把“到达序”误当成不变量而必然脆断）。
    p_rev = rebuild(list(reversed(events)))
    assert sorted((i.canonical() for i in p_rev.items), key=str) == \
        sorted((i.canonical() for i in p_rebuild.items), key=str), \
        "逆序重放的条目集合不一致"
    assert p_rev.last_seen == p_rebuild.last_seen, "逆序重放 last_seen 不一致"
    assert p_rev.tombstone_refs == p_rebuild.tombstone_refs

    print(f"[PASS] {len(events)} 条事件 rebuild≡fold，重放确定，顺序无关")
    return True


def test_compile_purity():
    """§7.1-#1: 编译纯度——同 (projection, query) 连编 N 次逐字节全等，
    且编译不修改投影。"""
    print("\n=== test_compile_purity (§7.1-#1 编译纯度) ===")
    projection = _golden_projection()
    query = ContextQuery(
        scope=XC,
        goal="资料分析 速算",
        budget_tokens=1000,
        as_of=AS_OF,
        negations=(NegationQuery(type="act.asked", within_days=7),),
        allow_l1_deref=True,
        handoff_packet={"text": "交接：资料分析专项突破", "source_event_ids": ["e-024"]},
    )
    proj_before = projection.canonical()

    first = compile(projection, query)
    for n in range(4):
        again = compile(projection, query)
        assert again.to_json() == first.to_json(), f"第 {n + 2} 次编译输出不一致"
        assert again.replay_digest == first.replay_digest, "replay_digest 不稳定"
    assert projection.canonical() == proj_before, "compile 修改了 projection"

    assert first.items, "该查询应有产出"
    print(f"[PASS] 连编 5 次逐字节全等，digest={first.replay_digest[:16]}…，投影未被修改")
    return True


# ---------------------------------------------------------------- §7.1-#2

def test_budget_never_exceeded():
    """§7.1-#2: 任何一次组装输出不超过 token 预算；超预算走降级（整项跳过），
    不得截断（装入的条目必须与投影条目逐字节全等）。"""
    print("\n=== test_budget_never_exceeded (§7.1-#2 预算纪律) ===")
    events = load_golden()
    projection = rebuild(events)
    item_by_event_id = {i.event_id: i for i in projection.items}

    saw_degraded = False
    for budget in (0, 1, 10, 40, 80, 150, 300, 600, 1200, HUGE):
        bundle = compile(projection, _full_query(budget_tokens=budget))
        assert bundle.total_tokens <= budget, f"预算 {budget} 被超出: {bundle.total_tokens}"
        assert bundle.total_tokens == sum(i.tokens for i in bundle.items), "total 与 items 不自洽"
        assert bundle.budget_tokens == budget
        for item in bundle.items:
            if item.event_id:  # 来自投影的条目：必须整条装入，绝不半截
                src = item_by_event_id[item.event_id]
                assert item.text == src.text, f"条目 {item.event_id} 文本被截断/篡改"
                assert item.tokens == src.token_len, f"条目 {item.event_id} token 数不一致"
        if bundle.degraded:
            saw_degraded = True

    # 极小预算 → 空包 + degraded（"全超则返回空 bundle + degraded=True"）
    empty = compile(projection, _full_query(budget_tokens=0))
    assert empty.items == [] and empty.total_tokens == 0
    assert empty.degraded is True, "空包必须标记 degraded"
    assert saw_degraded, "预算扫描中应观察到降级"

    print("[PASS] 10 组预算下 total_tokens ≤ budget，无半截条目，空包正确降级")
    return True


# ---------------------------------------------------------------- §6.2

def test_ladder_descent():
    """§6.2: 按预算沿梯子下潜——预算由紧到松，层级序列单调呈
    L3 → L3+L2 → L3+L2+L1（allow_l1_deref 控制是否解引用原文）。"""
    print("\n=== test_ladder_descent (§6.2 梯子下潜) ===")
    projection = _golden_projection()

    full = compile(projection, _full_query())
    l3_total = sum(i.tokens for i in full.items if i.level == 3)
    l2_total = sum(i.tokens for i in full.items if i.level == 2)
    l1_total = sum(i.tokens for i in full.items if i.level == 1)
    assert l3_total and l2_total and l1_total, "黄金夹具应覆盖 L1/L2/L3"

    # 紧预算：只装得下 L3 断言
    tight = compile(projection, _full_query(budget_tokens=l3_total, allow_l1_deref=False))
    levels = {i.level for i in tight.items}
    assert levels == {3}, f"紧预算应只有 L3，实际 {levels}"
    assert tight.deepest_layer_reached == 3

    # 中预算：L3 + L2（未开 deref，绝不出现 L1）
    mid = compile(projection, _full_query(budget_tokens=l3_total + l2_total,
                                          allow_l1_deref=False))
    levels = {i.level for i in mid.items}
    assert levels == {2, 3}, f"中预算应为 L3+L2，实际 {levels}"
    assert mid.deepest_layer_reached == 2

    # 松预算 + deref：下潜到 L1 原文
    loose = compile(projection, _full_query(budget_tokens=l3_total + l2_total + l1_total))
    levels = {i.level for i in loose.items}
    assert levels == {1, 2, 3}, f"松预算应为 L3+L2+L1，实际 {levels}"
    assert loose.deepest_layer_reached == 1
    assert not loose.degraded, "全额预算不应降级"

    # 单调性（预算增大）：装入量不减少；层级只在三个锚点预算上精确断言。
    # 注：中间预算下允许混层（贪心逐项装入，装不下整项则跳过继续，
    # 绝不截断）——这是 §7.1-#2 降级策略的设计行为，不是层级回升。
    budgets = sorted({0, l3_total // 2, l3_total, l3_total + l2_total // 2,
                      l3_total + l2_total, HUGE})
    prev_total = -1
    for b in budgets:
        bundle = compile(projection, _full_query(budget_tokens=b))
        assert bundle.total_tokens >= prev_total, f"预算 {b} 装入量下降"
        assert bundle.total_tokens <= b
        prev_total = bundle.total_tokens

    print(f"[PASS] L3({l3_total}t) → +L2({l2_total}t) → +L1({l1_total}t) 单调下潜")
    return True


# ---------------------------------------------------------------- §7.1-#6

def test_scope_no_leak():
    """§7.1-#6: 原始事件不跨作用域——父 scope 请求下，bundle 不含子 scope 的
    L1/L2 原文，只允许其 L3 断言上行；无关 scope 完全排除。"""
    print("\n=== test_scope_no_leak (§7.1-#6 作用域纪律) ===")
    projection = _golden_projection()

    # 父作用域 /meta：后代只允许 L3 上行
    bundle = compile(projection, _full_query(scope="/meta"))
    assert bundle.items, "/meta 应有产出"
    for item in bundle.items:
        if item.scope != "/meta":
            assert item.level == 3, f"子 scope {item.scope} 的 L{item.level} 泄漏进父域包"
    texts = " ".join(i.text for i in bundle.items)
    for leak in ("第12题", "直除法", "申论大作文提纲", "相遇模型"):
        assert leak not in texts, f"子/孙 scope 原文泄漏: {leak}"
    # 后代 L3 确实上行了
    assert any(i.scope == XC and i.level == 3 for i in bundle.items), "子域 L3 应上行"
    assert any(i.scope == SL and i.level == 3 for i in bundle.items), "无关子域 L3 应上行"

    # 子作用域 /meta/xingce：不含兄弟域与父域的任何条目
    bundle_xc = compile(projection, _full_query(scope=XC))
    for item in bundle_xc.items:
        assert item.scope == XC, f"兄弟/父域条目混入: {item.scope}"
    texts_xc = " ".join(i.text for i in bundle_xc.items)
    assert "申论" not in texts_xc and "周反思" not in texts_xc, "跨域内容混入"

    # 完全无关根域 /other：与黄金夹具任何 scope 都无前缀关系（既非同级、也非
    # 父/子级）→ 无任何候选，编译产出空包（覆盖“无关域”边界，与上面的父域/
    # 子域/同级共同构成四种作用域边界）。
    bundle_other = compile(projection, _full_query(scope="/other"))
    assert bundle_other.items == [], \
        f"无关根域应产出空包，实际 {len(bundle_other.items)} 项"
    assert bundle_other.total_tokens == 0
    assert bundle_other.deepest_layer_reached == 0
    # 无候选的空包不是“超预算降级”：degraded 应为 False（区别于预算=0 的空包）
    assert bundle_other.degraded is False, "无关域空包属'无候选'，不应标记 degraded"

    print("[PASS] 父域只见子域 L3；子域不见兄弟/父域；无关根域空包；原文零泄漏")
    return True


# ---------------------------------------------------------------- §2.3

def test_negation_query():
    """§2.3: 否定式查询——固定 as_of 下"窗口内无同类事件"正确产出缺失型断言；
    窗口内有同类事件则不产出；支持 within_days 与显式 since/until 两种窗口。"""
    print("\n=== test_negation_query (§2.3 否定式查询) ===")
    projection = _golden_projection()

    def absence_texts(negations) -> list[str]:
        bundle = compile(projection, _full_query(negations=negations))
        return [i.text for i in bundle.items
                if i.provenance.get("source") == "context-compiler/negation"]

    # 缺席成立：/meta/xingce 最后一次 act.asked 在 -20 天，7 天窗口内无
    texts = absence_texts((NegationQuery(type="act.asked", within_days=7),))
    assert len(texts) == 1, f"应产出 1 条缺失断言，实际 {len(texts)}"
    assert "缺失断言" in texts[0] and "act.asked" in texts[0]

    # 带 topic 的缺席（数量关系自 -20 后无 asked）
    texts = absence_texts((NegationQuery(type="act.asked", topic="数量关系",
                                         within_days=7, label="提问行为"),) )
    assert len(texts) == 1 and "提问行为" in texts[0]

    # 缺席不成立：act.answered 在 -2 天发生过（窗口内）
    texts = absence_texts((NegationQuery(type="act.answered", within_days=7),))
    assert texts == [], f"窗口内有同类事件，不应产出缺失断言: {texts}"

    # 显式 since/until 窗口：[-10, -8] 天内无 asked（最后一次在 -20）
    texts = absence_texts((NegationQuery(type="act.asked",
                                         since=AS_OF - 10 * DAY,
                                         until=AS_OF - 8 * DAY),))
    assert len(texts) == 1, "显式窗口内缺席应成立"

    # 缺失断言无出处事件指针（"什么都没发生也是数据"）
    bundle = compile(projection, _full_query(
        negations=(NegationQuery(type="act.asked", within_days=7),)))
    neg_items = [i for i in bundle.items
                 if i.provenance.get("source") == "context-compiler/negation"]
    assert neg_items and neg_items[0].source_event_ids == ()
    assert neg_items[0].level == 3, "缺失断言属断言层"

    print("[PASS] 缺席成立/不成立/显式窗口/topic 维度均正确")
    return True


# ---------------------------------------------------------------- §7.1-#5

def test_delete_propagation():
    """§7.1-#5/§5.4: 资料删除后，派生状态重建不含该资料任何痕迹
    （content_ref 墓碑 + provenance 墓碑两条路径都验证）。"""
    print("\n=== test_delete_propagation (§7.1-#5 删除无痕迹) ===")
    events = load_golden()

    # 墓碑（e-013 @ -11.0 / e-019 @ -4.0）之前：痕迹存在
    prefix = [e for e in events if e.timestamp <= AS_OF - 11.5 * DAY]
    p_pre = rebuild(prefix)
    b_pre = compile(p_pre, _full_query(budget_tokens=HUGE))
    ids_pre = {i.event_id for i in b_pre.items}
    assert {"e-002", "e-005"} <= ids_pre, "删除前应能看到 e-002/e-005"
    assert any(i.content_ref == "res://xingce/fenxi-1.pdf" for i in b_pre.items), \
        "删除前 content_ref 应可见"

    # 全量重放（含墓碑）：痕迹消失
    p_full = rebuild(events)
    assert "res://xingce/fenxi-1.pdf" in p_full.tombstone_refs
    assert "ct://xingce/e-002" in p_full.tombstone_refs
    assert (("plugin", "classifier-quadruple"), ("version", "1.0.0")) \
        in p_full.tombstone_provenance, "provenance 墓碑未登记"

    b_full = compile(p_full, _full_query(budget_tokens=HUGE))
    ids_full = {i.event_id for i in b_full.items}
    # content_ref 墓碑：e-002(L1) 消失；provenance 墓碑：e-003/e-005/e-009/e-010 消失
    assert not ({"e-002", "e-003", "e-005", "e-009", "e-010"} & ids_full), \
        f"被删痕迹仍在包中: {({'e-002', 'e-003', 'e-005', 'e-009', 'e-010'} & ids_full)}"
    for item in b_full.items:
        assert item.content_ref != "res://xingce/fenxi-1.pdf", "content_ref 痕迹残留"
        assert (str(item.provenance.get("plugin")), str(item.provenance.get("version"))) \
            != ("classifier-quadruple", "1.0.0"), "provenance 痕迹残留"
    texts = " ".join(i.text for i in b_full.items)
    for trace in ("直除法", "真题集（旧版）", "基期与现期的区别"):
        assert trace not in texts, f"文本痕迹残留: {trace}"
    # 新版本分类器产出不受墓碑影响（删除是精确的，不是全域清空）
    assert "e-015" in ids_full, "v2.0.0 分类器产出被误删"

    print("[PASS] content_ref 与 provenance 双路径删除传播，无痕迹残留且不误删")
    return True


# ---------------------------------------------------------------- §7.1-#7

def test_additive_schema_replay():
    """§7.1-#7: 全部历史事件在新版本 schema 下仍可重放——
    schema_version 0/1 混合编译成功；旧格式（缺字段）事件仍可 from_dict。"""
    print("\n=== test_additive_schema_replay (§7.1-#7 additive 演进) ===")
    events = load_golden()
    versions = {e.schema_version for e in events}
    assert {0, 1} <= versions, f"夹具应混合 schema 0/1，实际 {versions}"

    projection = rebuild(events)
    item_versions = {i.schema_version for i in projection.items}
    assert {0, 1} <= item_versions, "两个 schema 版本的事件都应产出投影条目"
    bundle = compile(projection, _full_query())
    assert bundle.items and not bundle.degraded, "混合版本编译应成功且不降级"
    assert {i.schema_version for i in projection.items if i.event_id} >= {0, 1}

    # 旧格式事件：缺 scope/provenance/payload/schema_version 仍可解析（默认值兜底）
    legacy = LHEvent.from_dict({"id": "old-1", "timestamp": 100.0,
                                "type": "interaction.answer"})
    assert legacy.actor == "system" and legacy.scope == "/"
    assert legacy.schema_version == 0
    assert legacy.provenance == {} and legacy.payload == {}
    assert legacy.content_ref == "" and legacy.session_ref == ""
    # 未知新字段不炸旧解析器（前向兼容）
    forward = LHEvent.from_dict({"id": "new-1", "timestamp": 1.0, "type": "act.asked",
                                 "future_axis": {"depth": "transfer"}})
    assert forward.id == "new-1"
    # to_dict/from_dict 往返无损
    for e in events[:5]:
        assert LHEvent.from_dict(e.to_dict()) == e, f"往返不一致: {e.id}"
    # 旧事件参与重放：混入 legacy 事件后 rebuild 不炸
    mixed = rebuild([*events, legacy, forward])
    assert len(mixed.items) == len(projection.items) + 2

    print(f"[PASS] schema {sorted(versions)} 混合重放成功；缺字段旧事件默认值兜底")
    return True


# ---------------------------------------------------------------- §7.1-#8

def test_store_not_mutated():
    """§7.1-#8/§7.1-#4: 编译与投影不改写事实源——store 快照前后逐字节相等；
    同时验证 store 的索引点查（scope 前缀 / topic / O(1) last_seen）。"""
    print("\n=== test_store_not_mutated (§7.1-#8 事实源不可变) ===")
    events = load_golden()
    store = InMemoryEventStore(events)
    assert len(store) == len(events)

    snap_before = store.snapshot()
    projection = rebuild(store)
    compile(projection, _full_query(scope="/meta"))
    compile(projection, _full_query(scope=XC, goal="资料分析",
                                    negations=(NegationQuery(type="act.asked",
                                                             within_days=7),)))
    snap_after = store.snapshot()
    assert snap_before == snap_after, "store 快照在编译前后不一致"
    assert len(store) == len(events), "store 长度变化"

    # append-only 幂等：重复追加不改变事实源
    store.append(events[0])
    assert store.snapshot() == snap_before, "重复 append 改变了快照"

    # 索引点查
    xc = store.scan(scope_prefix=XC)
    assert xc and all(e.scope == XC for e in xc)
    assert len(store.scan(scope_prefix="/meta")) >= len(xc)
    drills = store.scan(topic="资料分析")
    assert drills and all("资料分析" in e.topics for e in drills)
    assert store.get(events[0].id) == events[0]
    assert store.get("不存在的id") is None
    ls = store.last_seen(scope=XC, type="act.asked")
    assert ls == AS_OF - 20 * DAY, f"last_seen 应为 -20 天，实际偏移 {(ls - AS_OF) / DAY}"
    assert store.last_seen(scope=XC, type="act.asked", topic="数量关系") == ls
    assert store.last_seen(scope=XC, verb="从未发生") is None, "未发生应返回 None"

    print(f"[PASS] {len(events)} 条事件快照前后全等；scope/topic/last_seen 索引正确")
    return True


# ---------------------------------------------------------------- 桥接（附加）

def test_event_log_adapter_bridge():
    """附加: 既有 EventLog → LHEvent 单向只读桥接（envelope.py 双轨说明）。
    只读遍历、最小默认兜底、provenance 注明来源；不改动 EventLog。"""
    print("\n=== test_event_log_adapter_bridge (EventLog 桥接) ===")
    from core.event_log import EventLog, EventType

    log = EventLog(session_id="bridge-test")
    log.record(EventType.TOOL_CALL_START, data={"tool": "demo"})
    log.record(EventType.CUSTOM, data={"scope": XC, "actor": "user",
                                       "topics": ["资料分析"], "text": "自定义学习事件"})
    n_before = len(log)

    adapted = adapt_event_log(log)
    assert len(adapted) == n_before == len(log), "适配数量应与日志事件数一致"
    assert len(log) == n_before, "适配器改写了 EventLog"
    assert all(isinstance(e, LHEvent) for e in adapted)

    first, second = adapted
    assert first.actor == "system" and first.scope == "/", "无信封字段应给最小默认"
    assert first.schema_version == 0
    assert first.provenance.get("source") == "core.event_log", "provenance 应注明来源"
    assert first.payload.get("tool") == "demo", "data 应保留进 payload"
    assert second.scope == XC and second.actor == "user", "data 中信封字段应被采用"
    assert second.topics == ("资料分析",), "topics 应经 payload 可读"
    assert first.id and first.timestamp > 0

    # 桥接产物可进内核管道：rebuild + compile 全链路（两条都是 L1 原文，
    # 需 allow_l1_deref=True 才装入——梯子纪律：L1 只在显式解引用时入包）
    projection = rebuild(adapted)
    assert len(projection.items) == 2
    bundle = compile(projection, ContextQuery(scope=XC, budget_tokens=HUGE,
                                              as_of=first.timestamp,
                                              allow_l1_deref=True))
    assert any(i.event_id == second.id for i in bundle.items), "桥接事件应可装入上下文包"

    print("[PASS] EventLog 只读桥接 → LHEvent → rebuild → compile 全链路可用")
    return True


# ---------------------------------------------------------------- runner

_TESTS = [
    ("§7.1-#1 重放一致", test_replay_determinism),
    ("§7.1-#1 编译纯度", test_compile_purity),
    ("§7.1-#2 预算纪律", test_budget_never_exceeded),
    ("§6.2 梯子下潜", test_ladder_descent),
    ("§7.1-#6 作用域纪律", test_scope_no_leak),
    ("§2.3 否定式查询", test_negation_query),
    ("§7.1-#5 删除传播", test_delete_propagation),
    ("§7.1-#7 additive 演进", test_additive_schema_replay),
    ("§7.1-#8 事实源不可变", test_store_not_mutated),
    ("附加: EventLog 桥接", test_event_log_adapter_bridge),
]


async def main():
    print("=" * 60)
    print("第一层不变量测试（契约 §7.1）—— 纯函数，无 LLM/网络依赖")
    print("=" * 60)

    if not (Path(__file__).resolve().parent / "fixtures" / "golden_log.jsonl").exists():
        write_golden()  # 夹具缺失时自动重建（确定性生成）

    results = []
    for name, fn in _TESTS:
        try:
            fn()
            results.append((name, True))
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}")
            results.append((name, False))

    print()
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(r[1] for r in results)
    print()
    print("全部通过!" if all_passed else "存在失败项!")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
