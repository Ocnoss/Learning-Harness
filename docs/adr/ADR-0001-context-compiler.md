# ADR-0001: 上下文编译器薄切片（core/context/）的架构决策

- **状态**：已接受（Accepted）
- **日期**：2026-09
- **相关契约**：`LH内核契约-v0.1.md` §1.2、§2.2、§2.3、§2.4、§5.4、§6.1、§6.2、§7.1、§7.3、§8.3
- **相关代码**：`core/context/`（envelope / store / tokens / projection / compiler / event_log_adapter）、`core/stub_llm.py`
- **相关测试**：`tests/context_compiler/test_layer1_invariants.py`、`tests/context_compiler/test_layer2_scenarios.py`、`tests/context_compiler/test_layer3_ab_eval.py`、`tests/context_compiler/run_all.py`、`tests/context_compiler/harness/`（`fake_learner.py`、`judge_rubric.py`）、`tests/context_compiler/fixtures/`（`golden_log.jsonl`、`build_golden.py`）

> 契约 §7.3 要求：agent 修改内核前必须读 ADR，以防跨会话架构漂移。本文件记录上下文编译器薄切片（MVP）的全部关键决策、理由、后果与被否决的替代方案。

---

## 1. 背景

LH 的核心痛点之一是上下文管理：一次学习跨越数周乃至数月，上下文观是"一段有结构的成长史"而非"一段对话"（契约 §0）。契约把上下文重建定位为**编译问题而非检索问题**（§6.1-2）：输入当前情境 + 事件历史 + 学习者模型，输出 token 预算约束下的上下文包，并回答"以什么粒度装、装不下的以摘要还是指针形式存在"（§6.2）。

本切片是这一能力的最小可运行落地（MVP），目标不是产品，而是**检验架构**（§8.2-1 寄生验证）：把 §7.1 的不变量从"陈述"变成"会自动报警的断言"（§7.3 fitness functions）。

约束前提：
- **零回归**：既有 `core/event_log.py`、`core/tool_agent.py` 等运行时代码一律不改、不碰；`import core` 的既有行为不变。
- **确定性优先**：编译器与投影必须是纯函数，支撑 §7.1-#1 重放不变量。
- **一人 + coding agent 工作流**：优先机制简单、可独立验证，避免过早引入重型基础设施。

---

## 2. 决策

### 决策 1：事件与投影分离——层级（L0–L4）是派生投影，不是事件固有字段

**决策**：`LHEvent`（`core/context/envelope.py`）只承载契约 §2.2 的信封九字段 + `payload`，**不含** `level`/`layer` 作为固有属性。抽象梯子的层级（L1/L2/L3）由投影阶段的纯函数 `derive_level(event)`（`core/context/projection.py`）派生。

**理由**：
- 契约 §2.3 明言"投影是插件；原始的还是原始的，派生的永远可以重新派生"，§1.2"一切状态是日志的投影"。层级是**一种读法**，写回事件就把派生视图冻进了事实源。
- 分类器会错、会升级（§3.1"换分类器重放即翻新"）。若层级是事件字段，翻新就要改写历史事件——违反 §7.1-#8（事实源不被改写）与 §2.1 事件不动纪律。
- `derive_level` 仍**允许** `payload["layer"] ∈ {1,2,3}` 显式声明优先，但这是 additive 的 payload 内容（旧事件没有仍合法，§2.2 演进纪律），不是信封字段——机制在内核，词汇/裁决归插件（§1.1）。

**后果**：
- 正面：同一事件序列在不同投影规则下可重算出不同层级视图；重放不变量天然成立。
- 负面：每次编译前都要跑一遍 `rebuild`/`fold` 派生层级（O(n)）。MVP 规模下可接受；Phase 2 可缓存投影快照。

### 决策 2：独立 `LHEvent` 信封 + 从既有 `EventLog` 的单向只读适配器，而非扩展既有 `Event`

**决策**：新建 frozen dataclass `LHEvent`（§2.2 信封第一等实现）。既有 `core.event_log.Event`（agent tracing 事件）**保持不动**；通过 `core/context/event_log_adapter.py::adapt_event_log(log) -> list[LHEvent]` 做**单向、只读**桥接。

**理由**：
- **零回归 + 契约保真**：既有 `Event` 是 tracing 事件（`event_id`/`event_type`/`data`/`duration_ms`…），字段语义与 §2.2 信封（`scope`/`provenance`/`content_ref`/`schema_version`）不重合。强行扩展 `Event` 会污染既有语义、破坏所有既有调用点，且无法保证信封的 additive 演进纪律。
- **适配器可无痛下线**：`envelope.py` 顶部双轨说明明确——未来 `Event` 若演进为承载完整信封，适配器直接删除即可，内核管道（只吃 `list[LHEvent]`/`InMemoryEventStore`）不受影响。
- **防循环导入**：适配器对 `core.event_log` 的 import 放在**函数体内延迟执行**（既有先例：`core/llm_client.py` 在 `_resolve` 内延迟导入 `ModelTier`）；`TYPE_CHECKING` 块仅类型检查期导入。因此 `core/context` 子包运行期不依赖任何既有 core 运行时模块，既有代码也不 import 本子包——双向零耦合。

**后果**：
- 正面：MVP 完全独立可验证；`test_event_log_adapter_bridge` 证明桥接产物能进 `rebuild → compile` 全链路。
- 负面：短期内存在"双事件模型"的认知负担，靠 `envelope.py` docstring 的双轨说明消解。适配器只读遍历 `log._events`，不映射 `duration_ms` 等 tracing 专有语义（那些不是学习证据）。

### 决策 3：`InMemoryEventStore` = id→对象映射 + `(scope, verb, topic)→last_ts` 的 O(1) 否定查询索引；SQLite 推迟到 Phase 2

**决策**：`core/context/store.py` 提供 `EventStore` Protocol（内核冻结语法）与 `InMemoryEventStore` 内存实现。内部索引：
- `_by_id: dict[str, LHEvent]`——**id → 事件对象**的直接映射（刻意不复现既有 `EventLog` 的 id→位置索引缺陷，见决策 7）；
- `_by_scope` / `_by_type` / `_by_topic`——scope 前缀、事件类型、topic 的倒排索引（满足 §2.3"L2 事件须按 topic 可索引"）；
- `_last_seen: dict[(scope, verb, topic|None), float]`——使否定式查询（§2.3"自事件 X 后 N 天内无同类事件"）成为 **O(1) 点查**；`topic=None` 分量是"该 scope+verb 下任意 topic"的聚合键，支撑粗细两级查询。

`append()` 是唯一写入口，重复 id 幂等忽略（append-only，§7.1-#4）。SQLite / 文件持久化 / checkpoint **明确推迟到 Phase 2**。

**理由**：
- 契约 §9-2 把"事件存储选型（文件/SQLite/专用库）"列为**下一阶段待决**。MVP 阶段物理存储不是检验架构的瓶颈，机制（append-only + 否定查询索引）才是。
- 内存实现零依赖、纯确定性，天然满足 §8.3"测试替身免费（内存事件存储 → 系统确定性化）"。
- Protocol 定义冻结了存储语法，未来换 SQLite 实现不改消费方（编译器只依赖 `EventStore` 协议）。

**后果**：
- 正面：否定查询 O(1)；存储可替换性由 Protocol 保证。
- 负面：进程退出即丢数据，不能承载真实数月学习史——这是 MVP 的**已知边界**，Phase 2 用 SQLite 实现同一 Protocol 补齐。

### 决策 4：可注入的确定性 token 估算器（`CharBasedEstimator`）；tiktoken 推迟

**决策**：`core/context/tokens.py` 定义 `TokenEstimator` Protocol（`count(text) -> int`，必须确定性）与默认实现 `CharBasedEstimator`（CJK≈len/1.6、ASCII≈len/4，分别向上取整相加，保守上界）。估算器通过 `Projection(estimator=...)` 注入，投影阶段**预计算** `token_len`，编译器只做整数加法比较。tiktoken 等真实 tokenizer 后端**明确推迟**。

**理由**：
- 预算裁剪（§6.1）只需要"稳定的相对度量 + 保守上界"，不需要与真实 tokenizer 逐 token 对齐。
- `CharBasedEstimator` 零依赖、纯确定性（不查网络、不加载词表、不调时钟/随机），满足编译器确定性纪律（决策 5）。
- 可注入设计：未来要精度时换一个实现 `TokenEstimator` 的后端即可，投影/编译逻辑不动。

**后果**：
- 正面：MVP 无原生依赖（tiktoken 需编译/下载词表），跨平台开箱即跑。
- 负面：估算与真实 token 有偏差（保守偏多）。因取"宁多算不少算"的上界，预算纪律（§7.1-#2 不超预算）方向安全。

### 决策 5：确定性纪律——compile/fold 禁时钟/随机/IO，时间只走 `query.as_of`

**决策**：`compile(projection, query)`（`core/context/compiler.py`）与 `fold(state, event)`（`core/context/projection.py`）是**确定性纯函数**：
- 禁用 `time.time()` / `random` / `uuid4` / 文件 IO / 网络；
- 一切时间判断只来自显式入参 `query.as_of`（否定查询窗口锚定于此）；
- `fold` 不修改入参 `state`，返回新 `Projection`（`_clone()` 实现）；
- `compile` 不修改 `projection`；
- 产物 `ContextBundle.replay_digest = sha256(规范化 items JSON)`，同输入必同 digest。

**理由**：直接支撑 §7.1-#1（两次重放结果一致）与 §7.3 fitness function——确定性是可执行断言的前提。若编译掺入时钟/随机，重放不变量无从断言，黄金场景测试（§7.2）也无法逐字节比较。

**后果**：
- 正面：`test_replay_determinism` / `test_compile_purity` 可断言"连编 5 次逐字节全等、逆序重放条目集合一致、fold 不改入参"。
- 负面：真实运行时的"当前时间"必须由调用方显式注入 `as_of`——这是刻意的控制反转，把不确定性挡在编译器边界之外。

### 决策 6：三层测试策略；不用 web 前端

**决策**：测试分三层，**L1/L2/L3 三层均已在本切片落地**：
- **L1 确定性不变量**（`tests/context_compiler/test_layer1_invariants.py`）：纯函数，无 LLM/网络/原生依赖/真实时钟；覆盖计划契约映射表所列 8 项（§7.1 的 #1/#2/#5/#6/#7/#8 + §6.2 梯子下潜 + §2.3 否定式查询）；§7.1 的 #3（每个教学动作事件携带预测字段并引用至少一条证据）与 #4（插件只能追加、不得直写投影）列为后续补齐项；另附加 EventLog 桥接用例。
- **L2 模拟学习者黄金场景**（`tests/context_compiler/test_layer2_scenarios.py`）：假用户把数周事件压缩进几分钟灌入（§7.3 模拟学习者），跑 §7.2 五个端到端场景。
- **L3 headless LLM-judge A/B**（`tests/context_compiler/test_layer3_ab_eval.py`）：默认用 `core/stub_llm.py` 的 `StubLLMClient` 顺序回放做确定性评估；`RecordedLLMClient`/`RecordingLLMClient` 提供磁带领制-回放能力（工厂注册名 `stub`/`recorded`）；live 模式由环境变量门控（见 README）。

**为何不用 web 前端**：**测试 ≠ 分发**。本切片目标是检验架构、让不变量自动报警，不是给学习者做界面。web 前端会引入浏览器/构建/网络不确定性，与"确定性优先"冲突；契约 §9-6 把前端形态列为下一阶段待决。headless + 测试替身（§8.3"测试替身免费"）足以覆盖三层验证。

**理由**：分层让"确定性核心"（L1）与"需要替身/真实模型"（L2/L3）解耦——L1 永远可在任何环境零依赖跑绿，是回归的第一道闸门。

**后果**：L1/L2/L3 三层均为本切片的验收基线，`python tests/context_compiler/run_all.py` 一键跑绿；L2/L3 依赖 `core/stub_llm.py` 已注册的 `stub`/`recorded` 工厂后端。

### 决策 7：已知问题（显式推迟）——既有 `core/event_log.py` 的 append-only 违规与索引错位

**问题**：既有 `EventLog._add_event`（`core/event_log.py` 约 L264–274）在 `len(self._events) >= self.max_events` 时执行 `self._events.pop(0)` 移除最旧事件，并 `del self._event_index[removed.event_id]`。这有两处缺陷：
1. **违反 append-only**（§1.2 / §7.1-#4）：事件日志是唯一事实源，`pop(0)` 丢弃历史事件，使"重放全部历史"不再可能。
2. **`_event_index` 错位**：`_event_index` 是 `event_id → 位置(int)` 索引（约 L125）。`pop(0)` 后所有剩余事件位置整体前移 1，但索引里存的位置**未同步更新**——此后 `end_event(event_id)` 用旧位置 `self._events[idx]` 会取到**错误的事件**（off-by-one，溢出后每次 pop 累积错位）。

**本切片的处置**：**完全绕开，不改既有代码**。上下文编译器管道只吃 `InMemoryEventStore`（id→对象映射，无位置索引、无容量弹出）或不可变 `list[LHEvent]`；`adapt_event_log` 只读遍历 `log._events`，不触发 `_add_event`、不依赖 `_event_index`。因此本切片不受该缺陷影响。

**列为后续独立修复项**（对齐 §7.1-#1 重放一致 / §7.1-#7 additive 演进）：建议未来用"墓碑/归档"替代 `pop(0)`（容量管理不丢事实源），并把 `_event_index` 改为 id→对象映射或溢出后重建位置索引。**该修复属既有运行时改动，需独立 ADR 与回归验证，不在本切片范围。**

---

## 3. 被否决的替代方案

| 替代方案 | 否决理由 |
|---|---|
| **把层级（layer）写入 `Event`/`LHEvent` 信封字段** | 违反 §2.3"派生的永远可以重新派生"与 §7.1-#8"事实源不被改写"；分类器升级翻新（§3.1）将被迫改写历史事件，破坏 additive 演进（§2.2）。层级留在投影侧派生（决策 1）。 |
| **MVP 即上 SQLite + checkpoint + 30k 事件性能夹具** | 契约 §9-2 把存储选型列为下一阶段待决；MVP 目标是检验架构而非压测。过早引入持久化/性能基础设施增加依赖与不确定性，违背"一人 + agent、机制简单可独立验证"。`EventStore` Protocol 已冻结语法，Phase 2 换实现不改消费方（决策 3）。 |
| **扩展既有 `core.event_log.Event` 承载 §2.2 信封** | 既有 `Event` 是 tracing 事件，字段语义不重合；扩展会污染既有语义、破坏所有既有调用点，无法保证零回归。独立 `LHEvent` + 单向只读适配器实现零耦合，且适配器未来可无痛下线（决策 2）。 |
| **现在就把编译器深嵌入 `tool_agent`（pre-step 挂载）** | 契约 §8.2-1 的 pre-step 挂载是寄生验证的**后续**步骤；本切片先让编译器作为独立纯函数库可验证（L1 全绿）。深嵌入会把编译器与 agent 运行时耦合，使确定性测试受 agent 异步/IO 干扰，且违反"零回归"前提。挂载留给后续切片。 |
| **用 web 前端做测试可视化** | 测试 ≠ 分发；web 前端引入浏览器/构建/网络不确定性，与确定性优先冲突（决策 6）。 |

**本切片显式推迟项补录**：ToolContext 承载 event_log/scope/view 的完整事件溯源接入推迟；本切片编译器只吃 `InMemoryEventStore` / `list[LHEvent]`，接缝已由 `EventStore` Protocol 与 `adapt_event_log` 预留。

---

## 4. 验证

本切片以 `tests/context_compiler/` 三层测试为验收基线（`python tests/context_compiler/run_all.py` 一键跑绿）。契约 → 测试映射如下：

**L1 确定性不变量**（`test_layer1_invariants.py`）：

| 契约条款 | 测试函数 |
|---|---|
| §7.1-#1 重放一致 | `test_replay_determinism` / `test_compile_purity` |
| §7.1-#2 预算不超 + 不截断 | `test_budget_never_exceeded` |
| §6.2 梯子下潜 | `test_ladder_descent` |
| §7.1-#6 原始事件不跨域 | `test_scope_no_leak` |
| §2.3 否定式查询 | `test_negation_query` |
| §7.1-#5 / §5.4 删除无痕迹 | `test_delete_propagation` |
| §7.1-#7 additive 演进 | `test_additive_schema_replay` |
| §7.1-#8 事实源不被改写 | `test_store_not_mutated` |
| 附加：EventLog 单向桥接 | `test_event_log_adapter_bridge` |

> §7.1-#3（每个教学动作事件携带预测字段并引用至少一条证据）与 §7.1-#4（插件只能追加、不得直写投影）列为后续补齐项，L1 当前不承载这两项的可执行断言。

**L2 模拟学习者黄金场景**（`test_layer2_scenarios.py`，契约 §7.2 五场景）：

| 契约场景 | 测试函数 |
|---|---|
| §7.2 中断回归 | `test_scenario_interrupt_resume` |
| §7.2 跨课程 handoff | `test_scenario_cross_course_handoff` |
| §7.2 资料生命周期 | `test_scenario_resource_lifecycle` |
| §7.2 分类器升级 | `test_scenario_classifier_upgrade` |
| §7.2 用户争议 | `test_scenario_user_dispute` |

**L3 headless LLM-judge A/B**（`test_layer3_ab_eval.py`，契约 §6.1 核心命题）：

| 契约条款 | 测试函数 | 核心指标 |
|---|---|---|
| §6.1 上下文重建=编译问题 | `test_ab_eval_stub`（默认离线） | 编译组 `input_tokens < 0.3 ×` 全量组，且质量分 `≥` 全量组 `− 0.05` |
| §6.1 上下文重建=编译问题 | `test_ab_eval_live`（`LH_LIVE_LLM=1` 门控） | 同上，使用真实 LLM 打分 |

运行方式见 README“上下文编译器与三层测试”章节。
