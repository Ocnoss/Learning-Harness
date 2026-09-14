# Learning Plugins - 学习工具插件架构文档

## 项目概述

本项目是一个基于 LLM 驱动的学习工具插件集合。每个工具都是独立的插件，可以单独使用，也可以与其他工具组合形成工作流。

## 目录结构规范

```
Learning plugins/
├── README.md                    # 本文件 - 项目架构文档
├── core/                        # 核心基础插件（LLM 装载器）
│   ├── plugin.json             # 插件元数据
│   ├── llm_client.py           # LLM 客户端封装
│   ├── base_tool.py            # 工具基类
│   └── workflow.py             # 工作流引擎
├── tools/                       # 工具插件目录
│   ├── tool-template/          # 工具模板（复制此目录创建新工具）
│   │   ├── plugin.json
│   │   ├── main.py
│   │   └── README.md
│   ├── flashcard-generator/    # 示例：闪卡生成工具
│   ├── note-summarizer/        # 示例：笔记摘要工具
│   └── ...                     # 其他工具
└── workflows/                   # 预定义工作流配置
    └── example-workflow.json
```

## 核心插件 (core/)

### plugin.json 规范

```json
{
  "name": "core",
  "version": "1.0.0",
  "description": "LLM 基础插件，提供统一的 LLM 调用接口",
  "type": "core",
  "provides": ["llm_client", "base_tool", "workflow_engine"],
  "config_schema": {
    "llm_provider": {
      "type": "string",
      "enum": ["openai", "anthropic", "local", "custom"],
      "default": "openai"
    },
    "api_key": {
      "type": "string",
      "required": true,
      "env": "LLM_API_KEY"
    },
    "model": {
      "type": "string",
      "default": "gpt-4"
    },
    "base_url": {
      "type": "string",
      "required": false
    }
  }
}
```

### LLM 客户端接口 (llm_client.py)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Literal

@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

@dataclass
class CompletionResponse:
    content: str
    usage: dict
    model: str

class LLMClient(ABC):
    """LLM 客户端抽象基类"""
    
    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> CompletionResponse:
        """同步完成请求"""
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """流式完成请求"""
        pass

class LLMClientFactory:
    """LLM 客户端工厂"""
    
    _clients: dict[str, type[LLMClient]] = {}
    
    @classmethod
    def register(cls, provider: str, client_class: type[LLMClient]):
        cls._clients[provider] = client_class
    
    @classmethod
    def create(cls, provider: str, config: dict) -> LLMClient:
        if provider not in cls._clients:
            raise ValueError(f"Unknown LLM provider: {provider}")
        return cls._clients[provider](config)
```

### 工具基类 (base_tool.py)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class ToolContext:
    """工具执行上下文"""
    llm_client: "LLMClient"
    config: dict
    shared_state: dict  # 跨工具共享状态

@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any
    error: str | None = None
    metadata: dict = None

class BaseTool(ABC):
    """学习工具基类"""
    
    # 工具元数据（子类必须定义）
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    # 输入/输出 Schema（用于工作流编排）
    input_schema: dict = {}
    output_schema: dict = {}
    
    # 依赖的其他工具
    dependencies: list[str] = []
    
    # 推荐模型层级（子类可覆盖）
    # 可选值: "fast"（简单任务）, "balanced"（中等任务）, "flagship"（复杂推理）
    # 为 None 时使用 balanced
    recommended_tier: str | None = None
    
    def __init__(self, context: ToolContext):
        self.context = context
        self.llm = context.llm_client
    
    @abstractmethod
    async def execute(self, input_data: dict) -> ToolResult:
        """执行工具逻辑"""
        pass
    
    def validate_input(self, input_data: dict) -> bool:
        """验证输入数据"""
        # 实现 JSON Schema 验证
        return True
    
    async def call_tool(
        self,
        tool_name: str,
        input_data: dict
    ) -> ToolResult:
        """调用其他工具"""
        # 通过工具注册表调用
        pass
```

### 工作流引擎 (workflow.py)

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class WorkflowStep:
    tool_name: str
    input_mapping: dict  # 输入映射：{目标字段: 来源字段或值}
    condition: str | None = None  # 执行条件表达式

@dataclass
class Workflow:
    name: str
    description: str
    steps: list[WorkflowStep]

class WorkflowEngine:
    """工作流执行引擎"""
    
    def __init__(self, tool_registry: "ToolRegistry"):
        self.registry = tool_registry
    
    async def execute(
        self,
        workflow: Workflow,
        initial_input: dict
    ) -> dict[str, Any]:
        """执行工作流"""
        results = {}
        context = {"input": initial_input, "results": results}
        
        for step in workflow.steps:
            # 解析输入映射
            step_input = self._resolve_input(step.input_mapping, context)
            
            # 检查执行条件
            if step.condition and not self._eval_condition(step.condition, context):
                continue
            
            # 执行工具
            tool = self.registry.get_tool(step.tool_name)
            result = await tool.execute(step_input)
            results[step.tool_name] = result
        
        return results
```

## 工具插件规范 (tools/)

### plugin.json 规范

每个工具目录必须包含 `plugin.json`：

```json
{
  "name": "flashcard-generator",
  "version": "1.0.0",
  "description": "从文本内容生成学习闪卡",
  "type": "tool",
  "author": "",
  "entry": "main.py",
  "dependencies": [],
  "recommended_tier": "balanced",
  "input_schema": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "要转换的学习内容"
      },
      "card_count": {
        "type": "integer",
        "default": 10,
        "description": "生成闪卡数量"
      },
      "card_type": {
        "type": "string",
        "enum": ["qa", "cloze", "definition"],
        "default": "qa"
      }
    },
    "required": ["content"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "cards": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "front": {"type": "string"},
            "back": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}}
          }
        }
      }
    }
  }
}
```

### main.py 模板

```python
from core.base_tool import BaseTool, ToolContext, ToolResult

class FlashcardGeneratorTool(BaseTool):
    name = "flashcard-generator"
    description = "从文本内容生成学习闪卡"
    version = "1.0.0"
    
    # 推荐模型层级：闪卡生成属于中等复杂度任务
    recommended_tier = "balanced"
    
    input_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "card_count": {"type": "integer", "default": 10},
            "card_type": {"type": "string", "default": "qa"}
        },
        "required": ["content"]
    }
    
    output_schema = {
        "type": "object",
        "properties": {
            "cards": {"type": "array"}
        }
    }
    
    async def execute(self, input_data: dict) -> ToolResult:
        content = input_data["content"]
        card_count = input_data.get("card_count", 10)
        card_type = input_data.get("card_type", "qa")
        
        # 构建 LLM 提示词
        prompt = self._build_prompt(content, card_count, card_type)
        
        # 调用 LLM
        # tier: 由 _resolve_tier() 从 recommended_tier 解析
        # temperature / max_tokens: 由工具层根据任务需求决定
        response = await self.llm.complete(
            messages=[
                {"role": "system", "content": "你是一个专业的学习闪卡制作助手。"},
                {"role": "user", "content": prompt}
            ],
            tier=self._resolve_tier(),
            temperature=0.7,  # 中等温度，平衡创造性和准确性
            max_tokens=4096,  # 闪卡生成需要较长输出
        )
        
        # 解析结果
        cards = self._parse_response(response.content)
        
        return ToolResult(
            success=True,
            data={"cards": cards},
            metadata={"count": len(cards)}
        )
    
    def _build_prompt(self, content: str, count: int, card_type: str) -> str:
        return f"""请将以下内容转换为 {count} 张 {card_type} 类型的学习闪卡：

内容：
{content}

请以 JSON 格式返回，格式如下：
[{{"front": "问题", "back": "答案", "tags": ["标签1"]}}]"""
    
    def _parse_response(self, response: str) -> list[dict]:
        import json
        # 提取 JSON 部分并解析
        # ... 实现解析逻辑
        pass

# 工具入口点
def create_tool(context: ToolContext) -> FlashcardGeneratorTool:
    return FlashcardGeneratorTool(context)
```

## 工具间调用

### 直接调用

```python
class NoteSummarizerTool(BaseTool):
    name = "note-summarizer"
    dependencies = ["flashcard-generator"]  # 声明依赖
    
    async def execute(self, input_data: dict) -> ToolResult:
        # 先生成摘要
        summary = await self._generate_summary(input_data["content"])
        
        # 调用闪卡生成工具
        flashcard_result = await self.call_tool(
            "flashcard-generator",
            {"content": summary, "card_count": 5}
        )
        
        return ToolResult(
            success=True,
            data={
                "summary": summary,
                "flashcards": flashcard_result.data["cards"]
            }
        )
```

### 工作流配置 (workflows/)

```json
{
  "name": "note-to-flashcard",
  "description": "笔记 → 摘要 → 闪卡 工作流",
  "steps": [
    {
      "tool_name": "note-summarizer",
      "input_mapping": {
        "content": "$.input.note_content",
        "max_length": 500
      }
    },
    {
      "tool_name": "flashcard-generator",
      "input_mapping": {
        "content": "$.results.note-summarizer.summary",
        "card_count": 10
      }
    }
  ]
}
```

## 开发指南

### 创建新工具

1. 复制模板目录：
   ```bash
   cp -r tools/tool-template tools/your-tool-name
   ```

2. 修改 `plugin.json`：
   - 更新 `name`、`description`
   - 定义 `input_schema` 和 `output_schema`
   - 声明 `dependencies`（如有）
   - 设置 `recommended_tier`（推荐模型层级）

3. 实现 `main.py`：
   - 继承 `BaseTool`
   - 设置 `recommended_tier` 类属性
   - 实现 `execute` 方法
   - 使用 `self.llm` 调用 LLM，传入 `tier=self._resolve_tier()`

4. 测试工具：
   ```python
   # 在工具目录创建 test.py
   import asyncio
   from core.llm_client import LLMClientFactory
   from core.base_tool import ToolContext
   from main import create_tool
   
   async def test():
       llm = LLMClientFactory.create("openai", {"api_key": "..."})
       context = ToolContext(llm_client=llm, config={}, shared_state={})
       tool = create_tool(context)
       result = await tool.execute({"content": "测试内容"})
       print(result)
   
   asyncio.run(test())
   ```

### 模型层级选择指南

| 层级 | 适用场景 | 示例工具 |
|------|----------|----------|
| `fast` | 简单任务、低成本、高速度 | 文件索引、格式转换、简单分类 |
| `balanced` | 中等复杂度任务（默认） | 内容摘要、问答生成、翻译 |
| `flagship` | 复杂推理、高质量输出 | 深度分析、创意写作、多步规划 |

### 调用参数指南

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `temperature` | 控制输出随机性 | 0.1-0.3: 准确稳定; 0.5-0.7: 平衡; 0.8-1.5: 创造性 |
| `max_tokens` | 控制输出长度 | 100-500: 短输出; 500-2048: 中输出; 2048-8192: 长输出 |

### 工具命名规范

- 使用 kebab-case：`flashcard-generator`、`note-summarizer`
- 名称应清晰表达功能
- 避免使用缩写，除非广为人知

### 提示词工程建议

1. **系统提示词**：定义角色和输出格式
2. **用户提示词**：包含具体任务和输入数据
3. **输出解析**：始终要求 LLM 返回结构化 JSON
4. **错误处理**：实现重试机制和结果验证

## 配置管理

### 环境变量

```bash
# .env 文件
LLM_API_KEY=your-api-key
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
```

### 全局配置 (config.json)

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "temperature": 0.7
  },
  "tools": {
    "flashcard-generator": {
      "default_card_count": 10
    }
  }
}
```

## 上下文编译器与三层测试

> 本节对应内核契约 `LH内核契约-v0.1.md` §6.1/§6.2（上下文管理）与 §7（架构验证）。
> 完整架构决策记录见 [`docs/adr/ADR-0001-context-compiler.md`](docs/adr/ADR-0001-context-compiler.md)。

### 编译器是什么

上下文重建被定位为**编译问题，不是检索问题**（契约 §6.1-2）：

```
compile(projection, query) -> ContextBundle
```

输入当前情境（作用域 scope、目标 goal、token 预算、时间锚点 as_of、否定查询、交接包）+ 事件历史的投影，输出 **token 预算约束下的上下文包**。核心工作方式是**按预算沿抽象梯子下潜**（契约 §6.2）：预算紧只装 L3 断言，预算松展开 L2 证据，需要逐字证据且显式允许时再沿指针取 L1 原文；装不下整项则跳过（降级不截断，§7.1-#2）。

`compile` 与 `fold` 是**确定性纯函数**：禁用时钟/随机/IO，时间只走 `query.as_of`，同一输入永远产出逐字节相同的 `replay_digest`——这是 §7.1-#1 重放不变量的实现基础。

### `core/context/` 模块地图

| 模块 | 职责 | 关键符号 |
|------|------|----------|
| `envelope.py` | 契约 §2.2 事件信封（frozen，additive 演进） | `LHEvent` |
| `store.py` | append-only 事件存储协议 + 内存实现；`(scope,verb,topic)→last_ts` 的 O(1) 否定查询索引 | `EventStore` / `InMemoryEventStore` |
| `tokens.py` | 可注入的确定性 token 估算（默认字符法，tiktoken 推迟） | `TokenEstimator` / `CharBasedEstimator` |
| `projection.py` | 事件日志的派生只读视图；层级在此**派生**（非事件固有字段）；rebuild/fold 两条等价路径 | `Projection` / `ProjectionItem` / `rebuild` / `fold` / `derive_level` |
| `compiler.py` | 按预算沿梯子下潜的确定性编译器 | `ContextQuery` / `NegationQuery` / `ContextItem` / `ContextBundle` / `compile` |
| `event_log_adapter.py` | 既有 `EventLog` → `LHEvent` 的**单向只读**桥接（未来可无痛下线） | `adapt_event_log` |

该子包运行期不 import 任何既有 core 运行时模块（`event_log` 仅在适配器函数体内延迟导入），既有代码也不 import 本子包——零回归、可独立验证。

### 如何运行三层测试

测试分三层（L1 确定性不变量 / L2 模拟学习者黄金场景 / L3 headless LLM-judge A/B），均为**可独立运行的脚本**（末尾 `asyncio.run(main())` 风格，退出码反映通过/失败）。在仓库根目录用解释器直跑：

```bash
# 第一层：确定性不变量（纯函数，无 LLM/网络/原生依赖/真实时钟）
python tests/context_compiler/test_layer1_invariants.py

# 第二层：模拟学习者黄金场景（契约 §7.2 五个端到端场景）
python tests/context_compiler/test_layer2_scenarios.py

# 第三层：headless LLM-judge A/B（默认用测试替身离线回放）
python tests/context_compiler/test_layer3_ab_eval.py

# 三层一键全跑
python tests/context_compiler/run_all.py
```

> 本仓库未安装 pytest，上述脚本均设计为零依赖直跑。Windows PowerShell 下 `python` 若不在 PATH，请用绝对路径 `D:\miniconda\python.exe`（如 `D:\miniconda\python.exe tests/context_compiler/run_all.py`）。`test_*` 函数名仅为“未来 pytest 可发现”的兼容命名；若要用 pytest 跑，需先安装 pytest，**且 layer3 与 layer2 的 `test_scenario_*` 同为 async 协程**，均需 `pytest-asyncio`（否则裸跑 pytest 会静默收集跳过）。因此**明确推荐用 `python tests/context_compiler/run_all.py` 运行三层**。

### §7.1 不变量 → 测试映射（第一层）

| 契约条款 | 测试函数 |
|----------|----------|
| §7.1-#1 重放一致 | `test_replay_determinism` / `test_compile_purity` |
| §7.1-#2 预算不超 + 不截断 | `test_budget_never_exceeded` |
| §6.2 梯子下潜 | `test_ladder_descent` |
| §7.1-#6 原始事件不跨作用域 | `test_scope_no_leak` |
| §2.3 否定式查询 | `test_negation_query` |
| §7.1-#5 / §5.4 删除无痕迹 | `test_delete_propagation` |
| §7.1-#7 additive 演进 | `test_additive_schema_replay` |
| §7.1-#8 事实源不被改写 | `test_store_not_mutated` |
| 附加：EventLog 单向桥接 | `test_event_log_adapter_bridge` |

### 第三层 live 模式的环境变量门控

第三层默认**离线确定性**：用 `core/stub_llm.py` 的 `StubLLMClient` 顺序回放做确定性评估，不触网、不需 API key；`RecordedLLMClient` / `RecordingLLMClient` 提供磁带领制-回放能力（工厂注册名 `stub` / `recorded`，契约 §8.3“测试替身免费”）。仅当显式开启 live 模式才调用真实 LLM：

```bash
# PowerShell
$env:LH_LIVE_LLM = "1"          # 打开 live 门控（默认关闭 → 走替身回放）
$env:LLM_API_KEY = "your-key"    # live 模式必填
python tests/context_compiler/run_all.py
```

```bash
# bash
LH_LIVE_LLM=1 LLM_API_KEY=your-key python tests/context_compiler/run_all.py
```

未设置 `LH_LIVE_LLM` 时，第三层一律走替身回放，保证 CI 与本地零依赖跑绿；`LLM_API_KEY` 缺失时 live 模式会跳过而非报错。

### 为何不用 web 前端做测试

**测试 ≠ 分发**。本切片目标是检验架构、让不变量自动报警（契约 §7.3 fitness functions），不是给学习者做界面。web 前端会引入浏览器/构建/网络不确定性，与“确定性优先”冲突；前端形态属契约 §9-6 的下一阶段待决。详见 [ADR-0001 决策 6](docs/adr/ADR-0001-context-compiler.md)。

## 版本历史

- v1.2.0 (2026-09)【文档/切片里程碑，非包版本升级；core/__init__.py 的 `__version__` 仍为 `1.0.0`】：上下文编译器薄切片（`core/context/`）+ 三层测试 + ADR-0001
- v1.1.0 (2026-09-06): 添加多层级模型配置系统（fast/balanced/flagship）
- v1.0.0 (2026-09-05): 初始架构设计
