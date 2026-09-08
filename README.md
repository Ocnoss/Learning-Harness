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

## 版本历史

- v1.1.0 (2026-09-06): 添加多层级模型配置系统（fast/balanced/flagship）
- v1.0.0 (2026-09-05): 初始架构设计
