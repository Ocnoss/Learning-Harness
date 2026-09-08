"""Token Counter - LLM 调用统计工具

功能：
1. 记录每次 LLM 调用的 token 消耗（输入/输出）
2. 计算输出速度（tokens/秒）
3. 生成统计数据和可视化报告

使用方式：
- 作为独立工具调用，查询统计数据
- 作为装饰器包装 LLM 客户端，自动记录调用
"""

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# 确保能导入 core 模块
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.base_tool import BaseTool, ToolContext, ToolResult


@dataclass
class CallRecord:
    """单次 LLM 调用记录"""
    timestamp: str  # ISO 8601 格式
    model: str
    tier: str | None
    input_tokens: int
    output_tokens: int
    duration_ms: float
    tool_name: str | None = None

    @property
    def speed(self) -> float:
        """输出速度 (tokens/s)"""
        if self.duration_ms <= 0:
            return 0.0
        return self.output_tokens / (self.duration_ms / 1000)


@dataclass
class TokenStats:
    """Token 统计数据"""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration_ms: float = 0.0
    records: list[CallRecord] = field(default_factory=list)

    @property
    def avg_speed(self) -> float:
        """平均输出速度 (tokens/s)"""
        if self.total_duration_ms <= 0:
            return 0.0
        return self.total_output_tokens / (self.total_duration_ms / 1000)

    def add_record(self, record: CallRecord):
        """添加记录"""
        self.records.append(record)
        self.total_calls += 1
        self.total_input_tokens += record.input_tokens
        self.total_output_tokens += record.output_tokens
        self.total_duration_ms += record.duration_ms

    def filter_by_time(
        self,
        start_time: str | None = None,
        end_time: str | None = None
    ) -> "TokenStats":
        """按时间范围过滤"""
        filtered = TokenStats()
        for record in self.records:
            if start_time and record.timestamp < start_time:
                continue
            if end_time and record.timestamp > end_time:
                continue
            filtered.add_record(record)
        return filtered


class TokenCounter:
    """Token 计数器 - 单例模式"""

    _instance: "TokenCounter | None" = None
    _storage_path: Path | None = None

    def __new__(cls, storage_path: Path | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._stats = TokenStats()
            cls._instance._storage_path = storage_path
            if storage_path and storage_path.exists():
                cls._instance._load()
        return cls._instance

    def _load(self):
        """从文件加载统计数据"""
        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._stats = TokenStats(
                    total_calls=data.get("total_calls", 0),
                    total_input_tokens=data.get("total_input_tokens", 0),
                    total_output_tokens=data.get("total_output_tokens", 0),
                    total_duration_ms=data.get("total_duration_ms", 0.0),
                    records=[
                        CallRecord(**r) for r in data.get("records", [])
                    ]
                )
        except Exception:
            self._stats = TokenStats()

    def _save(self):
        """保存统计数据到文件"""
        if not self._storage_path:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_calls": self._stats.total_calls,
            "total_input_tokens": self._stats.total_input_tokens,
            "total_output_tokens": self._stats.total_output_tokens,
            "total_duration_ms": self._stats.total_duration_ms,
            "records": [asdict(r) for r in self._stats.records]
        }
        with open(self._storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        tier: str | None = None,
        tool_name: str | None = None
    ):
        """记录一次 LLM 调用"""
        record = CallRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            tier=tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            tool_name=tool_name
        )
        self._stats.add_record(record)
        self._save()

    def get_stats(
        self,
        start_time: str | None = None,
        end_time: str | None = None
    ) -> TokenStats:
        """获取统计数据"""
        if start_time or end_time:
            return self._stats.filter_by_time(start_time, end_time)
        return self._stats

    def clear(self):
        """清除所有记录"""
        self._stats = TokenStats()
        self._save()

    def generate_report(self, stats: TokenStats | None = None) -> str:
        """生成统计报告（Markdown 格式）"""
        if stats is None:
            stats = self._stats

        lines = [
            "# LLM 调用统计报告",
            "",
            f"生成时间: {datetime.now().isoformat()}",
            "",
            "## 总体统计",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 总调用次数 | {stats.total_calls} |",
            f"| 总输入 Token | {stats.total_input_tokens:,} |",
            f"| 总输出 Token | {stats.total_output_tokens:,} |",
            f"| 总耗时 | {stats.total_duration_ms/1000:.2f} 秒 |",
            f"| 平均速度 | {stats.avg_speed:.2f} tokens/s |",
            "",
        ]

        if stats.records:
            lines.extend([
                "## 最近调用记录",
                "",
                "| 时间 | 模型 | 层级 | 输入 | 输出 | 耗时 | 速度 |",
                "|------|------|------|------|------|------|------|",
            ])
            for record in stats.records[-20:]:  # 最近 20 条
                lines.append(
                    f"| {record.timestamp[:19]} | {record.model} | "
                    f"{record.tier or '-'} | {record.input_tokens} | "
                    f"{record.output_tokens} | {record.duration_ms:.0f}ms | "
                    f"{record.speed:.1f} t/s |"
                )

        return "\n".join(lines)


class TokenCounterTool(BaseTool):
    """Token 计数器工具"""

    name = "token-counter"
    description = "LLM 调用统计工具"
    version = "1.0.0"

    # 不需要 LLM，设为 None
    recommended_tier = None

    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["record", "stats", "report", "clear"],
                "description": "操作类型"
            },
            "start_time": {"type": "string", "description": "开始时间"},
            "end_time": {"type": "string", "description": "结束时间"},
            "record_data": {"type": "object", "description": "调用数据"}
        },
        "required": ["action"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "total_calls": {"type": "integer"},
            "total_input_tokens": {"type": "integer"},
            "total_output_tokens": {"type": "integer"},
            "avg_speed": {"type": "number"},
            "records": {"type": "array"},
            "report": {"type": "string"}
        }
    }

    # 默认存储路径
    DEFAULT_STORAGE = Path(__file__).parent / "token_stats.json"

    def __init__(self, context: ToolContext):
        super().__init__(context)
        storage_path = context.config.get("storage_path", self.DEFAULT_STORAGE)
        self.counter = TokenCounter(Path(storage_path))

    async def execute(self, input_data: dict) -> ToolResult:
        """执行统计操作"""
        action = input_data["action"]

        if action == "record":
            return self._handle_record(input_data.get("record_data", {}))
        elif action == "stats":
            return self._handle_stats(
                input_data.get("start_time"),
                input_data.get("end_time")
            )
        elif action == "report":
            return self._handle_report(
                input_data.get("start_time"),
                input_data.get("end_time")
            )
        elif action == "clear":
            return self._handle_clear()
        else:
            return ToolResult.fail(f"未知操作: {action}")

    def _handle_record(self, record_data: dict) -> ToolResult:
        """处理记录操作"""
        self.counter.record(
            model=record_data.get("model", "unknown"),
            tier=record_data.get("tier"),
            input_tokens=record_data.get("input_tokens", 0),
            output_tokens=record_data.get("output_tokens", 0),
            duration_ms=record_data.get("duration_ms", 0),
            tool_name=record_data.get("tool_name")
        )
        return ToolResult.ok({"message": "记录成功"})

    def _handle_stats(
        self,
        start_time: str | None,
        end_time: str | None
    ) -> ToolResult:
        """处理统计查询"""
        stats = self.counter.get_stats(start_time, end_time)
        return ToolResult.ok({
            "total_calls": stats.total_calls,
            "total_input_tokens": stats.total_input_tokens,
            "total_output_tokens": stats.total_output_tokens,
            "avg_speed": stats.avg_speed,
            "records": [asdict(r) for r in stats.records[-100:]]  # 最近 100 条
        })

    def _handle_report(
        self,
        start_time: str | None,
        end_time: str | None
    ) -> ToolResult:
        """处理报告生成"""
        stats = self.counter.get_stats(start_time, end_time)
        report = self.counter.generate_report(stats)
        return ToolResult.ok({
            "report": report,
            "total_calls": stats.total_calls,
            "total_input_tokens": stats.total_input_tokens,
            "total_output_tokens": stats.total_output_tokens,
            "avg_speed": stats.avg_speed
        })

    def _handle_clear(self) -> ToolResult:
        """处理清除操作"""
        self.counter.clear()
        return ToolResult.ok({"message": "已清除所有记录"})


# ---- 装饰器：自动记录 LLM 调用 ----
def count_tokens(tool_name: str | None = None):
    """装饰器：自动记录 LLM 调用的 token 消耗

    使用示例:
        @count_tokens(tool_name="my-tool")
        async def my_llm_call():
            response = await llm.complete(...)
            return response
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000

            # 尝试从结果中提取 token 信息
            if hasattr(result, 'usage'):
                usage = result.usage
                counter = TokenCounter()
                counter.record(
                    model=getattr(result, 'model', 'unknown'),
                    tier=getattr(result, 'tier', None),
                    input_tokens=usage.get('prompt_tokens', 0),
                    output_tokens=usage.get('completion_tokens', 0),
                    duration_ms=duration_ms,
                    tool_name=tool_name
                )

            return result
        return wrapper
    return decorator


# ---- 工具入口点 ----
def create_tool(context: ToolContext) -> TokenCounterTool:
    """创建工具实例"""
    return TokenCounterTool(context)
