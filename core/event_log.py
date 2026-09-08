"""事件记录系统 - 内核级事件日志

记录 Tool-Agent 执行过程中的所有关键事件，支持：
- 结构化事件存储
- 事件查询与过滤
- 执行轨迹回放
- 性能分析
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


class EventType(Enum):
    """事件类型"""
    # Agent 生命周期
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"

    # 思考与决策
    THINK_START = "think_start"
    THINK_END = "think_end"
    DECISION_MADE = "decision_made"

    # 工具调用
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_CALL_ERROR = "tool_call_error"

    # LLM 调用
    LLM_CALL_START = "llm_call_start"
    LLM_CALL_END = "llm_call_end"
    LLM_CALL_ERROR = "llm_call_error"

    # 用户交互
    USER_CONFIRM_REQUIRED = "user_confirm_required"
    USER_CONFIRMED = "user_confirmed"
    USER_CANCELLED = "user_cancelled"

    # 自定义
    CUSTOM = "custom"


@dataclass
class Event:
    """事件记录"""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: EventType = EventType.CUSTOM
    session_id: str = ""
    iteration: int = 0

    # 事件数据
    data: dict = field(default_factory=dict)

    # 关联信息
    parent_event_id: str | None = None  # 父事件（如工具调用属于某个思考步骤）
    correlation_id: str | None = None   # 关联ID（如 LLM 调用与工具调用）

    # 性能指标
    duration_ms: float | None = None

    def to_dict(self) -> dict:
        """序列化为字典"""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["datetime"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        """从字典反序列化"""
        data = dict(data)
        data["event_type"] = EventType(data.pop("event_type"))
        data.pop("datetime", None)
        return cls(**data)


class EventLog:
    """事件日志

    记录和管理所有事件，支持持久化和查询。

    使用示例:
        log = EventLog(session_id="test-session")

        # 记录事件
        event = log.record(
            EventType.TOOL_CALL_START,
            data={"tool": "resource-indexer", "input": {...}}
        )

        # 结束事件（记录耗时）
        log.end_event(event.event_id, data={"result": "success"})

        # 查询事件
        tool_calls = log.query(event_type=EventType.TOOL_CALL_START)
    """

    def __init__(
        self,
        session_id: str | None = None,
        persist_path: str | Path | None = None,
        max_events: int = 10000,
    ):
        """初始化事件日志

        Args:
            session_id: 会话ID（默认自动生成）
            persist_path: 持久化文件路径（可选）
            max_events: 最大事件数量（防止内存溢出）
        """
        self.session_id = session_id or str(uuid4())
        self.persist_path = Path(persist_path) if persist_path else None
        self.max_events = max_events

        self._events: list[Event] = []
        self._event_index: dict[str, int] = {}  # event_id -> index
        self._listeners: list[Callable[[Event], None]] = []

        # 统计信息
        self._stats = {
            "total_events": 0,
            "tool_calls": 0,
            "llm_calls": 0,
            "errors": 0,
            "total_duration_ms": 0,
        }

    def record(
        self,
        event_type: EventType,
        data: dict | None = None,
        iteration: int = 0,
        parent_event_id: str | None = None,
        correlation_id: str | None = None,
    ) -> Event:
        """记录事件

        Args:
            event_type: 事件类型
            data: 事件数据
            iteration: 当前迭代次数
            parent_event_id: 父事件ID
            correlation_id: 关联ID

        Returns:
            创建的事件对象
        """
        event = Event(
            event_type=event_type,
            session_id=self.session_id,
            iteration=iteration,
            data=data or {},
            parent_event_id=parent_event_id,
            correlation_id=correlation_id,
        )

        self._add_event(event)
        return event

    def end_event(
        self,
        event_id: str,
        data: dict | None = None,
        error: str | None = None,
    ) -> Event | None:
        """结束事件（记录耗时）

        Args:
            event_id: 事件ID
            data: 附加数据
            error: 错误信息（如有）

        Returns:
            更新后的事件对象
        """
        if event_id not in self._event_index:
            return None

        idx = self._event_index[event_id]
        event = self._events[idx]

        # 计算耗时
        event.duration_ms = (time.time() - event.timestamp) * 1000

        # 更新数据
        if data:
            event.data.update(data)
        if error:
            event.data["error"] = error

        # 更新统计
        self._stats["total_duration_ms"] += event.duration_ms

        # 触发监听器
        self._notify(event)

        # 持久化
        self._persist_event(event)

        return event

    def query(
        self,
        event_type: EventType | None = None,
        iteration: int | None = None,
        min_duration_ms: float | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """查询事件

        Args:
            event_type: 过滤事件类型
            iteration: 过滤迭代次数
            min_duration_ms: 过滤最小耗时
            limit: 返回数量限制

        Returns:
            匹配的事件列表
        """
        results = []
        for event in reversed(self._events):  # 最新的在前
            if event_type and event.event_type != event_type:
                continue
            if iteration is not None and event.iteration != iteration:
                continue
            if min_duration_ms is not None and (event.duration_ms or 0) < min_duration_ms:
                continue

            results.append(event)
            if len(results) >= limit:
                break

        return results

    def get_tool_call_trace(self, correlation_id: str) -> list[Event]:
        """获取工具调用轨迹（包括相关的 LLM 调用）

        Args:
            correlation_id: 关联ID

        Returns:
            相关事件列表，按时间排序
        """
        events = [e for e in self._events if e.correlation_id == correlation_id]
        return sorted(events, key=lambda e: e.timestamp)

    def get_stats(self) -> dict:
        """获取统计信息"""
        return dict(self._stats)

    def add_listener(self, callback: Callable[[Event], None]):
        """添加事件监听器"""
        self._listeners.append(callback)

    def _add_event(self, event: Event):
        """添加事件到日志"""
        # 检查容量
        if len(self._events) >= self.max_events:
            # 移除最旧的事件
            removed = self._events.pop(0)
            del self._event_index[removed.event_id]

        # 添加新事件
        self._event_index[event.event_id] = len(self._events)
        self._events.append(event)

        # 更新统计
        self._stats["total_events"] += 1
        if event.event_type == EventType.TOOL_CALL_START:
            self._stats["tool_calls"] += 1
        elif event.event_type == EventType.LLM_CALL_START:
            self._stats["llm_calls"] += 1
        elif event.event_type in (EventType.AGENT_ERROR, EventType.TOOL_CALL_ERROR, EventType.LLM_CALL_ERROR):
            self._stats["errors"] += 1

        # 触发监听器
        self._notify(event)

        # 持久化
        self._persist_event(event)

    def _notify(self, event: Event):
        """通知监听器"""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass  # 忽略监听器错误

    def _persist_event(self, event: Event):
        """持久化事件到文件"""
        if not self.persist_path:
            return

        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass  # 忽略持久化错误

    def export_trace(self, output_path: str | Path):
        """导出完整执行轨迹

        Args:
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trace = {
            "session_id": self.session_id,
            "exported_at": datetime.now().isoformat(),
            "stats": self.get_stats(),
            "events": [e.to_dict() for e in self._events],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self._events)
