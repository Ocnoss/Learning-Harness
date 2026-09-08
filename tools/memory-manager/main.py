"""Memory Manager - 记忆管理工具

功能：
1. 存储对话历史和执行记忆
2. 检索相关记忆
3. 搜索历史记录
4. 生成历史摘要
5. 清理过期记忆

设计要点：
- 作为可插拔工具，而非内核组件
- 使用 fast 模型 + 低思考强度
- 支持多种存储后端（内存/文件/数据库）
"""

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

# 确保能导入 core 模块
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.base_tool import BaseTool, ToolContext, ToolResult, ToolManifest


@dataclass
class MemoryEntry:
    """记忆条目"""
    entry_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    entry_type: str = "message"  # message | tool_call | summary | fact
    content: str = ""
    metadata: dict = field(default_factory=dict)

    # 向量嵌入（用于语义搜索，可选）
    embedding: list[float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(**data)


class MemoryManagerTool(BaseTool):
    """记忆管理工具"""

    name = "memory-manager"
    description = "存储、检索和管理对话历史与执行记忆"
    version = "1.0.0"

    # 推荐配置
    recommended_tier = "fast"
    recommended_thinking = "low"

    # AI 对任务的理解
    task_understanding = """
    管理 Tool-Agent 的记忆系统，包括：
    1. 存储对话历史、工具调用结果、重要事实
    2. 检索相关记忆以支持上下文理解
    3. 搜索历史记录
    4. 生成历史摘要
    5. 清理过期或无用记忆

    适用场景：
    - 需要记住之前的对话内容
    - 需要检索相关的历史执行记录
    - 需要总结长期对话历史
    - 需要清理无用记忆以释放空间
    """

    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["store", "retrieve", "search", "summarize", "clear", "stats"],
                "description": "执行动作"
            },
            "session_id": {
                "type": "string",
                "description": "会话ID（可选，默认使用当前会话）"
            },
            "entry_type": {
                "type": "string",
                "enum": ["message", "tool_call", "summary", "fact"],
                "description": "记忆类型（store 时必填）"
            },
            "content": {
                "type": "string",
                "description": "记忆内容（store 时必填）"
            },
            "metadata": {
                "type": "object",
                "description": "元数据（store 时可选）"
            },
            "query": {
                "type": "string",
                "description": "搜索查询（search 时必填）"
            },
            "entry_id": {
                "type": "string",
                "description": "条目ID（retrieve 时必填）"
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "description": "返回数量限制"
            },
            "max_age_hours": {
                "type": "integer",
                "description": "最大年龄（小时），用于清理"
            }
        },
        "required": ["action"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "entries": {"type": "array"},
            "summary": {"type": "string"},
            "stats": {"type": "object"},
            "cleared_count": {"type": "integer"}
        }
    }

    def __init__(self, context: ToolContext):
        super().__init__(context)
        # 内存存储（可扩展为文件/数据库）
        self._memories: dict[str, list[MemoryEntry]] = {}  # session_id -> entries
        self._index: dict[str, MemoryEntry] = {}  # entry_id -> entry

    async def execute(self, input_data: dict) -> ToolResult:
        """执行记忆管理"""
        action = input_data["action"]
        session_id = input_data.get("session_id", "default")

        handlers = {
            "store": self._store,
            "retrieve": self._retrieve,
            "search": self._search,
            "summarize": self._summarize,
            "clear": self._clear,
            "stats": self._stats,
        }

        handler = handlers.get(action)
        if not handler:
            return ToolResult.fail(f"未知动作: {action}")

        return await handler(session_id, input_data)

    async def _store(self, session_id: str, input_data: dict) -> ToolResult:
        """存储记忆"""
        entry_type = input_data.get("entry_type", "message")
        content = input_data.get("content", "")
        metadata = input_data.get("metadata", {})

        if not content:
            return ToolResult.fail("存储记忆需要提供 content 参数")

        entry = MemoryEntry(
            session_id=session_id,
            entry_type=entry_type,
            content=content,
            metadata=metadata,
        )

        # 存储
        if session_id not in self._memories:
            self._memories[session_id] = []
        self._memories[session_id].append(entry)
        self._index[entry.entry_id] = entry

        return ToolResult.ok(
            data={"entry_id": entry.entry_id, "stored": True},
            summary_for_llm=f"已存储 {entry_type} 记忆",
        )

    async def _retrieve(self, session_id: str, input_data: dict) -> ToolResult:
        """检索记忆"""
        entry_id = input_data.get("entry_id")
        limit = input_data.get("limit", 10)

        if entry_id:
            # 按 ID 检索
            entry = self._index.get(entry_id)
            if not entry:
                return ToolResult.fail(f"记忆不存在: {entry_id}")
            return ToolResult.ok(
                data={"entries": [entry.to_dict()]},
                summary_for_llm="已检索指定记忆",
            )
        else:
            # 检索最近记忆
            entries = self._memories.get(session_id, [])[-limit:]
            return ToolResult.ok(
                data={"entries": [e.to_dict() for e in entries]},
                summary_for_llm=f"已检索最近 {len(entries)} 条记忆",
            )

    async def _search(self, session_id: str, input_data: dict) -> ToolResult:
        """搜索记忆"""
        query = input_data.get("query", "")
        limit = input_data.get("limit", 10)

        if not query:
            return ToolResult.fail("搜索需要提供 query 参数")

        # 简单文本匹配（可扩展为语义搜索）
        query_lower = query.lower()
        matches = []

        for entry in self._memories.get(session_id, []):
            if query_lower in entry.content.lower():
                matches.append(entry)
                if len(matches) >= limit:
                    break

        return ToolResult.ok(
            data={"entries": [e.to_dict() for e in matches], "query": query},
            summary_for_llm=f"找到 {len(matches)} 条匹配记忆",
        )

    async def _summarize(self, session_id: str, input_data: dict) -> ToolResult:
        """生成历史摘要"""
        entries = self._memories.get(session_id, [])

        if not entries:
            return ToolResult.ok(
                data={"summary": "（无历史记录）"},
                summary_for_llm="无历史记录可总结",
            )

        # 统计
        type_counts = {}
        for e in entries:
            type_counts[e.entry_type] = type_counts.get(e.entry_type, 0) + 1

        # 最近内容预览
        recent = entries[-5:]
        previews = [e.content[:50] + "..." if len(e.content) > 50 else e.content for e in recent]

        summary = f"""会话 {session_id} 历史摘要：
- 总条目: {len(entries)}
- 类型分布: {type_counts}
- 最近内容: {'; '.join(previews)}"""

        return ToolResult.ok(
            data={"summary": summary, "stats": type_counts},
            summary_for_llm=f"已生成摘要：{len(entries)} 条记录",
        )

    async def _clear(self, session_id: str, input_data: dict) -> ToolResult:
        """清理记忆"""
        max_age_hours = input_data.get("max_age_hours")

        if session_id not in self._memories:
            return ToolResult.ok(
                data={"cleared_count": 0},
                summary_for_llm="无记忆可清理",
            )

        if max_age_hours:
            # 按年龄清理
            cutoff = time.time() - (max_age_hours * 3600)
            old_count = len(self._memories[session_id])
            self._memories[session_id] = [
                e for e in self._memories[session_id]
                if e.timestamp > cutoff
            ]
            cleared = old_count - len(self._memories[session_id])
        else:
            # 全部清理
            cleared = len(self._memories[session_id])
            self._memories[session_id] = []

        # 清理索引
        self._index = {
            eid: e for eid, e in self._index.items()
            if e.session_id != session_id or e in self._memories.get(session_id, [])
        }

        return ToolResult.ok(
            data={"cleared_count": cleared},
            summary_for_llm=f"已清理 {cleared} 条记忆",
        )

    async def _stats(self, session_id: str, input_data: dict) -> ToolResult:
        """获取统计信息"""
        entries = self._memories.get(session_id, [])

        type_counts = {}
        for e in entries:
            type_counts[e.entry_type] = type_counts.get(e.entry_type, 0) + 1

        return ToolResult.ok(
            data={
                "stats": {
                    "total_entries": len(entries),
                    "by_type": type_counts,
                    "sessions": list(self._memories.keys()),
                }
            },
            summary_for_llm=f"统计：{len(entries)} 条记忆",
        )

    def get_manifest(self) -> ToolManifest:
        """获取工具清单"""
        return ToolManifest(
            name=self.name,
            description_for_llm=self.description,
            applicable_scenarios=[
                "需要记住之前的对话内容",
                "需要检索相关的历史执行记录",
                "需要总结长期对话历史",
                "需要清理无用记忆以释放空间",
            ],
            input_requirements={
                "action": "执行动作：store/retrieve/search/summarize/clear/stats",
                "content": "记忆内容（store 时必填）",
                "query": "搜索查询（search 时必填）",
            },
            output_description="记忆条目列表、摘要文本或统计信息",
            examples=[
                {
                    "input": {"action": "store", "entry_type": "fact", "content": "用户偏好使用 fast 模型"},
                    "output_summary": "存储一条事实记忆"
                },
                {
                    "input": {"action": "search", "query": "考公资料"},
                    "output_summary": "搜索包含'考公资料'的记忆"
                }
            ],
            limitations=[
                "当前使用内存存储，重启后丢失",
                "语义搜索需要额外嵌入模型支持",
            ],
            recommended_tier=self.recommended_tier or "fast",
            recommended_thinking=self.recommended_thinking or "low",
        )


# ---- 工具入口点 ----
def create_tool(context: ToolContext) -> MemoryManagerTool:
    """创建工具实例"""
    return MemoryManagerTool(context)
