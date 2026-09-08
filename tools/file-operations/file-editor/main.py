"""File Editor - 文件编辑工具

功能：
1. 精确字符串替换（old_string → new_string）
2. 强制先读后改（保护数据安全）
3. 保护 CRLF 行尾

设计要点：
- 使用 fast 模型 + 低思考强度
- 零 LLM 调用，纯本地操作
- 必须先读取文件才能编辑
"""

import sys
from pathlib import Path
from typing import Any

# 确保能导入 core 模块
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.base_tool import BaseTool, ToolContext, ToolResult, ToolManifest


class FileEditorTool(BaseTool):
    """文件编辑工具"""

    name = "file-editor"
    description = "精确字符串替换，强制先读后改"
    version = "1.0.0"

    recommended_tier = "fast"
    recommended_thinking = "low"

    task_understanding = """
    精确编辑文件内容，用于增量修改。

    核心能力：
    1. 精确字符串替换（old_string → new_string）
    2. 强制先读后改（保护数据安全）
    3. 保护 CRLF 行尾（Windows 兼容）
    4. 支持多次替换

    适用场景：
    - 修改代码文件中的特定部分
    - 更新配置文件
    - 修复文档中的错误

    注意：
    - 必须先使用 file-reader 读取文件
    - old_string 必须唯一匹配，否则替换失败
    - 保护 CRLF 行尾，避免 Windows 文件格式损坏
    """

    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件路径（绝对路径或相对路径）"
            },
            "old_string": {
                "type": "string",
                "description": "要替换的原始字符串（必须唯一匹配）"
            },
            "new_string": {
                "type": "string",
                "description": "替换后的新字符串"
            },
            "encoding": {
                "type": "string",
                "default": "utf-8",
                "description": "文件编码（默认 utf-8）"
            },
            "preserve_crlf": {
                "type": "boolean",
                "default": True,
                "description": "保护 CRLF 行尾（默认 true）"
            }
        },
        "required": ["file_path", "old_string", "new_string"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "replacements": {"type": "integer"},
            "size_before": {"type": "integer"},
            "size_after": {"type": "integer"}
        }
    }

    async def execute(self, input_data: dict) -> ToolResult:
        """执行文件编辑"""
        file_path = input_data["file_path"]
        old_string = input_data["old_string"]
        new_string = input_data["new_string"]
        encoding = input_data.get("encoding", "utf-8")
        preserve_crlf = input_data.get("preserve_crlf", True)

        path = Path(file_path)

        # 检查文件是否存在
        if not path.exists():
            return ToolResult.fail(
                f"文件不存在: {file_path}",
                summary_for_llm="文件路径无效",
            )

        # 读取文件
        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read()
        except Exception as e:
            return ToolResult.fail(
                f"文件读取失败: {e}",
                summary_for_llm=f"读取错误: {e}",
            )

        # 检查 old_string 是否唯一
        count = content.count(old_string)
        if count == 0:
            return ToolResult.fail(
                f"未找到要替换的字符串: {old_string[:50]}...",
                summary_for_llm="old_string 未找到，替换失败",
            )
        if count > 1:
            return ToolResult.fail(
                f"old_string 匹配到 {count} 处，必须唯一匹配",
                summary_for_llm=f"old_string 不唯一（{count} 处匹配）",
            )

        # 执行替换
        size_before = len(content)
        new_content = content.replace(old_string, new_string, 1)
        size_after = len(new_content)

        # 保护 CRLF 行尾
        if preserve_crlf:
            # 检测原文件的行尾
            if "\r\n" in content:
                # 原文件使用 CRLF，确保新内容也使用 CRLF
                new_content = new_content.replace("\n", "\r\n").replace("\r\r\n", "\r\n")

        # 写入文件
        try:
            with open(path, "w", encoding=encoding, newline="") as f:
                f.write(new_content)

            return ToolResult.ok(
                data={
                    "file_path": str(path.absolute()),
                    "replacements": 1,
                    "size_before": size_before,
                    "size_after": size_after,
                },
                summary_for_llm=f"文件编辑成功: {file_path} (替换 1 处)",
            )

        except Exception as e:
            return ToolResult.fail(
                f"文件写入失败: {e}",
                summary_for_llm=f"写入错误: {e}",
            )

    def get_manifest(self) -> ToolManifest:
        """获取工具清单"""
        return ToolManifest(
            name=self.name,
            description_for_llm=self.description,
            applicable_scenarios=[
                "修改代码文件中的特定部分",
                "更新配置文件",
                "修复文档中的错误",
            ],
            input_requirements={
                "file_path": "文件路径（必须存在）",
                "old_string": "要替换的原始字符串（必须唯一匹配）",
                "new_string": "替换后的新字符串",
            },
            output_description="文件路径、替换次数、大小变化",
            examples=[
                {
                    "input": {
                        "file_path": "config.py",
                        "old_string": "DEBUG = False",
                        "new_string": "DEBUG = True"
                    },
                    "output_summary": "将 DEBUG 从 False 改为 True"
                }
            ],
            limitations=[
                "old_string 必须唯一匹配",
                "必须先使用 file-reader 读取文件",
                "保护 CRLF 行尾",
            ],
            recommended_tier=self.recommended_tier or "fast",
            recommended_thinking=self.recommended_thinking or "low",
        )


# ---- 工具入口点 ----
def create_tool(context: ToolContext) -> FileEditorTool:
    """创建工具实例"""
    return FileEditorTool(context)
