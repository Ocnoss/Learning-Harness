"""File Writer - 文件写入工具

功能：
1. 创建新文件
2. 覆盖现有文件
3. 追加内容到文件
4. 自动创建父目录

设计要点：
- 使用 fast 模型 + 低思考强度
- 零 LLM 调用，纯本地操作
- 禁止用它做增量修改（使用 file-editor 代替）
"""

import sys
from pathlib import Path
from typing import Any

# 确保能导入 core 模块
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.base_tool import BaseTool, ToolContext, ToolResult, ToolManifest


class FileWriterTool(BaseTool):
    """文件写入工具"""

    name = "file-writer"
    description = "创建/覆盖/追加文件，自动创建父目录"
    version = "1.0.0"

    recommended_tier = "fast"
    recommended_thinking = "low"

    task_understanding = """
    创建或修改文件内容。

    核心能力：
    1. 创建新文件（自动创建父目录）
    2. 覆盖现有文件（整体替换）
    3. 追加内容到文件末尾
    4. 支持多种编码（默认 UTF-8）

    适用场景：
    - 保存生成的内容到文件
    - 创建新的代码文件
    - 追加日志内容

    注意：
    - 禁止用于增量修改（应使用 file-editor）
    - 覆盖操作会丢失原内容，请谨慎使用
    """

    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件路径（绝对路径或相对路径）"
            },
            "content": {
                "type": "string",
                "description": "要写入的内容"
            },
            "mode": {
                "type": "string",
                "enum": ["create", "overwrite", "append"],
                "default": "create",
                "description": "写入模式：create(创建)/overwrite(覆盖)/append(追加)"
            },
            "encoding": {
                "type": "string",
                "default": "utf-8",
                "description": "文件编码（默认 utf-8）"
            },
            "create_dirs": {
                "type": "boolean",
                "default": True,
                "description": "自动创建父目录（默认 true）"
            }
        },
        "required": ["file_path", "content"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "mode": {"type": "string"},
            "size": {"type": "integer"},
            "created": {"type": "boolean"}
        }
    }

    async def execute(self, input_data: dict) -> ToolResult:
        """执行文件写入"""
        file_path = input_data["file_path"]
        content = input_data["content"]
        mode = input_data.get("mode", "create")
        encoding = input_data.get("encoding", "utf-8")
        create_dirs = input_data.get("create_dirs", True)

        path = Path(file_path)

        # 检查模式
        if mode == "create" and path.exists():
            return ToolResult.fail(
                f"文件已存在: {file_path}。使用 overwrite 模式覆盖或 append 模式追加",
                summary_for_llm="文件已存在，create 模式失败",
            )

        if mode in ("overwrite", "append") and not path.exists():
            return ToolResult.fail(
                f"文件不存在: {file_path}。使用 create 模式创建",
                summary_for_llm="文件不存在，overwrite/append 模式失败",
            )

        # 创建父目录
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        # 写入文件
        try:
            if mode == "append":
                with open(path, "a", encoding=encoding) as f:
                    f.write(content)
                created = False
            else:  # create 或 overwrite
                with open(path, "w", encoding=encoding) as f:
                    f.write(content)
                created = (mode == "create")

            size = path.stat().st_size

            return ToolResult.ok(
                data={
                    "file_path": str(path.absolute()),
                    "mode": mode,
                    "size": size,
                    "created": created,
                },
                summary_for_llm=f"文件写入成功: {file_path} ({size} 字节)",
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
                "保存生成的内容到文件",
                "创建新的代码文件",
                "追加日志内容",
            ],
            input_requirements={
                "file_path": "文件路径",
                "content": "要写入的内容",
                "mode": "写入模式：create/overwrite/append",
            },
            output_description="文件路径、写入模式、文件大小",
            examples=[
                {
                    "input": {"file_path": "output/result.md", "content": "# 结果", "mode": "create"},
                    "output_summary": "创建新文件 result.md"
                }
            ],
            limitations=[
                "禁止用于增量修改（应使用 file-editor）",
                "覆盖操作会丢失原内容",
                "默认使用 UTF-8 编码",
            ],
            recommended_tier=self.recommended_tier or "fast",
            recommended_thinking=self.recommended_thinking or "low",
        )


# ---- 工具入口点 ----
def create_tool(context: ToolContext) -> FileWriterTool:
    """创建工具实例"""
    return FileWriterTool(context)
