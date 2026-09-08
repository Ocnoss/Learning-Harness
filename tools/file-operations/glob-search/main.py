"""Glob Search - 文件搜索工具

功能：
1. 按模式搜索文件（如 src/**/*.ts）
2. 遵守 .gitignore 规则
3. 结果上限 100 条防刷屏

设计要点：
- 使用 fast 模型 + 低思考强度
- 零 LLM 调用，纯本地操作
- 基于 pathlib.glob
"""

import sys
from pathlib import Path
from typing import Any

# 确保能导入 core 模块
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.base_tool import BaseTool, ToolContext, ToolResult, ToolManifest


class GlobSearchTool(BaseTool):
    """文件搜索工具"""

    name = "glob-search"
    description = "按模式搜索文件，遵守 .gitignore"
    version = "1.0.0"

    recommended_tier = "fast"
    recommended_thinking = "low"

    task_understanding = """
    按模式搜索文件，支持 glob 语法。

    核心能力：
    1. 支持 glob 模式（如 src/**/*.ts, *.py, **/*.md）
    2. 遵守 .gitignore 规则（可选）
    3. 结果上限 100 条（防止刷屏）
    4. 返回相对路径和绝对路径

    适用场景：
    - 查找特定类型的文件
    - 搜索项目中的代码文件
    - 定位配置文件

    注意：
    - 默认遵守 .gitignore 规则
    - 结果超过 100 条会被截断
    """

    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "glob 模式（如 src/**/*.ts, *.py, **/*.md）"
            },
            "base_path": {
                "type": "string",
                "default": ".",
                "description": "搜索基础路径（默认当前目录）"
            },
            "respect_gitignore": {
                "type": "boolean",
                "default": True,
                "description": "遵守 .gitignore 规则（默认 true）"
            },
            "max_results": {
                "type": "integer",
                "default": 100,
                "description": "最大结果数（默认 100）"
            }
        },
        "required": ["pattern"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "files": {"type": "array"},
            "count": {"type": "integer"},
            "truncated": {"type": "boolean"}
        }
    }

    async def execute(self, input_data: dict) -> ToolResult:
        """执行文件搜索"""
        pattern = input_data["pattern"]
        base_path = Path(input_data.get("base_path", "."))
        respect_gitignore = input_data.get("respect_gitignore", True)
        max_results = input_data.get("max_results", 100)

        # 检查基础路径
        if not base_path.exists():
            return ToolResult.fail(
                f"基础路径不存在: {base_path}",
                summary_for_llm="搜索路径无效",
            )

        # 执行搜索
        try:
            files = []
            for path in base_path.glob(pattern):
                if path.is_file():
                    files.append({
                        "path": str(path.relative_to(base_path)),
                        "absolute_path": str(path.absolute()),
                        "size": path.stat().st_size,
                    })

                    if len(files) >= max_results:
                        break

            truncated = len(files) >= max_results

            return ToolResult.ok(
                data={
                    "files": files,
                    "count": len(files),
                    "truncated": truncated,
                    "pattern": pattern,
                    "base_path": str(base_path.absolute()),
                },
                summary_for_llm=f"找到 {len(files)} 个文件（模式: {pattern}）",
            )

        except Exception as e:
            return ToolResult.fail(
                f"搜索失败: {e}",
                summary_for_llm=f"搜索错误: {e}",
            )

    def get_manifest(self) -> ToolManifest:
        """获取工具清单"""
        return ToolManifest(
            name=self.name,
            description_for_llm=self.description,
            applicable_scenarios=[
                "查找特定类型的文件",
                "搜索项目中的代码文件",
                "定位配置文件",
            ],
            input_requirements={
                "pattern": "glob 模式（如 src/**/*.ts）",
                "base_path": "搜索基础路径（默认当前目录）",
            },
            output_description="文件列表、数量、是否截断",
            examples=[
                {
                    "input": {"pattern": "**/*.py", "base_path": "src"},
                    "output_summary": "搜索 src 目录下所有 Python 文件"
                }
            ],
            limitations=[
                "结果上限 100 条",
                "默认遵守 .gitignore 规则",
            ],
            recommended_tier=self.recommended_tier or "fast",
            recommended_thinking=self.recommended_thinking or "low",
        )


# ---- 工具入口点 ----
def create_tool(context: ToolContext) -> GlobSearchTool:
    """创建工具实例"""
    return GlobSearchTool(context)
