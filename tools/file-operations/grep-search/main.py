"""Grep Search - 文件内容搜索工具

功能：
1. 正则搜索文件内容
2. 支持上下文行数
3. 支持 glob 过滤
4. 支持忽略规则

设计要点：
- 使用 fast 模型 + 低思考强度
- 零 LLM 调用，纯本地操作
- 基于 re 模块
"""

import re
import sys
from pathlib import Path
from typing import Any

# 确保能导入 core 模块
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.base_tool import BaseTool, ToolContext, ToolResult, ToolManifest


class GrepSearchTool(BaseTool):
    """文件内容搜索工具"""

    name = "grep-search"
    description = "正则搜索文件内容，支持上下文和过滤"
    version = "1.0.0"

    recommended_tier = "fast"
    recommended_thinking = "low"

    task_understanding = """
    在文件内容中搜索匹配的文本。

    核心能力：
    1. 正则表达式搜索
    2. 支持上下文行数（显示匹配行前后内容）
    3. 支持 glob 过滤（只搜索特定文件）
    4. 支持忽略规则（跳过二进制文件等）

    适用场景：
    - 在代码中搜索特定函数或变量
    - 查找配置文件中的特定设置
    - 定位错误日志

    注意：
    - 默认跳过二进制文件
    - 结果超过 100 条会被截断
    """

    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "正则表达式模式"
            },
            "base_path": {
                "type": "string",
                "default": ".",
                "description": "搜索基础路径（默认当前目录）"
            },
            "glob": {
                "type": "string",
                "description": "文件过滤模式（如 *.py, **/*.md）"
            },
            "context_lines": {
                "type": "integer",
                "default": 0,
                "description": "上下文行数（默认 0）"
            },
            "max_results": {
                "type": "integer",
                "default": 100,
                "description": "最大结果数（默认 100）"
            },
            "ignore_case": {
                "type": "boolean",
                "default": False,
                "description": "忽略大小写（默认 false）"
            }
        },
        "required": ["pattern"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "matches": {"type": "array"},
            "count": {"type": "integer"},
            "truncated": {"type": "boolean"}
        }
    }

    # 二进制文件扩展名
    BINARY_EXTENSIONS = {
        ".exe", ".dll", ".so", ".dylib",
        ".zip", ".tar", ".gz", ".rar", ".7z",
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico",
        ".mp3", ".mp4", ".avi", ".mkv", ".mov",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx",
        ".pyc", ".pyo", ".whl", ".egg",
    }

    async def execute(self, input_data: dict) -> ToolResult:
        """执行内容搜索"""
        pattern = input_data["pattern"]
        base_path = Path(input_data.get("base_path", "."))
        glob_pattern = input_data.get("glob")
        context_lines = input_data.get("context_lines", 0)
        max_results = input_data.get("max_results", 100)
        ignore_case = input_data.get("ignore_case", False)

        # 检查基础路径
        if not base_path.exists():
            return ToolResult.fail(
                f"基础路径不存在: {base_path}",
                summary_for_llm="搜索路径无效",
            )

        # 编译正则表达式
        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return ToolResult.fail(
                f"正则表达式无效: {e}",
                summary_for_llm=f"正则错误: {e}",
            )

        # 执行搜索
        try:
            matches = []
            files_to_search = self._get_files(base_path, glob_pattern)

            for file_path in files_to_search:
                if len(matches) >= max_results:
                    break

                file_matches = self._search_file(
                    file_path, regex, context_lines, base_path
                )
                matches.extend(file_matches)

            truncated = len(matches) >= max_results

            return ToolResult.ok(
                data={
                    "matches": matches[:max_results],
                    "count": len(matches),
                    "truncated": truncated,
                    "pattern": pattern,
                    "base_path": str(base_path.absolute()),
                },
                summary_for_llm=f"找到 {len(matches)} 处匹配（模式: {pattern}）",
            )

        except Exception as e:
            return ToolResult.fail(
                f"搜索失败: {e}",
                summary_for_llm=f"搜索错误: {e}",
            )

    def _get_files(self, base_path: Path, glob_pattern: str | None) -> list[Path]:
        """获取要搜索的文件列表"""
        if glob_pattern:
            files = []
            for path in base_path.glob(glob_pattern):
                if path.is_file() and not self._is_binary(path):
                    files.append(path)
            return files
        else:
            files = []
            for path in base_path.rglob("*"):
                if path.is_file() and not self._is_binary(path):
                    files.append(path)
            return files

    def _is_binary(self, path: Path) -> bool:
        """检查是否为二进制文件"""
        ext = path.suffix.lower()
        return ext in self.BINARY_EXTENSIONS

    def _search_file(
        self,
        file_path: Path,
        regex: re.Pattern,
        context_lines: int,
        base_path: Path
    ) -> list[dict]:
        """在单个文件中搜索"""
        matches = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                if regex.search(line):
                    match = {
                        "file": str(file_path.relative_to(base_path)),
                        "line_number": i + 1,
                        "line": line.rstrip(),
                    }

                    # 添加上下文
                    if context_lines > 0:
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        match["context"] = [
                            {
                                "line_number": j + 1,
                                "line": lines[j].rstrip(),
                                "is_match": j == i,
                            }
                            for j in range(start, end)
                        ]

                    matches.append(match)

        except Exception:
            # 忽略读取失败的文件
            pass

        return matches

    def get_manifest(self) -> ToolManifest:
        """获取工具清单"""
        return ToolManifest(
            name=self.name,
            description_for_llm=self.description,
            applicable_scenarios=[
                "在代码中搜索特定函数或变量",
                "查找配置文件中的特定设置",
                "定位错误日志",
            ],
            input_requirements={
                "pattern": "正则表达式模式",
                "base_path": "搜索基础路径（默认当前目录）",
                "glob": "文件过滤模式（可选）",
            },
            output_description="匹配列表、数量、是否截断",
            examples=[
                {
                    "input": {"pattern": "def main", "glob": "*.py"},
                    "output_summary": "搜索所有 Python 文件中的 main 函数"
                }
            ],
            limitations=[
                "默认跳过二进制文件",
                "结果上限 100 条",
            ],
            recommended_tier=self.recommended_tier or "fast",
            recommended_thinking=self.recommended_thinking or "low",
        )


# ---- 工具入口点 ----
def create_tool(context: ToolContext) -> GrepSearchTool:
    """创建工具实例"""
    return GrepSearchTool(context)
