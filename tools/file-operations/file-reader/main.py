"""File Reader - 文件读取工具

功能：
1. 读取文本文件内容
2. 支持分页（offset/limit）
3. 自动处理 UTF-16 转码
4. 拒绝读取二进制和密钥文件

设计要点：
- 使用 fast 模型 + 低思考强度
- 零 LLM 调用，纯本地操作
"""

import sys
from pathlib import Path
from typing import Any

# 确保能导入 core 模块
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.base_tool import BaseTool, ToolContext, ToolResult, ToolManifest


class FileReaderTool(BaseTool):
    """文件读取工具"""

    name = "file-reader"
    description = "读取文本文件内容，支持分页和编码转换"
    version = "1.0.0"

    recommended_tier = "fast"
    recommended_thinking = "low"

    task_understanding = """
    读取文本文件内容，支持大文件分页读取。

    核心能力：
    1. 读取文本文件（txt, md, py, json, yaml 等）
    2. 支持分页读取（offset/limit）
    3. 自动检测并转换编码（UTF-8, UTF-16, GBK 等）
    4. 拒绝读取二进制文件和敏感文件（.env, 私钥等）

    适用场景：
    - 查看文件内容
    - 读取大文件的一部分
    - 获取代码文件内容
    """

    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件路径（绝对路径或相对路径）"
            },
            "offset": {
                "type": "integer",
                "default": 0,
                "description": "起始行号（从 0 开始）"
            },
            "limit": {
                "type": "integer",
                "default": 100,
                "description": "读取行数限制"
            },
            "encoding": {
                "type": "string",
                "description": "指定编码（可选，默认自动检测）"
            }
        },
        "required": ["file_path"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "total_lines": {"type": "integer"},
            "encoding": {"type": "string"},
            "is_binary": {"type": "boolean"},
            "truncated": {"type": "boolean"}
        }
    }

    # 禁止读取的文件模式
    FORBIDDEN_PATTERNS = {
        ".env", ".env.local", ".env.production",
        "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
        ".pem", ".key", ".p12", ".pfx",
        "credentials", "secrets",
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
        """执行文件读取"""
        file_path = input_data["file_path"]
        offset = input_data.get("offset", 0)
        limit = input_data.get("limit", 100)
        encoding = input_data.get("encoding")

        path = Path(file_path)

        # 检查文件是否存在
        if not path.exists():
            return ToolResult.fail(
                f"文件不存在: {file_path}",
                summary_for_llm="文件路径无效",
            )

        # 检查是否为文件
        if not path.is_file():
            return ToolResult.fail(
                f"路径不是文件: {file_path}",
                summary_for_llm="路径指向目录而非文件",
            )

        # 检查禁止读取的文件
        if self._is_forbidden(path):
            return ToolResult.fail(
                f"禁止读取敏感文件: {file_path}",
                summary_for_llm="文件包含敏感信息，禁止读取",
            )

        # 检查二进制文件
        if self._is_binary(path):
            return ToolResult.fail(
                f"二进制文件不支持读取: {file_path}",
                summary_for_llm="文件为二进制格式，请使用其他工具",
            )

        # 读取文件
        try:
            content, detected_encoding = self._read_file(path, encoding)
        except Exception as e:
            return ToolResult.fail(
                f"文件读取失败: {e}",
                summary_for_llm=f"读取错误: {e}",
            )

        # 分页处理
        lines = content.split("\n")
        total_lines = len(lines)

        if offset >= total_lines:
            return ToolResult.ok(
                data={
                    "content": "",
                    "total_lines": total_lines,
                    "encoding": detected_encoding,
                    "is_binary": False,
                    "truncated": False,
                },
                summary_for_llm=f"文件共 {total_lines} 行，offset {offset} 超出范围",
            )

        selected_lines = lines[offset:offset + limit]
        selected_content = "\n".join(selected_lines)
        truncated = (offset + limit) < total_lines

        return ToolResult.ok(
            data={
                "content": selected_content,
                "total_lines": total_lines,
                "encoding": detected_encoding,
                "is_binary": False,
                "truncated": truncated,
                "offset": offset,
                "limit": limit,
                "returned_lines": len(selected_lines),
            },
            summary_for_llm=f"读取 {len(selected_lines)} 行（共 {total_lines} 行）",
        )

    def _is_forbidden(self, path: Path) -> bool:
        """检查是否为禁止读取的文件"""
        name_lower = path.name.lower()
        return any(pattern in name_lower for pattern in self.FORBIDDEN_PATTERNS)

    def _is_binary(self, path: Path) -> bool:
        """检查是否为二进制文件"""
        ext = path.suffix.lower()
        return ext in self.BINARY_EXTENSIONS

    def _read_file(self, path: Path, encoding: str | None) -> tuple[str, str]:
        """读取文件并处理编码

        Returns:
            (内容, 检测到的编码)
        """
        # 如果指定了编码，直接使用
        if encoding:
            with open(path, "r", encoding=encoding) as f:
                return f.read(), encoding

        # 自动检测编码
        encodings_to_try = ["utf-8", "utf-16", "gbk", "gb2312", "latin-1"]

        for enc in encodings_to_try:
            try:
                with open(path, "r", encoding=enc) as f:
                    content = f.read()
                    return content, enc
            except (UnicodeDecodeError, UnicodeError):
                continue

        # 所有编码都失败，使用错误处理模式
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), "utf-8 (with errors)"

    def get_manifest(self) -> ToolManifest:
        """获取工具清单"""
        return ToolManifest(
            name=self.name,
            description_for_llm=self.description,
            applicable_scenarios=[
                "查看文件内容",
                "读取大文件的一部分",
                "获取代码文件内容",
            ],
            input_requirements={
                "file_path": "文件路径（必须存在）",
                "offset": "起始行号（默认 0）",
                "limit": "读取行数限制（默认 100）",
            },
            output_description="文件内容、总行数、编码信息",
            examples=[
                {
                    "input": {"file_path": "README.md", "limit": 50},
                    "output_summary": "读取 README.md 前 50 行"
                }
            ],
            limitations=[
                "不支持二进制文件",
                "禁止读取敏感文件（.env、私钥等）",
                "大文件需要分页读取",
            ],
            recommended_tier=self.recommended_tier or "fast",
            recommended_thinking=self.recommended_thinking or "low",
        )


# ---- 工具入口点 ----
def create_tool(context: ToolContext) -> FileReaderTool:
    """创建工具实例"""
    return FileReaderTool(context)
