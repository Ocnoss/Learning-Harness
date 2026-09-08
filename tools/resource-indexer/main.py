"""资料索引工具 - 扫描文件夹生成学习资料索引文档

功能：
1. 递归扫描指定文件夹，收集文件列表和文件夹结构
2. 使用 LLM 智能判断是否为学习资料（二次确认）
3. 使用 LLM 生成资料索引文档（含内容简介）

设计要点：
- 使用 fast 模型（简单任务，低成本）
- 低思考强度（任务明确，无需深度推理）
- 移除硬编码规则，让 LLM 自主判断
- 增强 LLM 对结果的控制权
"""

import sys
from pathlib import Path
from typing import Any

# 确保能导入 core 模块
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.base_tool import BaseTool, ToolContext, ToolResult, ToolManifest


class ResourceIndexerTool(BaseTool):
    """资料索引工具"""

    name = "resource-indexer"
    description = "扫描文件夹生成学习资料索引文档，智能识别资料类型和内容"
    version = "2.0.0"

    # 推荐配置：fast 模型 + 低思考强度
    recommended_tier = "fast"
    recommended_thinking = "low"

    # AI 对任务的理解
    task_understanding = """
    扫描指定文件夹，分析其中的文件，生成结构化的学习资料索引文档。

    核心能力：
    1. 递归扫描文件夹，收集文件信息（名称、大小、类型、路径）
    2. 智能判断文件夹是否为学习资料（基于文件名、类型分布）
    3. 生成 Markdown 格式的索引文档，包含内容摘要和学习建议

    适用场景：
    - 用户想要整理某个文件夹中的学习资料
    - 需要快速了解资料的内容概览和结构
    - 为后续的分类、分题等处理做准备
    """

    input_schema = {
        "type": "object",
        "properties": {
            "folder_path": {
                "type": "string",
                "description": "要扫描的文件夹绝对路径"
            },
            "max_depth": {
                "type": "integer",
                "default": 3,
                "description": "最大递归扫描深度"
            },
            "include_extensions": {
                "type": "array",
                "items": {"type": "string"},
                "default": [".pdf", ".doc", ".docx", ".txt", ".md", ".ppt", ".pptx", ".epub", ".mobi"],
                "description": "包含的文件扩展名列表"
            },
            "exclude_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["node_modules", "__pycache__", ".git", ".venv", "venv"],
                "description": "排除的文件夹/文件名模式"
            },
            "user_context": {
                "type": "string",
                "default": "",
                "description": "用户补充的先验信息（如'这是考研数学资料'）"
            },
            "skip_confirmation": {
                "type": "boolean",
                "default": False,
                "description": "跳过二次确认（用户已确认是学习资料）"
            }
        },
        "required": ["folder_path"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "index_document": {"type": "string"},
            "file_count": {"type": "integer"},
            "folder_structure": {"type": "object"},
            "needs_confirmation": {"type": "boolean"},
            "confirmation_message": {"type": "string"},
            "llm_judgment": {"type": "object"},
            "content_analysis": {"type": "object"},
            "file_paths": {"type": "array", "items": {"type": "string"}},  # 新增：具体文件路径列表
            "base_path": {"type": "string"},  # 新增：基础路径
        }
    }

    async def execute(self, input_data: dict) -> ToolResult:
        """执行资料索引"""
        folder_path = input_data["folder_path"]
        max_depth = input_data.get("max_depth", 3)
        include_extensions = set(input_data.get("include_extensions", []))
        exclude_patterns = set(input_data.get("exclude_patterns", []))
        user_context = input_data.get("user_context", "")
        skip_confirmation = input_data.get("skip_confirmation", False)

        # 1. 验证文件夹存在
        path = Path(folder_path)
        if not path.exists():
            return ToolResult.fail(
                f"文件夹不存在: {folder_path}",
                summary_for_llm=f"路径无效，请检查文件夹是否存在",
            )
        if not path.is_dir():
            return ToolResult.fail(
                f"路径不是文件夹: {folder_path}",
                summary_for_llm=f"路径指向文件而非文件夹",
            )

        # 2. 扫描文件夹
        scan_result = self._scan_folder(
            path, max_depth, include_extensions, exclude_patterns
        )

        # 3. 使用 LLM 智能分析（除非用户已确认）
        llm_judgment = None
        content_analysis = None

        if not skip_confirmation and not user_context:
            # 让 LLM 全面分析文件夹内容
            analysis_result = await self._analyze_with_llm(scan_result, path.name)

            if analysis_result["needs_confirmation"]:
                return ToolResult.need_confirmation(
                    message=analysis_result["confirmation_message"],
                    data={
                        "index_document": "",
                        "file_count": scan_result["file_count"],
                        "folder_structure": scan_result["structure"],
                        "needs_confirmation": True,
                        "confirmation_message": analysis_result["confirmation_message"],
                        "llm_judgment": analysis_result,
                    },
                    metadata={"model": analysis_result.get("model", "")}
                )

            llm_judgment = analysis_result
            content_analysis = analysis_result.get("content_analysis")

        # 4. 生成索引文档
        prompt = self._build_index_prompt(scan_result, user_context, path.name, content_analysis)

        try:
            response = await self.llm.complete(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个学习资料索引专家。基于文件列表和结构，生成清晰的资料索引文档。\n"
                            "文档应包含：资料概览、文件夹结构、各文件内容简介、学习建议。\n"
                            "如果文件名无法推断内容，标注'[内容待确认]'。\n"
                            "保持客观，不要过度推测。"
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                tier=self._resolve_tier(),
                temperature=0.3,
                max_tokens=4096,
            )

            # 检查层级降级提示
            metadata = {"model": response.model}
            if hasattr(response, 'tier_notice') and response.tier_notice:
                metadata["tier_notice"] = response.tier_notice.message

            return ToolResult.ok(
                data={
                    "index_document": response.content,
                    "file_count": scan_result["file_count"],
                    "folder_structure": scan_result["structure"],
                    "needs_confirmation": False,
                    "confirmation_message": "",
                    "llm_judgment": llm_judgment,
                    "content_analysis": content_analysis,
                    "file_paths": [f["path"] for f in scan_result["files"]],  # 相对路径列表
                    "base_path": str(path.absolute()),  # 绝对基础路径
                },
                metadata=metadata,
                summary_for_llm=f"成功生成索引文档，包含 {scan_result['file_count']} 个文件",
                suggested_next_actions=[
                    {
                        "tool": "resource-classifier",
                        "reason": "对资料进行详细分类",
                        "input_hint": {"folder_path": folder_path}
                    }
                ] if scan_result["file_count"] > 0 else [],
            )

        except Exception as e:
            return ToolResult.fail(
                f"LLM 调用失败: {e}",
                summary_for_llm=f"索引文档生成失败: {e}",
            )

    def _scan_folder(
        self,
        root_path: Path,
        max_depth: int,
        include_extensions: set[str],
        exclude_patterns: set[str],
    ) -> dict[str, Any]:
        """扫描文件夹，收集文件列表和结构

        Returns:
            {
                "file_count": int,
                "structure": dict,  # 文件夹层级结构
                "files": list[dict],  # 文件列表 [{path, name, size, ext}, ...]
            }
        """
        files: list[dict] = []
        structure: dict[str, Any] = {"name": root_path.name, "type": "folder", "children": []}

        def should_exclude(name: str) -> bool:
            """检查是否应该排除"""
            name_lower = name.lower()
            return any(pattern.lower() in name_lower for pattern in exclude_patterns)

        def scan_recursive(current_path: Path, current_depth: int, parent_node: dict):
            """递归扫描"""
            if current_depth > max_depth:
                return

            try:
                entries = sorted(current_path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            except PermissionError:
                return

            for entry in entries:
                if should_exclude(entry.name):
                    continue

                if entry.is_dir():
                    # 子文件夹
                    folder_node = {
                        "name": entry.name,
                        "type": "folder",
                        "children": [],
                        "file_count": 0,
                    }
                    parent_node["children"].append(folder_node)
                    scan_recursive(entry, current_depth + 1, folder_node)
                    # 统计文件数
                    folder_node["file_count"] = sum(
                        1 for child in folder_node["children"]
                        if child["type"] == "file"
                    )

                elif entry.is_file():
                    # 文件
                    ext = entry.suffix.lower()
                    if include_extensions and ext not in include_extensions:
                        continue

                    file_info = {
                        "name": entry.name,
                        "type": "file",
                        "extension": ext,
                        "size": entry.stat().st_size,
                        "path": str(entry.relative_to(root_path)),
                    }
                    parent_node["children"].append(file_info)
                    files.append(file_info)

        scan_recursive(root_path, 1, structure)

        return {
            "file_count": len(files),
            "structure": structure,
            "files": files,
        }

    async def _analyze_with_llm(
        self,
        scan_result: dict[str, Any],
        folder_name: str,
    ) -> dict[str, Any]:
        """使用 LLM 全面分析文件夹内容

        Returns:
            {
                "is_learning_material": bool,
                "confidence": str,  # "high" | "medium" | "low"
                "content_analysis": {
                    "subject": str,  # 学科/主题
                    "material_type": str,  # 资料类型
                    "quality_assessment": str,  # 质量评估
                },
                "needs_confirmation": bool,
                "confirmation_message": str,
                "model": str,
            }
        """
        files = scan_result["files"]
        file_count = scan_result["file_count"]

        # 构建文件概要
        file_summary = self._build_file_summary(files, file_count)

        prompt = f"""分析以下文件夹内容，判断是否为学习资料，并分析内容特征：

文件夹名称: {folder_name}
文件数量: {file_count}

{file_summary}

请输出 JSON：
{{
    "is_learning_material": true/false,
    "confidence": "high/medium/low",
    "content_analysis": {{
        "subject": "学科或主题（如：数学、编程、英语等）",
        "material_type": "资料类型（如：教材、题集、笔记、视频等）",
        "quality_assessment": "资料质量评估（如：系统完整、零散、过时等）"
    }},
    "reason": "判断理由",
    "needs_confirmation": true/false,
    "confirmation_message": "如果需要确认，给出提示信息"
}}

判断标准：
- 如果文件明显是学习资料（如教材、笔记、题集），is_learning_material=true，needs_confirmation=false
- 如果明显不是学习资料（如系统文件、软件安装包），is_learning_material=false，needs_confirmation=true
- 如果不确定，is_learning_material=false，needs_confirmation=true，并说明需要用户确认的原因"""

        try:
            result = await self.llm.complete_json(
                messages=[
                    {"role": "system", "content": "你是一个学习资料分析专家，擅长识别资料类型和内容特征。"},
                    {"role": "user", "content": prompt}
                ],
                tier=self._resolve_tier(),
                temperature=0.2,
            )

            return {
                "is_learning_material": result.get("is_learning_material", False),
                "confidence": result.get("confidence", "low"),
                "content_analysis": result.get("content_analysis", {}),
                "reason": result.get("reason", ""),
                "needs_confirmation": result.get("needs_confirmation", True),
                "confirmation_message": result.get("confirmation_message", "请确认是否为学习资料"),
                "model": "",  # 由调用方填充
            }

        except Exception as e:
            # LLM 分析失败时，默认需要确认
            return {
                "is_learning_material": False,
                "confidence": "low",
                "content_analysis": {},
                "reason": f"LLM 分析失败: {e}",
                "needs_confirmation": True,
                "confirmation_message": f"无法分析文件夹内容（{e}）。请补充说明或设置 skip_confirmation=true。",
                "model": "",
            }

    def _build_file_summary(self, files: list[dict], file_count: int) -> str:
        """构建文件概要信息（用于 LLM 分析）"""
        if file_count == 0:
            return "（无文件）"

        # 统计扩展名分布
        ext_counts: dict[str, int] = {}
        for f in files:
            ext = f["extension"] or "(无扩展名)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

        # 按数量排序
        sorted_exts = sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)
        ext_summary = ", ".join(f"{ext}({count})" for ext, count in sorted_exts)

        # 取前 30 个文件名作为示例（增加样本量）
        sample_names = [f["name"] for f in files[:30]]
        sample_text = "\n".join(f"- {name}" for name in sample_names)

        if file_count > 30:
            sample_text += f"\n... 还有 {file_count - 30} 个文件"

        return f"扩展名分布: {ext_summary}\n\n文件示例:\n{sample_text}"

    def _build_index_prompt(
        self,
        scan_result: dict[str, Any],
        user_context: str,
        folder_name: str,
        content_analysis: dict | None,
    ) -> str:
        """构建索引文档生成提示词"""
        files = scan_result["files"]
        structure = scan_result["structure"]

        # 构建文件列表文本
        file_list_lines = []
        for f in files[:100]:
            size_kb = f["size"] / 1024
            size_str = f"{size_kb:.1f}KB" if size_kb < 1024 else f"{size_kb/1024:.1f}MB"
            file_list_lines.append(f"- {f['path']} ({size_str})")

        if len(files) > 100:
            file_list_lines.append(f"... 还有 {len(files) - 100} 个文件未列出")

        file_list_text = "\n".join(file_list_lines)

        # 构建文件夹结构文本
        def format_structure(node: dict, indent: int = 0) -> str:
            lines = []
            prefix = "  " * indent
            if node["type"] == "folder":
                file_count = node.get("file_count", 0)
                lines.append(f"{prefix}[文件夹] {node['name']} ({file_count} 个文件)")
                for child in node.get("children", []):
                    lines.append(format_structure(child, indent + 1))
            else:
                lines.append(f"{prefix}[文件] {node['name']}")
            return "\n".join(lines)

        structure_text = format_structure(structure)

        # 组装提示词
        prompt_parts = []

        if user_context:
            prompt_parts.append(f"用户补充说明: {user_context}")
            prompt_parts.append("")

        if content_analysis:
            prompt_parts.append("内容分析:")
            prompt_parts.append(f"- 学科/主题: {content_analysis.get('subject', '未知')}")
            prompt_parts.append(f"- 资料类型: {content_analysis.get('material_type', '未知')}")
            prompt_parts.append(f"- 质量评估: {content_analysis.get('quality_assessment', '未知')}")
            prompt_parts.append("")

        prompt_parts.extend([
            f"文件夹名称: {folder_name}",
            f"文件总数: {scan_result['file_count']}",
            "",
            "文件夹结构:",
            structure_text,
            "",
            "文件列表:",
            file_list_text,
        ])

        return "\n".join(prompt_parts)

    def get_manifest(self) -> ToolManifest:
        """获取工具清单"""
        return ToolManifest(
            name=self.name,
            description_for_llm=self.description,
            applicable_scenarios=[
                "用户想要整理学习资料文件夹",
                "需要了解资料的内容概览和结构",
                "为后续分类、分题等处理做准备",
            ],
            input_requirements={
                "folder_path": "文件夹绝对路径（必须存在）",
                "user_context": "可选，用户补充说明（如'这是考研数学资料'）",
                "skip_confirmation": "可选，跳过二次确认",
            },
            output_description="Markdown 格式的索引文档，包含资料概览、结构说明、内容简介和学习建议",
            examples=[
                {
                    "input": {"folder_path": "D:/考研数学", "user_context": "考研数学复习资料"},
                    "output_summary": "生成包含 15 个文件的索引文档，按章节分类"
                }
            ],
            limitations=[
                "索引文档基于文件名推断内容，准确性有限",
                "如果文件夹包含大量系统文件，会要求用户确认",
                "扫描深度默认限制为 3 层",
            ],
            recommended_tier=self.recommended_tier or "fast",
            recommended_thinking=self.recommended_thinking or "low",
        )


# ---- 工具入口点 ----
def create_tool(context: ToolContext) -> ResourceIndexerTool:
    """创建工具实例"""
    return ResourceIndexerTool(context)
