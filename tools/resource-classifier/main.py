"""Resource Classifier - 学习资源分类工具

功能：
1. 将学习资源分类为：教材/讲义/笔记/题集/题解/其他
2. 支持单文件和批量文件夹分类
3. 智能策略：LLM 主导分类，规则辅助验证

设计要点：
- 使用 fast 模型，控制成本
- 低思考强度（分类是模式识别任务）
- 减少硬编码规则，让 LLM 承担更多分类决策
- 规则仅用于明显特征的快速判断
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

# 确保能导入 core 模块
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.base_tool import BaseTool, ToolContext, ToolResult, ToolManifest

# 动态导入本地模块（因为目录名含连字符）
import importlib.util

_module_dir = Path(__file__).parent

# 动态导入 content_sampler
_spec = importlib.util.spec_from_file_location(
    "content_sampler",
    _module_dir / "content_sampler.py"
)
content_sampler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(content_sampler)
extract_sample = content_sampler.extract_sample
clean_sample = content_sampler.clean_sample


class ResourceClassifierTool(BaseTool):
    """学习资源分类工具"""

    name = "resource-classifier"
    description = "智能分类学习资源：教材/讲义/笔记/题集/题解/其他"
    version = "2.0.0"

    # 推荐配置：fast 模型 + 低思考强度
    recommended_tier = "fast"
    recommended_thinking = "low"

    # AI 对任务的理解
    task_understanding = """
    将学习资源文件分类为预定义的类别。

    核心能力：
    1. 分析文件名和内容样本，判断资源类型
    2. 支持单文件和批量文件夹分类
    3. 结合文件名特征和内容特征进行综合判断

    分类体系：
    - 教材：系统性的教科书，通常有完整的章节结构
    - 讲义：课程讲义、课件，通常按课时组织
    - 笔记：个人学习笔记、总结，格式较自由
    - 题集：题目集合，通常无解析或解析较少
    - 题解：题目+详细解析，答案和解题过程完整
    - 其他：不属于以上类别的资源

    适用场景：
    - 整理学习资料时需要按类型分类
    - 为不同类型的资料采用不同的处理策略
    - 统计学习资源的类型分布
    """

    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "单个文件路径"},
            "folder_path": {"type": "string", "description": "文件夹路径"},
            "max_depth": {"type": "integer", "default": 3},
            "sample_pages": {"type": "integer", "default": 3},
            "use_content": {"type": "boolean", "default": True},
            "batch_mode": {"type": "boolean", "default": False, "description": "批量模式：减少 LLM 调用次数"}
        }
    }

    output_schema = {
        "type": "object",
        "properties": {
            "classifications": {"type": "array"},
            "summary": {"type": "object"},
            "llm_calls": {"type": "integer"}
        }
    }

    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = {
        ".pdf", ".txt", ".md", ".doc", ".docx",
        ".mp4", ".avi", ".mkv", ".mov",
        ".ppt", ".pptx", ".epub", ".mobi"
    }

    def __init__(self, context: ToolContext):
        super().__init__(context)
        self.llm_calls = 0

    async def execute(self, input_data: dict) -> ToolResult:
        """执行分类"""
        file_path = input_data.get("file_path")
        folder_path = input_data.get("folder_path")
        max_depth = input_data.get("max_depth", 3)
        sample_pages = input_data.get("sample_pages", 3)
        use_content = input_data.get("use_content", True)
        batch_mode = input_data.get("batch_mode", False)

        # 验证输入
        if not file_path and not folder_path:
            return ToolResult.fail(
                "必须提供 file_path 或 folder_path",
                summary_for_llm="缺少输入路径",
            )

        # 收集文件列表
        files = []
        if file_path:
            path = Path(file_path)
            if not path.exists():
                return ToolResult.fail(
                    f"文件不存在: {file_path}",
                    summary_for_llm=f"文件路径无效",
                )
            files.append(path)
        else:
            folder = Path(folder_path)
            if not folder.exists():
                return ToolResult.fail(
                    f"文件夹不存在: {folder_path}",
                    summary_for_llm=f"文件夹路径无效",
                )
            files = self._scan_folder(folder, max_depth)

        if not files:
            return ToolResult.fail(
                "未找到可分类的文件",
                summary_for_llm="指定路径下没有支持的文件格式",
            )

        # 分类文件
        if batch_mode and len(files) > 1:
            # 批量模式：一次 LLM 调用分类多个文件
            classifications = await self._classify_batch(files, sample_pages, use_content)
        else:
            # 逐个分类
            classifications = []
            for file in files:
                result = await self._classify_file(file, sample_pages, use_content)
                classifications.append(result)

        # 生成统计
        summary = self._generate_summary(classifications)

        return ToolResult.ok(
            data={
                "classifications": classifications,
                "summary": summary,
                "llm_calls": self.llm_calls
            },
            metadata={
                "total_files": len(files),
                "use_content": use_content,
                "batch_mode": batch_mode,
            },
            summary_for_llm=f"分类完成：{len(files)} 个文件，{summary['by_category']}",
            suggested_next_actions=[
                {
                    "tool": "question-splitter",
                    "reason": "对题集/题解类文件进行分题处理",
                    "input_hint": {"folder_path": folder_path} if folder_path else {}
                }
            ] if any(c["category"] in ["题集", "题解"] for c in classifications) else [],
        )

    def _scan_folder(self, folder: Path, max_depth: int) -> list[Path]:
        """扫描文件夹"""
        files = []

        def scan_recursive(path: Path, depth: int):
            if depth > max_depth:
                return

            try:
                for entry in path.iterdir():
                    if entry.is_dir():
                        scan_recursive(entry, depth + 1)
                    elif entry.is_file():
                        if entry.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                            files.append(entry)
            except PermissionError:
                pass

        scan_recursive(folder, 1)
        return files

    async def _classify_file(
        self,
        file_path: Path,
        sample_pages: int,
        use_content: bool
    ) -> dict:
        """分类单个文件"""
        # 提取内容样本
        sample = ""
        if use_content:
            sample = extract_sample(file_path, sample_pages)
            if sample:
                sample = clean_sample(sample)

        # 使用 LLM 分类
        return await self._classify_with_llm(file_path, sample)

    async def _classify_batch(
        self,
        files: list[Path],
        sample_pages: int,
        use_content: bool
    ) -> list[dict]:
        """批量分类（减少 LLM 调用次数）"""
        # 为每个文件准备信息
        file_infos = []
        for file in files:
            sample = ""
            if use_content:
                sample = extract_sample(file, sample_pages)
                if sample:
                    sample = clean_sample(sample)
                    sample = sample[:500]  # 限制样本长度

            file_infos.append({
                "name": file.name,
                "path": str(file),
                "sample": sample,
            })

        # 构建批量分类提示词
        files_text = "\n\n".join([
            f"文件 {i+1}:\n文件名: {info['name']}\n内容样本: {info['sample'][:300] if info['sample'] else '(无内容)'}"
            for i, info in enumerate(file_infos[:20])  # 限制批量大小
        ])

        prompt = f"""请对以下 {len(file_infos[:20])} 个学习资源文件进行分类：

{files_text}

分类标准：
- 教材：系统性的教科书，有完整章节结构
- 讲义：课程讲义、课件，按课时组织
- 笔记：个人学习笔记、总结，格式自由
- 题集：题目集合，无解析或解析较少
- 题解：题目+详细解析，答案完整
- 其他：不属于以上类别

输出 JSON 数组，每个元素对应一个文件：
[
    {{"file_index": 1, "category": "类别", "confidence": "high/medium/low", "reason": "判断理由"}},
    ...
]"""

        try:
            self.llm_calls += 1
            results = await self.llm.complete_json(
                messages=[
                    {"role": "system", "content": "你是一个学习资源分类专家，擅长快速准确地识别资料类型。"},
                    {"role": "user", "content": prompt}
                ],
                tier=self._resolve_tier(),
                temperature=0.2,
            )

            # 解析结果
            classifications = []
            result_map = {r["file_index"]: r for r in results if isinstance(r, dict)}

            for i, info in enumerate(file_infos[:20], 1):
                if i in result_map:
                    r = result_map[i]
                    classifications.append({
                        "file_path": info["path"],
                        "file_name": info["name"],
                        "category": r.get("category", "其他"),
                        "confidence": r.get("confidence", "medium"),
                        "reason": r.get("reason", ""),
                        "method": "llm_batch"
                    })
                else:
                    classifications.append({
                        "file_path": info["path"],
                        "file_name": info["name"],
                        "category": "其他",
                        "confidence": "low",
                        "reason": "批量分类未返回结果",
                        "method": "llm_batch"
                    })

            # 处理剩余文件（如果有）
            for info in file_infos[20:]:
                classifications.append({
                    "file_path": info["path"],
                    "file_name": info["name"],
                    "category": "其他",
                    "confidence": "low",
                    "reason": "超出批量处理限制",
                    "method": "skipped"
                })

            return classifications

        except Exception as e:
            # 批量分类失败，回退到逐个分类
            return [await self._classify_file(f, sample_pages, use_content) for f in files]

    async def _classify_with_llm(self, file_path: Path, sample: str) -> dict:
        """使用 LLM 分类单个文件"""
        self.llm_calls += 1

        # 构建提示词
        prompt = f"""请对以下学习资源进行分类：

文件名: {file_path.name}

内容样本:
{sample[:1000] if sample else "(无法提取内容)"}

分类标准：
- 教材：系统性的教科书，有完整章节结构
- 讲义：课程讲义、课件，按课时组织
- 笔记：个人学习笔记、总结，格式自由
- 题集：题目集合，无解析或解析较少
- 题解：题目+详细解析，答案完整
- 其他：不属于以上类别

输出 JSON：
{{
    "category": "类别",
    "confidence": "high/medium/low",
    "reason": "判断理由（简要说明依据）",
    "key_features": ["识别的关键特征"]
}}"""

        try:
            result = await self.llm.complete_json(
                messages=[
                    {"role": "system", "content": "你是一个学习资源分类专家，基于文件名和内容特征进行综合判断。"},
                    {"role": "user", "content": prompt}
                ],
                tier=self._resolve_tier(),
                temperature=0.2,
            )

            # 验证类别
            valid_categories = ["教材", "讲义", "笔记", "题集", "题解", "其他"]
            category = result.get("category", "其他")
            if category not in valid_categories:
                category = "其他"

            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "category": category,
                "confidence": result.get("confidence", "medium"),
                "reason": result.get("reason", ""),
                "key_features": result.get("key_features", []),
                "method": "llm"
            }

        except Exception as e:
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "category": "其他",
                "confidence": "low",
                "reason": f"LLM 分类失败: {e}",
                "method": "llm"
            }

    def _generate_summary(self, classifications: list[dict]) -> dict:
        """生成统计摘要"""
        by_category = {}
        by_confidence = {"high": 0, "medium": 0, "low": 0}

        for c in classifications:
            category = c["category"]
            confidence = c.get("confidence", "medium")

            by_category[category] = by_category.get(category, 0) + 1
            by_confidence[confidence] = by_confidence.get(confidence, 0) + 1

        return {
            "total": len(classifications),
            "by_category": by_category,
            "by_confidence": by_confidence,
        }

    def get_manifest(self) -> ToolManifest:
        """获取工具清单"""
        return ToolManifest(
            name=self.name,
            description_for_llm=self.description,
            applicable_scenarios=[
                "整理学习资料时需要按类型分类",
                "为不同类型的资料采用不同的处理策略",
                "统计学习资源的类型分布",
            ],
            input_requirements={
                "file_path": "单个文件路径（与 folder_path 二选一）",
                "folder_path": "文件夹路径（与 file_path 二选一）",
                "batch_mode": "批量模式：减少 LLM 调用次数，适合大量文件",
            },
            output_description="分类结果列表，每个文件包含类别、置信度、判断理由",
            examples=[
                {
                    "input": {"folder_path": "D:/学习资料", "batch_mode": True},
                    "output_summary": "批量分类 20 个文件，识别出 5 本教材、10 套题集、5 份笔记"
                }
            ],
            limitations=[
                "视频文件仅基于文件名分类，无法分析内容",
                "批量模式最多处理 20 个文件",
                "分类准确性依赖文件名和内容样本的质量",
            ],
            recommended_tier=self.recommended_tier or "fast",
            recommended_thinking=self.recommended_thinking or "low",
        )


# ---- 工具入口点 ----
def create_tool(context: ToolContext) -> ResourceClassifierTool:
    """创建工具实例"""
    return ResourceClassifierTool(context)
