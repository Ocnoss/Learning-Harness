"""Question Splitter - PDF 分题工具

功能：
1. 从 PDF 提取文本
2. 使用规则初步分割题目
3. 使用 LLM 校验和优化分割结果
4. 输出结构化的题目数据

设计要点：
- 规则分割 + LLM 校验的混合策略
- LLM 主导边界判断和完整性验证
- 使用 fast 模型，低思考强度
"""

import json
import sys
from pathlib import Path
from typing import Any

# 确保能导入 core 模块
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.base_tool import BaseTool, ToolContext, ToolResult, ToolManifest

# 导入本地模块（使用绝对导入，因为目录名含连字符）
import importlib.util

_module_dir = Path(__file__).parent

# 动态导入 pdf_extractor
_spec = importlib.util.spec_from_file_location(
    "pdf_extractor",
    _module_dir / "pdf_extractor.py"
)
pdf_extractor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pdf_extractor)
extract_text_from_pdf = pdf_extractor.extract_text_from_pdf
clean_text = pdf_extractor.clean_text

# 动态导入 rule_splitter
_spec = importlib.util.spec_from_file_location(
    "rule_splitter",
    _module_dir / "rule_splitter.py"
)
rule_splitter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rule_splitter)
RuleSplitter = rule_splitter.RuleSplitter


class QuestionSplitterTool(BaseTool):
    """PDF 分题工具"""

    name = "question-splitter"
    description = "从题库 PDF 中智能提取单题和解析，支持多种题号格式"
    version = "2.0.0"

    # 推荐配置：fast 模型 + 低思考强度
    recommended_tier = "fast"
    recommended_thinking = "low"

    # AI 对任务的理解
    task_understanding = """
    从 PDF 格式的题库中提取结构化的题目数据。

    核心能力：
    1. 从 PDF 提取文本内容
    2. 识别题目边界（题号、选项、答案、解析）
    3. 分割单个题目并提取结构化信息
    4. 验证分割结果的完整性和准确性

    适用场景：
    - 将题库 PDF 转换为可编程处理的 JSON 格式
    - 提取题目用于闪卡生成、在线测试等
    - 分析题目类型和难度分布
    """

    input_schema = {
        "type": "object",
        "properties": {
            "pdf_path": {
                "type": "string",
                "description": "PDF 文件路径"
            },
            "output_dir": {
                "type": "string",
                "description": "输出目录路径"
            },
            "use_llm_validation": {
                "type": "boolean",
                "default": True,
                "description": "是否使用 LLM 校验边界和完整性"
            },
            "question_pattern": {
                "type": "string",
                "description": "自定义题号正则表达式（可选，LLM 可自动识别）"
            },
            "extract_metadata": {
                "type": "boolean",
                "default": True,
                "description": "是否提取题目元数据（难度、知识点等）"
            }
        },
        "required": ["pdf_path"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "questions": {"type": "array"},
            "total_count": {"type": "integer"},
            "llm_calls": {"type": "integer"},
            "output_file": {"type": "string"},
            "extraction_stats": {"type": "object"}
        }
    }

    async def execute(self, input_data: dict) -> ToolResult:
        """执行分题"""
        pdf_path = input_data["pdf_path"]
        output_dir = input_data.get("output_dir")
        use_llm_validation = input_data.get("use_llm_validation", True)
        question_pattern = input_data.get("question_pattern")
        extract_metadata = input_data.get("extract_metadata", True)

        # 1. 验证 PDF 文件存在
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return ToolResult.fail(
                f"PDF 文件不存在: {pdf_path}",
                summary_for_llm="文件路径无效",
            )

        # 2. 提取文本
        try:
            raw_text = extract_text_from_pdf(pdf_file)
            text = clean_text(raw_text)
        except ImportError as e:
            return ToolResult.fail(
                str(e),
                summary_for_llm="PDF 提取依赖缺失",
            )
        except Exception as e:
            return ToolResult.fail(
                f"PDF 提取失败: {e}",
                summary_for_llm=f"PDF 解析错误: {e}",
            )

        if not text:
            return ToolResult.fail(
                "PDF 提取结果为空。可能原因：\n"
                "1. PDF 是图片型（扫描版），无法提取文本\n"
                "2. PDF 文件损坏\n"
                "建议：使用 OCR 工具先将图片型 PDF 转换为文本型 PDF",
                summary_for_llm="PDF 无文本内容，可能是扫描版",
            )

        # 3. 规则分割（初步）
        splitter = RuleSplitter(custom_pattern=question_pattern)
        split_result = splitter.split(text)

        if not split_result.questions:
            # 规则分割失败，尝试 LLM 直接分割
            return await self._llm_direct_split(text, pdf_file, output_dir)

        # 4. LLM 校验和优化（可选）
        llm_calls = 0
        extraction_stats = {
            "rule_split_count": len(split_result.questions),
            "llm_validated": 0,
            "boundary_warnings": 0,
            "completeness_issues": 0,
        }

        if use_llm_validation:
            validated_questions = []
            for i, question in enumerate(split_result.questions):
                # LLM 校验边界和完整性
                validated = await self._validate_question_with_llm(
                    question, i, split_result.questions
                )
                validated_questions.append(validated)
                llm_calls += 1

                if validated.get("_boundary_warning"):
                    extraction_stats["boundary_warnings"] += 1
                if validated.get("_completeness_issue"):
                    extraction_stats["completeness_issues"] += 1

            split_result.questions = validated_questions
            extraction_stats["llm_validated"] = len(validated_questions)

        # 5. 提取元数据（可选）
        if extract_metadata:
            for question in split_result.questions:
                metadata = await self._extract_metadata_with_llm(question)
                question.update(metadata)
                llm_calls += 1

        # 6. 清理内部字段
        for question in split_result.questions:
            question.pop("_boundary_confidence", None)
            question.pop("_needs_validation", None)
            question.pop("_boundary_warning", None)
            question.pop("_completeness_issue", None)

        # 7. 保存结果
        output_file = None
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            output_file = output_path / f"{pdf_file.stem}_questions.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    split_result.questions,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        return ToolResult.ok(
            data={
                "questions": split_result.questions,
                "total_count": len(split_result.questions),
                "llm_calls": llm_calls,
                "output_file": str(output_file) if output_file else None,
                "extraction_stats": extraction_stats,
            },
            metadata={
                "pdf_path": str(pdf_file),
                "use_llm_validation": use_llm_validation,
                "extract_metadata": extract_metadata,
            },
            summary_for_llm=f"成功提取 {len(split_result.questions)} 道题目",
            suggested_next_actions=[
                {
                    "tool": "quiz-app",
                    "reason": "将提取的题目用于在线测试",
                    "input_hint": {"questions_file": str(output_file)} if output_file else {}
                }
            ] if output_file else [],
        )

    async def _llm_direct_split(
        self,
        text: str,
        pdf_file: Path,
        output_dir: str | None,
    ) -> ToolResult:
        """LLM 直接分割（当规则分割失败时）"""
        prompt = f"""请从以下文本中提取所有题目，输出结构化 JSON：

文本内容：
{text[:8000]}  # 限制长度

请识别每道题目，提取：
- id: 题号
- content: 题目内容（不含答案）
- options: 选项列表（如果是选择题）
- answer: 答案
- analysis: 解析（如果有）

输出 JSON 数组：
[
    {{
        "id": "1",
        "content": "题目内容",
        "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
        "answer": "A",
        "analysis": "解析内容"
    }},
    ...
]"""

        try:
            questions = await self.llm.complete_json(
                messages=[
                    {"role": "system", "content": "你是一个题目提取专家，擅长从文本中识别和提取结构化题目数据。"},
                    {"role": "user", "content": prompt}
                ],
                tier=self._resolve_tier(),
                temperature=0.2,
                max_tokens=4096,
            )

            if not isinstance(questions, list):
                questions = []

            # 保存结果
            output_file = None
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / f"{pdf_file.stem}_questions.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(questions, f, ensure_ascii=False, indent=2)

            return ToolResult.ok(
                data={
                    "questions": questions,
                    "total_count": len(questions),
                    "llm_calls": 1,
                    "output_file": str(output_file) if output_file else None,
                    "extraction_stats": {"method": "llm_direct"},
                },
                summary_for_llm=f"LLM 直接提取 {len(questions)} 道题目",
            )

        except Exception as e:
            return ToolResult.fail(
                f"LLM 分割失败: {e}",
                summary_for_llm="无法从 PDF 提取题目",
            )

    async def _validate_question_with_llm(
        self,
        question: dict,
        index: int,
        all_questions: list[dict],
    ) -> dict:
        """使用 LLM 校验题目边界和完整性"""
        # 获取前后文本
        text_before = ""
        if index > 0:
            prev_q = all_questions[index - 1]
            text_before = prev_q.get("content", "")[-200:]

        text_after = question.get("content", "")[:200]

        prompt = f"""校验以下题目分割是否正确：

前一段文本结尾：
...{text_before}

当前题目（题号: {question.get('id', '?')}）：
{question.get('content', '')[:500]}

请检查：
1. 边界是否正确（是否在前一题和后一题之间正确分割）
2. 内容是否完整（是否包含完整的题干、选项、答案、解析）

输出 JSON：
{{
    "boundary_correct": true/false,
    "boundary_issue": "边界问题描述（如有）",
    "content_complete": true/false,
    "completeness_issue": "完整性问题描述（如有）",
    "suggested_fix": "建议的修复方案（如有）"
}}"""

        try:
            result = await self.llm.complete_json(
                messages=[
                    {"role": "system", "content": "你是一个题目校验专家，检查分割结果的准确性。"},
                    {"role": "user", "content": prompt}
                ],
                tier=self._resolve_tier(),
                temperature=0.1,
            )

            # 添加校验标记
            if not result.get("boundary_correct", True):
                question["_boundary_warning"] = result.get("boundary_issue", "边界可能不正确")

            if not result.get("content_complete", True):
                question["_completeness_issue"] = result.get("completeness_issue", "内容可能不完整")

            return question

        except Exception:
            # 校验失败，返回原题目
            return question

    async def _extract_metadata_with_llm(self, question: dict) -> dict:
        """使用 LLM 提取题目元数据"""
        prompt = f"""分析以下题目，提取元数据：

题目内容：
{question.get('content', '')[:500]}

答案：{question.get('answer', '未知')}

请提取：
- difficulty: 难度（简单/中等/困难）
- knowledge_points: 知识点列表
- question_type: 题型（选择题/填空题/解答题等）
- estimated_time: 预估做题时间（分钟）

输出 JSON：
{{
    "difficulty": "简单/中等/困难",
    "knowledge_points": ["知识点1", "知识点2"],
    "question_type": "题型",
    "estimated_time": 5
}}"""

        try:
            result = await self.llm.complete_json(
                messages=[
                    {"role": "system", "content": "你是一个题目分析专家，提取题目的元数据信息。"},
                    {"role": "user", "content": prompt}
                ],
                tier=self._resolve_tier(),
                temperature=0.2,
            )

            return {"metadata": result}

        except Exception:
            return {"metadata": {}}

    def get_manifest(self) -> ToolManifest:
        """获取工具清单"""
        return ToolManifest(
            name=self.name,
            description_for_llm=self.description,
            applicable_scenarios=[
                "将题库 PDF 转换为结构化 JSON 格式",
                "提取题目用于闪卡生成、在线测试",
                "分析题目类型和难度分布",
            ],
            input_requirements={
                "pdf_path": "PDF 文件路径（必须存在）",
                "use_llm_validation": "是否使用 LLM 校验边界和完整性",
                "extract_metadata": "是否提取题目元数据（难度、知识点等）",
            },
            output_description="结构化题目数据，包含题号、内容、选项、答案、解析和元数据",
            examples=[
                {
                    "input": {"pdf_path": "D:/题库/数学真题.pdf", "extract_metadata": True},
                    "output_summary": "提取 50 道题目，包含难度、知识点等元数据"
                }
            ],
            limitations=[
                "扫描版 PDF 无法提取文本，需要 OCR 预处理",
                "复杂排版可能导致分割不准确",
                "LLM 校验会增加处理时间和成本",
            ],
            recommended_tier=self.recommended_tier or "fast",
            recommended_thinking=self.recommended_thinking or "low",
        )


# ---- 工具入口点 ----
def create_tool(context: ToolContext) -> QuestionSplitterTool:
    """创建工具实例"""
    return QuestionSplitterTool(context)
