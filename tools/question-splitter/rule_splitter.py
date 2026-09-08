"""规则分题模块 - 使用正则表达式初步分割题目"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuestionBoundary:
    """题目边界"""
    start: int  # 起始位置
    end: int  # 结束位置
    question_id: int | None  # 题号
    confidence: float  # 置信度 (0-1)
    needs_validation: bool = False  # 是否需要 LLM 校验


@dataclass
class SplitResult:
    """分割结果"""
    questions: list[dict] = field(default_factory=list)
    boundaries: list[QuestionBoundary] = field(default_factory=list)
    raw_text: str = ""


class RuleSplitter:
    """规则分题器"""

    # 题号模式（按优先级排序）
    QUESTION_PATTERNS = [
        r"^(\d+)[.、．]\s*",  # 1. 或 1、
        r"^【(\d+)】\s*",  # 【1】
        r"^第\s*(\d+)\s*题\s*",  # 第1题
        r"^(\d+)\s*[:：]\s*",  # 1: 或 1：
    ]

    # 选项模式
    OPTION_PATTERN = r"^([A-D])[.、．]\s*(.+?)$"

    # 答案/解析标记
    ANSWER_MARKERS = [
        r"【答案】",
        r"【解析】",
        r"答案[:：]",
        r"解析[:：]",
        r"正确答案[:：]",
    ]

    def __init__(self, custom_pattern: str | None = None):
        """初始化

        Args:
            custom_pattern: 自定义题号正则表达式
        """
        if custom_pattern:
            self.QUESTION_PATTERNS.insert(0, custom_pattern)

        # 编译正则
        self._question_res = [
            re.compile(p, re.MULTILINE) for p in self.QUESTION_PATTERNS
        ]
        self._option_re = re.compile(self.OPTION_PATTERN, re.MULTILINE)
        self._answer_res = [
            re.compile(p, re.IGNORECASE) for p in self.ANSWER_MARKERS
        ]

    def split(self, text: str) -> SplitResult:
        """分割文本为题目列表

        Args:
            text: 原始文本

        Returns:
            SplitResult: 分割结果
        """
        result = SplitResult(raw_text=text)

        # 1. 找到所有题目边界
        boundaries = self._find_boundaries(text)
        result.boundaries = boundaries

        # 2. 根据边界分割题目
        for i, boundary in enumerate(boundaries):
            question_text = text[boundary.start:boundary.end]
            question = self._parse_question(
                question_text,
                question_id=boundary.question_id or (i + 1)
            )
            question["_boundary_confidence"] = boundary.confidence
            question["_needs_validation"] = boundary.needs_validation
            result.questions.append(question)

        return result

    def _find_boundaries(self, text: str) -> list[QuestionBoundary]:
        """找到所有题目边界"""
        boundaries = []
        lines = text.split("\n")

        current_start = 0
        current_id = None
        current_confidence = 0.0

        for i, line in enumerate(lines):
            line_start = sum(len(l) + 1 for l in lines[:i])  # +1 for \n

            # 检查是否匹配题号模式
            match = self._match_question_start(line)
            if match:
                # 保存上一个题目
                if current_id is not None:
                    boundaries.append(QuestionBoundary(
                        start=current_start,
                        end=line_start,
                        question_id=current_id,
                        confidence=current_confidence,
                        needs_validation=current_confidence < 0.8
                    ))

                # 开始新题目
                current_start = line_start
                current_id = int(match.group(1))
                current_confidence = self._calculate_confidence(line, match)

        # 保存最后一个题目
        if current_id is not None:
            boundaries.append(QuestionBoundary(
                start=current_start,
                end=len(text),
                question_id=current_id,
                confidence=current_confidence,
                needs_validation=current_confidence < 0.8
            ))

        return boundaries

    def _match_question_start(self, line: str) -> re.Match | None:
        """检查行是否匹配题号模式"""
        for pattern in self._question_res:
            match = pattern.match(line)
            if match:
                return match
        return None

    def _calculate_confidence(self, line: str, match: re.Match) -> float:
        """计算匹配置信度"""
        confidence = 0.5  # 基础置信度

        # 题号在行首，置信度高
        if match.start() == 0:
            confidence += 0.2

        # 题号后面紧跟标点，置信度高
        if match.group(0).rstrip() != match.group(1):
            confidence += 0.1

        # 行长度合理（不太长也不太短），置信度高
        if 10 < len(line) < 200:
            confidence += 0.1

        # 行内包含选项标记，置信度高
        if any(c in line for c in ["A.", "B.", "C.", "D."]):
            confidence += 0.1

        return min(confidence, 1.0)

    def _parse_question(self, text: str, question_id: int) -> dict:
        """解析单个题目

        Args:
            text: 题目文本
            question_id: 题号

        Returns:
            dict: 题目数据
        """
        lines = text.split("\n")

        # 分离题干和选项
        content_lines = []
        options = []
        answer = ""
        analysis = ""

        in_options = False
        in_answer = False
        in_analysis = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检查是否是答案标记（如 【答案】A）
            answer_match = re.search(r"【答案】\s*([A-D]+)", line, re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1)
                in_answer = False
                in_analysis = False
                continue

            # 检查是否是解析标记（如 【解析】...）
            analysis_match = re.search(r"【解析】\s*(.*)", line, re.IGNORECASE)
            if analysis_match:
                analysis = analysis_match.group(1)
                in_analysis = True
                in_answer = False
                continue

            # 检查是否是选项
            option_match = self._option_re.match(line)
            if option_match:
                in_options = True
                options.append(option_match.group(2).strip())
                continue

            # 根据状态分配内容
            if in_analysis:
                analysis += " " + line
            elif in_answer:
                answer += line
            elif in_options:
                # 选项内容延续
                if options:
                    options[-1] += " " + line
            else:
                content_lines.append(line)

        return {
            "id": question_id,
            "content": "\n".join(content_lines).strip(),
            "options": options,
            "answer": answer.strip(),
            "analysis": analysis.strip()
        }

    def _is_answer_marker(self, line: str) -> bool:
        """检查是否是答案标记"""
        for pattern in self._answer_res:
            if pattern.search(line):
                return True
        return False

    def _is_analysis_marker(self, line: str) -> bool:
        """检查是否是解析标记"""
        return "解析" in line or "分析" in line
