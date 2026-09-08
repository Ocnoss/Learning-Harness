"""LLM 校验模块 - 使用 LLM 校验可疑的题目边界"""

import sys
from pathlib import Path
from typing import Any

# 确保能导入 core 模块
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class LLMValidator:
    """LLM 边界校验器"""

    def __init__(self, llm_client, tier=None):
        """初始化

        Args:
            llm_client: LLM 客户端
            tier: 模型层级
        """
        self.llm = llm_client
        self.tier = tier
        self.call_count = 0

    async def validate_boundary(
        self,
        text_before: str,
        text_after: str,
        question_id: int
    ) -> bool:
        """校验边界是否正确

        Args:
            text_before: 边界前的文本（上一题结尾）
            text_after: 边界后的文本（当前题开头）
            question_id: 当前题号

        Returns:
            bool: 边界是否正确
        """
        self.call_count += 1

        # 构建提示词
        prompt = f"""判断以下文本分割是否正确。

上一题结尾：
...{text_before[-200:]}

当前题开头（题号 {question_id}）：
{text_after[:200]}...

这个分割点是否正确？只回答：是/否"""

        try:
            response = await self.llm.complete(
                messages=[
                    {"role": "system", "content": "判断文本分割点是否正确。只回答：是/否"},
                    {"role": "user", "content": prompt}
                ],
                tier=self.tier,
                temperature=0.1,
                max_tokens=10
            )

            content = response.content.strip().lower()
            return "是" in content or "yes" in content

        except Exception:
            # LLM 调用失败时，默认接受边界
            return True

    async def validate_question(self, question: dict) -> dict:
        """校验单个题目的完整性

        Args:
            question: 题目数据

        Returns:
            dict: 校验后的题目数据（可能包含修正建议）
        """
        self.call_count += 1

        # 检查是否有明显问题
        issues = []

        if not question.get("content"):
            issues.append("缺少题干")

        if not question.get("options"):
            issues.append("缺少选项")

        if not question.get("answer"):
            issues.append("缺少答案")

        if not issues:
            return question

        # 有问题时，调用 LLM 尝试修复
        prompt = f"""以下题目数据有问题：{', '.join(issues)}

题目内容：
{question.get('content', '(无)')[:500]}

选项：
{chr(10).join(question.get('options', ['(无)']))}

请判断这个题目是否完整。只回答：完整/不完整"""

        try:
            response = await self.llm.complete(
                messages=[
                    {"role": "system", "content": "判断题目数据是否完整。只回答：完整/不完整"},
                    {"role": "user", "content": prompt}
                ],
                tier=self.tier,
                temperature=0.1,
                max_tokens=10
            )

            content = response.content.strip()
            question["_validation"] = {
                "issues": issues,
                "is_complete": "完整" in content,
                "llm_response": content
            }

        except Exception as e:
            question["_validation"] = {
                "issues": issues,
                "is_complete": False,
                "error": str(e)
            }

        return question
