"""Quiz Engine - 刷题引擎"""

import json
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class AnswerRecord:
    """答题记录"""
    question_id: int
    user_answer: str
    correct_answer: str
    is_correct: bool
    duration_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class QuizProgress:
    """刷题进度"""
    current_index: int = 0
    total_answered: int = 0
    total_correct: int = 0
    wrong_question_ids: list[int] = field(default_factory=list)
    history: list[AnswerRecord] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """正确率"""
        if self.total_answered == 0:
            return 0.0
        return self.total_correct / self.total_answered

    def add_record(self, record: AnswerRecord):
        """添加答题记录"""
        self.history.append(record)
        self.total_answered += 1
        if record.is_correct:
            self.total_correct += 1
        else:
            if record.question_id not in self.wrong_question_ids:
                self.wrong_question_ids.append(record.question_id)

    def save(self, file_path: Path):
        """保存进度"""
        data = {
            "current_index": self.current_index,
            "total_answered": self.total_answered,
            "total_correct": self.total_correct,
            "wrong_question_ids": self.wrong_question_ids,
            "history": [asdict(r) for r in self.history]
        }
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: Path) -> "QuizProgress":
        """加载进度"""
        if not file_path.exists():
            return cls()

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        progress = cls(
            current_index=data.get("current_index", 0),
            total_answered=data.get("total_answered", 0),
            total_correct=data.get("total_correct", 0),
            wrong_question_ids=data.get("wrong_question_ids", [])
        )
        progress.history = [
            AnswerRecord(**r) for r in data.get("history", [])
        ]
        return progress


class QuizEngine:
    """刷题引擎"""

    def __init__(self, questions: list[dict]):
        """初始化

        Args:
            questions: 题目列表
        """
        self.questions = questions
        self.progress = QuizProgress()

    def get_question(self, index: int) -> dict | None:
        """获取题目"""
        if 0 <= index < len(self.questions):
            return self.questions[index]
        return None

    def get_questions(
        self,
        mode: str = "sequential",
        count: int = 10,
        start_index: int = 0
    ) -> list[dict]:
        """获取题目列表

        Args:
            mode: 模式（sequential/random/review）
            count: 数量
            start_index: 起始索引

        Returns:
            题目列表
        """
        if mode == "random":
            indices = random.sample(
                range(len(self.questions)),
                min(count, len(self.questions))
            )
            return [self.questions[i] for i in indices]

        elif mode == "review":
            # 复习错题
            wrong_ids = self.progress.wrong_question_ids
            if not wrong_ids:
                return []
            questions = [
                q for q in self.questions
                if q.get("id") in wrong_ids
            ]
            return questions[:count]

        else:  # sequential
            end_index = min(start_index + count, len(self.questions))
            return self.questions[start_index:end_index]

    def check_answer(self, question: dict, user_answer: str) -> bool:
        """检查答案

        Args:
            question: 题目
            user_answer: 用户答案

        Returns:
            是否正确
        """
        correct_answer = question.get("answer", "").strip().upper()
        user_answer = user_answer.strip().upper()

        # 单选题
        if len(correct_answer) == 1:
            return user_answer == correct_answer

        # 多选题（答案可能是 "AB" 或 "A,B" 格式）
        correct_set = set(correct_answer.replace(",", "").replace(" ", ""))
        user_set = set(user_answer.replace(",", "").replace(" ", ""))
        return user_set == correct_set

    def answer_question(
        self,
        question: dict,
        user_answer: str,
        duration_ms: float
    ) -> AnswerRecord:
        """回答问题

        Args:
            question: 题目
            user_answer: 用户答案
            duration_ms: 答题用时

        Returns:
            答题记录
        """
        is_correct = self.check_answer(question, user_answer)

        record = AnswerRecord(
            question_id=question.get("id", 0),
            user_answer=user_answer,
            correct_answer=question.get("answer", ""),
            is_correct=is_correct,
            duration_ms=duration_ms
        )

        self.progress.add_record(record)
        return record

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total": self.progress.total_answered,
            "correct": self.progress.total_correct,
            "accuracy": self.progress.accuracy,
            "wrong_count": len(self.progress.wrong_question_ids)
        }
