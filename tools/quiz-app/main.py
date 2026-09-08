"""Quiz App - 刷题工具

功能：
1. 加载 question-splitter 生成的题目文件
2. 提供刷题界面（CLI）
3. 记录答题进度和正确率

使用方式：
- 作为独立工具调用，获取题目和统计
- 作为交互式 CLI 运行，进行刷题练习
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

# 确保能导入 core 模块
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.base_tool import BaseTool, ToolContext, ToolResult

# 动态导入 quiz_engine（因为目录名含连字符）
import importlib.util

_module_dir = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "quiz_engine",
    _module_dir / "quiz_engine.py"
)
quiz_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(quiz_engine)
QuizEngine = quiz_engine.QuizEngine
QuizProgress = quiz_engine.QuizProgress


class QuizAppTool(BaseTool):
    """刷题工具"""

    name = "quiz-app"
    description = "加载题目文件进行刷题练习"
    version = "1.0.0"

    # 不需要 LLM
    recommended_tier = None

    # 声明依赖
    dependencies = ["question-splitter"]

    input_schema = {
        "type": "object",
        "properties": {
            "question_file": {
                "type": "string",
                "description": "题目文件路径（JSON 格式）"
            },
            "mode": {
                "type": "string",
                "enum": ["sequential", "random", "review"],
                "default": "sequential",
                "description": "刷题模式"
            },
            "count": {
                "type": "integer",
                "default": 10,
                "description": "刷题数量"
            },
            "progress_file": {
                "type": "string",
                "description": "进度文件路径"
            }
        },
        "required": ["question_file"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "total": {"type": "integer"},
            "correct": {"type": "integer"},
            "accuracy": {"type": "number"},
            "history": {"type": "array"},
            "wrong_questions": {"type": "array"}
        }
    }

    async def execute(self, input_data: dict) -> ToolResult:
        """执行刷题"""
        question_file = input_data["question_file"]
        mode = input_data.get("mode", "sequential")
        count = input_data.get("count", 10)
        progress_file = input_data.get("progress_file")

        # 1. 加载题目
        questions = self._load_questions(question_file)
        if not questions:
            return ToolResult.fail(f"无法加载题目文件: {question_file}")

        # 2. 创建引擎
        engine = QuizEngine(questions)

        # 3. 加载进度（如果有）
        if progress_file:
            progress_path = Path(progress_file)
            engine.progress = QuizProgress.load(progress_path)

        # 4. 获取题目
        quiz_questions = engine.get_questions(
            mode=mode,
            count=count,
            start_index=engine.progress.current_index
        )

        if not quiz_questions:
            return ToolResult.fail("没有可刷的题目")

        # 5. 返回题目和统计（非交互模式）
        return ToolResult.ok(
            data={
                "questions": quiz_questions,
                "total": len(questions),
                "current_index": engine.progress.current_index,
                "stats": engine.get_stats()
            },
            metadata={
                "mode": mode,
                "count": len(quiz_questions)
            }
        )

    def _load_questions(self, file_path: str) -> list[dict]:
        """加载题目文件"""
        path = Path(file_path)
        if not path.exists():
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 支持两种格式：
        # 1. 直接的题目列表
        # 2. 包含 questions 字段的对象
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "questions" in data:
            return data["questions"]

        return []


# ---- 交互式 CLI ----
class QuizCLI:
    """刷题 CLI"""

    def __init__(self, question_file: str, progress_file: str | None = None):
        """初始化

        Args:
            question_file: 题目文件路径
            progress_file: 进度文件路径
        """
        self.question_file = Path(question_file)
        self.progress_file = Path(progress_file) if progress_file else None

        # 加载题目
        with open(self.question_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            self.questions = data
        elif isinstance(data, dict) and "questions" in data:
            self.questions = data["questions"]
        else:
            raise ValueError("无效的题目文件格式")

        # 创建引擎
        self.engine = QuizEngine(self.questions)

        # 加载进度
        if self.progress_file and self.progress_file.exists():
            self.engine.progress = QuizProgress.load(self.progress_file)

    def run(self, mode: str = "sequential", count: int = 10):
        """运行刷题"""
        print("=" * 60)
        print("刷题模式")
        print("=" * 60)
        print(f"总题数: {len(self.questions)}")
        print(f"已答题: {self.engine.progress.total_answered}")
        print(f"正确率: {self.engine.progress.accuracy:.1%}")
        print("=" * 60)

        # 获取题目
        questions = self.engine.get_questions(
            mode=mode,
            count=count,
            start_index=self.engine.progress.current_index
        )

        if not questions:
            print("没有可刷的题目")
            return

        # 开始刷题
        for i, question in enumerate(questions, 1):
            self._show_question(question, i, len(questions))
            user_answer = input("\n你的答案: ").strip()

            start_time = time.time()
            record = self.engine.answer_question(
                question,
                user_answer,
                duration_ms=(time.time() - start_time) * 1000
            )

            self._show_result(record, question)

            # 更新进度
            self.engine.progress.current_index += 1

            # 询问是否继续
            if i < len(questions):
                choice = input("\n按 Enter 继续，输入 q 退出: ").strip().lower()
                if choice == "q":
                    break

        # 保存进度
        if self.progress_file:
            self.engine.progress.save(self.progress_file)
            print(f"\n进度已保存到: {self.progress_file}")

        # 显示统计
        self._show_stats()

    def _show_question(self, question: dict, current: int, total: int):
        """显示题目"""
        print("\n" + "=" * 60)
        print(f"第 {current}/{total} 题 (ID: {question.get('id', '?')})")
        print("=" * 60)
        print(f"\n{question.get('content', '(无题干)')}")

        options = question.get("options", [])
        if options:
            print()
            for j, option in enumerate(options):
                label = chr(ord('A') + j)
                print(f"{label}. {option}")

    def _show_result(self, record, question: dict):
        """显示结果"""
        print("\n" + "-" * 60)
        if record.is_correct:
            print("[正确]")
        else:
            print(f"[错误] 正确答案: {record.correct_answer}")

        # 显示解析
        analysis = question.get("analysis", "")
        if analysis:
            print(f"\n解析: {analysis}")

    def _show_stats(self):
        """显示统计"""
        stats = self.engine.get_stats()
        print("\n" + "=" * 60)
        print("刷题统计")
        print("=" * 60)
        print(f"本次答题: {stats['total']}")
        print(f"正确: {stats['correct']}")
        print(f"正确率: {stats['accuracy']:.1%}")
        print(f"错题数: {stats['wrong_count']}")


# ---- 工具入口点 ----
def create_tool(context: ToolContext) -> QuizAppTool:
    """创建工具实例"""
    return QuizAppTool(context)


# ---- CLI 入口 ----
def main():
    """CLI 入口"""
    import argparse

    parser = argparse.ArgumentParser(description="刷题工具")
    parser.add_argument("question_file", help="题目文件路径")
    parser.add_argument("--mode", choices=["sequential", "random", "review"],
                        default="sequential", help="刷题模式")
    parser.add_argument("--count", type=int, default=10, help="刷题数量")
    parser.add_argument("--progress", help="进度文件路径")

    args = parser.parse_args()

    cli = QuizCLI(args.question_file, args.progress)
    cli.run(mode=args.mode, count=args.count)


if __name__ == "__main__":
    main()
