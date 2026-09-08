"""文件名分类模块 - 基于文件名关键词的初步分类"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FilenameMatch:
    """文件名匹配结果"""
    category: str
    confidence: str  # high, medium, low
    matched_keywords: list[str]


class FilenameClassifier:
    """文件名分类器"""

    # 分类关键词（按优先级排序）
    CATEGORY_KEYWORDS = {
        "教材": [
            "教材", "教程", "课本", "教科书", "讲义",
            "textbook", "coursebook", "manual"
        ],
        "讲义": [
            "讲义", "课件", "ppt", "slides", "教案",
            "lecture", "courseware", "handout"
        ],
        "笔记": [
            "笔记", "随堂", "记录", "总结", "整理",
            "note", "summary", "review"
        ],
        "题集": [
            "题", "题库", "习题", "试题", "真题", "模拟题",
            "600题", "1000题", "练习", "刷题",
            "exercise", "quiz", "test", "practice"
        ],
        "题解": [
            "解析", "答案", "解答", "详解", "题解",
            "solution", "answer", "explanation"
        ],
    }

    # 高置信度模式（完整匹配）
    HIGH_CONFIDENCE_PATTERNS = [
        r"教材|讲义|笔记|题集|题解|解析|答案",
        r"^\d+题|600题|1000题",
        r"textbook|lecture|note|solution"
    ]

    def classify(self, file_path: Path) -> FilenameMatch:
        """分类文件

        Args:
            file_path: 文件路径

        Returns:
            FilenameMatch: 匹配结果
        """
        file_name = file_path.name.lower()
        stem = file_path.stem.lower()

        # 收集所有匹配的关键词
        matches = {}

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            matched = []
            for keyword in keywords:
                if keyword.lower() in file_name:
                    matched.append(keyword)
            if matched:
                matches[category] = matched

        # 确定最佳分类
        if not matches:
            return FilenameMatch(
                category="其他",
                confidence="low",
                matched_keywords=[]
            )

        # 优先级：讲义 > 教材 > 笔记 > 题解 > 题集
        priority_order = ["讲义", "教材", "笔记", "题解", "题集"]

        for category in priority_order:
            if category in matches:
                # 检查置信度
                confidence = self._calculate_confidence(file_name, matches[category])
                return FilenameMatch(
                    category=category,
                    confidence=confidence,
                    matched_keywords=matches[category]
                )

        # 默认返回第一个匹配
        category = list(matches.keys())[0]
        return FilenameMatch(
            category=category,
            confidence="medium",
            matched_keywords=matches[category]
        )

    def _calculate_confidence(self, file_name: str, keywords: list[str]) -> str:
        """计算置信度"""
        # 多个关键词匹配，置信度高
        if len(keywords) >= 2:
            return "high"

        # 检查高置信度模式
        for pattern in self.HIGH_CONFIDENCE_PATTERNS:
            if re.search(pattern, file_name, re.IGNORECASE):
                return "high"

        # 关键词在文件名开头，置信度高
        for keyword in keywords:
            if file_name.startswith(keyword.lower()):
                return "high"

        return "medium"
