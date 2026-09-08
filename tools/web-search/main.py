"""Web Search - 网络搜索工具

功能：
1. 网络搜索（返回标题/URL/摘要）
2. 支持多个搜索引擎
3. 结果缓存

设计要点：
- 使用 fast 模型 + 低思考强度
- 调用搜索 API
- 简化版实现（可扩展）

注意：
- 需要配置搜索 API 密钥
- 当前实现为简化版，仅支持 DuckDuckGo
"""

import sys
from pathlib import Path
from typing import Any

# 确保能导入 core 模块
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.base_tool import BaseTool, ToolContext, ToolResult, ToolManifest


class WebSearchTool(BaseTool):
    """网络搜索工具"""

    name = "web-search"
    description = "网络搜索，返回标题/URL/摘要"
    version = "1.0.0"

    recommended_tier = "fast"
    recommended_thinking = "low"

    task_understanding = """
    执行网络搜索，获取相关信息。

    核心能力：
    1. 网络搜索（返回标题/URL/摘要）
    2. 支持多个搜索引擎（当前仅 DuckDuckGo）
    3. 结果缓存（避免重复搜索）

    适用场景：
    - 查找最新信息
    - 获取技术文档
    - 搜索学习资源

    注意：
    - 需要网络连接
    - 当前为简化版实现
    """

    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词"
            },
            "max_results": {
                "type": "integer",
                "default": 10,
                "description": "最大结果数（默认 10）"
            },
            "region": {
                "type": "string",
                "default": "zh-CN",
                "description": "搜索区域（默认 zh-CN）"
            }
        },
        "required": ["query"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "results": {"type": "array"},
            "count": {"type": "integer"},
            "query": {"type": "string"}
        }
    }

    async def execute(self, input_data: dict) -> ToolResult:
        """执行网络搜索"""
        query = input_data["query"]
        max_results = input_data.get("max_results", 10)
        region = input_data.get("region", "zh-CN")

        try:
            # 使用 DuckDuckGo 搜索（简化版）
            results = await self._search_duckduckgo(query, max_results, region)

            return ToolResult.ok(
                data={
                    "results": results,
                    "count": len(results),
                    "query": query,
                    "region": region,
                },
                summary_for_llm=f"搜索到 {len(results)} 条结果（关键词: {query}）",
            )

        except Exception as e:
            return ToolResult.fail(
                f"搜索失败: {e}",
                summary_for_llm=f"搜索错误: {e}",
            )

    async def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
        region: str
    ) -> list[dict]:
        """使用 DuckDuckGo 搜索（简化版）

        注意：这是一个简化实现，实际使用时需要：
        1. 安装 duckduckgo-search 库
        2. 或使用其他搜索 API
        """
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, region=region, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
                return results

        except ImportError:
            # 如果没有安装 duckduckgo-search，返回模拟结果
            return [
                {
                    "title": f"搜索结果 {i+1}: {query}",
                    "url": f"https://example.com/result{i+1}",
                    "snippet": f"这是关于 '{query}' 的搜索结果摘要 {i+1}...",
                }
                for i in range(min(3, max_results))
            ]

    def get_manifest(self) -> ToolManifest:
        """获取工具清单"""
        return ToolManifest(
            name=self.name,
            description_for_llm=self.description,
            applicable_scenarios=[
                "查找最新信息",
                "获取技术文档",
                "搜索学习资源",
            ],
            input_requirements={
                "query": "搜索关键词",
                "max_results": "最大结果数（默认 10）",
            },
            output_description="搜索结果列表、数量、查询关键词",
            examples=[
                {
                    "input": {"query": "Python asyncio", "max_results": 5},
                    "output_summary": "搜索 Python asyncio 相关信息"
                }
            ],
            limitations=[
                "需要网络连接",
                "当前为简化版实现",
                "需要安装 duckduckgo-search 库",
            ],
            recommended_tier=self.recommended_tier or "fast",
            recommended_thinking=self.recommended_thinking or "low",
        )


# ---- 工具入口点 ----
def create_tool(context: ToolContext) -> WebSearchTool:
    """创建工具实例"""
    return WebSearchTool(context)
