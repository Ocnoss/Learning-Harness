"""Tool Develop - 工具开发元工具

功能：
1. 分析用户需求，理解工具目标
2. 设计工具 Schema（输入/输出/配置）
3. 生成工具代码框架
4. 审查现有工具，提出改进建议
5. 迭代优化工具实现

设计要点：
- 使用 balanced 或 flagship 模型（需要较强推理能力）
- 推荐 high 思考强度（复杂设计任务）
- 输出结构化设计文档和代码
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


class ToolDevelopTool(BaseTool):
    """工具开发元工具"""

    name = "tool-develop"
    description = "开发新工具或改进现有工具：分析需求 → 设计 Schema → 生成代码"
    version = "1.0.0"

    # 推荐配置
    recommended_tier = "balanced"
    recommended_thinking = "high"

    # AI 对任务的理解
    task_understanding = """
    这是一个元工具，用于开发其他工具。它能够：
    1. 理解用户的功能需求，分析所需工具的核心能力
    2. 设计符合规范的 JSON Schema（输入/输出）
    3. 生成符合项目架构的 Python 代码框架
    4. 审查现有工具代码，识别机械性、硬编码问题
    5. 提出改进建议，增强工具的 LLM 自主性

    适用场景：
    - 用户想要创建新工具但不确定如何设计
    - 需要审查现有工具是否符合"LLM 驱动"理念
    - 希望改进工具的灵活性和智能性
    """

    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["analyze", "design", "generate", "review", "improve"],
                "description": "执行动作：分析需求/设计Schema/生成代码/审查工具/改进建议"
            },
            "requirement": {
                "type": "string",
                "description": "用户需求描述（analyze/design 时必填）"
            },
            "tool_name": {
                "type": "string",
                "description": "目标工具名称（review/improve 时必填）"
            },
            "tool_path": {
                "type": "string",
                "description": "工具代码路径（review 时可选，默认从 registry 加载）"
            },
            "design_doc": {
                "type": "object",
                "description": "设计文档（generate 时必填）"
            },
            "output_dir": {
                "type": "string",
                "description": "代码输出目录（generate 时可选）"
            }
        },
        "required": ["action"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "analysis": {"type": "object"},
            "design": {"type": "object"},
            "code_files": {"type": "object"},
            "review_report": {"type": "object"},
            "suggestions": {"type": "array"}
        }
    }

    async def execute(self, input_data: dict) -> ToolResult:
        """执行工具开发任务"""
        action = input_data["action"]

        handlers = {
            "analyze": self._analyze_requirement,
            "design": self._design_schema,
            "generate": self._generate_code,
            "review": self._review_tool,
            "improve": self._suggest_improvements,
        }

        handler = handlers.get(action)
        if not handler:
            return ToolResult.fail(f"未知动作: {action}")

        return await handler(input_data)

    async def _analyze_requirement(self, input_data: dict) -> ToolResult:
        """分析用户需求"""
        requirement = input_data.get("requirement", "")
        if not requirement:
            return ToolResult.fail("analyze 动作需要提供 requirement 参数")

        prompt = f"""分析以下工具开发需求，提取关键信息：

需求描述：
{requirement}

请分析并输出 JSON：
{{
    "tool_name": "建议的工具名称（kebab-case）",
    "core_function": "核心功能描述（一句话）",
    "input_types": ["输入数据类型列表"],
    "output_types": ["输出数据类型列表"],
    "key_challenges": ["实现难点列表"],
    "llm_usage": "LLM 在工具中的作用（如：分类、生成、验证等）",
    "recommended_tier": "fast/balanced/flagship",
    "recommended_thinking": "low/medium/high",
    "similar_tools": ["项目中相似的现有工具"]
}}"""

        try:
            result = await self.llm.complete_json(
                messages=[
                    {"role": "system", "content": "你是一个工具设计专家，擅长分析需求并设计工具架构。"},
                    {"role": "user", "content": prompt}
                ],
                tier=self._resolve_tier(),
                temperature=0.3,
            )

            return ToolResult.ok(
                data={"analysis": result},
                summary_for_llm=f"需求分析完成：建议工具名 {result.get('tool_name', 'unknown')}",
                suggested_next_actions=[{
                    "tool": "tool-develop",
                    "action": "design",
                    "reason": "基于分析结果设计详细 Schema",
                    "input_hint": {"action": "design", "requirement": requirement}
                }]
            )

        except Exception as e:
            return ToolResult.fail(f"需求分析失败: {e}")

    async def _design_schema(self, input_data: dict) -> ToolResult:
        """设计工具 Schema"""
        requirement = input_data.get("requirement", "")
        if not requirement:
            return ToolResult.fail("design 动作需要提供 requirement 参数")

        prompt = f"""基于以下需求，设计工具的完整 Schema：

需求描述：
{requirement}

请设计并输出 JSON：
{{
    "tool_name": "工具名称",
    "description": "工具描述",
    "task_understanding": "AI 对任务的理解（用于 Tool-Agent 决策）",
    "input_schema": {{
        "type": "object",
        "properties": {{...}},
        "required": [...]
    }},
    "output_schema": {{
        "type": "object",
        "properties": {{...}}
    }},
    "recommended_tier": "fast/balanced/flagship",
    "recommended_thinking": "low/medium/high",
    "dependencies": ["依赖的其他工具"],
    "limitations": ["使用限制和注意事项"]
}}

确保 Schema 符合 JSON Schema 规范，字段命名清晰。"""

        try:
            result = await self.llm.complete_json(
                messages=[
                    {"role": "system", "content": "你是一个工具架构师，擅长设计清晰、灵活的工具接口。"},
                    {"role": "user", "content": prompt}
                ],
                tier=self._resolve_tier(),
                temperature=0.2,
            )

            return ToolResult.ok(
                data={"design": result},
                summary_for_llm=f"Schema 设计完成：{result.get('tool_name', 'unknown')}",
                suggested_next_actions=[{
                    "tool": "tool-develop",
                    "action": "generate",
                    "reason": "基于设计文档生成代码框架",
                    "input_hint": {"action": "generate", "design_doc": result}
                }]
            )

        except Exception as e:
            return ToolResult.fail(f"Schema 设计失败: {e}")

    async def _generate_code(self, input_data: dict) -> ToolResult:
        """生成工具代码框架"""
        design_doc = input_data.get("design_doc")
        if not design_doc:
            return ToolResult.fail("generate 动作需要提供 design_doc 参数")

        output_dir = input_data.get("output_dir", "")

        prompt = f"""基于以下设计文档，生成工具的 Python 代码框架：

设计文档：
{json.dumps(design_doc, ensure_ascii=False, indent=2)}

项目架构要求：
1. 继承 BaseTool 类
2. 定义 name, description, version, input_schema, output_schema
3. 声明 recommended_tier 和 recommended_thinking
4. 实现 execute 方法
5. 提供 create_tool 工厂函数
6. 使用 ToolResult.ok() / ToolResult.fail() 返回结果

请生成以下文件内容：
1. plugin.json
2. main.py

输出 JSON 格式：
{{
    "plugin_json": {{...}},
    "main_py": "Python 代码内容"
}}"""

        try:
            result = await self.llm.complete_json(
                messages=[
                    {"role": "system", "content": "你是一个 Python 开发专家，熟悉 LLM 工具开发最佳实践。"},
                    {"role": "user", "content": prompt}
                ],
                tier=self._resolve_tier(),
                temperature=0.3,
                max_tokens=4096,
            )

            code_files = {
                "plugin.json": result.get("plugin_json", {}),
                "main.py": result.get("main_py", ""),
            }

            # 如果指定了输出目录，写入文件
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                with open(output_path / "plugin.json", "w", encoding="utf-8") as f:
                    json.dump(code_files["plugin.json"], f, ensure_ascii=False, indent=2)

                with open(output_path / "main.py", "w", encoding="utf-8") as f:
                    f.write(code_files["main.py"])

            return ToolResult.ok(
                data={"code_files": code_files, "output_dir": output_dir},
                summary_for_llm=f"代码生成完成：{design_doc.get('tool_name', 'unknown')}",
            )

        except Exception as e:
            return ToolResult.fail(f"代码生成失败: {e}")

    async def _review_tool(self, input_data: dict) -> ToolResult:
        """审查现有工具"""
        tool_name = input_data.get("tool_name")
        if not tool_name:
            return ToolResult.fail("review 动作需要提供 tool_name 参数")

        # 尝试从 registry 获取工具
        tool = self._tool_registry.get(tool_name)
        tool_code = ""

        if tool:
            # 读取工具代码
            tool_path = Path(tool.__module__.replace(".", "/") + ".py")
            if tool_path.exists():
                tool_code = tool_path.read_text(encoding="utf-8")

        if not tool_code:
            return ToolResult.fail(f"无法获取工具 {tool_name} 的代码")

        prompt = f"""审查以下工具代码，识别问题并提出改进建议：

工具名称：{tool_name}

代码：
```python
{tool_code[:8000]}  # 限制长度
```

请从以下维度审查：
1. **机械性问题**：是否过度依赖硬编码规则，而非 LLM 智能判断
2. **灵活性问题**：是否充分利用 LLM 的推理能力
3. **Schema 设计**：输入/输出定义是否清晰、完整
4. **错误处理**：是否妥善处理异常情况
5. **文档完整性**：描述、示例、限制说明是否充分

输出 JSON：
{{
    "overall_score": 1-10,
    "issues": [
        {{"type": "机械性/灵活性/Schema/错误处理/文档", "severity": "high/medium/low", "description": "...", "suggestion": "..."}}
    ],
    "strengths": ["优点列表"],
    "recommended_changes": ["建议的修改"]
}}"""

        try:
            result = await self.llm.complete_json(
                messages=[
                    {"role": "system", "content": "你是一个代码审查专家，专注于 LLM 工具的设计质量。"},
                    {"role": "user", "content": prompt}
                ],
                tier=self._resolve_tier(),
                temperature=0.3,
            )

            return ToolResult.ok(
                data={"review_report": result},
                summary_for_llm=f"审查完成：评分 {result.get('overall_score', '?')}/10，发现 {len(result.get('issues', []))} 个问题",
                suggested_next_actions=[{
                    "tool": "tool-develop",
                    "action": "improve",
                    "reason": "基于审查结果生成改进方案",
                    "input_hint": {"action": "improve", "tool_name": tool_name}
                }]
            )

        except Exception as e:
            return ToolResult.fail(f"工具审查失败: {e}")

    async def _suggest_improvements(self, input_data: dict) -> ToolResult:
        """提出改进建议"""
        tool_name = input_data.get("tool_name")
        if not tool_name:
            return ToolResult.fail("improve 动作需要提供 tool_name 参数")

        # 先执行审查
        review_result = await self._review_tool({"tool_name": tool_name})
        if not review_result.success:
            return review_result

        review_report = review_result.data.get("review_report", {})

        prompt = f"""基于以下审查报告，生成具体的代码改进方案：

工具名称：{tool_name}

审查报告：
{json.dumps(review_report, ensure_ascii=False, indent=2)}

请生成：
1. 改进后的代码框架（关键部分）
2. 具体的修改说明
3. 迁移指南（如何从旧版本迁移）

输出 JSON：
{{
    "improved_code": "改进后的代码框架",
    "changes": [
        {{"file": "...", "change": "...", "reason": "..."}}
    ],
    "migration_guide": "迁移步骤说明"
}}"""

        try:
            result = await self.llm.complete_json(
                messages=[
                    {"role": "system", "content": "你是一个重构专家，擅长改进 LLM 工具的设计。"},
                    {"role": "user", "content": prompt}
                ],
                tier=self._resolve_tier(),
                temperature=0.3,
                max_tokens=4096,
            )

            return ToolResult.ok(
                data={"suggestions": result},
                summary_for_llm=f"改进方案生成完成：{len(result.get('changes', []))} 处修改建议",
            )

        except Exception as e:
            return ToolResult.fail(f"改进建议生成失败: {e}")

    def get_manifest(self) -> ToolManifest:
        """获取工具清单"""
        return ToolManifest(
            name=self.name,
            description_for_llm=self.description,
            applicable_scenarios=[
                "需要创建新工具时",
                "需要审查现有工具设计时",
                "需要改进工具灵活性时",
                "不确定如何设计工具 Schema 时",
            ],
            input_requirements={
                "action": "执行动作：analyze/design/generate/review/improve",
                "requirement": "用户需求描述（analyze/design 时必填）",
                "tool_name": "目标工具名称（review/improve 时必填）",
            },
            output_description="根据动作类型输出分析结果、设计文档、代码框架或审查报告",
            examples=[
                {
                    "input": {"action": "analyze", "requirement": "我需要一个工具来总结 PDF 文档的主要内容"},
                    "output_summary": "分析需求，建议工具名 pdf-summarizer，推荐 balanced 模型"
                },
                {
                    "input": {"action": "review", "tool_name": "resource-indexer"},
                    "output_summary": "审查工具代码，识别机械性问题和改进建议"
                }
            ],
            limitations=[
                "代码生成结果需要人工审查和调整",
                "复杂工具可能需要多次迭代设计",
            ],
            recommended_tier=self.recommended_tier or "balanced",
            recommended_thinking=self.recommended_thinking or "high",
        )


# ---- 工具入口点 ----
def create_tool(context: ToolContext) -> ToolDevelopTool:
    """创建工具实例"""
    return ToolDevelopTool(context)
