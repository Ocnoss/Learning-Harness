"""工作流引擎模块 - 编排多个工具形成工作流"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.base_tool import BaseTool, ToolResult


@dataclass
class WorkflowStep:
    """工作流步骤

    Attributes:
        tool_name: 工具名称
        input_mapping: 输入映射，key 为目标字段，value 为来源表达式
            支持 JSONPath 风格:
            - "$.input.xxx" - 引用初始输入
            - "$.results.<tool_name>.xxx" - 引用前序工具的输出
            - 其他值直接作为字面量传递
        condition: 执行条件表达式（可选），如 "$.results.xxx.success == true"
    """
    tool_name: str
    input_mapping: dict = field(default_factory=dict)
    condition: str | None = None


@dataclass
class Workflow:
    """工作流定义

    Attributes:
        name: 工作流名称
        description: 工作流描述
        steps: 步骤列表
    """
    name: str
    description: str = ""
    steps: list[WorkflowStep] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Workflow":
        """从字典创建工作流"""
        steps = [
            WorkflowStep(
                tool_name=s["tool_name"],
                input_mapping=s.get("input_mapping", {}),
                condition=s.get("condition"),
            )
            for s in data.get("steps", [])
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Workflow":
        """从 JSON 文件加载工作流"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class WorkflowEngine:
    """工作流执行引擎

    负责按顺序执行工作流中的各个步骤，处理输入映射和条件判断。

    使用示例:
        engine = WorkflowEngine()
        engine.register_tool(tool_a)
        engine.register_tool(tool_b)

        workflow = Workflow.from_file("workflows/my-workflow.json")
        results = await engine.execute(workflow, {"note_content": "..."})
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool):
        """注册工具"""
        self._tools[tool.name] = tool

    def register_tools(self, tools: list[BaseTool]):
        """批量注册工具"""
        for tool in tools:
            self.register_tool(tool)

    async def execute(
        self,
        workflow: Workflow,
        initial_input: dict | None = None
    ) -> dict[str, Any]:
        """执行工作流

        Args:
            workflow: 工作流定义
            initial_input: 初始输入数据

        Returns:
            包含所有步骤结果的字典:
            {
                "workflow": "工作流名称",
                "success": True/False,
                "results": { "tool-name": ToolResult, ... },
                "errors": [ ... ]
            }
        """
        initial_input = initial_input or {}
        results: dict[str, ToolResult] = {}
        errors: list[str] = []

        context = {
            "input": initial_input,
            "results": results,
        }

        for i, step in enumerate(workflow.steps):
            step_label = f"步骤 {i + 1} ({step.tool_name})"

            # 检查工具是否已注册
            if step.tool_name not in self._tools:
                error_msg = f"{step_label}: 工具 '{step.tool_name}' 未注册"
                errors.append(error_msg)
                results[step.tool_name] = ToolResult.fail(error_msg)
                break

            # 检查执行条件
            if step.condition and not self._eval_condition(step.condition, context):
                continue

            # 解析输入映射
            try:
                step_input = self._resolve_input(step.input_mapping, context)
            except Exception as e:
                error_msg = f"{step_label}: 输入映射解析失败 - {e}"
                errors.append(error_msg)
                results[step.tool_name] = ToolResult.fail(error_msg)
                break

            # 验证输入
            tool = self._tools[step.tool_name]
            validation_errors = tool.validate_input(step_input)
            if validation_errors:
                error_msg = f"{step_label}: 输入验证失败 - {'; '.join(validation_errors)}"
                errors.append(error_msg)
                results[step.tool_name] = ToolResult.fail(error_msg)
                break

            # 执行工具
            try:
                result = await tool.execute(step_input)
                results[step.tool_name] = result
                if not result.success:
                    errors.append(f"{step_label}: 执行失败 - {result.error}")
                    break
            except Exception as e:
                error_msg = f"{step_label}: 执行异常 - {e}"
                errors.append(error_msg)
                results[step.tool_name] = ToolResult.fail(error_msg)
                break

        return {
            "workflow": workflow.name,
            "success": len(errors) == 0,
            "results": results,
            "errors": errors,
        }

    def _resolve_input(
        self,
        input_mapping: dict,
        context: dict
    ) -> dict:
        """解析输入映射

        将映射表达式解析为实际值:
        - "$.input.field" -> context["input"]["field"]
        - "$.results.tool_name.field" -> context["results"]["tool_name"].data["field"]
        - 其他值直接作为字面量
        """
        resolved = {}
        for target_field, source in input_mapping.items():
            if isinstance(source, str) and source.startswith("$."):
                resolved[target_field] = self._resolve_path(source, context)
            else:
                resolved[target_field] = source
        return resolved

    def _resolve_path(self, path: str, context: dict) -> Any:
        """解析 JSONPath 风格的路径表达式

        支持格式:
        - $.input.field.subfield
        - $.results.tool-name.field.subfield
        """
        # 去掉 "$." 前缀
        path = path[2:]
        parts = path.split(".")

        if not parts:
            raise ValueError(f"无效的路径表达式: '{path}'")

        root_key = parts[0]
        if root_key == "input":
            value = context["input"]
            remaining = parts[1:]
        elif root_key == "results":
            if len(parts) < 2:
                raise ValueError(f"results 路径需要指定工具名: '{path}'")
            tool_name = parts[1]
            if tool_name not in context["results"]:
                raise ValueError(f"工具 '{tool_name}' 的结果不存在")
            tool_result = context["results"][tool_name]
            if isinstance(tool_result, ToolResult):
                value = tool_result.data
            else:
                value = tool_result
            remaining = parts[2:]
        else:
            raise ValueError(f"未知的根路径: '{root_key}'，支持 'input' 和 'results'")

        # 逐级取值
        for part in remaining:
            if isinstance(value, dict):
                if part not in value:
                    raise KeyError(f"路径 '{path}' 中的键 '{part}' 不存在")
                value = value[part]
            elif isinstance(value, list):
                try:
                    value = value[int(part)]
                except (ValueError, IndexError):
                    raise ValueError(f"路径 '{path}' 中的索引 '{part}' 无效")
            else:
                raise ValueError(f"路径 '{path}' 在 '{part}' 处无法继续解析")

        return value

    def _eval_condition(self, condition: str, context: dict) -> bool:
        """评估条件表达式

        支持简单的比较表达式:
        - "$.results.tool.success == true"
        - "$.input.count > 5"
        - "$.results.tool.data.field == 'value'"
        """
        # 匹配: <path> <operator> <value>
        pattern = r"^\s*(\$\.[\w.\-]+)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$"
        match = re.match(pattern, condition)
        if not match:
            # 无法解析的条件默认不执行
            return False

        path_expr, operator, raw_value = match.groups()

        try:
            left = self._resolve_path(path_expr, context)
        except (ValueError, KeyError):
            return False

        # 解析右值
        right = self._parse_literal(raw_value)

        # 执行比较
        try:
            if operator == "==":
                return left == right
            elif operator == "!=":
                return left != right
            elif operator == ">":
                return left > right
            elif operator == "<":
                return left < right
            elif operator == ">=":
                return left >= right
            elif operator == "<=":
                return left <= right
        except TypeError:
            return False

        return False

    @staticmethod
    def _parse_literal(raw: str) -> Any:
        """解析字面量"""
        raw = raw.strip()
        if raw.lower() == "true":
            return True
        if raw.lower() == "false":
            return False
        if raw.lower() == "null" or raw.lower() == "none":
            return None
        # 尝试解析为数字
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            pass
        # 去除引号作为字符串
        if (raw.startswith("'") and raw.endswith("'")) or \
           (raw.startswith('"') and raw.endswith('"')):
            return raw[1:-1]
        return raw
