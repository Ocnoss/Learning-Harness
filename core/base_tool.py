"""工具基类模块 - 所有学习工具的基础类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm_client import LLMClient, TieredLLMClient
    from core.llm_config import ModelTier, ThinkingEffort


@dataclass
class ToolContext:
    """工具执行上下文

    包含工具运行所需的共享资源：
    - llm_client: LLM 客户端实例（LLMClient 或 TieredLLMClient）
    - config: 全局配置
    - shared_state: 跨工具共享状态（工作流中传递数据）
    """
    llm_client: "LLMClient | TieredLLMClient"
    config: dict = field(default_factory=dict)
    shared_state: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    """工具执行结果

    Attributes:
        success: 是否执行成功
        data: 输出数据（应符合 output_schema）
        error: 错误信息（失败时）
        metadata: 附加元数据
        summary_for_llm: LLM 友好的结果摘要（一句话）
        suggested_next_actions: 建议的下一步操作列表
        confidence: 结果置信度 (0-1)
        needs_confirmation: 是否需要用户确认
        confirmation_message: 确认提示信息
    """
    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)
    summary_for_llm: str = ""
    suggested_next_actions: list[dict] = field(default_factory=list)
    confidence: float = 1.0
    needs_confirmation: bool = False
    confirmation_message: str = ""

    @classmethod
    def ok(
        cls,
        data: Any,
        metadata: dict | None = None,
        summary_for_llm: str = "",
        suggested_next_actions: list[dict] | None = None,
        confidence: float = 1.0,
    ) -> "ToolResult":
        """创建成功结果"""
        return cls(
            success=True,
            data=data,
            metadata=metadata or {},
            summary_for_llm=summary_for_llm,
            suggested_next_actions=suggested_next_actions or [],
            confidence=confidence,
        )

    @classmethod
    def fail(
        cls,
        error: str,
        metadata: dict | None = None,
        summary_for_llm: str = "",
        suggested_next_actions: list[dict] | None = None,
    ) -> "ToolResult":
        """创建失败结果"""
        return cls(
            success=False,
            error=error,
            metadata=metadata or {},
            summary_for_llm=summary_for_llm or f"执行失败: {error}",
            suggested_next_actions=suggested_next_actions or [],
            confidence=0.0,
        )

    @classmethod
    def need_confirmation(
        cls,
        message: str,
        data: Any = None,
        metadata: dict | None = None,
    ) -> "ToolResult":
        """创建需要用户确认的结果"""
        return cls(
            success=True,
            data=data,
            metadata=metadata or {},
            needs_confirmation=True,
            confirmation_message=message,
            summary_for_llm=f"需要用户确认: {message}",
            confidence=0.5,
        )


@dataclass
class ToolManifest:
    """工具清单 - LLM 友好的工具描述

    用于 Tool-Agent 理解工具能力和使用场景。
    """
    name: str
    description_for_llm: str  # LLM 可读的详细描述
    applicable_scenarios: list[str] = field(default_factory=list)  # 适用场景
    input_requirements: dict = field(default_factory=dict)  # 输入要求说明
    output_description: str = ""  # 输出描述
    examples: list[dict] = field(default_factory=list)  # 使用示例
    limitations: list[str] = field(default_factory=list)  # 限制和注意事项
    recommended_tier: str = "balanced"  # 推荐模型层级
    recommended_thinking: str = "medium"  # 推荐思考强度


class BaseTool(ABC):
    """学习工具基类

    所有工具插件必须继承此类并实现 execute 方法。

    子类需要定义的类属性:
        name: 工具唯一标识（kebab-case）
        description: 工具描述
        version: 版本号
        input_schema: 输入 JSON Schema
        output_schema: 输出 JSON Schema
        dependencies: 依赖的其他工具名称列表
        recommended_tier: 推荐模型层级 ("fast"/"balanced"/"flagship")
        recommended_thinking: 推荐思考强度 ("low"/"medium"/"high")
        task_understanding: AI 对任务的理解描述（帮助 Tool-Agent 决策）

    使用示例:
        class MyTool(BaseTool):
            name = "my-tool"
            description = "我的工具"
            recommended_tier = "balanced"
            recommended_thinking = "medium"
            task_understanding = "处理用户输入的文本，生成结构化输出"

            async def execute(self, input_data: dict) -> ToolResult:
                response = await self.llm.complete([
                    {"role": "user", "content": input_data["prompt"]}
                ])
                return ToolResult.ok(
                    data={"text": response.content},
                    summary_for_llm="成功生成文本内容"
                )
    """

    # 工具元数据（子类必须覆盖）
    name: str = ""
    description: str = ""
    version: str = "1.0.0"

    # 输入/输出 Schema（用于验证和工作流编排）
    input_schema: dict = {}
    output_schema: dict = {}

    # 依赖的其他工具名称
    dependencies: list[str] = []

    # 推荐模型层级（子类可覆盖）
    # 可选值: "fast", "balanced", "flagship"
    # 为 None 时使用 balanced
    recommended_tier: str | None = None

    # 推荐思考强度（子类可覆盖）
    # 可选值: "low", "medium", "high"
    # 为 None 时由调用方（用户或 Agent）决定
    # 注意：这只是推荐值，实际调用时可被覆盖
    recommended_thinking: str | None = None

    # AI 对任务的理解描述（子类应覆盖）
    # 用于 Tool-Agent 理解工具的核心功能和适用场景
    task_understanding: str = ""

    def __init__(self, context: ToolContext):
        self.context = context
        self.llm = context.llm_client
        self._tool_registry: dict[str, "BaseTool"] = {}

    def _resolve_tier(self) -> "ModelTier | None":
        """将 recommended_tier 字符串解析为 ModelTier 枚举

        Returns:
            ModelTier 枚举值，或 None（使用默认层级）
        """
        if self.recommended_tier is None:
            return None

        from core.llm_config import ModelTier

        tier_map = {
            "fast": ModelTier.FAST,
            "balanced": ModelTier.BALANCED,
            "flagship": ModelTier.FLAGSHIP,
        }
        tier = tier_map.get(self.recommended_tier.lower())
        if tier is None:
            valid = ", ".join(tier_map.keys())
            raise ValueError(
                f"工具 '{self.name}' 的 recommended_tier "
                f"'{self.recommended_tier}' 无效。可选值: {valid}"
            )
        return tier

    def _resolve_thinking(self) -> "ThinkingEffort | None":
        """将 recommended_thinking 字符串解析为 ThinkingEffort 枚举

        Returns:
            ThinkingEffort 枚举值，或 None（由调用方决定）
        """
        if self.recommended_thinking is None:
            return None

        from core.llm_config import ThinkingEffort

        effort_map = {
            "low": ThinkingEffort.LOW,
            "medium": ThinkingEffort.MEDIUM,
            "high": ThinkingEffort.HIGH,
        }
        effort = effort_map.get(self.recommended_thinking.lower())
        if effort is None:
            valid = ", ".join(effort_map.keys())
            raise ValueError(
                f"工具 '{self.name}' 的 recommended_thinking "
                f"'{self.recommended_thinking}' 无效。可选值: {valid}"
            )
        return effort

    def get_llm_call_params(self) -> dict:
        """获取 LLM 调用参数（供 Tool-Agent 参考）

        Returns:
            包含推荐层级和思考强度的字典
        """
        return {
            "recommended_tier": self.recommended_tier,
            "recommended_thinking": self.recommended_thinking,
        }

    def get_manifest(self) -> ToolManifest:
        """获取工具清单（LLM 友好描述）

        子类可覆盖此方法提供更详细的描述。
        """
        return ToolManifest(
            name=self.name,
            description_for_llm=self.description,
            recommended_tier=self.recommended_tier or "balanced",
            recommended_thinking=self.recommended_thinking or "medium",
        )

    @abstractmethod
    async def execute(self, input_data: dict) -> ToolResult:
        """执行工具逻辑

        Args:
            input_data: 输入数据（应符合 input_schema）

        Returns:
            ToolResult: 执行结果
        """
        pass

    def validate_input(self, input_data: dict) -> list[str]:
        """验证输入数据是否符合 input_schema

        基于 JSON Schema 的简单验证，检查必填字段和类型。

        Args:
            input_data: 待验证的输入数据

        Returns:
            错误信息列表，空列表表示验证通过
        """
        errors: list[str] = []
        schema = self.input_schema

        if not schema:
            return errors

        # 检查必填字段
        required = schema.get("required", [])
        for field_name in required:
            if field_name not in input_data:
                errors.append(f"缺少必填字段: '{field_name}'")

        # 检查字段类型
        properties = schema.get("properties", {})
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        for field_name, field_schema in properties.items():
            if field_name not in input_data:
                continue
            expected_type = field_schema.get("type")
            if expected_type and expected_type in type_map:
                value = input_data[field_name]
                py_type = type_map[expected_type]
                # bool 是 int 的子类，需要特殊处理
                if expected_type == "integer" and isinstance(value, bool):
                    errors.append(f"字段 '{field_name}' 类型错误: 期望 integer，实际 boolean")
                elif not isinstance(value, py_type):
                    errors.append(
                        f"字段 '{field_name}' 类型错误: "
                        f"期望 {expected_type}，实际 {type(value).__name__}"
                    )

        return errors

    def register_tool(self, tool: "BaseTool"):
        """注册可调用的工具（由外部注入）"""
        self._tool_registry[tool.name] = tool

    async def call_tool(
        self,
        tool_name: str,
        input_data: dict
    ) -> ToolResult:
        """调用其他工具

        Args:
            tool_name: 目标工具名称
            input_data: 传递给目标工具的输入

        Returns:
            ToolResult: 目标工具的执行结果

        Raises:
            ValueError: 工具未注册
        """
        if tool_name not in self._tool_registry:
            available = ", ".join(self._tool_registry.keys()) or "(无)"
            raise ValueError(
                f"工具 '{tool_name}' 未注册。可用工具: {available}"
            )
        tool = self._tool_registry[tool_name]
        return await tool.execute(input_data)

    def get_info(self) -> dict:
        """获取工具元数据"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "dependencies": self.dependencies,
            "recommended_tier": self.recommended_tier,
            "recommended_thinking": self.recommended_thinking,
            "task_understanding": self.task_understanding,
        }
