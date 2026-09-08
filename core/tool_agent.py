"""Tool-Agent 核心模块

实现 LLM 驱动的工具调用 Agent，支持 ReAct 风格循环：
Thought (思考) → Action (行动) → Observation (观察) → 循环

核心特性：
- LLM 自主决策工具调用顺序
- 支持 auto/yolo/ask-when-needed 交互模式
- 失败恢复与重新规划
- 思考强度 (thinking_effort) 传递
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from core.base_tool import BaseTool, ToolContext, ToolResult
from core.interaction import InteractionHandler, InteractionMode
from core.llm_client import TieredLLMClient, Message
from core.llm_config import ModelTier, ThinkingEffort, LLMConfig
from core.tool_registry import ToolRegistry
from core.event_log import EventLog, EventType


@dataclass
class AgentStep:
    """Agent 执行步骤记录"""
    iteration: int
    thought: str
    action: str
    action_input: dict
    observation: ToolResult | None = None
    is_final: bool = False
    error: str | None = None


@dataclass
class AgentResult:
    """Agent 执行结果"""
    success: bool
    final_answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_iterations: int = 0
    error: str | None = None


class ToolAgent:
    """LLM 驱动的工具调用 Agent

    使用示例:
        config = LLMConfig.from_file("llm_config.json")
        llm = TieredLLMClient(config)
        context = ToolContext(llm_client=llm)

        registry = ToolRegistry(context)
        registry.discover_tools("tools/")

        agent = ToolAgent(
            llm=llm,
            registry=registry,
            interaction_mode=InteractionMode.ASK_WHEN_NEEDED,
        )

        result = await agent.run("帮我整理 D:/学习资料 文件夹")
    """

    # Agent 系统提示词模板
    SYSTEM_PROMPT_TEMPLATE = """你是一个智能工具调用 Agent。你的任务是通过调用可用工具完成用户目标。

{tools_description}

## 执行规则

1. **理解任务**: 分析用户目标，确定需要哪些工具
2. **规划步骤**: 按逻辑顺序规划工具调用
3. **执行观察**: 每次调用后分析结果，决定下一步
4. **灵活调整**: 如果工具失败或结果不符合预期，重新规划
5. **完成确认**: 当任务完成时，给出最终答案

## 输出格式

每次回复必须是 JSON 格式：
```json
{{
    "thought": "你的思考过程，分析当前状态和下一步计划",
    "action": "工具名称 或 'final'",
    "action_input": {{...}},  // 工具输入参数
    "is_final": false,  // 是否为最终答案
    "final_answer": ""  // 仅当 is_final=true 时填写
}}
```

## 注意事项

- 每次只调用一个工具
- 仔细阅读工具描述和输入要求
- 如果工具返回 needs_confirmation=true，向用户确认后再继续
- 如果工具失败，分析原因并尝试替代方案
- 保持思考过程简洁但清晰

## 当前交互模式

{interaction_mode_desc}
"""

    def __init__(
        self,
        llm: TieredLLMClient,
        registry: ToolRegistry,
        interaction_handler: InteractionHandler | None = None,
        max_iterations: int = 20,
        default_tier: ModelTier = ModelTier.BALANCED,
        default_thinking: ThinkingEffort | None = None,  # None 表示由工具决定
        session_id: str | None = None,
        event_log_path: str | Path | None = None,
    ):
        """初始化 Tool-Agent

        Args:
            llm: 多层级 LLM 客户端
            registry: 工具注册表
            interaction_handler: 交互处理器
            max_iterations: 最大迭代次数
            default_tier: 默认模型层级
            default_thinking: 默认思考强度（None 表示由工具决定）
            session_id: 会话ID（用于事件记录和历史追踪）
            event_log_path: 事件日志持久化路径（可选）
        """
        self.llm = llm
        self.registry = registry
        self.interaction = interaction_handler or InteractionHandler()
        self.max_iterations = max_iterations
        self.default_tier = default_tier
        self.default_thinking = default_thinking

        # 会话和事件日志
        self.session_id = session_id or str(uuid4())
        self.event_log = EventLog(session_id=self.session_id, persist_path=event_log_path)

        # 执行状态
        self._steps: list[AgentStep] = []
        self._current_iteration = 0

    async def run(
        self,
        task: str,
        context: dict | None = None,
    ) -> AgentResult:
        """执行任务的 Agent 主循环

        Args:
            task: 用户任务描述
            context: 额外上下文信息

        Returns:
            AgentResult: 执行结果
        """
        self._steps = []
        self._current_iteration = 0

        # 记录 Agent 启动事件
        self.event_log.record(
            EventType.AGENT_START,
            data={"task": task, "context": context},
        )

        # 构建系统提示词
        system_prompt = self._build_system_prompt()

        # 初始化消息历史（仅用于 LLM 调用，不持久化）
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._format_task(task, context)},
        ]

        await self.interaction.notify(
            "Agent 启动",
            f"任务: {task}\n模式: {self.interaction.get_mode_description()}"
        )

        while self._current_iteration < self.max_iterations:
            self._current_iteration += 1

            # 记录思考开始
            think_event = self.event_log.record(
                EventType.THINK_START,
                data={"iteration": self._current_iteration},
                iteration=self._current_iteration,
            )

            # LLM 决策下一步
            try:
                step = await self._think_and_act(messages)
                self.event_log.end_event(think_event.event_id, data={
                    "thought": step.thought[:200],
                    "action": step.action,
                })
            except Exception as e:
                self.event_log.end_event(think_event.event_id, error=str(e))
                self.event_log.record(
                    EventType.AGENT_ERROR,
                    data={"error": str(e), "iteration": self._current_iteration},
                )
                return AgentResult(
                    success=False,
                    final_answer="",
                    steps=self._steps,
                    total_iterations=self._current_iteration,
                    error=f"LLM 决策失败: {e}",
                )

            self._steps.append(step)

            # 记录决策
            self.event_log.record(
                EventType.DECISION_MADE,
                data={
                    "thought": step.thought,
                    "action": step.action,
                    "is_final": step.is_final,
                },
                iteration=self._current_iteration,
            )

            # 输出思考过程
            await self.interaction.notify(
                f"步骤 {self._current_iteration}",
                f"思考: {step.thought[:200]}..."
            )

            # 检查是否为最终答案
            if step.is_final:
                final_answer = step.action_input.get("final_answer", "")
                self.event_log.record(
                    EventType.AGENT_END,
                    data={"success": True, "final_answer": final_answer[:200]},
                )
                await self.interaction.notify("任务完成", final_answer)
                return AgentResult(
                    success=True,
                    final_answer=final_answer,
                    steps=self._steps,
                    total_iterations=self._current_iteration,
                )

            # 执行工具
            tool_event = self.event_log.record(
                EventType.TOOL_CALL_START,
                data={"tool": step.action, "input": step.action_input},
                iteration=self._current_iteration,
            )

            observation = await self._execute_tool(step)
            step.observation = observation

            self.event_log.end_event(tool_event.event_id, data={
                "success": observation.success,
                "summary": observation.summary_for_llm,
            }, error=observation.error)

            # 处理需要确认的情况
            if observation.needs_confirmation:
                self.event_log.record(
                    EventType.USER_CONFIRM_REQUIRED,
                    data={"message": observation.confirmation_message},
                    iteration=self._current_iteration,
                )

                confirmed = await self.interaction.confirm(
                    observation.confirmation_message,
                    default=False,
                    risk_level="normal",
                )

                if confirmed:
                    self.event_log.record(EventType.USER_CONFIRMED, iteration=self._current_iteration)
                else:
                    self.event_log.record(EventType.USER_CANCELLED, iteration=self._current_iteration)
                    # 用户取消，让 LLM 决定如何处理
                    messages.append({
                        "role": "user",
                        "content": f"用户取消了操作。请重新规划或给出替代方案。"
                    })
                    continue

            # 将结果反馈给 LLM
            messages.append({"role": "assistant", "content": json.dumps({
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "is_final": step.is_final,
            }, ensure_ascii=False)})

            messages.append({
                "role": "user",
                "content": self._format_observation(observation),
            })

            # 检查工具执行失败
            if not observation.success:
                await self.interaction.notify(
                    "工具执行失败",
                    f"工具: {step.action}\n错误: {observation.error}",
                    level="warning"
                )

        # 达到最大迭代次数
        self.event_log.record(
            EventType.AGENT_END,
            data={"success": False, "error": "max_iterations_reached"},
        )
        return AgentResult(
            success=False,
            final_answer="",
            steps=self._steps,
            total_iterations=self._current_iteration,
            error=f"达到最大迭代次数 ({self.max_iterations})，任务未完成",
        )

    async def _think_and_act(self, messages: list[dict]) -> AgentStep:
        """LLM 思考并决策下一步行动

        Args:
            messages: 消息历史

        Returns:
            AgentStep: 决策结果
        """
        # 构建调用参数
        call_params = {
            "messages": messages,
            "tier": self.default_tier,
            "temperature": 0.3,  # 较低温度，确保输出稳定
        }

        # 如果用户指定了思考强度，传递给 LLM（转换为字符串）
        if self.default_thinking is not None:
            call_params["thinking_effort"] = self.default_thinking.value

        # 使用 complete_json 获取结构化输出
        response = await self.llm.complete_json(**call_params)

        # 解析响应
        thought = response.get("thought", "")
        action = response.get("action", "")
        action_input = response.get("action_input", {})
        is_final = response.get("is_final", False)

        return AgentStep(
            iteration=self._current_iteration,
            thought=thought,
            action=action,
            action_input=action_input,
            is_final=is_final,
        )

    async def _execute_tool(self, step: AgentStep) -> ToolResult:
        """执行工具调用

        Args:
            step: 执行步骤

        Returns:
            ToolResult: 工具执行结果
        """
        tool = self.registry.get_tool(step.action)
        if tool is None:
            available = ", ".join(self.registry.get_all_tools().keys())
            return ToolResult.fail(
                f"工具 '{step.action}' 不存在。可用工具: {available}",
                summary_for_llm=f"工具不存在，请从可用工具中选择: {available}",
            )

        # 验证输入
        validation_errors = tool.validate_input(step.action_input)
        if validation_errors:
            return ToolResult.fail(
                f"输入验证失败: {'; '.join(validation_errors)}",
                summary_for_llm=f"输入参数错误: {'; '.join(validation_errors)}",
            )

        # 执行工具
        # 注意：工具内部会自行决定 thinking_effort（通过 recommended_thinking 或用户覆盖）
        try:
            result = await tool.execute(step.action_input)
            return result
        except Exception as e:
            return ToolResult.fail(
                f"工具执行异常: {e}",
                summary_for_llm=f"工具执行出错: {e}",
            )

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        tools_description = self.registry.get_tools_description_for_llm()
        interaction_mode_desc = self.interaction.get_mode_description()

        return self.SYSTEM_PROMPT_TEMPLATE.format(
            tools_description=tools_description,
            interaction_mode_desc=interaction_mode_desc,
        )

    def _format_task(self, task: str, context: dict | None) -> str:
        """格式化任务描述"""
        lines = [f"用户任务: {task}"]

        if context:
            lines.append("")
            lines.append("额外上下文:")
            for key, value in context.items():
                lines.append(f"- {key}: {value}")

        return "\n".join(lines)

    def _format_observation(self, result: ToolResult) -> str:
        """格式化工具执行结果（供 LLM 理解）"""
        lines = ["工具执行结果:"]

        if result.success:
            lines.append(f"状态: 成功")
            if result.summary_for_llm:
                lines.append(f"摘要: {result.summary_for_llm}")
            lines.append(f"数据: {json.dumps(result.data, ensure_ascii=False, default=str)}")
        else:
            lines.append(f"状态: 失败")
            lines.append(f"错误: {result.error}")
            if result.summary_for_llm:
                lines.append(f"说明: {result.summary_for_llm}")

        if result.suggested_next_actions:
            lines.append("建议下一步:")
            for action in result.suggested_next_actions:
                lines.append(f"  - {action.get('tool', 'unknown')}: {action.get('reason', '')}")

        if result.needs_confirmation:
            lines.append(f"需要确认: {result.confirmation_message}")

        return "\n".join(lines)

    async def stream_run(
        self,
        task: str,
        context: dict | None = None,
    ) -> AsyncIterator[dict]:
        """流式执行任务（实时输出中间状态）

        Args:
            task: 用户任务描述
            context: 额外上下文信息

        Yields:
            dict: 中间状态更新
        """
        # TODO: 实现流式执行
        raise NotImplementedError("流式执行尚未实现")


# 便捷函数
async def create_agent(
    config_path: str | Path = "llm_config.json",
    tools_dir: str | Path = "tools/",
    interaction_mode: InteractionMode = InteractionMode.ASK_WHEN_NEEDED,
    default_tier: ModelTier = ModelTier.BALANCED,
    default_thinking: ThinkingEffort | None = None,
    **kwargs,
) -> ToolAgent:
    """创建并初始化 Tool-Agent

    Args:
        config_path: LLM 配置文件路径
        tools_dir: 工具目录路径
        interaction_mode: 交互模式
        default_tier: 默认模型层级
        default_thinking: 默认思考强度（None 表示由工具决定）
        **kwargs: 传递给 ToolAgent 的其他参数

    Returns:
        初始化完成的 ToolAgent 实例
    """
    from core.providers.openai_client import OpenAIClient  # 触发注册

    # 加载配置
    config = LLMConfig.from_file(config_path)
    llm = TieredLLMClient(config)

    # 创建上下文
    context = ToolContext(llm_client=llm)

    # 加载工具
    registry = ToolRegistry(context)
    loaded = registry.discover_tools(tools_dir)
    print(f"已加载 {len(loaded)} 个工具: {', '.join(loaded)}")

    # 创建交互处理器
    interaction = InteractionHandler(mode=interaction_mode)

    return ToolAgent(
        llm=llm,
        registry=registry,
        interaction_handler=interaction,
        default_tier=default_tier,
        default_thinking=default_thinking,
        **kwargs,
    )
