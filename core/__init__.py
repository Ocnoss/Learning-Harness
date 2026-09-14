"""Core 基础插件 - LLM 客户端、工具基类、工作流引擎"""

from core.llm_client import (
    Message,
    CompletionResponse,
    LLMClient,
    LLMClientFactory,
    TieredLLMClient,
)
from core.llm_config import (
    ModelTier,
    ThinkingEffort,
    TierModelConfig,
    TierFallbackNotice,
    LLMConfig,
)
from core.base_tool import (
    ToolContext,
    ToolResult,
    ToolManifest,
    BaseTool,
)
from core.tool_registry import ToolRegistry
from core.interaction import (
    InteractionMode,
    InteractionHandler,
    create_auto_handler,
    create_yolo_handler,
    create_ask_handler,
)
from core.tool_agent import ToolAgent, AgentResult, AgentStep, create_agent
from core.event_log import EventLog, Event, EventType
from core.workflow import (
    WorkflowStep,
    Workflow,
    WorkflowEngine,
)
from core.context import (
    LHEvent,
    EventStore,
    InMemoryEventStore,
    TokenEstimator,
    CharBasedEstimator,
    Projection,
    ProjectionItem,
    rebuild,
    fold,
    derive_level,
    ContextQuery,
    ContextItem,
    ContextBundle,
    NegationQuery,
    compile,
    normalize_scope,
    is_within_scope,
    is_descendant_scope,
    adapt_event_log,
)

__all__ = [
    # LLM
    "Message",
    "CompletionResponse",
    "LLMClient",
    "LLMClientFactory",
    "TieredLLMClient",
    # LLM Config
    "ModelTier",
    "ThinkingEffort",
    "TierModelConfig",
    "TierFallbackNotice",
    "LLMConfig",
    # Tool
    "ToolContext",
    "ToolResult",
    "ToolManifest",
    "BaseTool",
    # Tool Registry
    "ToolRegistry",
    # Interaction
    "InteractionMode",
    "InteractionHandler",
    "create_auto_handler",
    "create_yolo_handler",
    "create_ask_handler",
    # Tool Agent
    "ToolAgent",
    "AgentResult",
    "AgentStep",
    "create_agent",
    # Event Log
    "EventLog",
    "Event",
    "EventType",
    # Workflow
    "WorkflowStep",
    "Workflow",
    "WorkflowEngine",
    # Context Compiler (core.context)
    "LHEvent",
    "EventStore",
    "InMemoryEventStore",
    "TokenEstimator",
    "CharBasedEstimator",
    "Projection",
    "ProjectionItem",
    "rebuild",
    "fold",
    "derive_level",
    "ContextQuery",
    "ContextItem",
    "ContextBundle",
    "NegationQuery",
    "compile",
    "normalize_scope",
    "is_within_scope",
    "is_descendant_scope",
    "adapt_event_log",
]

__version__ = "1.0.0"
