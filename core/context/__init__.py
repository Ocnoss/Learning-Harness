"""core.context —— LH 内核上下文编译子包（契约 §2.2/§2.3/§6.1/§6.2）。

公共符号一览:
    LHEvent                                §2.2 事件信封（frozen dataclass）
    EventStore / InMemoryEventStore        append-only 存储协议与内存实现
    TokenEstimator / CharBasedEstimator    确定性 token 估算
    Projection / ProjectionItem            派生快照；rebuild / fold 两条等价路径
    ContextQuery / NegationQuery           编译请求（含否定式查询）
    ContextItem / ContextBundle            编译产物
    compile                                确定性纯函数编译器
    adapt_event_log                        既有 EventLog → LHEvent 只读桥接

本子包不 import 任何既有 core 运行时模块（event_log 仅在适配器函数体内
延迟导入），既有代码也不 import 本子包——零回归、可独立验证。
"""

from core.context.envelope import LHEvent
from core.context.store import EventStore, InMemoryEventStore
from core.context.tokens import TokenEstimator, CharBasedEstimator
from core.context.projection import (
    Projection,
    ProjectionItem,
    rebuild,
    fold,
    derive_level,
)
from core.context.compiler import (
    ContextQuery,
    ContextItem,
    ContextBundle,
    NegationQuery,
    compile,
    normalize_scope,
    is_within_scope,
    is_descendant_scope,
)
from core.context.event_log_adapter import adapt_event_log

__all__ = [
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
