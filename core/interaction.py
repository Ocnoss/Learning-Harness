"""交互模式处理模块

实现 auto / yolo / ask-when-needed 三种用户交互模式。
参考主流 AI CLI 的设计，不重复造轮子。
"""

from enum import Enum
from typing import Callable, Awaitable


class InteractionMode(Enum):
    """交互模式"""
    AUTO = "auto"                    # 自动执行，不询问用户
    YOLO = "yolo"                    # 自动执行所有操作，包括高风险
    ASK_WHEN_NEEDED = "ask"          # 仅在需要时询问用户


class InteractionHandler:
    """交互处理器

    根据当前模式决定是否询问用户、如何展示信息。

    使用示例:
        handler = InteractionHandler(mode=InteractionMode.ASK_WHEN_NEEDED)

        # 需要确认时
        if result.needs_confirmation:
            approved = await handler.confirm(
                result.confirmation_message,
                default=False
            )
            if not approved:
                return "用户取消操作"
    """

    def __init__(
        self,
        mode: InteractionMode = InteractionMode.ASK_WHEN_NEEDED,
        confirm_callback: Callable[[str, bool], Awaitable[bool]] | None = None,
        notify_callback: Callable[[str, str], Awaitable[None]] | None = None,
    ):
        """初始化交互处理器

        Args:
            mode: 交互模式
            confirm_callback: 确认回调函数 (message, default) -> bool
            notify_callback: 通知回调函数 (title, message) -> None
        """
        self.mode = mode
        self._confirm_callback = confirm_callback
        self._notify_callback = notify_callback

    async def confirm(
        self,
        message: str,
        default: bool = False,
        risk_level: str = "normal",  # "normal" | "high" | "critical"
    ) -> bool:
        """请求用户确认

        Args:
            message: 确认提示信息
            default: 默认值（用户直接回车时）
            risk_level: 风险等级

        Returns:
            用户是否确认
        """
        # YOLO 模式：自动确认所有操作
        if self.mode == InteractionMode.YOLO:
            await self._notify("自动确认", f"[YOLO] {message}")
            return True

        # AUTO 模式：自动确认普通操作，高风险操作仍需确认
        if self.mode == InteractionMode.AUTO:
            if risk_level == "normal":
                await self._notify("自动确认", f"[AUTO] {message}")
                return True
            # 高风险操作继续向下执行，需要确认

        # ASK_WHEN_NEEDED 或 AUTO 高风险：调用回调或默认行为
        if self._confirm_callback:
            return await self._confirm_callback(message, default)

        # 默认行为：打印提示并返回默认值
        risk_marker = {
            "normal": "[?]",
            "high": "[!]",
            "critical": "[!!!]",
        }.get(risk_level, "[?]")

        print(f"\n{risk_marker} 需要确认: {message}")
        print(f"  默认: {'是' if default else '否'}")
        print("  (设置 confirm_callback 以支持交互式确认)")

        return default

    async def notify(self, title: str, message: str, level: str = "info"):
        """发送通知（无需确认）

        Args:
            title: 通知标题
            message: 通知内容
            level: 级别 ("info" | "warning" | "error")
        """
        await self._notify(title, message, level)

    async def _notify(self, title: str, message: str, level: str = "info"):
        """内部通知方法"""
        if self._notify_callback:
            await self._notify_callback(title, message)
        else:
            prefix = {
                "info": "[i]",
                "warning": "[!]",
                "error": "[x]",
            }.get(level, "[i]")
            print(f"{prefix} {title}: {message}")

    def should_auto_proceed(self, risk_level: str = "normal") -> bool:
        """判断是否应该自动继续（不询问用户）

        Args:
            risk_level: 风险等级

        Returns:
            是否自动继续
        """
        if self.mode == InteractionMode.YOLO:
            return True
        if self.mode == InteractionMode.AUTO:
            return risk_level == "normal"
        return False

    def get_mode_description(self) -> str:
        """获取当前模式描述"""
        return {
            InteractionMode.AUTO: "自动模式：普通操作自动执行，高风险操作需确认",
            InteractionMode.YOLO: "YOLO 模式：所有操作自动执行，无需确认",
            InteractionMode.ASK_WHEN_NEEDED: "询问模式：需要时询问用户",
        }[self.mode]


# 预定义的交互处理器实例
def create_auto_handler() -> InteractionHandler:
    """创建自动模式处理器"""
    return InteractionHandler(mode=InteractionMode.AUTO)


def create_yolo_handler() -> InteractionHandler:
    """创建 YOLO 模式处理器"""
    return InteractionHandler(mode=InteractionMode.YOLO)


def create_ask_handler(
    confirm_callback: Callable[[str, bool], Awaitable[bool]] | None = None
) -> InteractionHandler:
    """创建询问模式处理器"""
    return InteractionHandler(
        mode=InteractionMode.ASK_WHEN_NEEDED,
        confirm_callback=confirm_callback,
    )
