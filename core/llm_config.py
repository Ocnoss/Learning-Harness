"""LLM 多层级模型配置模块

提供 fast / balanced / flagship 三层级模型配置管理。
当推荐层级未配置时，不报错而是生成降级提示，由调用层决定是否继续。

思考强度 (thinking_effort) 由调用层（工具或用户）动态指定，不在配置层固定。
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ModelTier(Enum):
    """模型层级"""
    FAST = "fast"           # 快速模型：简单任务、低成本
    BALANCED = "balanced"   # 均衡模型：中等复杂度任务
    FLAGSHIP = "flagship"   # 旗舰模型：复杂推理、高质量输出


class ThinkingEffort(Enum):
    """思考强度"""
    LOW = "low"         # 低思考：快速响应，简单推理
    MEDIUM = "medium"   # 中思考：平衡速度与质量
    HIGH = "high"       # 高思考：深度推理，复杂分析


# 层级优先级（数值越大越强）
TIER_ORDER = {
    ModelTier.FAST: 1,
    ModelTier.BALANCED: 2,
    ModelTier.FLAGSHIP: 3,
}

TIER_LABELS = {
    ModelTier.FAST: "快速模型",
    ModelTier.BALANCED: "均衡模型",
    ModelTier.FLAGSHIP: "旗舰模型",
}

THINKING_LABELS = {
    ThinkingEffort.LOW: "低思考",
    ThinkingEffort.MEDIUM: "中思考",
    ThinkingEffort.HIGH: "高思考",
}


@dataclass
class TierModelConfig:
    """单个层级的模型配置

    只包含连接信息（provider/model/api_key/base_url），
    不包含 temperature/max_tokens/thinking_effort 等调用参数——这些由工具层按需传入。

    Attributes:
        provider: LLM 提供商标识（如 "openai"、"anthropic"）
        model: 模型名称（如 "gpt-4o-mini"、"gpt-4o"）
        api_key: API Key（可选，为空则从环境变量读取）
        base_url: 自定义 API 地址（可选）
        extra: 额外配置项（如 organization、timeout 等连接级参数）
    """
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    extra: dict = field(default_factory=dict)

    def is_configured(self) -> bool:
        """该层级是否已配置（provider 和 model 非空）"""
        return bool(self.provider and self.model)

    def to_client_config(self) -> dict:
        """转换为 LLMClientFactory.create() 所需的配置字典

        只包含连接信息，不包含 temperature/max_tokens/thinking_effort 等调用参数。
        """
        config: dict[str, Any] = {
            "api_key": self.api_key,
            "model": self.model,
        }
        if self.base_url:
            config["base_url"] = self.base_url
        config.update(self.extra)
        return config


@dataclass
class TierFallbackNotice:
    """层级降级提示

    当推荐层级未配置时生成，包含降级信息供调用层决策。

    Attributes:
        recommended_tier: 推荐的模型层级
        actual_tier: 实际可用的模型层级
        recommended_label: 推荐层级的中文标签
        actual_label: 实际层级的中文标签
        message: 人类可读的提示信息
        severity: 提示级别（"warning"=降级到更弱模型, "notice"=升级到更强模型）
    """
    recommended_tier: ModelTier
    actual_tier: ModelTier
    recommended_label: str
    actual_label: str
    message: str
    severity: str  # "warning" | "notice"


@dataclass
class LLMConfig:
    """LLM 多层级配置

    管理 fast / balanced / flagship 三个层级的模型配置，
    提供层级解析和降级提示生成。

    使用示例:
        config = LLMConfig.from_file("llm_config.json")

        # 解析层级，可能返回降级提示
        tier_config, notice = config.resolve_tier(ModelTier.FLAGSHIP)
        if notice:
            print(notice.message)  # "当前任务推荐调用旗舰模型..."

        # 使用解析后的配置创建客户端
        client = LLMClientFactory.create(tier_config.provider, tier_config.to_client_config())
    """

    fast: TierModelConfig = field(default_factory=TierModelConfig)
    balanced: TierModelConfig = field(default_factory=TierModelConfig)
    flagship: TierModelConfig = field(default_factory=TierModelConfig)

    def get_tier_config(self, tier: ModelTier) -> TierModelConfig:
        """获取指定层级的配置"""
        mapping = {
            ModelTier.FAST: self.fast,
            ModelTier.BALANCED: self.balanced,
            ModelTier.FLAGSHIP: self.flagship,
        }
        return mapping[tier]

    def resolve_tier(
        self,
        recommended: ModelTier,
    ) -> tuple[TierModelConfig, TierFallbackNotice | None]:
        """解析推荐层级，返回实际可用的配置和降级提示

        解析策略:
        1. 如果推荐层级已配置，直接返回，无提示
        2. 如果推荐层级未配置，向上查找更强的已配置层级
        3. 如果没有更强的，向下查找更弱的已配置层级
        4. 如果所有层级都未配置，抛出 ValueError

        Args:
            recommended: 推荐的模型层级

        Returns:
            (TierModelConfig, TierFallbackNotice | None)
            - 推荐层级已配置: (推荐层级配置, None)
            - 降级到更弱模型: (实际配置, warning 提示)
            - 升级到更强模型: (实际配置, notice 提示)

        Raises:
            ValueError: 所有层级均未配置
        """
        recommended_config = self.get_tier_config(recommended)

        # 推荐层级已配置，直接使用
        if recommended_config.is_configured():
            return recommended_config, None

        # 收集所有已配置的层级
        configured: list[ModelTier] = []
        for tier in ModelTier:
            if self.get_tier_config(tier).is_configured():
                configured.append(tier)

        if not configured:
            raise ValueError(
                "所有模型层级均未配置。"
                "请在 llm_config.json 中至少配置一个层级的 provider 和 model。"
            )

        recommended_order = TIER_ORDER[recommended]

        # 优先向上查找更强的模型
        stronger = [t for t in configured if TIER_ORDER[t] > recommended_order]
        if stronger:
            # 取最接近推荐层级的（最弱的强模型）
            actual = min(stronger, key=lambda t: TIER_ORDER[t])
            actual_config = self.get_tier_config(actual)
            notice = TierFallbackNotice(
                recommended_tier=recommended,
                actual_tier=actual,
                recommended_label=TIER_LABELS[recommended],
                actual_label=TIER_LABELS[actual],
                message=(
                    f"当前任务推荐调用{TIER_LABELS[recommended]}，"
                    f"该层级未配置。"
                    f"将使用{TIER_LABELS[actual]}（{actual_config.model}）执行，"
                    f"可能造成不必要的消耗。"
                ),
                severity="notice",
            )
            return actual_config, notice

        # 向下查找更弱的模型
        weaker = [t for t in configured if TIER_ORDER[t] < recommended_order]
        if weaker:
            # 取最接近推荐层级的（最强的弱模型）
            actual = max(weaker, key=lambda t: TIER_ORDER[t])
            actual_config = self.get_tier_config(actual)
            notice = TierFallbackNotice(
                recommended_tier=recommended,
                actual_tier=actual,
                recommended_label=TIER_LABELS[recommended],
                actual_label=TIER_LABELS[actual],
                message=(
                    f"当前任务推荐调用{TIER_LABELS[recommended]}，"
                    f"该层级未配置。"
                    f"将使用{TIER_LABELS[actual]}（{actual_config.model}）执行，"
                    f"可能造成执行效果不佳。"
                ),
                severity="warning",
            )
            return actual_config, notice

        # 理论上不会到达这里
        raise ValueError("层级解析失败")

    def get_configured_tiers(self) -> list[ModelTier]:
        """获取所有已配置的层级"""
        return [
            tier for tier in ModelTier
            if self.get_tier_config(tier).is_configured()
        ]

    def is_tier_configured(self, tier: ModelTier) -> bool:
        """检查指定层级是否已配置"""
        return self.get_tier_config(tier).is_configured()

    # ---- 序列化与加载 ----

    def to_dict(self) -> dict:
        """序列化为字典"""
        def tier_to_dict(tc: TierModelConfig) -> dict:
            d: dict[str, Any] = {}
            if tc.provider:
                d["provider"] = tc.provider
            if tc.model:
                d["model"] = tc.model
            if tc.api_key:
                d["api_key"] = tc.api_key
            if tc.base_url:
                d["base_url"] = tc.base_url
            if tc.extra:
                d["extra"] = tc.extra
            return d

        return {
            "fast": tier_to_dict(self.fast),
            "balanced": tier_to_dict(self.balanced),
            "flagship": tier_to_dict(self.flagship),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LLMConfig":
        """从字典加载"""
        def parse_tier(tier_data: dict | None) -> TierModelConfig:
            if not tier_data:
                return TierModelConfig()
            return TierModelConfig(
                provider=tier_data.get("provider", ""),
                model=tier_data.get("model", ""),
                api_key=tier_data.get("api_key", ""),
                base_url=tier_data.get("base_url", ""),
                extra=tier_data.get("extra", {}),
            )

        return cls(
            fast=parse_tier(data.get("fast")),
            balanced=parse_tier(data.get("balanced")),
            flagship=parse_tier(data.get("flagship")),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "LLMConfig":
        """从 JSON 文件加载配置

        Args:
            path: 配置文件路径

        Returns:
            LLMConfig 实例

        Raises:
            FileNotFoundError: 配置文件不存在
            json.JSONDecodeError: 配置文件格式错误
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LLM 配置文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def save(self, path: str | Path):
        """保存配置到 JSON 文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
