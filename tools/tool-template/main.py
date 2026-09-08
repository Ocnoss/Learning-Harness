"""工具模板 - 复制此文件并修改以实现新工具

使用步骤:
1. 复制整个 tool-template 目录，重命名为你的工具名（kebab-case）
2. 修改 plugin.json 中的 name、description、input_schema、output_schema
3. 修改本文件中的类名、name、description 和 execute 逻辑
"""

import sys
from pathlib import Path

# 确保能导入 core 模块（开发时使用）
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.base_tool import BaseTool, ToolContext, ToolResult


class TemplateTool(BaseTool):
    """工具模板类 - 请重命名为你的工具类名"""

    # ---- 必须修改的元数据 ----
    name = "tool-template"           # 工具唯一标识（kebab-case）
    description = "工具模板"          # 简短描述工具功能
    version = "1.0.0"

    # ---- 输入/输出 Schema ----
    input_schema = {
        "type": "object",
        "properties": {
            "input_text": {
                "type": "string",
                "description": "输入文本"
            }
        },
        "required": ["input_text"]
    }

    output_schema = {
        "type": "object",
        "properties": {
            "output_text": {
                "type": "string",
                "description": "输出文本"
            }
        }
    }

    # ---- 依赖的其他工具 ----
    dependencies: list[str] = []

    # ---- 推荐模型层级 ----
    # 声明本工具推荐的 LLM 模型层级，由 TieredLLMClient 自动路由到对应模型
    #
    # 可选值:
    #   "fast"     - 快速模型：简单任务、低成本、高速度
    #                适用场景：文件索引、格式转换、简单分类、关键词提取
    #   "balanced" - 均衡模型：中等复杂度任务（默认）
    #                适用场景：内容摘要、问答生成、翻译、代码解释
    #   "flagship" - 旗舰模型：复杂推理、高质量输出
    #                适用场景：深度分析、创意写作、复杂推理、多步规划
    #
    # 为 None 时使用 balanced
    #
    # 注意：当推荐层级未配置时，TieredLLMClient 会自动降级/升级并提示
    recommended_tier = "balanced"

    async def execute(self, input_data: dict) -> ToolResult:
        """执行工具逻辑

        在这里实现你的工具核心逻辑：
        1. 从 input_data 获取输入参数
        2. 构建 LLM 提示词
        3. 调用 self.llm.complete() 或 self.llm.complete_json()
           - tier: 由 _resolve_tier() 从 recommended_tier 解析
           - temperature: 控制输出随机性（0.0-2.0）
             * 低温度 (0.1-0.3): 需要准确、稳定的输出（如文件路径、结构化数据）
             * 中温度 (0.5-0.7): 平衡创造性和准确性（如内容摘要、问答）
             * 高温度 (0.8-1.5): 需要创造性、多样性（如创意写作、头脑风暴）
           - max_tokens: 控制输出长度
             * 短输出 (100-500): 判断、分类、简单回答
             * 中输出 (500-2048): 摘要、解释、问答
             * 长输出 (2048-8192): 完整文档、深度分析
        4. 处理结果并返回 ToolResult
        """
        input_text = input_data["input_text"]

        try:
            # 示例：调用 LLM
            # tier: 由 _resolve_tier() 从 recommended_tier 解析
            # temperature / max_tokens: 由工具层根据具体任务需求决定
            response = await self.llm.complete(
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个学习助手。"
                    },
                    {
                        "role": "user",
                        "content": f"请处理以下内容：{input_text}"
                    }
                ],
                # 使用 TieredLLMClient 时传入 tier：
                tier=self._resolve_tier(),
                # 调用参数由工具层按需指定（示例值）：
                temperature=0.7,   # 中等温度，平衡创造性和准确性
                max_tokens=2048,   # 中等长度输出
            )

            # 检查是否有层级降级提示
            result_metadata = {"model": response.model}
            if hasattr(response, 'tier_notice') and response.tier_notice:
                result_metadata["tier_notice"] = response.tier_notice.message

            return ToolResult.ok(
                data={"output_text": response.content},
                metadata=result_metadata
            )

        except Exception as e:
            return ToolResult.fail(f"执行失败: {e}")

    # ---- 可选：调用其他工具 ----
    # async def execute(self, input_data: dict) -> ToolResult:
    #     # 调用依赖的工具
    #     other_result = await self.call_tool("other-tool", {"key": "value"})
    #     if not other_result.success:
    #         return ToolResult.fail(f"依赖工具执行失败: {other_result.error}")
    #
    #     # 使用其他工具的结果继续处理
    #     ...


# ---- 工具入口点（必须） ----
def create_tool(context: ToolContext) -> TemplateTool:
    """创建工具实例

    每个工具模块必须暴露此函数，供插件加载器调用。
    """
    return TemplateTool(context)
