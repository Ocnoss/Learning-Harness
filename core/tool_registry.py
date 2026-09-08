"""工具注册与发现模块

负责扫描、加载和注册所有可用工具，为 Tool-Agent 提供工具查询服务。
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from core.base_tool import BaseTool, ToolContext, ToolManifest


class ToolRegistry:
    """工具注册表

    管理所有已加载的工具，支持按名称、能力、场景查询。

    使用示例:
        registry = ToolRegistry(context)
        registry.discover_tools("tools/")

        # 获取工具
        tool = registry.get_tool("resource-indexer")

        # 获取所有工具的 LLM 描述
        manifests = registry.get_all_manifests()
    """

    def __init__(self, context: ToolContext):
        self.context = context
        self._tools: dict[str, BaseTool] = {}
        self._manifests: dict[str, ToolManifest] = {}
        self._tool_dirs: list[Path] = []

    def discover_tools(self, tools_dir: str | Path) -> list[str]:
        """扫描目录发现并加载所有工具（支持嵌套目录）

        Args:
            tools_dir: 工具根目录路径

        Returns:
            成功加载的工具名称列表
        """
        tools_path = Path(tools_dir)
        if not tools_path.exists():
            raise FileNotFoundError(f"工具目录不存在: {tools_dir}")

        loaded = []
        for item in tools_path.rglob("*"):
            if not item.is_dir():
                continue
            if item.name.startswith("_") or item.name.startswith("."):
                continue

            # 检查是否为有效工具目录
            plugin_file = item / "plugin.json"
            main_file = item / "main.py"

            if not plugin_file.exists() or not main_file.exists():
                continue

            try:
                tool = self._load_tool(item)
                if tool:
                    self.register(tool)
                    loaded.append(tool.name)
            except Exception as e:
                print(f"[警告] 加载工具失败 {item.name}: {e}", file=sys.stderr)

        return loaded

    def _load_tool(self, tool_dir: Path) -> BaseTool | None:
        """加载单个工具

        Args:
            tool_dir: 工具目录路径

        Returns:
            工具实例，加载失败返回 None
        """
        # 读取 plugin.json
        plugin_file = tool_dir / "plugin.json"
        with open(plugin_file, "r", encoding="utf-8") as f:
            plugin_meta = json.load(f)

        # 动态导入 main.py
        main_file = tool_dir / "main.py"
        spec = importlib.util.spec_from_file_location(
            f"tools.{tool_dir.name}",
            main_file
        )
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)

        # 将工具目录添加到模块搜索路径（支持工具内相对导入）
        tool_dir_str = str(tool_dir)
        if tool_dir_str not in sys.path:
            sys.path.insert(0, tool_dir_str)

        try:
            spec.loader.exec_module(module)
        finally:
            # 恢复路径
            if tool_dir_str in sys.path:
                sys.path.remove(tool_dir_str)

        # 查找 create_tool 工厂函数
        if not hasattr(module, "create_tool"):
            raise ValueError(f"工具 {tool_dir.name} 缺少 create_tool 函数")

        # 创建工具实例
        tool = module.create_tool(self.context)

        # 补充 plugin.json 中的元数据
        if hasattr(tool, "_plugin_meta"):
            tool._plugin_meta = plugin_meta

        return tool

    def register(self, tool: BaseTool):
        """注册工具实例

        Args:
            tool: 工具实例
        """
        self._tools[tool.name] = tool
        self._manifests[tool.name] = tool.get_manifest()

        # 将工具注册到所有其他工具中（支持跨工具调用）
        for other_tool in self._tools.values():
            if other_tool.name != tool.name:
                other_tool.register_tool(tool)
                tool.register_tool(other_tool)

    def get_tool(self, name: str) -> BaseTool | None:
        """按名称获取工具"""
        return self._tools.get(name)

    def get_manifest(self, name: str) -> ToolManifest | None:
        """按名称获取工具清单"""
        return self._manifests.get(name)

    def get_all_tools(self) -> dict[str, BaseTool]:
        """获取所有工具"""
        return dict(self._tools)

    def get_all_manifests(self) -> dict[str, ToolManifest]:
        """获取所有工具清单"""
        return dict(self._manifests)

    def find_tools_by_scenario(self, scenario: str) -> list[ToolManifest]:
        """按适用场景查找工具

        Args:
            scenario: 场景关键词

        Returns:
            匹配的工具清单列表
        """
        scenario_lower = scenario.lower()
        results = []
        for manifest in self._manifests.values():
            # 检查描述和适用场景
            if (scenario_lower in manifest.description_for_llm.lower() or
                any(scenario_lower in s.lower() for s in manifest.applicable_scenarios)):
                results.append(manifest)
        return results

    def get_tools_description_for_llm(self) -> str:
        """生成 LLM 友好的工具列表描述

        Returns:
            格式化的工具描述文本，可直接用于 LLM 提示词
        """
        lines = ["可用工具列表:", ""]

        for name, manifest in sorted(self._manifests.items()):
            lines.append(f"## {name}")
            lines.append(f"描述: {manifest.description_for_llm}")
            lines.append(f"推荐模型: {manifest.recommended_tier} / 思考强度: {manifest.recommended_thinking}")

            if manifest.applicable_scenarios:
                lines.append(f"适用场景: {', '.join(manifest.applicable_scenarios)}")

            if manifest.input_requirements:
                lines.append("输入要求:")
                for key, desc in manifest.input_requirements.items():
                    lines.append(f"  - {key}: {desc}")

            if manifest.output_description:
                lines.append(f"输出: {manifest.output_description}")

            if manifest.limitations:
                lines.append(f"注意事项: {'; '.join(manifest.limitations)}")

            if manifest.examples:
                lines.append("示例:")
                for i, example in enumerate(manifest.examples[:2], 1):  # 最多显示2个示例
                    lines.append(f"  {i}. 输入: {json.dumps(example.get('input', {}), ensure_ascii=False)}")
                    if 'output_summary' in example:
                        lines.append(f"     输出: {example['output_summary']}")

            lines.append("")

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())
