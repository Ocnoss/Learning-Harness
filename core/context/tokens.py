"""Token 估算 —— 可注入的确定性估算器（契约 §6.1 预算分配的度量基础）。

MVP 默认实现 CharBasedEstimator 零依赖、纯确定性：同一输入永远同一输出，
不查网络、不加载词表。预算裁剪只需要“稳定的相对度量”，不需要与真实
tokenizer 逐 token 对齐（tiktoken 后端明确推迟，见计划“推迟项”）。

估算规则（按字符分类，分别向上取整后相加）:
    - CJK 字符（含中文标点）: 每 1.6 字符 ≈ 1 token
    - 其他字符（ASCII / emoji 等）: 每 4 字符 ≈ 1 token
准确度说明：对 CJK/ASCII 混合文本向上取整偏保守（宁多算不少算）；但 emoji 等
落入“其他字符”桶、真实 tokenizer 却可能拆成多 token 的输入会被低估，故整体
是稳定的相对度量而非严格上界（对 emoji 实为下界）。
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenEstimator(Protocol):
    """Token 估算器协议：count(text) -> int，必须确定性（同输入同输出）。"""

    def count(self, text: str) -> int:
        """返回 text 的估算 token 数（非负整数）。"""
        ...


def is_cjk(ch: str) -> bool:
    """判断单个字符是否按 CJK 计费（汉字 / 假名 / 谚文 / CJK 标点）。"""
    o = ord(ch)
    return (
        0x3000 <= o <= 0x303F      # CJK 符号与标点
        or 0x3040 <= o <= 0x30FF   # 假名
        or 0x3400 <= o <= 0x4DBF   # 扩展 A
        or 0x4E00 <= o <= 0x9FFF   # 基本汉字
        or 0xAC00 <= o <= 0xD7AF   # 谚文
        or 0xF900 <= o <= 0xFAFF   # 兼容汉字
        or 0xFF00 <= o <= 0xFFEF   # 全角字符
        or 0x20000 <= o <= 0x2FA1F  # 扩展 B–F
    )


class CharBasedEstimator:
    """基于字符分类的确定性估算器（默认实现，零依赖）。

    CJK≈len/1.6，ASCII≈len/4，分别向上取整后相加。对 CJK/ASCII 混合文本偏
    保守（宁多算）；但 emoji 等归入“其他字符”桶、真实可能占多 token 的输入
    会被低估，故为稳定的相对度量而非严格上界。绝不调用时钟/随机/IO，满足
    编译器确定性纪律。
    """

    def __init__(self, cjk_chars_per_token: float = 1.6, ascii_chars_per_token: float = 4.0):
        self.cjk_chars_per_token = cjk_chars_per_token
        self.ascii_chars_per_token = ascii_chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        cjk = 0
        other = 0
        for ch in text:
            if is_cjk(ch):
                cjk += 1
            else:
                other += 1
        total = math.ceil(cjk / self.cjk_chars_per_token) + math.ceil(
            other / self.ascii_chars_per_token
        )
        return total
