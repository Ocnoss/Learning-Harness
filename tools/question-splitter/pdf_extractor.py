"""PDF 文本提取模块"""

import re
from pathlib import Path
from typing import Any


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """从 PDF 提取文本

    优先使用 pdfplumber（更好的中文支持），
    如果没有安装则使用 pypdf。
    """
    pdf_path = Path(pdf_path)

    # 尝试使用 pdfplumber
    try:
        import pdfplumber
        return _extract_with_pdfplumber(pdf_path)
    except ImportError:
        pass

    # 尝试使用 pypdf
    try:
        import pypdf
        return _extract_with_pypdf(pdf_path)
    except ImportError:
        pass

    raise ImportError(
        "需要安装 PDF 处理库: pip install pdfplumber 或 pip install pypdf"
    )


def _extract_with_pdfplumber(pdf_path: Path) -> str:
    """使用 pdfplumber 提取文本"""
    import pdfplumber

    text_parts = []
    empty_pages = 0
    max_empty_pages = 10  # 最多允许连续 10 页无文本

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                text_parts.append(text)
                empty_pages = 0
            else:
                empty_pages += 1
                # 如果连续多页无文本，可能是图片型 PDF，提前退出
                if empty_pages >= max_empty_pages:
                    print(f"Warning: Page {i+1} onwards appears to be image-based, stopping extraction")
                    break

    return "\n\n".join(text_parts)


def _extract_with_pypdf(pdf_path: Path) -> str:
    """使用 pypdf 提取文本"""
    import pypdf

    text_parts = []
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

    return "\n\n".join(text_parts)


def clean_text(text: str) -> str:
    """清理提取的文本

    - 移除多余空白
    - 统一换行符
    - 移除页眉页脚（常见模式）
    """
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 移除多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 移除页码（常见模式：单独的数字行）
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # 移除页眉页脚（常见模式）
    text = re.sub(r"^.*?第\s*\d+\s*页.*?$", "", text, flags=re.MULTILINE)

    return text.strip()
