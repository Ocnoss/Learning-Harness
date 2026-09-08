"""内容采样模块 - 从文件中提取样本内容"""

import re
from pathlib import Path
from typing import Any


def extract_sample(file_path: Path, sample_pages: int = 3) -> str:
    """从文件中提取样本内容

    Args:
        file_path: 文件路径
        sample_pages: PDF 采样页数

    Returns:
        样本文本
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf_sample(file_path, sample_pages)
    elif suffix in [".txt", ".md"]:
        return _extract_text_sample(file_path)
    elif suffix in [".doc", ".docx"]:
        return _extract_docx_sample(file_path)
    elif suffix in [".mp4", ".avi", ".mkv", ".mov"]:
        return _extract_video_metadata(file_path)
    else:
        return ""


def _extract_pdf_sample(file_path: Path, sample_pages: int) -> str:
    """提取 PDF 前几页文本"""
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= sample_pages:
                    break
                text = page.extract_text()
                if text:
                    text_parts.append(text)

        return "\n\n".join(text_parts)

    except ImportError:
        # 尝试 pypdf
        try:
            import pypdf

            text_parts = []
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    if i >= sample_pages:
                        break
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)

            return "\n\n".join(text_parts)

        except ImportError:
            return ""


def _extract_text_sample(file_path: Path, max_chars: int = 3000) -> str:
    """提取文本文件样本"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def _extract_docx_sample(file_path: Path) -> str:
    """提取 DOCX 文件样本"""
    try:
        from docx import Document

        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs[:50]]  # 前 50 段
        return "\n".join(paragraphs)

    except ImportError:
        return ""
    except Exception:
        return ""


def _extract_video_metadata(file_path: Path) -> str:
    """提取视频文件元数据（简化版）"""
    # 仅返回文件名和大小信息
    stat = file_path.stat()
    size_mb = stat.st_size / (1024 * 1024)

    return f"视频文件: {file_path.name}\n大小: {size_mb:.1f} MB"


def clean_sample(text: str) -> str:
    """清理样本文本"""
    # 移除多余空白
    text = re.sub(r"\s+", " ", text)
    # 移除特殊字符
    text = re.sub(r"[^\w\s\u4e00-\u9fff.,;:!?()（）【】《》\"\"'']", "", text)
    return text.strip()
