"""
DOCX 文件段落提取器
从 .docx 文件中提取正文段落，过滤标题/图题/表题等
"""
import re
from typing import List, Dict, Optional


def extract_from_docx(
    file_path: str,
    min_length: int = 30,
    skip_headings: bool = True,
    skip_captions: bool = True,
    chapter_range: Optional[tuple] = None,
) -> List[Dict]:
    """
    从 DOCX 文件中提取正文段落

    Args:
        file_path: .docx 文件路径
        min_length: 最小段落长度
        skip_headings: 是否跳过章节标题
        skip_captions: 是否跳过图题表题
        chapter_range: 章节范围 (start, end)，如 (1, 7) 只提取第1-7章

    Returns:
        段落列表，每个元素为 dict: {text, chapter, section, index}
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("需要安装 python-docx: pip install python-docx")

    doc = Document(file_path)
    paragraphs = []
    in_body = False
    current_chapter = 0
    current_section = ""
    para_index = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # 检测章标题（如 "第1章 绪论"）
        ch_match = re.match(r'^第(\d+)章\s*(.*)', text)
        if ch_match:
            ch_num = int(ch_match.group(1))
            # 首次进入正文
            if ch_num == 1 and not in_body:
                in_body = True
                current_chapter = 1
                current_section = ch_match.group(2).strip()
                continue
            # 后续章标题
            if in_body:
                current_chapter = ch_num
                current_section = ch_match.group(2).strip()
                if skip_headings:
                    continue

        # 检测结束标记
        if text.startswith('结  论') or text.startswith('结论'):
            in_body = False
            continue
        if text.startswith('参考文献') or text.startswith('致  谢') or text.startswith('致谢'):
            in_body = False
            continue

        if not in_body:
            continue

        # 章节范围过滤
        if chapter_range:
            if current_chapter < chapter_range[0] or current_chapter > chapter_range[1]:
                continue

        # 跳过节标题（如 "1.1 xxx"、"3.2.1 xxx"）
        if skip_headings:
            if re.match(r'^\d+\.\d+', text):
                current_section = text
                continue

        # 跳过图题表题
        if skip_captions:
            if re.match(r'^[图表]\d', text):
                continue

        # 长度过滤
        if len(text) < min_length:
            continue

        para_index += 1
        paragraphs.append({
            'text': text,
            'chapter': current_chapter,
            'section': current_section,
            'index': para_index,
            'source': 'docx',
        })

    return paragraphs
