"""
纯文本文件段落提取器
"""
import re
from typing import List, Dict


def extract_from_txt(
    file_path: str,
    min_length: int = 30,
    delimiter: str = "auto",
) -> List[Dict]:
    """
    从纯文本文件中提取段落

    Args:
        file_path: .txt 文件路径
        min_length: 最小段落长度
        delimiter: 段落分隔方式。"auto" = 自动检测（空行/换行），"newline" = 按换行分

    Returns:
        段落列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if delimiter == "auto":
        # 尝试按双换行分段
        raw_paragraphs = re.split(r'\n\s*\n', content)
        if len(raw_paragraphs) <= 1:
            # 退化为按单换行分段
            raw_paragraphs = content.split('\n')
    else:
        raw_paragraphs = content.split('\n')

    paragraphs = []
    for i, para in enumerate(raw_paragraphs):
        text = para.strip()
        if not text or len(text) < min_length:
            continue
        paragraphs.append({
            'text': text,
            'chapter': 0,
            'section': '',
            'index': i + 1,
            'source': 'txt',
        })

    return paragraphs
