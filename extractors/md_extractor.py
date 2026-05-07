"""
Markdown 文件段落提取器
"""
import os
import re
import glob
from typing import List, Dict, Optional


def extract_from_md(
    file_path: str,
    min_length: int = 30,
    skip_headings: bool = True,
    skip_captions: bool = True,
    skip_code_blocks: bool = True,
) -> List[Dict]:
    """从单个 Markdown 文件提取段落"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return _parse_markdown(content, file_path, min_length, skip_headings, skip_captions, skip_code_blocks)


def extract_from_md_dir(
    dir_path: str,
    min_length: int = 30,
    skip_headings: bool = True,
    skip_captions: bool = True,
    skip_code_blocks: bool = True,
) -> List[Dict]:
    """从目录中所有 .md 文件提取段落（按文件名排序）"""
    files = sorted(glob.glob(os.path.join(dir_path, '*.md')))
    all_paragraphs = []
    for f in files:
        basename = os.path.basename(f)
        # 跳过非章节文件
        if not (basename.startswith('第') or basename.startswith('chapter')):
            continue
        paras = extract_from_md(f, min_length, skip_headings, skip_captions, skip_code_blocks)
        all_paragraphs.extend(paras)
    return all_paragraphs


def _parse_markdown(
    content: str,
    source: str,
    min_length: int,
    skip_headings: bool,
    skip_captions: bool,
    skip_code_blocks: bool,
) -> List[Dict]:
    """解析 Markdown 内容，提取正文段落"""
    paragraphs = []
    current_chapter = 0
    current_section = ""
    para_index = 0
    in_code_block = False

    for line in content.split('\n'):
        stripped = line.strip()

        # 代码块开关
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # 空行跳过
        if not stripped or len(stripped) < 3:
            continue

        # 章标题
        if stripped.startswith('# '):
            ch_match = re.match(r'^#+\s*第(\d+)章', stripped)
            if ch_match:
                current_chapter = int(ch_match.group(1))
            elif stripped.startswith('## '):
                pass  # 节标题
            if skip_headings:
                continue

        # 节/小节标题
        if re.match(r'^#{2,4}\s+\d+', stripped):
            current_section = re.sub(r'^#+\s*', '', stripped)
            if skip_headings:
                continue

        # 其他标题
        if stripped.startswith('#'):
            if skip_headings:
                continue

        # 图题表题
        if skip_captions and re.match(r'^[图表]\d', stripped):
            continue

        # 长度过滤
        if len(stripped) < min_length:
            continue

        para_index += 1
        paragraphs.append({
            'text': stripped,
            'chapter': current_chapter,
            'section': current_section,
            'index': para_index,
            'source': os.path.basename(source) if source else 'md',
        })

    return paragraphs
