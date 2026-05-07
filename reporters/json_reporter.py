"""JSON 报告生成器"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


def generate_json_report(
    paragraphs: List[Dict],
    results: List[Dict],
    engine_status: Dict,
    output_path: str,
    source_file: str = "",
):
    """生成 JSON 格式报告"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    valid = [r for r in results if r['label'] != 'unknown']
    scores = [r['aigc_score'] for r in valid if r['aigc_score'] is not None]

    summary = {
        'source_file': source_file,
        'detect_time': datetime.now().isoformat(),
        'engine': engine_status,
        'total_paragraphs': len(results),
        'valid_count': len(valid),
        'skipped_count': len(results) - len(valid),
        'ai_count': sum(1 for r in valid if r['label'] == 'ai'),
        'human_count': sum(1 for r in valid if r['label'] == 'human'),
        'ai_ratio': round(sum(1 for r in valid if r['label'] == 'ai') / len(valid) * 100, 1) if valid else 0,
        'human_ratio': round(sum(1 for r in valid if r['label'] == 'human') / len(valid) * 100, 1) if valid else 0,
        'avg_aigc_score': round(sum(scores) / len(scores) * 100, 1) if scores else 0,
        'max_aigc_score': round(max(scores) * 100, 1) if scores else 0,
        'min_aigc_score': round(min(scores) * 100, 1) if scores else 0,
    }

    # 逐段结果
    detail_results = []
    for para, result in zip(paragraphs, results):
        detail_results.append({
            'index': para['index'],
            'chapter': para.get('chapter', 0),
            'section': para.get('section', ''),
            'text': para['text'],
            'text_preview': para['text'][:120],
            'aigc_score': result.get('aigc_score'),
            'aigc_score_pct': round(result['aigc_score'] * 100, 1) if result.get('aigc_score') is not None else None,
            'label': result['label'],
            'confidence': result.get('confidence', 0),
            'method': result.get('method', ''),
            'engine_details': {
                eng: {
                    'score': round(r['aigc_score'] * 100, 1) if r.get('aigc_score') is not None else None,
                    'label': r.get('label', 'unknown'),
                    'confidence': r.get('confidence', 0),
                    'method': r.get('method'),
                    'features': r.get('features', {}),
                }
                for eng, r in result.get('engine_results', {}).items()
            }
        })

    report = {
        'summary': summary,
        'results': detail_results,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("JSON 报告已保存: %s", output_path)
    return output_path
