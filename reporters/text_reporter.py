"""文本文件报告生成器"""
import os
import logging
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


def generate_text_report(
    paragraphs: List[Dict],
    results: List[Dict],
    engine_status: Dict,
    output_path: str,
    source_file: str = "",
):
    """生成详细文本报告"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    total = len(results)
    valid = [r for r in results if r['label'] != 'unknown']
    ai_items = [r for r in valid if r['label'] == 'ai']
    human_items = [r for r in valid if r['label'] == 'human']
    scores = [r['aigc_score'] for r in valid if r['aigc_score'] is not None]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("                    AIGC 查重检测报告\n")
        f.write("=" * 80 + "\n\n")

        # 基本信息
        f.write(f"检测文件:   {source_file}\n")
        f.write(f"检测时间:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"检测引擎:   {engine_status['mode']}\n")
        if engine_status['mode'] == 'ensemble':
            f.write(f"  FengCi0:  {'可用' if engine_status['fengci_available'] else '不可用'} "
                    f"(权重 {engine_status['fengci_weight']:.0%})\n")
            f.write(f"  HC3+m3e:  {'可用' if engine_status['hc3_available'] else '不可用'} "
                    f"(权重 {engine_status['hc3_weight']:.0%})\n")
        f.write(f"判定阈值:   {engine_status['threshold']:.2f}\n\n")

        # 汇总统计
        f.write("-" * 80 + "\n")
        f.write("汇总统计\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"总段落数:     {total}\n")
        f.write(f"有效检测:     {len(valid)}\n")
        failed = total - len(valid)
        if failed:
            f.write(f"跳过/失败:    {failed}\n")
        f.write("\n")

        if valid:
            ai_ratio = len(ai_items) / len(valid) * 100
            human_ratio = len(human_items) / len(valid) * 100
            avg_score = sum(scores) / len(scores) * 100 if scores else 0

            f.write(f"判定为人工撰写: {len(human_items):4d} ({human_ratio:5.1f}%)\n")
            f.write(f"判定为AI生成:   {len(ai_items):4d} ({ai_ratio:5.1f}%)\n")
            f.write(f"平均AIGC分数:   {avg_score:.1f}%\n\n")

            # 区间分布
            f.write("分数区间分布:\n")
            brackets = [
                (0.0, 0.1, '极低 [0-10%)'),
                (0.1, 0.2, '低   [10-20%)'),
                (0.2, 0.3, '中低 [20-30%)'),
                (0.3, 0.4, '中等 [30-40%)'),
                (0.4, 0.5, '中高 [40-50%)'),
                (0.5, 0.7, '偏高 [50-70%)'),
                (0.7, 1.0, '高   [70-100%)'),
            ]
            for lo, hi, name in brackets:
                cnt = sum(1 for s in scores if lo <= s < hi)
                pct = cnt / len(valid) * 100
                f.write(f"  {name:20s}: {cnt:4d} ({pct:5.1f}%)\n")
            f.write("\n")

            # 按章节统计
            chapter_stats = {}
            for para, result in zip(paragraphs, results):
                ch = para.get('chapter', 0)
                if ch not in chapter_stats:
                    chapter_stats[ch] = {'total': 0, 'ai': 0, 'scores': []}
                chapter_stats[ch]['total'] += 1
                if result['label'] == 'ai':
                    chapter_stats[ch]['ai'] += 1
                if result['aigc_score'] is not None:
                    chapter_stats[ch]['scores'].append(result['aigc_score'])

            if any(ch > 0 for ch in chapter_stats.keys()):
                f.write("按章节统计:\n")
                for ch in sorted(chapter_stats.keys()):
                    if ch == 0:
                        continue
                    stat = chapter_stats[ch]
                    avg = sum(stat['scores']) / len(stat['scores']) * 100 if stat['scores'] else 0
                    ai_pct = stat['ai'] / stat['total'] * 100 if stat['total'] > 0 else 0
                    f.write(f"  第{ch}章: {stat['total']:3d}段, AI={stat['ai']:2d}({ai_pct:4.1f}%), 平均={avg:.1f}%\n")
                f.write("\n")

        # 逐段详情
        f.write("-" * 80 + "\n")
        f.write("逐段检测详情:\n")
        f.write("-" * 80 + "\n\n")

        for para, result in zip(paragraphs, results):
            score_pct = result['aigc_score'] * 100 if result['aigc_score'] is not None else -1
            label_str = 'AI生成' if result['label'] == 'ai' else ('人工撰写' if result['label'] == 'human' else '未知')
            conf = result.get('confidence', 0)
            ch = para.get('chapter', '?')

            f.write(f"[段落 {para['index']:3d}] 第{ch}章 | {label_str} (AIGC: {score_pct:.1f}%, 置信度: {conf:.2f})\n")
            f.write(f"  {para['text'][:200]}\n")

            # 引擎详情
            for eng_name, eng_result in result.get('engine_results', {}).items():
                if eng_result.get('aigc_score') is not None:
                    eng_score = eng_result['aigc_score'] * 100
                    f.write(f"  [{eng_name}] score={eng_score:.1f}% label={eng_result.get('label', '?')}")

                    # 特征信息
                    features = eng_result.get('features', {})
                    if features and eng_name == 'fengci':
                        top_feats = sorted(features.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)[:5]
                        feat_str = ', '.join([f"{k}={v:.3f}" for k, v in top_feats])
                        f.write(f"  特征: {feat_str}")
                    f.write("\n")

            f.write("\n")

    logger.info("文本报告已保存: %s", output_path)
    return output_path
