"""终端报告输出"""
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

ENGINE_LABELS = {
    "fengci": "FengCi0",
    "hc3": "HC3+m3e",
    "openai": "OpenAI API",
    "binoculars": "Binoculars",
    "local_logprob": "Local Logprob",
}


def print_summary(paragraphs: List[Dict], results: List[Dict], engine_status: Dict):
    """打印终端摘要"""
    total = len(results)
    valid = [r for r in results if r['label'] != 'unknown']
    failed = [r for r in results if r['label'] == 'unknown']

    ai_items = [r for r in valid if r['label'] == 'ai']
    human_items = [r for r in valid if r['label'] == 'human']
    scores = [r['aigc_score'] for r in valid if r['aigc_score'] is not None]

    print()
    print("=" * 60)
    print("              AIGC 查重检测报告")
    print("=" * 60)
    print(f"检测引擎:   {engine_status['mode']}")
    if engine_status['mode'] == 'ensemble':
        for name, label in ENGINE_LABELS.items():
            print(f"  {label:14s}: {'✓' if engine_status.get(f'{name}_available') else '✗'} "
                  f"(权重 {engine_status.get(f'{name}_weight', 0):.0%})")
    print(f"判定阈值:   {engine_status['threshold']:.2f}")
    print("-" * 60)
    print(f"总段落数:     {total}")
    print(f"有效检测:     {len(valid)}")
    if failed:
        print(f"跳过/失败:    {len(failed)} (文本过短或引擎不可用)")
    print("-" * 60)

    if valid:
        ai_ratio = len(ai_items) / len(valid) * 100
        human_ratio = len(human_items) / len(valid) * 100
        avg_score = sum(scores) / len(scores) * 100 if scores else 0
        max_score = max(scores) * 100 if scores else 0
        min_score = min(scores) * 100 if scores else 0

        print(f"判定为人工撰写: {len(human_items):4d} ({human_ratio:5.1f}%)")
        print(f"判定为AI生成:   {len(ai_items):4d} ({ai_ratio:5.1f}%)")
        print("-" * 60)
        print(f"平均AIGC分数:   {avg_score:.1f}%")
        print(f"最高AIGC分数:   {max_score:.1f}%")
        print(f"最低AIGC分数:   {min_score:.1f}%")
        print("=" * 60)

        # 区间分布
        print("\n分数区间分布:")
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
            bar = '█' * max(1, int(cnt / 2)) if cnt > 0 else ''
            print(f"  {name:20s}: {cnt:4d} ({pct:5.1f}%) {bar}")

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
            print("\n按章节统计:")
            for ch in sorted(chapter_stats.keys()):
                if ch == 0:
                    continue
                stat = chapter_stats[ch]
                avg = sum(stat['scores']) / len(stat['scores']) * 100 if stat['scores'] else 0
                ai_pct = stat['ai'] / stat['total'] * 100 if stat['total'] > 0 else 0
                print(f"  第{ch}章: {stat['total']:3d}段, AI={stat['ai']:2d}({ai_pct:4.1f}%), 平均={avg:.1f}%")

        # 高风险段落
        high_risk = [(p, r) for p, r in zip(paragraphs, results)
                     if r['aigc_score'] is not None and r['aigc_score'] > 0.5]
        if high_risk:
            high_risk.sort(key=lambda x: x[1]['aigc_score'], reverse=True)
            print(f"\n⚠️  AI判定段落 (>{engine_status['threshold']*100:.0f}%): {len(high_risk)} 段")
            for p, r in high_risk[:20]:
                score_pct = r['aigc_score'] * 100
                print(f"  [{p['index']:3d}] {score_pct:5.1f}% | {p['text'][:80]}...")

    print()


def print_detail(paragraphs: List[Dict], results: List[Dict]):
    """打印逐段详情（verbose 模式）"""
    for para, result in zip(paragraphs, results):
        score_pct = result['aigc_score'] * 100 if result['aigc_score'] is not None else -1
        label_str = '🔴AI' if result['label'] == 'ai' else ('🟢人' if result['label'] == 'human' else '⚪?')
        print(f"  [{para['index']:3d}] {score_pct:5.1f}% {label_str} | {para['text'][:100]}")
