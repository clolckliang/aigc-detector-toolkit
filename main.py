#!/usr/bin/env python3
"""
AIGC Detector Toolkit — 中文 AI 生成文本检测工具

四引擎融合检测：FengCi0 特征工程 + HC3 深度学习 + OpenAI 兼容 API logprobs + Binoculars

用法:
    python main.py detect <file>              检测单个文件
    python main.py detect <file> --engine fengci
    python main.py batch <dir>                批量检测目录
    python main.py status                     查看引擎状态
    python main.py test                       快速测试（内置样本）
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.engine import DetectionEngine
from extractors.docx_extractor import extract_from_docx
from extractors.md_extractor import extract_from_md
from extractors.text_extractor import extract_from_txt
from reporters.console_reporter import print_summary, print_detail
from reporters.text_reporter import generate_text_report
from reporters.json_reporter import generate_json_report


# ── 内置配置 ────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "engine": {
        "default": "ensemble",
        "ensemble": {
            "fengci_weight": 0.30,
            "hc3_weight": 0.20,
            "openai_weight": 0.25,
            "binoculars_weight": 0.25,
        },
    },
    "openai_api": {
        "api_base": os.environ.get(
            "OPENAI_BASE_URL",
            os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        ),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "strategy": "perplexity",
    },
    "binoculars": {
        "api_base": "",
        "api_key": "",
        "model": "",
    },
    "extraction": {
        "min_paragraph_length": 30,
        "skip_headings": True,
        "skip_captions": True,
        "skip_code_blocks": True,
    },
    "threshold": {"aigc_threshold": 0.5},
    "output": {"dir": None, "json": True, "text": True, "console": True, "verbose": False},
}


def load_config(config_path=None):
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    user = {}  # 记录用户配置中的顶层 key，用于兼容判断

    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                user = yaml.safe_load(f) or {}
            for section, values in user.items():
                if section in config and isinstance(values, dict) and isinstance(config[section], dict):
                    config[section].update(values)
                else:
                    config[section] = values
        except ImportError:
            pass
        except Exception as e:
            logging.warning("配置加载失败: %s", e)

    # 兼容旧配置: mimo_api → openai_api
    if "mimo_api" in config and "openai_api" not in user:
        config["openai_api"].update({k: v for k, v in config.pop("mimo_api").items() if v})

    if os.environ.get("OPENAI_API_KEY"):
        config["openai_api"]["api_key"] = os.environ["OPENAI_API_KEY"]

    if os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"):
        config["openai_api"]["api_base"] = os.environ.get("OPENAI_BASE_URL") or os.environ["OPENAI_API_BASE"]

    if os.environ.get("OPENAI_MODEL"):
        config["openai_api"]["model"] = os.environ["OPENAI_MODEL"]

    return config


def _models_dir():
    return os.path.join(PROJECT_ROOT, "models")


def build_engine(config, args):
    openai_cfg = config.get("openai_api", {})
    bino_cfg = config.get("binoculars", {})
    ensemble = config["engine"]["ensemble"]
    models = _models_dir()
    mode = getattr(args, "engine", None) or config["engine"]["default"]
    # openai 模式直接传递

    return DetectionEngine(
        mode=mode,
        fengci_model=os.path.join(models, "fengci", "aigc_detector_model.joblib"),
        hc3_model=os.path.join(models, "hc3", "OptimizedCNN_aigc_detector.pth"),
        fengci_weight=ensemble.get("fengci_weight", 0.30),
        hc3_weight=ensemble.get("hc3_weight", 0.20),
        openai_weight=ensemble.get("openai_weight", 0.25),
        binoculars_weight=ensemble.get("binoculars_weight", 0.25),
        aigc_threshold=getattr(args, "threshold", None) or config["threshold"]["aigc_threshold"],
        openai_api_base=openai_cfg.get("api_base"),
        openai_api_key=openai_cfg.get("api_key"),
        openai_model=openai_cfg.get("model", "gpt-4o-mini"),
        openai_strategy=openai_cfg.get("strategy", "perplexity"),
        binoculars_api_base=bino_cfg.get("api_base") or openai_cfg.get("api_base"),
        binoculars_api_key=bino_cfg.get("api_key") or openai_cfg.get("api_key"),
        binoculars_model=bino_cfg.get("model") or openai_cfg.get("model", "gpt-4o-mini"),
    )


def extract_paragraphs(file_path, config):
    ext = os.path.splitext(file_path)[1].lower()
    ec = config.get("extraction", {})
    min_len = ec.get("min_paragraph_length", 30)

    if ext == ".docx":
        return extract_from_docx(file_path, min_length=min_len,
                                 skip_headings=ec.get("skip_headings", True),
                                 skip_captions=ec.get("skip_captions", True))
    elif ext == ".md":
        return extract_from_md(file_path, min_length=min_len,
                               skip_headings=ec.get("skip_headings", True),
                               skip_captions=ec.get("skip_captions", True),
                               skip_code_blocks=ec.get("skip_code_blocks", True))
    elif ext == ".txt":
        return extract_from_txt(file_path, min_length=min_len)
    else:
        raise ValueError(f"不支持的文件格式: {ext}（支持 .docx / .md / .txt）")


def detect_file(engine, file_path, config, args):
    print(f"\n📄 正在检测: {file_path}")
    print("-" * 60)

    print("  提取段落中...")
    paragraphs = extract_paragraphs(file_path, config)
    print(f"  提取到 {len(paragraphs)} 个段落")

    if not paragraphs:
        print("  ❌ 未提取到有效段落！")
        return None

    texts = [p["text"] for p in paragraphs]
    print(f"  开始 AIGC 检测（引擎: {engine.mode}）...")
    results = engine.detect_batch(texts, show_progress=True)

    engine_status = engine.get_status()
    output_dir = getattr(args, "output", None) or config.get("output", {}).get("dir")
    if not output_dir:
        output_dir = os.path.dirname(os.path.abspath(file_path))

    basename = os.path.splitext(os.path.basename(file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if config.get("output", {}).get("console", True):
        print_summary(paragraphs, results, engine_status)

    if getattr(args, "verbose", False) or config.get("output", {}).get("verbose", False):
        print_detail(paragraphs, results)

    if config.get("output", {}).get("text", True):
        rp = os.path.join(output_dir, f"AIGC检测报告_{basename}_{timestamp}.txt")
        generate_text_report(paragraphs, results, engine_status, rp, file_path)
        print(f"  📝 文本报告: {rp}")

    if config.get("output", {}).get("json", True):
        jp = os.path.join(output_dir, f"AIGC检测结果_{basename}_{timestamp}.json")
        generate_json_report(paragraphs, results, engine_status, jp, file_path)
        print(f"  📊 JSON报告: {jp}")

    return results


# ── 子命令 ──────────────────────────────────────────────────

def cmd_detect(args, config):
    fp = os.path.abspath(args.file)
    if not os.path.exists(fp):
        print(f"❌ 文件不存在: {fp}")
        return 1
    engine = build_engine(config, args)
    detect_file(engine, fp, config, args)
    return 0


def cmd_batch(args, config):
    dp = os.path.abspath(args.dir)
    if not os.path.isdir(dp):
        print(f"❌ 目录不存在: {dp}")
        return 1

    files = []
    for ext in ("*.docx", "*.md", "*.txt"):
        files.extend(glob.glob(os.path.join(dp, "**", ext), recursive=True))
    files = sorted(set(files))
    if not files:
        print(f"❌ 目录下未找到文件: {dp}")
        return 1

    print(f"\n📁 批量检测: {dp}")
    print(f"  找到 {len(files)} 个文件")

    engine = build_engine(config, args)
    all_results = []
    for fp in files:
        try:
            r = detect_file(engine, fp, config, args)
            if r:
                all_results.append((fp, r))
        except Exception as e:
            print(f"  ❌ 失败 [{fp}]: {e}")

    if all_results:
        ta = sum(1 for _, rs in all_results for r in rs if r["label"] == "ai")
        tv = sum(1 for _, rs in all_results for r in rs if r["label"] != "unknown")
        print("\n" + "=" * 60)
        print(f"📊 批量完成: {len(all_results)} 个文件")
        if tv:
            print(f"   AI段落: {ta}/{tv} ({ta/tv*100:.1f}%)")
        print("=" * 60)
    return 0


def cmd_status(args, config):
    print("\n🔍 AIGC Detector Toolkit — 引擎状态")
    print("=" * 60)
    engine = build_engine(config, args)
    s = engine.get_status()
    labels = {
        "fengci": "FengCi0 特征工程",
        "hc3": "HC3+m3e 深度学习",
        "openai": "OpenAI 兼容 API logprobs",
        "binoculars": "Binoculars 双模型",
    }
    for name, label in labels.items():
        avail = s.get(f"{name}_available", False)
        w = s.get(f"{name}_weight", 0)
        print(f"  {label:20s}  {'✓ 可用' if avail else '✗ 不可用'}  （权重 {w:.0%}）")
    print(f"  判定阈值:         {s['threshold']}")
    print("=" * 60)
    return 0


def cmd_test(args, config):
    """快速测试：用内置样本验证引擎是否正常"""
    print("\n🧪 快速测试")
    print("=" * 60)
    engine = build_engine(config, args)

    samples = [
        ("人工风格", "今天在实验室调试了MAX30102的I2C通信，发现SDA线上的上拉电阻值选小了导致信号畸变，换了个4.7k的电阻后波形正常了。顺便把SpO2的查表逻辑优化了一下，原来那个184元素的数组可以用二分查找加速。"),
        ("AI风格", "综上所述，智能手表嵌入式系统的设计是一个涉及多学科交叉的综合性工程问题。在硬件层面，需要综合考虑处理器性能、功耗控制、传感器接口兼容性等因素；在软件层面，需要完成实时操作系统的移植与优化、图形库的适配与加速、传感器驱动的开发与调试等工作。"),
    ]

    for label, text in samples:
        r = engine.detect_single(text)
        score = r["aigc_score"]
        score_pct = f"{score*100:.1f}%" if score is not None else "N/A"
        print(f"  [{label}] AIGC={score_pct} label={r['label']} method={r['method']}")

    print("=" * 60)
    return 0


# ── 主入口 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AIGC Detector Toolkit — 中文 AI 生成文本检测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--config", default=None, help="YAML 配置文件路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="逐段详情输出")

    sub = parser.add_subparsers(dest="command", help="子命令")
    choices = ["fengci", "hc3", "openai", "binoculars", "ensemble"]

    p = sub.add_parser("detect", help="检测单个文件")
    p.add_argument("file", help="待检测文件（.docx / .md / .txt）")
    p.add_argument("--engine", choices=choices, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("-o", "--output", default=None)

    p = sub.add_parser("batch", help="批量检测目录")
    p.add_argument("dir", help="待检测目录")
    p.add_argument("--engine", choices=choices, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("-o", "--output", default=None)

    sub.add_parser("status", help="查看引擎状态")
    sub.add_parser("test", help="快速测试")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = args.config or os.path.join(PROJECT_ROOT, "configs", "default.yaml")
    config = load_config(config_path)

    cmds = {"detect": cmd_detect, "batch": cmd_batch, "status": cmd_status, "test": cmd_test}
    return cmds[args.command](args, config)


if __name__ == "__main__":
    sys.exit(main())
