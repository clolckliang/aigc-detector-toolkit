"""Evaluation and threshold calibration helpers for detector outputs."""
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


VALID_LABELS = {"human", "ai"}


def load_labeled_dataset(path: str) -> List[Dict]:
    """Load a JSONL or CSV dataset with at least text and label fields."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"评测集不存在: {path}")

    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSONL 第 {line_no} 行解析失败: {e}") from e
    elif p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError("评测集仅支持 .jsonl 或 .csv")

    dataset = []
    for idx, row in enumerate(rows, 1):
        text = str(row.get("text", "")).strip()
        label = str(row.get("label", "")).strip().lower()
        if not text:
            raise ValueError(f"第 {idx} 条缺少 text")
        if label not in VALID_LABELS:
            raise ValueError(f"第 {idx} 条 label 必须是 human 或 ai，当前为: {label!r}")
        dataset.append({"text": text, "label": label, "index": row.get("index", idx)})
    return dataset


def binary_metrics(y_true: List[str], scores: List[Optional[float]], threshold: float = 0.5) -> Dict:
    """Compute binary classification metrics, ignoring missing scores."""
    pairs = [(label, score) for label, score in zip(y_true, scores) if score is not None]
    skipped = len(y_true) - len(pairs)
    if not pairs:
        return _empty_metrics(skipped=skipped, threshold=threshold)

    tp = fp = tn = fn = 0
    for label, score in pairs:
        pred_ai = score > threshold
        true_ai = label == "ai"
        if pred_ai and true_ai:
            tp += 1
        elif pred_ai and not true_ai:
            fp += 1
        elif not pred_ai and true_ai:
            fn += 1
        else:
            tn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "threshold": round(threshold, 4),
        "count": len(pairs),
        "skipped": skipped,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(roc_auc(y_true, scores), 4) if len({label for label, _ in pairs}) == 2 else None,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def threshold_scan(
    y_true: List[str],
    scores: List[Optional[float]],
    low_fpr_target: float = 0.05,
) -> Dict:
    """Find best-F1 threshold and a low-FPR threshold over valid scores."""
    valid_scores = sorted({score for score in scores if score is not None})
    if not valid_scores:
        return {"best_f1": None, "low_fpr": None}

    candidates = [0.0, 1.0]
    candidates.extend(valid_scores)
    candidates.extend((valid_scores[i] + valid_scores[i + 1]) / 2 for i in range(len(valid_scores) - 1))

    scanned = []
    for threshold in sorted(set(candidates)):
        metrics = binary_metrics(y_true, scores, threshold)
        cm = metrics["confusion_matrix"]
        fpr = _safe_div(cm["fp"], cm["fp"] + cm["tn"])
        scanned.append({"threshold": threshold, "fpr": fpr, **metrics})

    best_f1 = max(scanned, key=lambda m: (m["f1"], m["recall"], m["accuracy"]))
    low_fpr_candidates = [m for m in scanned if m["fpr"] <= low_fpr_target]
    low_fpr = max(low_fpr_candidates, key=lambda m: (m["recall"], m["f1"])) if low_fpr_candidates else None

    return {
        "best_f1": _compact_scan_result(best_f1),
        "low_fpr": _compact_scan_result(low_fpr) if low_fpr else None,
        "low_fpr_target": low_fpr_target,
    }


def summarize_engine_metrics(dataset: List[Dict], results: List[Dict], threshold: float) -> Dict:
    """Summarize fused and per-engine metrics from DetectionEngine results."""
    labels = [row["label"] for row in dataset]
    summary = {
        "fused": {
            "metrics": binary_metrics(labels, [r.get("aigc_score") for r in results], threshold),
            "threshold_scan": threshold_scan(labels, [r.get("aigc_score") for r in results]),
        },
        "engines": {},
    }

    engine_names = sorted({
        name
        for result in results
        for name in result.get("engine_results", {}).keys()
    })
    for name in engine_names:
        scores = [
            result.get("engine_results", {}).get(name, {}).get("aigc_score")
            for result in results
        ]
        methods = _method_counts(
            result.get("engine_results", {}).get(name, {}).get("method")
            for result in results
        )
        summary["engines"][name] = {
            "metrics": binary_metrics(labels, scores, threshold),
            "threshold_scan": threshold_scan(labels, scores),
            "methods": methods,
        }
    return summary


def write_eval_report(
    output_path: str,
    dataset_path: str,
    dataset: List[Dict],
    results: List[Dict],
    engine_status: Dict,
    metrics: Dict,
) -> str:
    report = {
        "dataset": dataset_path,
        "engine_status": engine_status,
        "metrics": metrics,
        "results": [
            {
                "index": row["index"],
                "label": row["label"],
                "text_preview": row["text"][:120],
                "aigc_score": result.get("aigc_score"),
                "predicted_label": result.get("label"),
                "confidence": result.get("confidence"),
                "method": result.get("method"),
                "engine_results": result.get("engine_results", {}),
            }
            for row, result in zip(dataset, results)
        ],
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return str(out)


def roc_auc(y_true: List[str], scores: List[Optional[float]]) -> Optional[float]:
    """Compute ROC AUC with average ranks for ties."""
    pairs = [(1 if label == "ai" else 0, score) for label, score in zip(y_true, scores) if score is not None]
    positives = sum(label for label, _ in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(pairs, key=lambda item: item[1])
    rank_sum_pos = 0.0
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][1] == ranked[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            if ranked[k][0] == 1:
                rank_sum_pos += avg_rank
        i = j + 1

    return (rank_sum_pos - positives * (positives + 1) / 2) / (positives * negatives)


def _empty_metrics(skipped: int, threshold: float) -> Dict:
    return {
        "threshold": round(threshold, 4),
        "count": 0,
        "skipped": skipped,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auc": None,
        "confusion_matrix": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
    }


def _compact_scan_result(metrics: Dict) -> Dict:
    return {
        "threshold": round(metrics["threshold"], 4),
        "fpr": round(metrics["fpr"], 4),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def _method_counts(methods: Iterable[Optional[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for method in methods:
        if method:
            counts[method] = counts.get(method, 0) + 1
    return counts


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0
