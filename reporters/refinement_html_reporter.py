"""HTML report generator for refinement results."""
import html
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


STRATEGY_LABELS = {
    "refine_cn": "表达润色（中文）",
    "deai_cn": "去 AI 化（中文）",
    "deai_en": "去 AI 化（英文）",
    "logic_check_en": "逻辑检查（英文）",
    "skip": "跳过",
    "error": "异常",
}


REPORT_CSS = r"""
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --panel-soft: #f8fafc;
      --ink: #172033;
      --muted: #667085;
      --line: #d8dee8;
      --brand: #2459a6;
      --brand-soft: #edf4ff;
      --green: #1f7a5a;
      --green-soft: #ecfdf3;
      --red: #b42318;
      --red-soft: #fff1f0;
      --amber: #b76e00;
      --amber-soft: #fff7e6;
      --shadow: 0 12px 30px rgba(15, 23, 42, .08);
      --sans: "Noto Sans SC", "Source Han Sans SC", "Microsoft YaHei", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: var(--sans);
      line-height: 1.68;
    }
    .page {
      width: min(1360px, calc(100vw - 32px));
      margin: 18px auto 42px;
    }
    .hero, .panel, .card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 18px;
      align-items: end;
      padding: 22px 24px;
      background: #122033;
      color: #fff;
    }
    .eyebrow {
      color: #a8c7fa;
      font-size: 13px;
      font-weight: 700;
    }
    h1 {
      margin: 6px 0 8px;
      font-size: clamp(26px, 3.2vw, 40px);
      line-height: 1.15;
    }
    .meta { color: #c7d2e3; font-size: 14px; }
    .verdict {
      border: 1px solid rgba(255, 255, 255, .16);
      border-left: 5px solid #a8c7fa;
      border-radius: 8px;
      background: rgba(255, 255, 255, .08);
      padding: 14px;
    }
    .verdict strong { display: block; font-size: 20px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 8px;
      margin: 14px 0;
    }
    .metric {
      min-height: 76px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
    }
    .metric b { display: block; font-size: 24px; line-height: 1.1; }
    .metric span { color: var(--muted); font-size: 12px; font-weight: 700; }
    .toolbar {
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin: 14px 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, .96);
      box-shadow: 0 8px 20px rgba(15, 23, 42, .08);
    }
    .filter-btn, .copy-btn {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: inherit;
      cursor: pointer;
      padding: 8px 11px;
      font: 700 13px var(--sans);
    }
    .filter-btn.active, .filter-btn:hover, .copy-btn:hover {
      background: var(--brand);
      border-color: var(--brand);
      color: #fff;
    }
    .search {
      min-width: min(340px, 100%);
      flex: 1;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 12px;
      font: 14px var(--sans);
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 16px;
      align-items: start;
    }
    .panel { overflow: hidden; }
    .panel-title {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin: 0;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-soft);
      font-size: 16px;
    }
    .rows { display: grid; gap: 12px; padding: 14px; }
    .row {
      border: 1px solid var(--line);
      border-left: 5px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 14px;
      break-inside: avoid;
    }
    .row.changed { border-left-color: var(--green); }
    .row.unchanged { border-left-color: var(--muted); }
    .row.worse { border-left-color: var(--red); }
    .row-head {
      display: grid;
      grid-template-columns: 90px minmax(0, 1fr) auto;
      gap: 10px;
      align-items: start;
      margin-bottom: 10px;
    }
    .index { color: var(--muted); font-weight: 800; }
    .strategy { color: var(--brand); font-weight: 800; }
    .score {
      display: inline-grid;
      grid-template-columns: auto 16px auto;
      gap: 4px;
      align-items: center;
      border-radius: 999px;
      background: var(--brand-soft);
      color: var(--brand);
      padding: 4px 10px;
      font-weight: 800;
      white-space: nowrap;
    }
    .score i { color: var(--muted); font-style: normal; text-align: center; }
    .score-delta {
      display: inline-block;
      margin-left: 8px;
      font-weight: 800;
    }
    .score-delta.improved { color: var(--green); }
    .score-delta.worse { color: var(--red); }
    .compare {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 10px;
    }
    .text-block {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
      padding: 10px;
    }
    .text-block h3 {
      margin: 0 0 6px;
      color: var(--muted);
      font-size: 13px;
    }
    .text-block p {
      margin: 0;
      white-space: pre-wrap;
    }
    .note {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .history {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }
    .pill {
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
    }
    .sidebar {
      position: sticky;
      top: 78px;
      display: grid;
      gap: 14px;
    }
    .card { padding: 14px; box-shadow: none; }
    .card h2 {
      margin: 0 0 10px;
      font-size: 16px;
    }
    .bars { display: grid; gap: 8px; }
    .bar-label {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 3px;
    }
    .bar-track {
      height: 8px;
      border-radius: 999px;
      background: #e8edf3;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: inherit;
      background: var(--green);
    }
    .bar-fill.skip { background: var(--muted); }
    .bar-fill.worse { background: var(--red); }
    .priority {
      border-left: 3px solid var(--green);
      padding: 8px 0 8px 10px;
      margin-bottom: 8px;
      font-size: 13px;
    }
    .empty { color: var(--muted); }
    @media (max-width: 1100px) {
      .hero, .layout { grid-template-columns: 1fr; }
      .sidebar { position: static; }
      .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    @media (max-width: 720px) {
      .page { width: calc(100vw - 16px); margin: 8px auto 24px; }
      .hero { padding: 18px; }
      .metrics, .compare { grid-template-columns: 1fr; }
      .toolbar { position: static; align-items: stretch; }
      .row-head { grid-template-columns: 1fr; }
      .score { width: fit-content; }
    }
    @media print {
      body { background: #fff; }
      .page { width: auto; margin: 0; }
      .toolbar, script { display: none !important; }
      .hero, .panel, .card { box-shadow: none; }
      .layout { grid-template-columns: 1fr; }
      .sidebar { position: static; }
      .row { break-inside: avoid; page-break-inside: avoid; }
    }
"""


REPORT_JS = r"""
(function () {
  var active = "all";
  var input = document.getElementById("report-search");
  var count = document.getElementById("visible-count");
  var rows = Array.prototype.slice.call(document.querySelectorAll(".row"));

  function apply() {
    var q = ((input && input.value) || "").toLowerCase().trim();
    var visible = 0;
    rows.forEach(function (row) {
      var state = row.getAttribute("data-state") || "";
      var text = (row.getAttribute("data-text") || row.textContent || "").toLowerCase();
      var ok = (active === "all" || active === state) && (!q || text.indexOf(q) !== -1);
      row.style.display = ok ? "" : "none";
      if (ok) visible += 1;
    });
    if (count) count.textContent = "显示 " + visible + " / " + rows.length + " 段";
  }

  Array.prototype.slice.call(document.querySelectorAll(".filter-btn")).forEach(function (button) {
    button.addEventListener("click", function () {
      active = button.getAttribute("data-filter") || "all";
      document.querySelectorAll(".filter-btn.active").forEach(function (node) { node.classList.remove("active"); });
      button.classList.add("active");
      apply();
    });
  });

  if (input) input.addEventListener("input", apply);

  Array.prototype.slice.call(document.querySelectorAll(".copy-btn")).forEach(function (button) {
    button.addEventListener("click", function () {
      var text = button.getAttribute("data-copy") || "";
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text);
      }
      var old = button.textContent;
      button.textContent = "已复制";
      setTimeout(function () { button.textContent = old; }, 1200);
    });
  });

  apply();
})();
"""


def generate_refinement_html_report(
    refinement_results: List[Dict],
    output_path: str,
    source_file: str = "",
    aigc_before: Optional[Dict] = None,
    aigc_after: Optional[Dict] = None,
):
    """Generate a single-file HTML report for refinement results."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows = [_normalize_row(item) for item in refinement_results]
    changed = [row for row in rows if row["changed"]]
    unchanged = [row for row in rows if not row["changed"]]
    improved = [row for row in rows if row["delta"] is not None and row["delta"] < 0]
    worse = [row for row in rows if row["delta"] is not None and row["delta"] > 0]
    before_scores = [row["before"] for row in rows if row["before"] is not None]
    after_scores = [row["after"] for row in rows if row["after"] is not None]
    avg_before = sum(before_scores) / len(before_scores) if before_scores else None
    avg_after = sum(after_scores) / len(after_scores) if after_scores else None

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>降 AIGC 率润色报告</title>
  <style>
{REPORT_CSS}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <div>
        <div class="eyebrow">AIGC Refinement Report</div>
        <h1>降 AIGC 率润色报告</h1>
        <div class="meta">处理文件：{_esc(source_file or "-")} · 生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
      </div>
      <div class="verdict">
        <strong>{len(changed)} / {len(rows)} 段已修改</strong>
        <span>{_summary_sentence(aigc_before, aigc_after, avg_before, avg_after)}</span>
      </div>
    </header>

    <section class="metrics">
      <div class="metric"><b>{len(rows)}</b><span>总段落</span></div>
      <div class="metric"><b>{len(changed)}</b><span>已修改</span></div>
      <div class="metric"><b>{len(unchanged)}</b><span>未修改/跳过</span></div>
      <div class="metric"><b>{len(improved)}</b><span>分数下降</span></div>
      <div class="metric"><b>{_score_text(avg_before)}</b><span>平均润色前</span></div>
      <div class="metric"><b>{_score_text(avg_after)}</b><span>平均润色后</span></div>
    </section>

    <nav class="toolbar" aria-label="报告筛选工具">
      <button class="filter-btn active" data-filter="all">全部</button>
      <button class="filter-btn" data-filter="changed">已修改</button>
      <button class="filter-btn" data-filter="unchanged">未修改</button>
      <button class="filter-btn" data-filter="improved">分数下降</button>
      <button class="filter-btn" data-filter="worse">分数上升</button>
      <input id="report-search" class="search" type="search" placeholder="搜索原文、改写或说明">
      <span id="visible-count" class="empty"></span>
    </nav>

    <section class="layout">
      <article class="panel">
        <h2 class="panel-title"><span>逐段润色对比</span><span>原文 / 改写 / 分数变化</span></h2>
        <div class="rows">
          {_render_rows(rows)}
        </div>
      </article>
      <aside class="sidebar">
        {_render_distribution(rows)}
        {_render_priority(rows)}
        {_render_strategy(rows)}
      </aside>
    </section>
  </main>
  <script>
{REPORT_JS}
  </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    logger.info("润色 HTML 报告已保存: %s", output_path)
    return output_path


def _normalize_row(item: Dict) -> Dict:
    before = item.get("aigc_before")
    after = item.get("aigc_after")
    delta = after - before if isinstance(before, (int, float)) and isinstance(after, (int, float)) else None
    if delta is not None and delta < 0:
        state = "improved"
    elif delta is not None and delta > 0:
        state = "worse"
    elif item.get("changed"):
        state = "changed"
    else:
        state = "unchanged"
    return {
        "index": item.get("index", ""),
        "chapter": item.get("chapter", ""),
        "original": item.get("original", ""),
        "refined": item.get("refined", item.get("original", "")),
        "strategy": item.get("strategy", "unknown"),
        "strategy_label": STRATEGY_LABELS.get(item.get("strategy", "unknown"), item.get("strategy", "unknown")),
        "changed": bool(item.get("changed")),
        "note": item.get("note", ""),
        "before": before if isinstance(before, (int, float)) else None,
        "after": after if isinstance(after, (int, float)) else None,
        "delta": delta,
        "rounds": item.get("rounds", 0),
        "round_history": item.get("round_history", []) or [],
        "state": state,
    }


def _render_rows(rows: List[Dict]) -> str:
    if not rows:
        return '<p class="empty">没有润色结果。</p>'
    return "\n".join(_render_row(row) for row in rows)


def _render_row(row: Dict) -> str:
    delta_cls = "improved" if row["delta"] is not None and row["delta"] < 0 else "worse" if row["delta"] is not None and row["delta"] > 0 else ""
    row_cls = "changed" if row["changed"] else "unchanged"
    if row["state"] == "worse":
        row_cls = "worse"
    copy_text = f"段落 {row['index']}：{row['refined']}"
    return (
        f'<section class="row {row_cls}" data-state="{_esc(row["state"])}" data-text="{_esc(row["original"] + " " + row["refined"] + " " + row["note"])}">'
        '<div class="row-head">'
        f'<div class="index">段落 {row["index"]}</div>'
        f'<div><div class="strategy">{_esc(row["strategy_label"])}</div><div class="note">第 {_esc(row["chapter"] or "-")} 章 · 轮次 {_esc(row["rounds"])}</div></div>'
        f'<div><span class="score"><b>{_score_text(row["before"])}</b><i>→</i><b>{_score_text(row["after"])}</b></span>'
        f'<span class="score-delta {delta_cls}">{_delta_text(row["delta"])}</span></div>'
        '</div>'
        '<div class="compare">'
        f'<div class="text-block"><h3>原文</h3><p>{_esc(row["original"])}</p></div>'
        f'<div class="text-block"><h3>改写</h3><p>{_esc(row["refined"])}</p></div>'
        '</div>'
        f'<div class="note">说明：{_esc(row["note"] or "-")}</div>'
        f'{_render_history(row["round_history"])}'
        f'<div class="note"><button class="copy-btn" type="button" data-copy="{_esc(copy_text)}">复制改写</button></div>'
        '</section>'
    )


def _render_history(history: List[Dict]) -> str:
    if not history:
        return ""
    pills = []
    for item in history:
        pills.append(
            f'<span class="pill">第 {_esc(item.get("round", "-"))} 轮：{_score_text(item.get("aigc_score"))}</span>'
        )
    return '<div class="history">' + "".join(pills) + "</div>"


def _render_distribution(rows: List[Dict]) -> str:
    total = len(rows) or 1
    items = [
        ("已修改", sum(1 for row in rows if row["changed"]), "changed"),
        ("未修改/跳过", sum(1 for row in rows if not row["changed"]), "skip"),
        ("分数下降", sum(1 for row in rows if row["delta"] is not None and row["delta"] < 0), "improved"),
        ("分数上升", sum(1 for row in rows if row["delta"] is not None and row["delta"] > 0), "worse"),
    ]
    bars = []
    for label, count, cls in items:
        pct = count / total * 100
        bars.append(
            f'<div><div class="bar-label"><span>{_esc(label)}</span><span>{count} 段 · {pct:.1f}%</span></div>'
            f'<div class="bar-track"><div class="bar-fill {cls}" style="width:{pct:.1f}%"></div></div></div>'
        )
    return '<div class="card"><h2>结果分布</h2><div class="bars">' + "".join(bars) + "</div></div>"


def _render_priority(rows: List[Dict]) -> str:
    candidates = [row for row in rows if row["before"] is not None]
    candidates.sort(key=lambda row: row["before"], reverse=True)
    if not candidates:
        return '<div class="card"><h2>优先复核</h2><p class="empty">没有可排序的分数。</p></div>'
    items = []
    for row in candidates[:8]:
        items.append(
            f'<div class="priority"><strong>段落 {row["index"]} · {_score_text(row["before"])} → {_score_text(row["after"])}</strong><br>{_esc(_shorten(row["note"] or row["original"], 78))}</div>'
        )
    return '<div class="card"><h2>优先复核</h2>' + "".join(items) + "</div>"


def _render_strategy(rows: List[Dict]) -> str:
    counts = {}
    for row in rows:
        counts[row["strategy_label"]] = counts.get(row["strategy_label"], 0) + 1
    items = "".join(
        f'<div class="bar-label"><span>{_esc(label)}</span><span>{count} 段</span></div>'
        for label, count in sorted(counts.items(), key=lambda item: -item[1])
    )
    return '<div class="card"><h2>策略分布</h2>' + (items or '<p class="empty">无数据</p>') + "</div>"


def _summary_sentence(aigc_before: Optional[Dict], aigc_after: Optional[Dict], avg_before, avg_after) -> str:
    if aigc_before and aigc_after and aigc_before.get("ai_ratio") is not None and aigc_after.get("ai_ratio") is not None:
        delta = float(aigc_before.get("ai_ratio", 0)) - float(aigc_after.get("ai_ratio", 0))
        return f"AIGC 率 {aigc_before.get('ai_ratio')}% → {aigc_after.get('ai_ratio')}%，降幅 {delta:.1f} 个百分点。"
    if avg_before is not None and avg_after is not None:
        return f"平均分 {_score_text(avg_before)} → {_score_text(avg_after)}，变化 {_delta_text(avg_after - avg_before)}。"
    return "报告包含逐段原文、改写、策略、轮次和分数变化。"


def _score_text(score) -> str:
    return "-" if score is None else f"{float(score) * 100:.1f}%"


def _delta_text(delta) -> str:
    if delta is None:
        return "-"
    if delta == 0:
        return "无变化"
    sign = "+" if delta > 0 else ""
    return f"{sign}{float(delta) * 100:.1f}%"


def _shorten(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _esc(value) -> str:
    return html.escape(str(value), quote=True)
