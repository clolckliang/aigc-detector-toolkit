"""HTML visual report with highlighted paragraphs and revision guidance."""
import html
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


ENGINE_LABELS = {
    "fengci": "FengCi0",
    "hc3": "HC3+m3e",
    "openai": "OpenAI API",
    "binoculars": "Binoculars",
    "local_logprob": "Local Logprob",
    "lastde": "LastDe",
}


REPORT_CSS = r"""
    :root {
      --paper: #fffdf7;
      --paper-2: #fbf7ec;
      --ink: #1f2a24;
      --muted: #66736b;
      --line: #dfd6c4;
      --line-strong: #a5967e;
      --brand: #31523c;
      --brand-soft: #e7f0df;
      --red: #c82124;
      --red-bg: #fff0ee;
      --amber: #a65f00;
      --amber-bg: #fff5dc;
      --green: #247044;
      --green-bg: #edf7ef;
      --shadow: 0 18px 45px rgba(45, 35, 20, .16);
      --mono: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      --serif: "Noto Serif SC", "Source Han Serif SC", "Songti SC", "SimSun", serif;
      --sans: "Noto Sans SC", "Source Han Sans SC", "Microsoft YaHei", sans-serif;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 20% 0%, rgba(230, 210, 164, .35), transparent 34rem),
        linear-gradient(180deg, #f0eadc 0%, #e6dcc8 100%);
      font-family: var(--serif);
      line-height: 1.78;
    }
    .page {
      width: min(1440px, calc(100vw - 32px));
      margin: 24px auto 48px;
    }
    .hero {
      position: relative;
      overflow: hidden;
      border: 1px solid var(--line-strong);
      background: linear-gradient(135deg, #fffef9 0%, #f7f0df 72%, #edf5e9 100%);
      box-shadow: var(--shadow);
      padding: 28px 32px 24px;
    }
    .hero:before {
      content: "";
      position: absolute;
      inset: 12px;
      border: 1px solid rgba(117, 97, 64, .18);
      pointer-events: none;
    }
    .eyebrow {
      color: var(--brand);
      font-family: var(--sans);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: .12em;
      text-transform: uppercase;
    }
    h1 {
      margin: 8px 0 10px;
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1.15;
      letter-spacing: -.03em;
    }
    .meta {
      color: var(--muted);
      font-family: var(--sans);
      font-size: 14px;
    }
    .hero-grid {
      position: relative;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 24px;
      align-items: end;
    }
    .verdict {
      border-left: 5px solid var(--brand);
      background: rgba(255, 255, 255, .66);
      padding: 14px 16px;
      font-family: var(--sans);
    }
    .verdict.high { border-left-color: var(--red); }
    .verdict.medium { border-left-color: var(--amber); }
    .verdict.low { border-left-color: var(--green); }
    .verdict strong { display: block; font-size: 20px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin-top: 18px;
      position: relative;
    }
    .metric {
      border: 1px solid rgba(142, 123, 91, .35);
      background: rgba(255, 255, 255, .72);
      padding: 12px;
      min-height: 84px;
    }
    .metric b {
      display: block;
      font-family: var(--sans);
      font-size: 24px;
      line-height: 1.1;
    }
    .metric span {
      color: var(--muted);
      font-family: var(--sans);
      font-size: 12px;
    }
    .toolbar {
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin: 16px 0;
      padding: 12px;
      border: 1px solid var(--line);
      background: rgba(255, 253, 247, .94);
      backdrop-filter: blur(10px);
      box-shadow: 0 8px 24px rgba(45, 35, 20, .08);
      font-family: var(--sans);
    }
    .filter-btn, .plain-btn, .copy-btn {
      border: 1px solid var(--line-strong);
      background: #fffaf0;
      color: var(--ink);
      cursor: pointer;
      padding: 8px 11px;
      font: 600 13px var(--sans);
    }
    .filter-btn.active, .plain-btn:hover, .copy-btn:hover {
      background: var(--brand);
      border-color: var(--brand);
      color: white;
    }
    .search {
      min-width: min(320px, 100%);
      flex: 1;
      border: 1px solid var(--line-strong);
      background: white;
      padding: 9px 12px;
      font: 14px var(--sans);
    }
    .visible-count { color: var(--muted); font-size: 13px; }
    .review-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 18px;
      align-items: start;
    }
    .paper-panel, .side-card, .guidance-panel {
      border: 1px solid var(--line-strong);
      background: var(--paper);
      box-shadow: var(--shadow);
    }
    .panel-title {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin: 0;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--paper-2);
      font: 700 18px var(--sans);
    }
    .document {
      padding: 18px;
      counter-reset: para;
    }
    .para-card {
      position: relative;
      margin: 0 0 14px;
      padding: 14px 14px 14px 96px;
      border: 1px solid transparent;
      border-left: 5px solid transparent;
      background: rgba(255, 255, 255, .58);
      page-break-inside: avoid;
      transition: transform .16s ease, box-shadow .16s ease, background .16s ease;
    }
    .para-card:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 22px rgba(45, 35, 20, .08);
    }
    .para-card mark { background: #ffef99; color: inherit; padding: 0 .1em; }
    .para-index {
      position: absolute;
      left: 14px;
      top: 16px;
      color: var(--muted);
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.4;
    }
    .para-score {
      display: block;
      margin-top: 4px;
      font-weight: 800;
    }
    .para-text {
      white-space: pre-wrap;
      font-size: 16px;
    }
    .para-meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
      font-family: var(--sans);
      font-size: 12px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      border: 1px solid currentColor;
      border-radius: 999px;
      padding: 2px 8px;
      line-height: 1.5;
      background: rgba(255, 255, 255, .72);
    }
    .risk-ai {
      color: var(--red);
      background: var(--red-bg);
      border-color: rgba(200, 33, 36, .3);
      border-left-color: var(--red);
      font-weight: 700;
      text-decoration-line: underline;
      text-decoration-thickness: 1.4px;
      text-underline-offset: 4px;
    }
    .risk-review {
      color: var(--amber);
      background: var(--amber-bg);
      border-color: rgba(166, 95, 0, .28);
      border-left-color: var(--amber);
    }
    .risk-normal {
      border-left-color: #b9c9b1;
    }
    .engine-detail {
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px dashed rgba(109, 91, 61, .35);
      color: #526057;
      font: 12px/1.7 var(--sans);
    }
    body.hide-engine-details .engine-detail { display: none; }
    .sidebar {
      position: sticky;
      top: 76px;
      max-height: calc(100vh - 92px);
      overflow: auto;
    }
    .side-card {
      margin-bottom: 14px;
      padding: 14px;
      box-shadow: none;
    }
    .side-card h3 {
      margin: 0 0 10px;
      font: 700 16px var(--sans);
    }
    .risk-link {
      display: grid;
      grid-template-columns: 48px 64px 1fr;
      gap: 8px;
      align-items: start;
      width: 100%;
      border: 0;
      border-bottom: 1px solid #ece3d3;
      background: transparent;
      color: inherit;
      cursor: pointer;
      padding: 8px 0;
      text-align: left;
      font: 13px/1.45 var(--sans);
    }
    .risk-link:hover { color: var(--red); }
    .priority-item {
      border-left: 3px solid var(--red);
      padding: 8px 0 8px 10px;
      margin: 0 0 8px;
      font: 13px/1.55 var(--sans);
    }
    .bars { display: grid; gap: 8px; }
    .bar-row { font: 13px var(--sans); }
    .bar-label { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 3px; }
    .bar-track { height: 8px; background: #efe7d8; overflow: hidden; }
    .bar-fill { height: 100%; background: var(--brand); }
    .bar-fill.high { background: var(--red); }
    .bar-fill.review { background: var(--amber); }
    table {
      width: 100%;
      border-collapse: collapse;
      font: 14px/1.6 var(--sans);
    }
    th, td {
      border-bottom: 1px solid #ece3d3;
      padding: 10px;
      text-align: left;
      vertical-align: top;
    }
    th {
      background: #f7efe0;
      color: #5f553f;
      font-weight: 700;
    }
    .guidance-panel { margin-top: 18px; }
    .guidance-body { padding: 0 18px 18px; overflow-x: auto; }
    .advice-text { color: #36463b; }
    .muted { color: var(--muted); }
    .empty-state {
      padding: 18px;
      color: var(--muted);
      font-family: var(--sans);
    }
    @media (max-width: 1100px) {
      .hero-grid, .review-grid { grid-template-columns: 1fr; }
      .sidebar { position: static; max-height: none; }
      .metrics { grid-template-columns: repeat(3, 1fr); }
    }
    @media (max-width: 720px) {
      .page { width: calc(100vw - 16px); margin: 8px auto 24px; }
      .hero { padding: 20px 18px; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .toolbar { position: static; }
      .para-card { padding-left: 14px; }
      .para-index { position: static; display: block; margin-bottom: 8px; }
      .risk-link { grid-template-columns: 44px 56px 1fr; }
    }
    @media print {
      body { background: white; }
      .page { width: auto; margin: 0; }
      .hero, .paper-panel, .side-card, .guidance-panel { box-shadow: none; border-color: #b8b8b8; }
      .toolbar, .copy-btn, .plain-btn, script { display: none !important; }
      .review-grid { grid-template-columns: 1fr; }
      .sidebar { position: static; max-height: none; overflow: visible; }
      .para-card, tr, .side-card { page-break-inside: avoid; break-inside: avoid; }
      .risk-ai { color: #b00000 !important; text-decoration-line: underline; }
      .risk-review { color: #8a4b00 !important; }
    }

    /* Modern report skin: denser, calmer, and aligned with the WebUI. */
    :root {
      --paper: #ffffff;
      --paper-2: #f8fafc;
      --ink: #172033;
      --muted: #667085;
      --line: #d8dee8;
      --line-strong: #c6ceda;
      --brand: #2459a6;
      --brand-soft: #edf4ff;
      --red: #b42318;
      --red-bg: #fff1f0;
      --amber: #b76e00;
      --amber-bg: #fff7e6;
      --green: #1f7a5a;
      --green-bg: #ecfdf3;
      --shadow: 0 12px 30px rgba(15, 23, 42, .08);
      --mono: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      --serif: "Noto Sans SC", "Source Han Sans SC", "Microsoft YaHei", sans-serif;
      --sans: "Noto Sans SC", "Source Han Sans SC", "Microsoft YaHei", sans-serif;
    }
    body {
      background: #f4f6f8;
      font-family: var(--sans);
      line-height: 1.62;
    }
    .page {
      width: min(1480px, calc(100vw - 32px));
      margin: 16px auto 40px;
    }
    .hero {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #122033;
      color: #fff;
      padding: 22px 24px;
    }
    .hero:before { display: none; }
    .eyebrow {
      color: #a8c7fa;
      letter-spacing: 0;
      text-transform: none;
    }
    h1 {
      margin: 6px 0 8px;
      font-size: clamp(26px, 3.4vw, 40px);
      letter-spacing: 0;
    }
    .meta { color: #c7d2e3; }
    .verdict {
      border: 1px solid rgba(255, 255, 255, .16);
      border-left: 5px solid #a8c7fa;
      border-radius: 8px;
      background: rgba(255, 255, 255, .08);
      color: #fff;
    }
    .metrics {
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 8px;
    }
    .metric {
      border-color: rgba(255, 255, 255, .14);
      border-radius: 8px;
      background: rgba(255, 255, 255, .08);
      min-height: 76px;
    }
    .metric b { color: #fff; font-size: 24px; }
    .metric span { color: #c7d2e3; }
    .toolbar {
      border-radius: 8px;
      background: rgba(255, 255, 255, .96);
      box-shadow: 0 8px 20px rgba(15, 23, 42, .08);
    }
    .filter-btn, .plain-btn, .copy-btn {
      border-color: var(--line);
      border-radius: 6px;
      background: #fff;
    }
    .filter-btn.active, .plain-btn:hover, .copy-btn:hover {
      background: var(--brand);
      border-color: var(--brand);
    }
    .search {
      border-color: var(--line);
      border-radius: 6px;
    }
    .review-grid {
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 16px;
    }
    .paper-panel, .side-card, .guidance-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .panel-title {
      background: var(--paper-2);
      border-bottom-color: var(--line);
      font-size: 16px;
    }
    .document { padding: 14px; }
    .para-card {
      border-color: var(--line);
      border-left-width: 5px;
      border-radius: 8px;
      background: #fff;
      padding: 14px 14px 14px 104px;
      box-shadow: none;
    }
    .para-card:hover {
      transform: none;
      box-shadow: 0 8px 20px rgba(15, 23, 42, .08);
    }
    .para-text { font-size: 15px; line-height: 1.7; }
    .para-score { color: var(--brand); font-size: 14px; }
    .risk-ai {
      text-decoration-line: none;
      border-color: rgba(180, 35, 24, .22);
      border-left-color: var(--red);
    }
    .risk-review {
      border-color: rgba(183, 110, 0, .24);
      border-left-color: var(--amber);
    }
    .risk-normal {
      border-left-color: var(--green);
    }
    .pill {
      border: 1px solid var(--line);
      background: #f8fafc;
    }
    .engine-detail {
      border-top-color: var(--line);
      color: var(--muted);
    }
    .sidebar {
      top: 72px;
    }
    .side-card {
      box-shadow: none;
      padding: 14px;
    }
    .risk-link {
      grid-template-columns: 48px 68px 1fr;
      border-bottom-color: var(--line);
    }
    .priority-item {
      border-left-color: var(--red);
      background: #fff;
    }
    .bar-track {
      border-radius: 999px;
      background: #e8edf3;
    }
    .bar-fill {
      border-radius: inherit;
      background: var(--green);
    }
    table {
      font-size: 13px;
    }
    th, td {
      border-bottom-color: var(--line);
      padding: 10px 12px;
    }
    th {
      background: var(--paper-2);
      color: var(--muted);
    }
    .guidance-body {
      padding: 0 14px 14px;
    }
    @media (max-width: 1100px) {
      .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .review-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) {
      .page { width: calc(100vw - 16px); margin: 8px auto 24px; }
      .hero { padding: 18px; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .toolbar { position: static; align-items: stretch; }
      .search { flex-basis: 100%; }
      .para-card { padding-left: 14px; }
      .risk-link { grid-template-columns: 44px 62px 1fr; }
    }
"""


REPORT_JS = r"""
(function () {
  var activeRisk = "all";
  var searchInput = document.getElementById("report-search");
  var countNode = document.getElementById("visible-count");
  var paragraphs = Array.prototype.slice.call(document.querySelectorAll(".para-card"));

  function normalize(text) {
    return (text || "").toLowerCase().trim();
  }

  function applyFilters() {
    var query = normalize(searchInput ? searchInput.value : "");
    var visible = 0;
    paragraphs.forEach(function (node) {
      var risk = node.getAttribute("data-risk") || "unknown";
      var text = normalize(node.getAttribute("data-text") || node.textContent);
      var riskMatch = activeRisk === "all" || risk === activeRisk;
      var textMatch = !query || text.indexOf(query) !== -1;
      var show = riskMatch && textMatch;
      node.style.display = show ? "" : "none";
      if (show) visible += 1;
    });
    if (countNode) countNode.textContent = "显示 " + visible + " / " + paragraphs.length + " 段";
  }

  Array.prototype.slice.call(document.querySelectorAll(".filter-btn")).forEach(function (button) {
    button.addEventListener("click", function () {
      activeRisk = button.getAttribute("data-filter") || "all";
      document.querySelectorAll(".filter-btn.active").forEach(function (item) { item.classList.remove("active"); });
      button.classList.add("active");
      applyFilters();
    });
  });

  if (searchInput) searchInput.addEventListener("input", applyFilters);

  Array.prototype.slice.call(document.querySelectorAll(".risk-link")).forEach(function (button) {
    button.addEventListener("click", function () {
      var target = document.getElementById(button.getAttribute("data-target"));
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "center" });
        if (target.animate) {
          target.animate([{ outlineColor: "#c82124" }, { outlineColor: "transparent" }], { duration: 1100 });
        }
      }
    });
  });

  Array.prototype.slice.call(document.querySelectorAll(".copy-btn")).forEach(function (button) {
    button.addEventListener("click", function () {
      var text = button.getAttribute("data-copy") || "";
      function done() {
        var old = button.textContent;
        button.textContent = "已复制";
        setTimeout(function () { button.textContent = old; }, 1200);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(fallbackCopy);
      } else {
        fallbackCopy();
      }
      function fallbackCopy() {
        var area = document.createElement("textarea");
        area.value = text;
        area.style.position = "fixed";
        area.style.opacity = "0";
        document.body.appendChild(area);
        area.select();
        try { document.execCommand("copy"); } catch (e) {}
        document.body.removeChild(area);
        done();
      }
    });
  });

  var detailToggle = document.getElementById("toggle-engine-details");
  if (detailToggle) {
    detailToggle.addEventListener("click", function () {
      document.body.classList.toggle("hide-engine-details");
      detailToggle.textContent = document.body.classList.contains("hide-engine-details") ? "显示引擎明细" : "隐藏引擎明细";
    });
  }

  applyFilters();
})();
"""


def generate_html_report(
    paragraphs: List[Dict],
    results: List[Dict],
    engine_status: Dict,
    output_path: str,
    source_file: str = "",
):
    """Generate a single-file review report with highlighted paragraphs."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    threshold = float(engine_status.get("threshold", 0.5))
    rows = _build_rows(paragraphs, results, threshold)
    valid_rows = [row for row in rows if row["score"] is not None]
    ai_rows = [row for row in valid_rows if row["risk"] == "ai"]
    review_rows = [row for row in valid_rows if row["risk"] == "review"]
    normal_rows = [row for row in valid_rows if row["risk"] == "normal"]
    unknown_count = len(rows) - len(valid_rows)
    scores = [row["score"] for row in valid_rows]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0
    ai_ratio = len(ai_rows) / len(valid_rows) if valid_rows else 0.0
    verdict_class, verdict_title, verdict_text = _overall_verdict(ai_ratio, max_score, avg_score)
    guidance_rows = sorted(
        [row for row in rows if row["score"] is not None and row["score"] >= _review_cutoff(threshold)],
        key=lambda row: row["score"],
        reverse=True,
    )

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AIGC 可视化检测报告</title>
  <style>
{REPORT_CSS}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">AIGC Review Report</div>
          <h1>AIGC 可视化检测报告</h1>
          <div class="meta">
            检测文件：{_esc(source_file)} · 检测时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} · 引擎：{_esc(engine_status.get("mode", ""))}
          </div>
        </div>
        <div class="verdict {verdict_class}">
          <strong>{_esc(verdict_title)}</strong>
          <span>{_esc(verdict_text)}</span>
        </div>
      </div>
      <section class="metrics" aria-label="摘要指标">
        <div class="metric"><b>{len(rows)}</b><span>总段落</span></div>
        <div class="metric"><b>{len(valid_rows)}</b><span>有效检测</span></div>
        <div class="metric"><b>{len(ai_rows)}</b><span>高风险段落</span></div>
        <div class="metric"><b>{avg_score * 100:.1f}%</b><span>平均 AIGC 分数</span></div>
        <div class="metric"><b>{threshold * 100:.0f}%</b><span>判定阈值</span></div>
        <div class="metric"><b>{_esc(engine_status.get("api_concurrency", 1))}</b><span>API 并发数</span></div>
      </section>
    </header>

    <nav class="toolbar" aria-label="报告筛选工具">
      <button class="filter-btn active" data-filter="all">全部</button>
      <button class="filter-btn" data-filter="ai">高风险</button>
      <button class="filter-btn" data-filter="review">需复核</button>
      <button class="filter-btn" data-filter="normal">正常</button>
      <input id="report-search" class="search" type="search" placeholder="搜索原文关键词">
      <button id="toggle-engine-details" class="plain-btn" type="button">隐藏引擎明细</button>
      <span id="visible-count" class="visible-count"></span>
    </nav>

    <section class="review-grid">
      <article class="paper-panel">
        <h2 class="panel-title"><span>原文内容批注</span><span class="muted">红色为高风险，黄色为需复核</span></h2>
        <div class="document">
          {_render_annotated_text(rows)}
        </div>
      </article>
      <aside class="sidebar" aria-label="审阅侧栏">
        {_render_risk_distribution(len(ai_rows), len(review_rows), len(normal_rows), unknown_count, len(rows))}
        {_render_revision_index(ai_rows, threshold)}
        {_render_priority_list(guidance_rows)}
        {_render_engine_status(engine_status)}
      </aside>
    </section>

    <section class="guidance-panel">
      <h2 class="panel-title"><span>逐段修改指导</span><span class="muted">按风险分数从高到低排序</span></h2>
      <div class="guidance-body">
        {_render_guidance_table(guidance_rows, threshold)}
      </div>
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

    logger.info("HTML 可视化报告已保存: %s", output_path)
    return output_path


def _build_rows(paragraphs: List[Dict], results: List[Dict], threshold: float) -> List[Dict]:
    rows = []
    for i, (para, result) in enumerate(zip(paragraphs, results), start=1):
        score = result.get("aigc_score")
        risk, label = _risk_level(score, threshold)
        advice = _revision_advice(score, threshold) if score is not None else "未得到有效分数，建议结合上下文人工复核。"
        rows.append(
            {
                "id": f"para-{para.get('index', i)}",
                "index": para.get("index", i),
                "text": para.get("text", ""),
                "score": score,
                "risk": risk,
                "risk_label": label,
                "label": result.get("label", "unknown"),
                "confidence": result.get("confidence", 0),
                "method": result.get("method", ""),
                "engine_summary": _engine_summary(result.get("engine_results", {})),
                "advice": advice,
            }
        )
    return rows


def _risk_level(score, threshold: float) -> Tuple[str, str]:
    if score is None:
        return "unknown", "未知"
    if score > threshold:
        return "ai", "高风险"
    if score >= _review_cutoff(threshold):
        return "review", "需复核"
    return "normal", "正常"


def _review_cutoff(threshold: float) -> float:
    return max(0.4, threshold - 0.1)


def _overall_verdict(ai_ratio: float, max_score: float, avg_score: float) -> Tuple[str, str, str]:
    if ai_ratio >= 0.2 or max_score >= 0.8:
        return "high", "整体风险偏高", "优先处理高风险索引中的段落，再复核黄色段落。"
    if ai_ratio > 0 or avg_score >= 0.4 or max_score >= 0.6:
        return "medium", "存在局部风险", "建议按修改清单逐段优化，并保留具体实验和个人判断。"
    return "low", "整体风险较低", "当前结果以人工风格为主，仍建议抽查接近阈值段落。"


def _render_annotated_text(rows: List[Dict]) -> str:
    if not rows:
        return '<p class="empty-state">没有可显示的段落。</p>'
    parts = []
    for row in rows:
        score_text = _score_text(row["score"])
        risk_cls = {
            "ai": " risk-ai",
            "review": " risk-review",
            "normal": " risk-normal",
        }.get(row["risk"], "")
        engine = row["engine_summary"] or "无逐引擎明细"
        parts.append(
            f'<section id="{_esc(row["id"])}" class="para-card{risk_cls}" data-risk="{_esc(row["risk"])}" data-text="{_esc(row["text"])}">'
            f'<span class="para-index">段落 {row["index"]}<span class="para-score">{_esc(score_text)}</span></span>'
            f'<div class="para-text">{_esc(row["text"])}</div>'
            f'<div class="para-meta"><span class="pill">{_esc(row["risk_label"])}</span><span class="pill">label: {_esc(row["label"])}</span><span class="pill">method: {_esc(row["method"] or "-")}</span></div>'
            f'<div class="engine-detail">{_esc(engine)}</div>'
            '</section>'
        )
    return "\n".join(parts)


def _render_engine_status(engine_status: Dict) -> str:
    rows = []
    for name, label in ENGINE_LABELS.items():
        available = "可用" if engine_status.get(f"{name}_available") else "不可用"
        weight = float(engine_status.get(f"{name}_weight", 0) or 0)
        rows.append(f"<tr><td>{_esc(label)}</td><td>{_esc(available)}</td><td>{weight:.0%}</td></tr>")
    return (
        '<div class="side-card"><h3>检测引擎</h3><table>'
        "<tr><th>引擎</th><th>状态</th><th>权重</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )


def _render_risk_distribution(ai_count: int, review_count: int, normal_count: int, unknown_count: int, total: int) -> str:
    items = [
        ("高风险", ai_count, "high"),
        ("需复核", review_count, "review"),
        ("正常", normal_count, "normal"),
        ("未知", unknown_count, "unknown"),
    ]
    bars = []
    for label, count, cls in items:
        pct = (count / total * 100) if total else 0
        bars.append(
            f'<div class="bar-row"><div class="bar-label"><span>{_esc(label)}</span><span>{count} 段 · {pct:.1f}%</span></div>'
            f'<div class="bar-track"><div class="bar-fill {cls}" style="width:{pct:.1f}%"></div></div></div>'
        )
    return '<div class="side-card"><h3>风险分布</h3><div class="bars">' + "".join(bars) + "</div></div>"


def _render_revision_index(high_risk_rows: List[Dict], threshold: float) -> str:
    if not high_risk_rows:
        return '<div class="side-card"><h3>高风险索引</h3><p class="muted">没有超过阈值的段落。</p></div>'

    items = []
    for row in high_risk_rows[:24]:
        items.append(
            f'<button class="risk-link" type="button" data-target="{_esc(row["id"])}">'
            f'<strong>#{row["index"]}</strong><span>{row["score"] * 100:.1f}%</span><span>{_esc(_shorten(row["text"], 48))}</span></button>'
        )
    return (
        f'<div class="side-card"><h3>高风险索引 <span class="muted">阈值 {threshold * 100:.0f}%</span></h3>'
        + "".join(items)
        + "</div>"
    )


def _render_priority_list(guidance_rows: List[Dict]) -> str:
    if not guidance_rows:
        return '<div class="side-card"><h3>修改优先级清单</h3><p class="muted">没有需要重点修改的段落。</p></div>'
    items = []
    for row in guidance_rows[:8]:
        items.append(
            f'<div class="priority-item"><strong>段落 {row["index"]} · {_score_text(row["score"])}</strong><br>{_esc(_shorten(row["advice"], 72))}</div>'
        )
    return '<div class="side-card"><h3>修改优先级清单</h3>' + "".join(items) + "</div>"


def _render_guidance_table(rows: List[Dict], threshold: float) -> str:
    if not rows:
        return '<p class="empty-state">没有需要重点修改的段落。</p>'
    table_rows = []
    for row in rows:
        engine_text = row["engine_summary"] or "无逐引擎明细"
        copy_text = f"段落 {row['index']}（AIGC {_score_text(row['score'])}）：{row['advice']}"
        table_rows.append(
            "<tr>"
            f"<td>{row['index']}</td>"
            f"<td>{_score_text(row['score'])}</td>"
            f"<td>{_esc(row['risk_label'])}</td>"
            f"<td>{_esc(_shorten(row['text'], 140))}</td>"
            f"<td><div class=\"advice-text\">{_esc(row['advice'])}</div><p class=\"suggestion muted\">{_esc(engine_text)}</p></td>"
            f"<td><button class=\"copy-btn\" type=\"button\" data-copy=\"{_esc(copy_text)}\">复制建议</button></td>"
            "</tr>"
        )
    return (
        "<table><tr><th>段落</th><th>AIGC</th><th>风险</th><th>原文预览</th><th>修改建议</th><th>操作</th></tr>"
        + "".join(table_rows)
        + "</table>"
    )


def _engine_summary(engine_results: Dict) -> str:
    parts = []
    for name, result in engine_results.items():
        score = result.get("aigc_score")
        if score is None:
            continue
        method = result.get("method") or result.get("mode") or ""
        label = ENGINE_LABELS.get(name, name)
        parts.append(f"{label}: {score * 100:.1f}% {method}".strip())
    return "；".join(parts)


def _revision_advice(score, threshold: float) -> str:
    if score is None:
        return "未得到有效分数，建议结合上下文人工复核。"
    if score > max(0.7, threshold + 0.2):
        return "高风险：建议重写该段，加入具体实验过程、数据来源、个人判断或论文上下文，减少模板化总结句。"
    if score > threshold:
        return "超过阈值：建议拆分长句，替换泛化表达，补充具体对象、参数、问题原因和处理过程。"
    return "接近阈值：建议人工复核，优先改写连续抽象表述和过于整齐的并列结构。"


def _score_text(score) -> str:
    return "未知" if score is None else f"{score * 100:.1f}%"


def _shorten(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _esc(value) -> str:
    return html.escape(str(value), quote=True)
