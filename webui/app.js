const state = {
  rows: [],
  file: null,
  result: null,
  polling: null,
  config: null,
};

const $ = (id) => document.getElementById(id);

const statusPill = $("statusPill");
const fileInput = $("fileInput");
const fileName = $("fileName");
const dropZone = $("dropZone");
const resultRows = $("resultRows");
const labelFilter = $("labelFilter");
const filterInput = $("filterInput");
const progressFill = $("progressFill");
const progressText = $("progressText");
const runStage = $("runStage");
const runMessage = $("runMessage");

function setStatus(text, mode = "") {
  statusPill.textContent = text;
  statusPill.className = `status-pill ${mode}`.trim();
}

function getOptions() {
  const threshold = $("threshold").value;
  const options = {
    threshold: threshold === "" ? null : Number(threshold),
    min_length: Number($("minLength").value || 30),
  };
  if ($("taskMode").value === "refine") {
    options.refine_threshold = Number($("refineThreshold").value || 0.4);
    options.max_rounds = Number($("maxRounds").value || 2);
    options.detect_concurrency = Number($("detectConcurrency").value || 4);
  }
  return options;
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function formatRatio(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "0%";
  return `${value.toFixed(1)}%`;
}

function labelText(label) {
  if (label === "ai") return "AI";
  if (label === "human") return "人工";
  return "未知";
}

function updateSummary(data) {
  const summary = data.summary || {};
  $("totalMetric").textContent = summary.total ?? 0;
  $("aiMetric").textContent = summary.ai ?? 0;
  $("ratioMetric").textContent = formatRatio(summary.ai_ratio ?? 0);
  $("avgMetric").textContent = formatPercent(summary.avg_score);
  $("changedMetric").textContent = summary.changed ?? 0;
  $("roundsMetric").textContent = summary.rounds ?? 0;
  $("engineMeta").textContent = `${summary.engine || "-"} / 阈值 ${summary.threshold ?? "-"}`;
}

function updateProgress(job) {
  const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
  progressFill.style.width = `${progress}%`;
  progressText.textContent = `${progress.toFixed(0)}%`;
  runStage.textContent = job.stage || "运行中";
  runMessage.textContent = job.message || job.title || "";
}

function updateBars(rows) {
  const buckets = [
    [0, 0.1, "0-10%"],
    [0.1, 0.2, "10-20%"],
    [0.2, 0.3, "20-30%"],
    [0.3, 0.4, "30-40%"],
    [0.4, 0.5, "40-50%"],
    [0.5, 0.7, "50-70%"],
    [0.7, 1.01, "70-100%"],
  ];
  const scores = rows.map((r) => r.aigc_score).filter((v) => typeof v === "number");
  const total = Math.max(scores.length, 1);
  $("bars").innerHTML = buckets.map(([lo, hi, name]) => {
    const count = scores.filter((score) => score >= lo && score < hi).length;
    const width = (count / total) * 100;
    return `
      <div class="bar-row">
        <span>${name}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
        <span>${count}</span>
      </div>
    `;
  }).join("");
}

function renderRows() {
  const label = labelFilter.value;
  const query = filterInput.value.trim().toLowerCase();
  const rows = state.rows.filter((row) => {
    if (label !== "all" && row.label !== label) return false;
    if (query && !row.text.toLowerCase().includes(query)) return false;
    return true;
  });

  if (!rows.length) {
    resultRows.innerHTML = '<tr class="empty-row"><td colspan="7">没有匹配结果</td></tr>';
    return;
  }

  resultRows.innerHTML = rows.map((row) => `
    <tr>
      <td>${row.index ?? ""}</td>
      <td class="score">${formatPercent(row.aigc_score)}</td>
      <td><span class="label ${row.label || "unknown"}">${labelText(row.label)}</span></td>
      <td>${row.method || "-"}</td>
      <td class="engine-cell">${renderEngineScores(row.engine_scores || [])}</td>
      <td class="refine-cell">${renderRefineInfo(row)}</td>
      <td class="paragraph-cell">${escapeHtml(row.text)}</td>
    </tr>
  `).join("");
}

function applyResult(data) {
  state.result = data;
  state.rows = data.paragraphs || [];
  updateSummary(data);
  updateBars(state.rows);
  renderRows();
  $("exportJsonBtn").disabled = !state.rows.length;
  $("exportCsvBtn").disabled = !state.rows.length;
}

function renderEngineScores(scores) {
  if (!scores.length) return '<span class="muted-inline">-</span>';
  return scores.map((item) => {
    const score = item.score === null || item.score === undefined ? "-" : `${(item.score * 100).toFixed(1)}%`;
    return `<span class="engine-chip" title="${escapeHtml(item.method || "")}">${escapeHtml(item.label || item.engine)} ${score}</span>`;
  }).join("");
}

function renderRefineInfo(row) {
  if (!row.refined && !row.rounds) return '<span class="muted-inline">-</span>';
  const before = row.aigc_before === null || row.aigc_before === undefined ? "-" : `${(row.aigc_before * 100).toFixed(1)}%`;
  const after = row.aigc_after === null || row.aigc_after === undefined ? "-" : `${(row.aigc_after * 100).toFixed(1)}%`;
  const changed = row.changed ? "已改" : "未改";
  const refined = row.refined && row.refined !== row.text
    ? `<details><summary>${changed} / ${row.rounds || 0}轮 / ${before}→${after}</summary><div class="refined-text">${escapeHtml(row.refined)}</div></details>`
    : `${changed} / ${row.rounds || 0}轮 / ${before}→${after}`;
  return refined;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function submitJob(url, options) {
  setStatus("检测中", "busy");
  updateProgress({ progress: 2, stage: "提交任务", message: "正在发送请求" });
  toggleButtons(true);
  try {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "请求失败");
    if (!data.job_id) throw new Error("服务端未返回任务 ID");
    await pollJob(data.job_id);
  } catch (error) {
    setStatus("失败", "error");
    runStage.textContent = "失败";
    runMessage.textContent = error.message;
    resultRows.innerHTML = `<tr class="empty-row"><td colspan="7">${escapeHtml(error.message)}</td></tr>`;
    toggleButtons(false);
  }
}

async function pollJob(jobId) {
  if (state.polling) clearInterval(state.polling);
  return new Promise((resolve, reject) => {
    state.polling = setInterval(async () => {
      try {
        const response = await fetch(`/api/jobs/${jobId}`);
        const job = await response.json();
        if (!response.ok) throw new Error(job.error || "任务查询失败");
        updateProgress(job);
        if (job.state === "done") {
          clearInterval(state.polling);
          state.polling = null;
          applyResult(job.result);
          setStatus("完成");
          toggleButtons(false);
          resolve(job.result);
        } else if (job.state === "error") {
          clearInterval(state.polling);
          state.polling = null;
          throw new Error(job.error || "检测失败");
        }
      } catch (error) {
        clearInterval(state.polling);
        state.polling = null;
        setStatus("失败", "error");
        runStage.textContent = "失败";
        runMessage.textContent = error.message;
        toggleButtons(false);
        reject(error);
      }
    }, 500);
  });
}

function toggleButtons(disabled) {
  $("detectFileBtn").disabled = disabled;
  $("detectTextBtn").disabled = disabled;
}

async function detectText() {
  const text = $("textInput").value.trim();
  if (!text) {
    setStatus("请输入文本", "error");
    return;
  }
  const endpoint = $("taskMode").value === "refine" ? "/api/refine-text" : "/api/detect-text";
  await submitJob(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, ...getOptions() }),
  });
}

async function detectFile() {
  if (!state.file) {
    setStatus("请选择文件", "error");
    return;
  }
  const form = new FormData();
  form.append("file", state.file);
  const options = getOptions();
  if (options.threshold !== null) form.append("threshold", String(options.threshold));
  if ($("taskMode").value === "refine") {
    form.append("refine_threshold", String(options.refine_threshold));
    form.append("max_rounds", String(options.max_rounds));
    form.append("detect_concurrency", String(options.detect_concurrency));
  }
  const endpoint = $("taskMode").value === "refine" ? "/api/refine-file" : "/api/detect-file";
  await submitJob(endpoint, {
    method: "POST",
    body: form,
  });
}

function download(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function exportJson() {
  if (!state.result) return;
  download("aigc-detect-result.json", JSON.stringify(state.result, null, 2), "application/json;charset=utf-8");
}

function exportCsv() {
  if (!state.rows.length) return;
  const headers = ["index", "score", "label", "method", "engine_scores", "refined", "text"];
  const lines = [headers.join(",")];
  for (const row of state.rows) {
    const engineScores = (row.engine_scores || []).map((item) => `${item.label}:${item.score_text || "-"}`).join(";");
    const values = [
      row.index ?? "",
      row.aigc_score ?? "",
      row.label ?? "",
      row.method ?? "",
      engineScores,
      row.refined ?? "",
      row.text ?? "",
    ].map(csvEscape);
    lines.push(values.join(","));
  }
  download("aigc-detect-result.csv", lines.join("\n"), "text/csv;charset=utf-8");
}

function csvEscape(value) {
  const text = String(value).replaceAll('"', '""');
  return `"${text}"`;
}

function setFile(file) {
  state.file = file;
  fileName.textContent = file ? file.name : "未选择文件";
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".tab-page").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    $(`${button.dataset.tab}Page`).classList.add("active");
  });
});

fileInput.addEventListener("change", () => setFile(fileInput.files[0] || null));
$("detectTextBtn").addEventListener("click", detectText);
$("detectFileBtn").addEventListener("click", detectFile);
$("exportJsonBtn").addEventListener("click", exportJson);
$("exportCsvBtn").addEventListener("click", exportCsv);
$("saveConfigBtn").addEventListener("click", saveConfig);
$("taskMode").addEventListener("change", syncMode);
labelFilter.addEventListener("change", renderRows);
filterInput.addEventListener("input", renderRows);

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) setFile(file);
});

updateBars([]);
loadConfig();
syncMode();

function syncMode() {
  const refine = $("taskMode").value === "refine";
  $("refineOptions").classList.toggle("active", refine);
  $("detectFileBtn").textContent = refine ? "润色文件" : "检测文件";
  $("detectTextBtn").textContent = refine ? "润色文本" : "检测文本";
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    const config = await response.json();
    if (!response.ok) throw new Error(config.error || "读取配置失败");
    state.config = config;
    fillConfig(config);
  } catch (error) {
    setStatus("配置读取失败", "error");
  }
}

function fillConfig(config) {
  const openai = config.openai_api || {};
  const refiner = config.refiner_api || {};
  const threshold = config.threshold || {};
  const perf = config.performance || {};
  $("cfgOpenaiBase").value = openai.api_base || "";
  $("cfgOpenaiKey").value = openai.api_key || "";
  $("cfgOpenaiModel").value = openai.model || "";
  $("cfgRefinerBase").value = refiner.api_base || "";
  $("cfgRefinerKey").value = refiner.api_key || "";
  $("cfgRefinerModel").value = refiner.model || "";
  $("cfgRefinerTemperature").value = refiner.temperature ?? 0.3;
  $("cfgApiConcurrency").value = perf.api_concurrency || 1;
  $("cfgAigcThreshold").value = threshold.aigc_threshold ?? 0.5;
  renderWeights((config.engine || {}).ensemble || {});
}

function renderWeights(weights) {
  const names = ["fengci", "hc3", "openai", "binoculars", "lastde", "local_logprob"];
  $("weightsGrid").innerHTML = names.map((name) => `
    <label>${name}<input data-weight="${name}" type="number" min="0" max="1" step="0.01" value="${weights[`${name}_weight`] ?? 0}"></label>
  `).join("");
}

async function saveConfig() {
  const ensemble = {};
  document.querySelectorAll("[data-weight]").forEach((input) => {
    ensemble[`${input.dataset.weight}_weight`] = Number(input.value || 0);
  });
  const payload = {
    engine: { default: "ensemble", ensemble },
    openai_api: {
      api_base: $("cfgOpenaiBase").value,
      api_key: $("cfgOpenaiKey").value,
      model: $("cfgOpenaiModel").value,
      strategy: "perplexity",
    },
    refiner_api: {
      api_base: $("cfgRefinerBase").value,
      api_key: $("cfgRefinerKey").value,
      model: $("cfgRefinerModel").value,
      temperature: Number($("cfgRefinerTemperature").value || 0.3),
    },
    threshold: { aigc_threshold: Number($("cfgAigcThreshold").value || 0.5) },
    performance: { api_concurrency: Number($("cfgApiConcurrency").value || 1) },
  };
  setStatus("保存配置", "busy");
  const response = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus("保存失败", "error");
    return;
  }
  state.config = data.config;
  setStatus("配置已保存");
}
