import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  Download,
  FileText,
  Gauge,
  Layers3,
  PauseCircle,
  Play,
  RefreshCw,
  Save,
  Search,
  Settings2,
  ShieldAlert,
  Sparkles,
  UploadCloud,
  Wand2,
} from "lucide-react";
import "./styles.css";

const ENGINE_NAMES = ["fengci", "hc3", "openai", "binoculars", "lastde", "local_logprob"];
const ENGINE_LABELS = {
  fengci: "FengCi0",
  hc3: "HC3+m3e",
  openai: "OpenAI API",
  binoculars: "Binoculars",
  local_logprob: "Local Logprob",
  lastde: "LastDe",
};

const INITIAL_PROGRESS = {
  progress: 0,
  stage: "等待任务",
  message: "选择文件或粘贴文本后开始检测",
  state: "idle",
};

function App() {
  const [taskMode, setTaskMode] = useState("detect");
  const [sourceMode, setSourceMode] = useState("file");
  const [threshold, setThreshold] = useState(0.5);
  const [minLength, setMinLength] = useState(30);
  const [refineThreshold, setRefineThreshold] = useState(0.4);
  const [maxRounds, setMaxRounds] = useState(2);
  const [detectConcurrency, setDetectConcurrency] = useState(4);
  const [textInput, setTextInput] = useState("");
  const [file, setFile] = useState(null);
  const [config, setConfig] = useState(null);
  const [configOpen, setConfigOpen] = useState(false);
  const [result, setResult] = useState(null);
  const [progress, setProgress] = useState(INITIAL_PROGRESS);
  const [status, setStatus] = useState({ text: "就绪", tone: "ready" });
  const [currentJobId, setCurrentJobId] = useState(null);
  const [filter, setFilter] = useState("all");
  const [sortMode, setSortMode] = useState("index");
  const [query, setQuery] = useState("");
  const [expandedRows, setExpandedRows] = useState(() => new Set());
  const [editedRows, setEditedRows] = useState({});
  const [dragging, setDragging] = useState(false);
  const pollRef = useRef(null);

  const running = Boolean(currentJobId);
  const rows = result?.paragraphs || [];
  const summary = result?.summary || {};
  const engineStatus = result?.engine_status || {};

  useEffect(() => {
    loadConfig();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const visibleRows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const selected = rows.filter((row) => {
      if (filter === "ai" && row.label !== "ai") return false;
      if (filter === "human" && row.label !== "human") return false;
      if (filter === "unknown" && row.label !== "unknown") return false;
      if (filter === "changed" && !row.changed) return false;
      if (filter === "high" && !isHighRisk(row)) return false;
      const body = `${row.text || ""} ${row.refined || ""}`.toLowerCase();
      return !q || body.includes(q);
    });
    return selected.sort((a, b) => {
      if (sortMode === "score_desc") return scoreValue(b) - scoreValue(a);
      if (sortMode === "score_asc") return scoreValue(a) - scoreValue(b);
      if (sortMode === "changed") return Number(Boolean(b.changed)) - Number(Boolean(a.changed));
      return Number(a.index || 0) - Number(b.index || 0);
    });
  }, [rows, filter, sortMode, query]);

  const distribution = useMemo(() => buildDistribution(rows), [rows]);

  async function loadConfig() {
    try {
      const response = await fetch("/api/config");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "读取配置失败");
      setConfig(data);
      setThreshold(data.threshold?.aigc_threshold ?? 0.5);
      setDetectConcurrency(data.performance?.api_concurrency ?? 4);
    } catch (error) {
      setStatus({ text: "配置读取失败", tone: "error" });
    }
  }

  async function saveConfig(nextConfig) {
    setStatus({ text: "保存配置", tone: "busy" });
    const response = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(nextConfig),
    });
    const data = await response.json();
    if (!response.ok) {
      setStatus({ text: "保存失败", tone: "error" });
      return;
    }
    setConfig(data.config);
    setStatus({ text: "配置已保存", tone: "ready" });
  }

  function requestOptions() {
    const options = {
      threshold: threshold === "" ? null : Number(threshold),
      min_length: Number(minLength || 30),
    };
    if (taskMode === "refine") {
      options.refine_threshold = Number(refineThreshold || 0.4);
      options.max_rounds = Number(maxRounds || 2);
      options.detect_concurrency = Number(detectConcurrency || 4);
    }
    return options;
  }

  async function submitJob(endpoint, request) {
    setStatus({ text: taskMode === "refine" ? "润色中" : "检测中", tone: "busy" });
    setProgress({ progress: 2, stage: "提交任务", message: "正在发送请求", state: "running" });
    setExpandedRows(new Set());
    try {
      const response = await fetch(endpoint, request);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "请求失败");
      if (!data.job_id) throw new Error("服务端未返回任务 ID");
      setCurrentJobId(data.job_id);
      await pollJob(data.job_id);
    } catch (error) {
      setStatus({ text: "失败", tone: "error" });
      setProgress({ progress: 100, stage: "失败", message: error.message, state: "error" });
      setCurrentJobId(null);
    }
  }

  async function submitRowJob(row, mode, text) {
    const clean = String(text || "").trim();
    if (!clean) {
      setStatus({ text: "段落为空", tone: "error" });
      return;
    }
    const endpoint = mode === "refine" ? "/api/refine-text" : "/api/detect-text";
    const payload = {
      text: clean,
      ...requestOptions(),
      min_length: 1,
    };
    if (mode === "refine") {
      payload.refine_threshold = Number(refineThreshold || 0.4);
      payload.max_rounds = Number(maxRounds || 2);
      payload.detect_concurrency = Number(detectConcurrency || 4);
    }
    setStatus({ text: mode === "refine" ? "单段润色中" : "单段复检中", tone: "busy" });
    setProgress({ progress: 2, stage: "提交单段任务", message: `段落 #${row.index}`, state: "running" });
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "请求失败");
      setCurrentJobId(data.job_id);
      const jobResult = await pollJob(data.job_id, { applyResult: false });
      if (jobResult?.paragraphs?.[0]) {
        replaceRow(row.index, normalizeSingleRow(row, jobResult.paragraphs[0], clean, mode));
        setStatus({ text: "单段已更新", tone: "ready" });
      }
    } catch (error) {
      setStatus({ text: "单段任务失败", tone: "error" });
      setProgress({ progress: 100, stage: "失败", message: error.message, state: "error" });
      setCurrentJobId(null);
    }
  }

  function replaceRow(index, nextRow) {
    setResult((old) => {
      if (!old) return old;
      const paragraphs = (old.paragraphs || []).map((row) => (row.index === index ? nextRow : row));
      return { ...old, paragraphs, summary: summarizeRows(paragraphs, old.summary || {}) };
    });
  }

  function saveManualEdit(row, text) {
    const clean = String(text || "").trim();
    if (!clean) {
      setStatus({ text: "段落为空", tone: "error" });
      return;
    }
    replaceRow(row.index, {
      ...row,
      text: clean,
      refined: clean,
      changed: clean !== row.text,
      aigc_score: null,
      label: "unknown",
      confidence: null,
      method: "manual_edit",
      engine_scores: [],
    });
    setEditedRows((old) => {
      const next = { ...old };
      delete next[row.index];
      return next;
    });
    setStatus({ text: "手动修改已保存", tone: "ready" });
  }

  function pollJob(jobId, options = {}) {
    const shouldApplyResult = options.applyResult !== false;
    if (pollRef.current) clearInterval(pollRef.current);
    return new Promise((resolve, reject) => {
      pollRef.current = setInterval(async () => {
        try {
          const response = await fetch(`/api/jobs/${jobId}`);
          const job = await response.json();
          if (!response.ok) throw new Error(job.error || "任务查询失败");
          setProgress(job);
          if (job.state === "done") {
            clearInterval(pollRef.current);
            pollRef.current = null;
            setCurrentJobId(null);
            if (shouldApplyResult) setResult(job.result);
            setStatus({ text: "完成", tone: "ready" });
            resolve(job.result);
          } else if (job.state === "canceled") {
            clearInterval(pollRef.current);
            pollRef.current = null;
            setCurrentJobId(null);
            setStatus({ text: "已中断", tone: "ready" });
            setProgress({ ...job, progress: 100, stage: "已中断", message: job.message || "任务已中断" });
            resolve(null);
          } else if (job.state === "error") {
            clearInterval(pollRef.current);
            pollRef.current = null;
            throw new Error(job.error || "检测失败");
          }
        } catch (error) {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setCurrentJobId(null);
          setStatus({ text: "失败", tone: "error" });
          setProgress({ progress: 100, stage: "失败", message: error.message, state: "error" });
          reject(error);
        }
      }, 500);
    });
  }

  async function runTextJob() {
    const text = textInput.trim();
    if (!text) {
      setStatus({ text: "请输入文本", tone: "error" });
      return;
    }
    const endpoint = taskMode === "refine" ? "/api/refine-text" : "/api/detect-text";
    await submitJob(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, ...requestOptions() }),
    });
  }

  async function runFileJob() {
    if (!file) {
      setStatus({ text: "请选择文件", tone: "error" });
      return;
    }
    const form = new FormData();
    form.append("file", file);
    const options = requestOptions();
    if (options.threshold !== null) form.append("threshold", String(options.threshold));
    if (taskMode === "refine") {
      form.append("refine_threshold", String(options.refine_threshold));
      form.append("max_rounds", String(options.max_rounds));
      form.append("detect_concurrency", String(options.detect_concurrency));
    }
    const endpoint = taskMode === "refine" ? "/api/refine-file" : "/api/detect-file";
    await submitJob(endpoint, { method: "POST", body: form });
  }

  async function cancelJob() {
    if (!currentJobId) return;
    setStatus({ text: "中断中", tone: "busy" });
    setProgress((old) => ({ ...old, state: "canceling", stage: "正在中断", message: "等待当前步骤结束" }));
    try {
      const response = await fetch(`/api/jobs/${currentJobId}/cancel`, { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "中断失败");
      setProgress(data);
    } catch (error) {
      setStatus({ text: "中断失败", tone: "error" });
      setProgress((old) => ({ ...old, message: error.message }));
    }
  }

  function runCurrentSource() {
    if (sourceMode === "file") return runFileJob();
    return runTextJob();
  }

  function toggleRow(index) {
    setExpandedRows((old) => {
      const next = new Set(old);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }

  function onDrop(event) {
    event.preventDefault();
    setDragging(false);
    const selected = event.dataTransfer.files?.[0];
    if (selected) {
      setFile(selected);
      setSourceMode("file");
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><ShieldAlert size={22} /></div>
          <div>
            <h1>AIGC Detector Toolkit</h1>
            <p>中文文档检测与降重工作台</p>
          </div>
        </div>
        <div className="topbar-actions">
          <button className="ghost-btn" type="button" onClick={() => setConfigOpen(true)}>
            <Settings2 size={16} /> 配置
          </button>
          <span className={`status-pill ${status.tone}`}>{status.text}</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="control-panel">
          <div className="panel-heading">
            <span>任务控制</span>
            <strong>{taskMode === "refine" ? "润色降重" : "AI检测"}</strong>
          </div>

          <SegmentedControl
            label="运行模式"
            value={taskMode}
            onChange={setTaskMode}
            options={[
              { value: "detect", label: "检测", icon: Gauge },
              { value: "refine", label: "润色", icon: Wand2 },
            ]}
          />

          <SegmentedControl
            label="输入来源"
            value={sourceMode}
            onChange={setSourceMode}
            options={[
              { value: "file", label: "文件", icon: FileText },
              { value: "text", label: "文本", icon: Layers3 },
            ]}
          />

          <div className="field-row">
            <NumberField label="判定阈值" value={threshold} onChange={setThreshold} min={0} max={1} step={0.01} />
            <NumberField label="最小段长" value={minLength} onChange={setMinLength} min={1} step={1} />
          </div>

          {taskMode === "refine" && (
            <div className="refine-options">
              <div className="field-row">
                <NumberField label="润色阈值" value={refineThreshold} onChange={setRefineThreshold} min={0} max={1} step={0.01} />
                <NumberField label="最大轮数" value={maxRounds} onChange={setMaxRounds} min={1} max={8} step={1} />
              </div>
              <NumberField label="复测并发" value={detectConcurrency} onChange={setDetectConcurrency} min={1} max={32} step={1} />
            </div>
          )}

          {sourceMode === "file" ? (
            <label
              className={`drop-zone ${dragging ? "dragging" : ""}`}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={(event) => {
                event.preventDefault();
                setDragging(false);
              }}
              onDrop={onDrop}
            >
              <UploadCloud size={28} />
              <strong>{file ? file.name : "选择或拖入文件"}</strong>
              <span>支持 .docx / .md / .txt</span>
              <input
                type="file"
                accept=".docx,.md,.txt"
                onChange={(event) => setFile(event.target.files?.[0] || null)}
              />
            </label>
          ) : (
            <textarea
              className="text-input"
              value={textInput}
              onChange={(event) => setTextInput(event.target.value)}
              spellCheck={false}
              placeholder="粘贴需要检测的正文，段落可用空行分隔。"
            />
          )}

          <button className="primary-action" type="button" onClick={runCurrentSource} disabled={running}>
            {running ? <RefreshCw className="spin" size={17} /> : <Play size={17} />}
            {taskMode === "refine" ? "开始润色" : "开始检测"}
          </button>
        </aside>

        <section className="results-panel">
          <RunPanel progress={progress} running={running} onCancel={cancelJob} />
          <MetricGrid summary={summary} rows={rows} />
          <InsightBoard summary={summary} rows={rows} distribution={distribution} engineStatus={engineStatus} />

          <section className="analysis-card">
            <div className="section-head">
              <div>
                <h2>逐段解释</h2>
                <p>按段查看综合判定、引擎贡献、风险原因和降重差异。</p>
              </div>
              <div className="tools">
                <button className="tool-btn" type="button" disabled={!result} onClick={() => downloadJson(result)}>
                  <Download size={15} /> JSON
                </button>
                <button className="tool-btn" type="button" disabled={!rows.length} onClick={() => downloadCsv(rows)}>
                  <Download size={15} /> CSV
                </button>
              </div>
            </div>
            <ResultToolbar
              filter={filter}
              setFilter={setFilter}
              sortMode={sortMode}
              setSortMode={setSortMode}
              query={query}
              setQuery={setQuery}
            />
            <ResultsTable
              rows={visibleRows}
              expandedRows={expandedRows}
              editedRows={editedRows}
              onToggle={toggleRow}
              onEditChange={(index, value) => setEditedRows((old) => ({ ...old, [index]: value }))}
              onSaveEdit={saveManualEdit}
              onRowDetect={(row, text) => submitRowJob(row, "detect", text)}
              onRowRefine={(row, text) => submitRowJob(row, "refine", text)}
              running={running}
            />
          </section>
        </section>
      </section>

      {configOpen && config && (
        <ConfigDrawer
          config={config}
          onClose={() => setConfigOpen(false)}
          onSave={(payload) => saveConfig(payload)}
        />
      )}
    </main>
  );
}

function SegmentedControl({ label, value, onChange, options }) {
  return (
    <div className="field-group">
      <label>{label}</label>
      <div className="segmented">
        {options.map((option) => {
          const Icon = option.icon;
          return (
            <button
              key={option.value}
              type="button"
              className={value === option.value ? "active" : ""}
              onClick={() => onChange(option.value)}
            >
              <Icon size={15} /> {option.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function NumberField({ label, value, onChange, min, max, step }) {
  return (
    <div className="field-group">
      <label>{label}</label>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  );
}

function RunPanel({ progress, running, onCancel }) {
  const value = Math.max(0, Math.min(100, Number(progress.progress || 0)));
  return (
    <section className="run-panel">
      <div className="run-meta">
        <strong>{progress.stage || "等待任务"}</strong>
        <span>{progress.message || "提交任务后会显示运行状态"}</span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${value}%` }} />
      </div>
      <span className="progress-text">{value.toFixed(0)}%</span>
      <button className="cancel-btn" type="button" disabled={!running} onClick={onCancel}>
        <PauseCircle size={15} /> 中断
      </button>
    </section>
  );
}

function MetricGrid({ summary, rows }) {
  const changed = summary.changed ?? rows.filter((row) => row.changed).length;
  return (
    <section className="summary-grid">
      <Metric title="段落" value={summary.total ?? rows.length ?? 0} icon={FileText} />
      <Metric title="AI 判定" value={summary.ai ?? 0} icon={ShieldAlert} tone="danger" />
      <Metric title="AI 比例" value={formatRatio(summary.ai_ratio ?? 0)} icon={BarChart3} />
      <Metric title="平均分" value={formatPercent(summary.avg_score)} icon={Gauge} />
      <Metric title="已修改" value={changed ?? 0} icon={Sparkles} tone="success" />
      <Metric title="总轮次" value={summary.rounds ?? 0} icon={RefreshCw} />
    </section>
  );
}

function Metric({ title, value, icon: Icon, tone = "" }) {
  return (
    <div className={`metric ${tone}`}>
      <span><Icon size={15} /> {title}</span>
      <strong>{value}</strong>
    </div>
  );
}

function InsightBoard({ summary, rows, distribution, engineStatus }) {
  const engineEntries = ENGINE_NAMES.map((name) => ({
    name,
    label: ENGINE_LABELS[name] || name,
    available: engineStatus?.[`${name}_available`],
    weight: Number(engineStatus?.[`${name}_weight`] || 0),
  })).filter((item) => item.available || item.weight > 0);
  const highRisk = rows.filter(isHighRisk).length;

  return (
    <section className="insight-grid">
      <div className="analysis-card distribution-card">
        <div className="section-head compact">
          <h2>分数分布</h2>
          <span>{summary.engine || "ensemble"} / 阈值 {summary.threshold ?? "-"}</span>
        </div>
        <div className="bars">
          {distribution.map((bucket) => (
            <div className="bar-row" key={bucket.label}>
              <span>{bucket.label}</span>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${bucket.width}%` }} />
              </div>
              <strong>{bucket.count}</strong>
            </div>
          ))}
        </div>
      </div>
      <div className="analysis-card">
        <div className="section-head compact">
          <h2>解释概览</h2>
          <span>{highRisk} 个高风险段落</span>
        </div>
        <div className="engine-stack">
          {engineEntries.length ? engineEntries.map((engine) => (
            <div className="engine-weight" key={engine.name}>
              <div>
                <strong>{engine.label}</strong>
                <span>{engine.available ? "可用" : "不可用"}</span>
              </div>
              <div className="weight-bar">
                <i style={{ width: `${Math.min(100, engine.weight * 100)}%` }} />
              </div>
              <b>{Math.round(engine.weight * 100)}%</b>
            </div>
          )) : <p className="empty-copy">完成检测后会显示引擎权重。</p>}
        </div>
      </div>
    </section>
  );
}

function ResultToolbar({ filter, setFilter, sortMode, setSortMode, query, setQuery }) {
  return (
    <div className="result-toolbar">
      <div className="filter-tabs">
        {[
          ["all", "全部"],
          ["ai", "AI"],
          ["human", "人工"],
          ["unknown", "未知"],
          ["changed", "已修改"],
          ["high", "高风险"],
        ].map(([value, label]) => (
          <button key={value} type="button" className={filter === value ? "active" : ""} onClick={() => setFilter(value)}>
            {label}
          </button>
        ))}
      </div>
      <div className="toolbar-controls">
        <label className="search-box">
          <Search size={15} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索段落" />
        </label>
        <select value={sortMode} onChange={(event) => setSortMode(event.target.value)}>
          <option value="index">按序号</option>
          <option value="score_desc">分数从高到低</option>
          <option value="score_asc">分数从低到高</option>
          <option value="changed">已修改优先</option>
        </select>
      </div>
    </div>
  );
}

function ResultsTable({
  rows,
  expandedRows,
  editedRows,
  onToggle,
  onEditChange,
  onSaveEdit,
  onRowDetect,
  onRowRefine,
  running,
}) {
  if (!rows.length) {
    return (
      <div className="empty-state">
        <Activity size={28} />
        <strong>暂无匹配结果</strong>
        <span>运行检测后，逐段解释会显示在这里。</span>
      </div>
    );
  }
  return (
    <div className="result-list">
      {rows.map((row) => (
        <ResultRow
          key={row.index}
          row={row}
          expanded={expandedRows.has(row.index)}
          editValue={editedRows[row.index] ?? row.refined ?? row.text ?? ""}
          onToggle={() => onToggle(row.index)}
          onEditChange={(value) => onEditChange(row.index, value)}
          onSaveEdit={(value) => onSaveEdit(row, value)}
          onRowDetect={(value) => onRowDetect(row, value)}
          onRowRefine={(value) => onRowRefine(row, value)}
          running={running}
        />
      ))}
    </div>
  );
}

function ResultRow({ row, expanded, editValue, onToggle, onEditChange, onSaveEdit, onRowDetect, onRowRefine, running }) {
  const score = scoreValue(row);
  const reasons = riskReasons(row);
  return (
    <article className={`result-row ${expanded ? "expanded" : ""}`}>
      <button className="row-summary" type="button" onClick={onToggle}>
        <span className="row-index">#{row.index}</span>
        <span className="score-pill">{formatPercent(score)}</span>
        <span className={`label ${row.label || "unknown"}`}>{labelText(row.label)}</span>
        <span className="row-text">{row.text || ""}</span>
        <ChevronDown className="chevron" size={18} />
      </button>
      {expanded && (
        <div className="row-detail">
          <div className="detail-main">
            <div className="explain-block">
              <h3>引擎贡献</h3>
              <EngineContributions scores={row.engine_scores || []} finalScore={score} />
            </div>
            <div className="explain-block">
              <h3>风险原因</h3>
              <div className="reason-list">
                {reasons.map((reason) => (
                  <span key={reason}>{reason}</span>
                ))}
              </div>
            </div>
          </div>
          <DiffPanel row={row} />
          <div className="row-actions">
            <div className="editor-box">
              <div className="editor-head">
                <h3>单段操作</h3>
                <span>可先手动修改，再复检或重新润色</span>
              </div>
              <textarea value={editValue} onChange={(event) => onEditChange(event.target.value)} />
              <div className="editor-actions">
                <button type="button" className="tool-btn" disabled={running} onClick={() => onSaveEdit(editValue)}>
                  <Save size={15} /> 保存修改
                </button>
                <button type="button" className="tool-btn" disabled={running} onClick={() => onRowDetect(editValue)}>
                  <CheckCircle2 size={15} /> 单段复检
                </button>
                <button type="button" className="tool-btn primary-inline" disabled={running} onClick={() => onRowRefine(editValue)}>
                  <Wand2 size={15} /> 重新润色
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </article>
  );
}

function EngineContributions({ scores, finalScore }) {
  if (!scores.length) return <p className="empty-copy">没有逐引擎明细。</p>;
  return (
    <div className="contribution-list">
      {scores.map((item) => {
        const value = Number(item.score || 0);
        const delta = Math.abs(value - finalScore);
        return (
          <div className="contribution" key={item.engine || item.label}>
            <div>
              <strong>{item.label || ENGINE_LABELS[item.engine] || item.engine}</strong>
              <span>{item.method || "score"} · 差异 {formatPercent(delta)}</span>
            </div>
            <div className="contribution-track">
              <i style={{ width: `${Math.max(2, Math.min(100, value * 100))}%` }} />
            </div>
            <b>{formatPercent(value)}</b>
          </div>
        );
      })}
    </div>
  );
}

function DiffPanel({ row }) {
  const original = row.text || "";
  const refined = row.refined || "";
  if (!refined || refined === original) {
    return (
      <div className="diff-panel">
        <div className="diff-column">
          <h3>原文</h3>
          <p>{original}</p>
        </div>
      </div>
    );
  }
  return (
    <div className="diff-panel three">
      <div className="diff-column">
        <h3>原文</h3>
        <p>{original}</p>
      </div>
      <div className="diff-column">
        <h3>改写</h3>
        <p>{refined}</p>
      </div>
      <div className="diff-column">
        <h3>Diff</h3>
        <p className="diff-text" dangerouslySetInnerHTML={{ __html: renderDiff(original, refined) }} />
      </div>
    </div>
  );
}

function ConfigDrawer({ config, onClose, onSave }) {
  const [draft, setDraft] = useState(() => ({
    openaiBase: config.openai_api?.api_base || "",
    openaiKey: config.openai_api?.api_key || "",
    openaiModel: config.openai_api?.model || "",
    refinerBase: config.refiner_api?.api_base || "",
    refinerKey: config.refiner_api?.api_key || "",
    refinerModel: config.refiner_api?.model || "",
    refinerTemperature: config.refiner_api?.temperature ?? 0.3,
    apiConcurrency: config.performance?.api_concurrency || 1,
    defaultThreshold: config.threshold?.aigc_threshold ?? 0.5,
    weights: { ...(config.engine?.ensemble || {}) },
  }));

  function setField(name, value) {
    setDraft((old) => ({ ...old, [name]: value }));
  }

  function setWeight(name, value) {
    setDraft((old) => ({ ...old, weights: { ...old.weights, [`${name}_weight`]: Number(value || 0) } }));
  }

  function submit() {
    onSave({
      engine: { default: "ensemble", ensemble: draft.weights },
      openai_api: {
        api_base: draft.openaiBase,
        api_key: draft.openaiKey,
        model: draft.openaiModel,
        strategy: "perplexity",
      },
      refiner_api: {
        api_base: draft.refinerBase,
        api_key: draft.refinerKey,
        model: draft.refinerModel,
        temperature: Number(draft.refinerTemperature || 0.3),
      },
      threshold: { aigc_threshold: Number(draft.defaultThreshold || 0.5) },
      performance: { api_concurrency: Number(draft.apiConcurrency || 1) },
    });
  }

  return (
    <div className="drawer-backdrop" onMouseDown={onClose}>
      <aside className="config-drawer" onMouseDown={(event) => event.stopPropagation()}>
        <div className="drawer-head">
          <div>
            <h2>配置中心</h2>
            <p>检测、润色和融合权重设置</p>
          </div>
          <button className="icon-btn" type="button" onClick={onClose}>×</button>
        </div>
        <div className="drawer-section">
          <h3>检测 API</h3>
          <TextField label="API Base" value={draft.openaiBase} onChange={(v) => setField("openaiBase", v)} />
          <TextField label="API Key" type="password" value={draft.openaiKey} onChange={(v) => setField("openaiKey", v)} />
          <TextField label="模型" value={draft.openaiModel} onChange={(v) => setField("openaiModel", v)} />
        </div>
        <div className="drawer-section">
          <h3>润色 API</h3>
          <TextField label="API Base" value={draft.refinerBase} onChange={(v) => setField("refinerBase", v)} />
          <TextField label="API Key" type="password" value={draft.refinerKey} onChange={(v) => setField("refinerKey", v)} />
          <TextField label="模型" value={draft.refinerModel} onChange={(v) => setField("refinerModel", v)} />
          <NumberField label="Temperature" value={draft.refinerTemperature} onChange={(v) => setField("refinerTemperature", v)} min={0} max={2} step={0.1} />
        </div>
        <div className="drawer-section">
          <h3>运行参数</h3>
          <div className="field-row">
            <NumberField label="API 并发" value={draft.apiConcurrency} onChange={(v) => setField("apiConcurrency", v)} min={1} max={64} step={1} />
            <NumberField label="默认阈值" value={draft.defaultThreshold} onChange={(v) => setField("defaultThreshold", v)} min={0} max={1} step={0.01} />
          </div>
          <div className="weights-grid">
            {ENGINE_NAMES.map((name) => (
              <NumberField
                key={name}
                label={ENGINE_LABELS[name] || name}
                value={draft.weights[`${name}_weight`] ?? 0}
                onChange={(v) => setWeight(name, v)}
                min={0}
                max={1}
                step={0.01}
              />
            ))}
          </div>
        </div>
        <button className="primary-action" type="button" onClick={submit}>
          <Save size={17} /> 保存配置
        </button>
      </aside>
    </div>
  );
}

function TextField({ label, value, onChange, type = "text" }) {
  return (
    <div className="field-group">
      <label>{label}</label>
      <input type={type} value={value} onChange={(event) => onChange(event.target.value)} />
    </div>
  );
}

function buildDistribution(rows) {
  const buckets = [
    [0, 0.1, "0-10%"],
    [0.1, 0.2, "10-20%"],
    [0.2, 0.3, "20-30%"],
    [0.3, 0.4, "30-40%"],
    [0.4, 0.5, "40-50%"],
    [0.5, 0.7, "50-70%"],
    [0.7, 1.01, "70-100%"],
  ];
  const scores = rows.map(scoreValue).filter((score) => Number.isFinite(score));
  const total = Math.max(scores.length, 1);
  return buckets.map(([lo, hi, label]) => {
    const count = scores.filter((score) => score >= lo && score < hi).length;
    return { label, count, width: (count / total) * 100 };
  });
}

function scoreValue(row) {
  if (typeof row === "number") return row;
  return typeof row?.aigc_score === "number" ? row.aigc_score : 0;
}

function isHighRisk(row) {
  return scoreValue(row) >= 0.7 || row.label === "ai";
}

function normalizeSingleRow(previous, incoming, sourceText, mode) {
  const next = {
    ...previous,
    ...incoming,
    index: previous.index,
  };
  if (mode === "detect") {
    next.text = sourceText;
    next.refined = previous.refined && previous.refined !== previous.text ? previous.refined : "";
    next.changed = Boolean(next.refined && next.refined !== next.text);
  }
  if (mode === "refine") {
    next.text = incoming.text || sourceText;
    next.refined = incoming.refined || incoming.text || sourceText;
    next.changed = Boolean(incoming.changed);
  }
  return next;
}

function summarizeRows(rows, previousSummary = {}) {
  const valid = rows.filter((row) => row.label !== "unknown");
  const ai = valid.filter((row) => row.label === "ai").length;
  const human = valid.filter((row) => row.label === "human").length;
  const scores = valid.map((row) => row.aigc_score).filter((score) => typeof score === "number");
  return {
    ...previousSummary,
    total: rows.length,
    valid: valid.length,
    ai,
    human,
    unknown: rows.length - valid.length,
    ai_ratio: valid.length ? (ai / valid.length) * 100 : 0,
    avg_score: scores.length ? scores.reduce((sum, score) => sum + score, 0) / scores.length : null,
    max_score: scores.length ? Math.max(...scores) : null,
    min_score: scores.length ? Math.min(...scores) : null,
    changed: rows.filter((row) => row.changed).length,
    rounds: rows.reduce((sum, row) => sum + Number(row.rounds || 0), 0),
  };
}

function riskReasons(row) {
  const reasons = [];
  const score = scoreValue(row);
  if (score >= 0.7) reasons.push("综合分超过 70%");
  if (row.label === "ai") reasons.push("最终标签为 AI");
  const engineScores = (row.engine_scores || []).filter((item) => typeof item.score === "number");
  if (engineScores.length) {
    const max = Math.max(...engineScores.map((item) => item.score));
    const min = Math.min(...engineScores.map((item) => item.score));
    if (max - min > 0.25) reasons.push("引擎分歧明显");
    const highEngines = engineScores.filter((item) => item.score >= 0.7).map((item) => item.label || ENGINE_LABELS[item.engine] || item.engine);
    if (highEngines.length) reasons.push(`${highEngines.join("、")} 判为高风险`);
  }
  if (row.changed) reasons.push("降重流程已改写该段");
  if (!reasons.length) reasons.push("未发现明显高风险信号");
  return reasons;
}

function tokenizeText(text) {
  if (window.Intl && Intl.Segmenter) {
    const segmenter = new Intl.Segmenter("zh", { granularity: "word" });
    return Array.from(segmenter.segment(text), (item) => item.segment);
  }
  return Array.from(text);
}

function renderDiff(before, after) {
  const left = tokenizeText(before);
  const right = tokenizeText(after);
  const rows = Array.from({ length: left.length + 1 }, () => Array(right.length + 1).fill(0));
  for (let i = left.length - 1; i >= 0; i -= 1) {
    for (let j = right.length - 1; j >= 0; j -= 1) {
      rows[i][j] = left[i] === right[j] ? rows[i + 1][j + 1] + 1 : Math.max(rows[i + 1][j], rows[i][j + 1]);
    }
  }
  const parts = [];
  let i = 0;
  let j = 0;
  while (i < left.length || j < right.length) {
    if (i < left.length && j < right.length && left[i] === right[j]) {
      parts.push(`<span>${escapeHtml(left[i])}</span>`);
      i += 1;
      j += 1;
    } else if (j < right.length && (i === left.length || rows[i][j + 1] >= rows[i + 1][j])) {
      parts.push(`<ins>${escapeHtml(right[j])}</ins>`);
      j += 1;
    } else if (i < left.length) {
      parts.push(`<del>${escapeHtml(left[i])}</del>`);
      i += 1;
    }
  }
  return parts.join("");
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatRatio(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "0%";
  return `${Number(value).toFixed(1)}%`;
}

function labelText(label) {
  if (label === "ai") return "AI";
  if (label === "human") return "人工";
  return "未知";
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
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

function downloadJson(data) {
  if (!data) return;
  download("aigc-detect-result.json", JSON.stringify(data, null, 2), "application/json;charset=utf-8");
}

function downloadCsv(rows) {
  const headers = ["index", "score", "label", "engine_scores", "refined", "text"];
  const lines = [headers.join(",")];
  for (const row of rows) {
    const engineScores = (row.engine_scores || []).map((item) => `${item.label}:${item.score_text || "-"}`).join(";");
    const values = [
      row.index ?? "",
      row.aigc_score ?? "",
      row.label ?? "",
      engineScores,
      row.refined ?? "",
      row.text ?? "",
    ].map(csvEscape);
    lines.push(values.join(","));
  }
  download("aigc-detect-result.csv", lines.join("\n"), "text/csv;charset=utf-8");
}

function csvEscape(value) {
  return `"${String(value).replaceAll('"', '""')}"`;
}

createRoot(document.getElementById("root")).render(<App />);
