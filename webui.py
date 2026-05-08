#!/usr/bin/env python3
"""Local Web UI for AIGC Detector Toolkit."""
from __future__ import annotations

import argparse
import errno
import json
import mimetypes
import os
import re
import tempfile
import threading
import time
import uuid
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from main import PROJECT_ROOT, build_engine, extract_paragraphs, load_config, normalize_config
from core.refiner import RefinementEngine


WEB_ROOT = Path(PROJECT_ROOT) / "webui"
CONFIG_PATH = Path(PROJECT_ROOT) / "configs" / "default.yaml"
MAX_BODY_BYTES = 50 * 1024 * 1024
JOBS = {}
JOBS_LOCK = threading.Lock()


ENGINE_LABELS = {
    "fengci": "FengCi0",
    "hc3": "HC3+m3e",
    "openai": "OpenAI API",
    "binoculars": "Binoculars",
    "local_logprob": "Local Logprob",
    "lastde": "LastDe",
}


class JobCancelled(RuntimeError):
    """Raised when a WebUI job has been cancelled by the user."""


def is_job_cancelled(job_id: str | None) -> bool:
    if not job_id:
        return False
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return bool(job and job.get("cancel_requested"))


def raise_if_cancelled(job_id: str | None):
    if is_job_cancelled(job_id):
        raise JobCancelled("任务已中断")


def mark_job_cancel_requested(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        if job.get("state") in ("done", "error", "canceled"):
            return dict(job)
        job["cancel_requested"] = True
        job["state"] = "canceling"
        job["stage"] = "正在中断"
        job["message"] = "等待当前步骤结束"
        job["updated_at"] = time.time()
        return dict(job)


def split_text_to_paragraphs(text: str, min_length: int = 30):
    raw = re.split(r"\n\s*\n", text)
    if len(raw) <= 1:
        raw = text.splitlines()
    paragraphs = []
    for i, item in enumerate(raw):
        clean = item.strip()
        if len(clean) < min_length:
            continue
        paragraphs.append({
            "text": clean,
            "chapter": 0,
            "section": "",
            "index": i + 1,
            "source": "input",
        })
    return paragraphs


def summarize(paragraphs, results, engine_status):
    valid = [r for r in results if r.get("label") != "unknown"]
    unknown = len(results) - len(valid)
    ai = [r for r in valid if r.get("label") == "ai"]
    human = [r for r in valid if r.get("label") == "human"]
    scores = [r.get("aigc_score") for r in valid if r.get("aigc_score") is not None]
    avg_score = sum(scores) / len(scores) if scores else None
    return {
        "total": len(results),
        "valid": len(valid),
        "ai": len(ai),
        "human": len(human),
        "unknown": unknown,
        "ai_ratio": (len(ai) / len(valid) * 100) if valid else 0,
        "avg_score": avg_score,
        "max_score": max(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "engine": engine_status.get("mode"),
        "threshold": engine_status.get("threshold"),
    }


def serialize_results(paragraphs, results):
    rows = []
    for para, result in zip(paragraphs, results):
        rows.append({
            "index": para.get("index"),
            "chapter": para.get("chapter", 0),
            "section": para.get("section", ""),
            "source": para.get("source", ""),
            "text": para.get("text", ""),
            "aigc_score": result.get("aigc_score"),
            "label": result.get("label"),
            "confidence": result.get("confidence"),
            "method": result.get("method"),
            "engine_scores": serialize_engine_scores(result.get("engine_results", {})),
        })
    return rows


def serialize_engine_scores(engine_results):
    scores = []
    for name, item in (engine_results or {}).items():
        score = item.get("aigc_score")
        scores.append({
            "engine": name,
            "label": ENGINE_LABELS.get(name, name),
            "score": score,
            "score_text": None if score is None else f"{score * 100:.1f}%",
            "decision": item.get("label"),
            "method": item.get("method") or item.get("mode") or "",
            "confidence": item.get("confidence"),
        })
    return scores


def update_job(job_id: str, **fields):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        if job.get("state") == "canceling" and fields.get("state") == "running":
            fields.pop("state", None)
        job.update(fields)
        job["updated_at"] = time.time()


def finish_cancelled_job(job_id: str, message: str = "任务已中断"):
    update_job(job_id, state="canceled", stage="已中断", progress=100, error=None, message=message)


def read_web_config():
    config = load_config(str(CONFIG_PATH))
    return {
        "engine": config.get("engine", {}),
        "openai_api": config.get("openai_api", {}),
        "refiner_api": config.get("refiner_api", {}),
        "threshold": config.get("threshold", {}),
        "performance": config.get("performance", {}),
        "extraction": config.get("extraction", {}),
        "lastde": config.get("lastde", {}),
        "binoculars": config.get("binoculars", {}),
        "local_logprob": config.get("local_logprob", {}),
    }


def write_web_config(payload: dict):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("保存配置需要安装 pyyaml") from exc

    current = read_web_config()
    for section in ("engine", "openai_api", "refiner_api", "threshold", "performance", "extraction", "lastde", "binoculars", "local_logprob"):
        if section in payload and isinstance(payload[section], dict):
            if isinstance(current.get(section), dict):
                current[section].update(payload[section])
            else:
                current[section] = payload[section]

    normalize_config(current)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write("# AIGC Detector Toolkit 默认配置\n\n")
        yaml.safe_dump(current, f, allow_unicode=True, sort_keys=False)
    return read_web_config()


def snapshot_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        return dict(job)


def create_job(title: str, worker, *args):
    job_id = uuid.uuid4().hex
    now = time.time()
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "title": title,
            "state": "queued",
            "stage": "排队中",
            "progress": 0,
            "message": "",
            "result": None,
            "error": None,
            "cancel_requested": False,
            "created_at": now,
            "updated_at": now,
        }
    thread = threading.Thread(target=worker, args=(job_id, *args), daemon=True)
    thread.start()
    return job_id


def run_detection(paragraphs, threshold: float | None, job_id: str | None = None):
    raise_if_cancelled(job_id)
    if job_id:
        update_job(job_id, state="running", stage="初始化引擎", progress=20, message=f"段落数 {len(paragraphs)}")
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    args = argparse.Namespace(engine=None, threshold=threshold)
    engine = build_engine(config, args)
    raise_if_cancelled(job_id)
    if job_id:
        update_job(job_id, stage="执行检测", progress=42, message=f"引擎 {engine.mode}")
    texts = [p["text"] for p in paragraphs]
    results = engine.detect_batch(texts, show_progress=False)
    raise_if_cancelled(job_id)
    if job_id:
        update_job(job_id, stage="汇总结果", progress=88, message="生成摘要和逐段结果")
    engine_status = engine.get_status()
    return {
        "summary": summarize(paragraphs, results, engine_status),
        "paragraphs": serialize_results(paragraphs, results),
        "engine_status": engine_status,
    }


def build_refiner(config, overrides=None):
    overrides = overrides or {}
    openai_cfg = config.get("openai_api", {})
    refiner_cfg = config.get("refiner_api", {})
    perf_cfg = config.get("performance", {})
    api_base = overrides.get("refiner_api_base") or refiner_cfg.get("api_base") or openai_cfg.get("api_base", "https://api.openai.com/v1")
    api_key = overrides.get("refiner_api_key") or refiner_cfg.get("api_key") or openai_cfg.get("api_key", "")
    model = overrides.get("refiner_model") or refiner_cfg.get("model") or openai_cfg.get("model", "gpt-4o-mini")
    temperature = overrides.get("refiner_temperature")
    if temperature in (None, ""):
        temperature = refiner_cfg.get("temperature", 0.3)
    return RefinementEngine(
        api_base=api_base,
        api_key=api_key,
        model=model,
        temperature=float(temperature),
        concurrency=int(perf_cfg.get("api_concurrency", 4) or 4),
    )


def summarize_refinement(paragraphs, refinement_results, before_summary):
    changed = [r for r in refinement_results if r.get("changed")]
    after_known = [r for r in refinement_results if r.get("aigc_after") is not None]
    ai_after = [r for r in after_known if r.get("aigc_after_label") == "ai"]
    scores = [r.get("aigc_after") for r in after_known if r.get("aigc_after") is not None]
    return {
        "total": len(refinement_results),
        "valid": len(after_known),
        "ai": len(ai_after),
        "human": len(after_known) - len(ai_after),
        "unknown": len(refinement_results) - len(after_known),
        "ai_ratio": (len(ai_after) / len(after_known) * 100) if after_known else 0,
        "avg_score": (sum(scores) / len(scores)) if scores else None,
        "max_score": max(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "engine": before_summary.get("engine"),
        "threshold": before_summary.get("threshold"),
        "changed": len(changed),
        "rounds": sum(r.get("rounds", 0) for r in refinement_results),
        "ai_after": len(ai_after),
        "ai_after_ratio": (len(ai_after) / len(after_known) * 100) if after_known else 0,
        "before": before_summary,
    }


def serialize_refinement(paragraphs, refinement_results):
    rows = []
    for para, item in zip(paragraphs, refinement_results):
        after_score = item.get("aigc_after")
        after_label = item.get("aigc_after_label")
        if not after_label:
            after_label = "unknown" if after_score is None else ("human" if after_score <= 0.5 else "ai")
        rows.append({
            "index": para.get("index"),
            "chapter": para.get("chapter", 0),
            "section": para.get("section", ""),
            "source": para.get("source", ""),
            "text": item.get("original", para.get("text", "")),
            "refined": item.get("refined", para.get("text", "")),
            "aigc_score": item.get("aigc_after", item.get("aigc_before")),
            "aigc_before": item.get("aigc_before"),
            "aigc_after": item.get("aigc_after"),
            "label": after_label,
            "method": item.get("strategy"),
            "changed": item.get("changed", False),
            "rounds": item.get("rounds", 0),
            "note": item.get("note", ""),
            "round_history": item.get("round_history", []),
            "engine_scores": [],
        })
    return rows


async def refine_paragraphs_async(paragraphs, initial_results, engine, config, options, job_id):
    update_job(job_id, stage="初始化润色器", progress=52)
    refiner = build_refiner(config, options)
    if not refiner.api_key:
        raise ValueError("未配置润色模型 API Key")
    threshold = float(options.get("refine_threshold") or options.get("threshold") or 0.4)
    max_rounds = int(options.get("max_rounds") or 2)
    detect_concurrency = int(options.get("detect_concurrency") or config.get("performance", {}).get("api_concurrency", 4) or 4)
    update_job(job_id, stage="循环润色", progress=60, message=f"最多 {max_rounds} 轮")
    results = await refiner.refine_batch_iterative_async(
        paragraphs,
        initial_results,
        detect_fn=engine.detect_single,
        score_threshold=threshold,
        max_rounds=max_rounds,
        detect_concurrency=detect_concurrency,
        should_cancel=lambda: is_job_cancelled(job_id),
        show_progress=False,
    )
    raise_if_cancelled(job_id)
    update_job(job_id, stage="汇总润色结果", progress=92)
    return results


def run_text_job(job_id: str, payload: dict):
    try:
        raise_if_cancelled(job_id)
        update_job(job_id, state="running", stage="解析文本", progress=8)
        text = payload.get("text", "")
        min_length = int(payload.get("min_length") or 30)
        paragraphs = split_text_to_paragraphs(text, min_length=min_length)
        if not paragraphs:
            raise ValueError("没有提取到有效段落")
        result = run_detection(
            paragraphs,
            float(payload["threshold"]) if payload.get("threshold") not in (None, "") else None,
            job_id=job_id,
        )
        update_job(job_id, state="done", stage="完成", progress=100, message="检测完成", result=result)
    except JobCancelled:
        finish_cancelled_job(job_id)
    except Exception as exc:
        update_job(job_id, state="error", stage="失败", progress=100, error=str(exc), message=str(exc))


def run_text_refine_job(job_id: str, payload: dict):
    try:
        raise_if_cancelled(job_id)
        update_job(job_id, state="running", stage="解析文本", progress=6)
        text = payload.get("text", "")
        min_length = int(payload.get("min_length") or 30)
        paragraphs = split_text_to_paragraphs(text, min_length=min_length)
        if not paragraphs:
            raise ValueError("没有提取到有效段落")
        config = load_config(str(CONFIG_PATH))
        args = argparse.Namespace(engine=None, threshold=float(payload["threshold"]) if payload.get("threshold") not in (None, "") else None)
        raise_if_cancelled(job_id)
        update_job(job_id, stage="初始化检测引擎", progress=16, message=f"段落数 {len(paragraphs)}")
        engine = build_engine(config, args)
        raise_if_cancelled(job_id)
        update_job(job_id, stage="初检", progress=30, message=f"引擎 {engine.mode}")
        initial_results = engine.detect_batch([p["text"] for p in paragraphs], show_progress=False)
        raise_if_cancelled(job_id)
        initial_summary = summarize(paragraphs, initial_results, engine.get_status())
        refinement_results = asyncio.run(refine_paragraphs_async(paragraphs, initial_results, engine, config, payload, job_id))
        result = {
            "summary": summarize_refinement(paragraphs, refinement_results, initial_summary),
            "paragraphs": serialize_refinement(paragraphs, refinement_results),
            "engine_status": engine.get_status(),
            "mode": "refine",
        }
        update_job(job_id, state="done", stage="完成", progress=100, message="润色完成", result=result)
    except JobCancelled:
        finish_cancelled_job(job_id)
    except Exception as exc:
        update_job(job_id, state="error", stage="失败", progress=100, error=str(exc), message=str(exc))


def run_file_job(job_id: str, fields: dict, uploaded: dict):
    tmp_path = None
    try:
        raise_if_cancelled(job_id)
        update_job(job_id, state="running", stage="保存上传文件", progress=6, message=uploaded["filename"])
        suffix = Path(uploaded["filename"]).suffix.lower()
        if suffix not in (".docx", ".md", ".txt"):
            raise ValueError("仅支持 .docx / .md / .txt")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded["content"])
            tmp_path = tmp.name

        raise_if_cancelled(job_id)
        update_job(job_id, stage="提取段落", progress=12, message=uploaded["filename"])
        config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
        paragraphs = extract_paragraphs(tmp_path, config)
        if not paragraphs:
            raise ValueError("没有提取到有效段落")
        threshold = fields.get("threshold")
        result = run_detection(
            paragraphs,
            float(threshold) if threshold not in (None, "") else None,
            job_id=job_id,
        )
        result["filename"] = uploaded["filename"]
        update_job(job_id, state="done", stage="完成", progress=100, message="检测完成", result=result)
    except JobCancelled:
        finish_cancelled_job(job_id)
    except Exception as exc:
        update_job(job_id, state="error", stage="失败", progress=100, error=str(exc), message=str(exc))
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def run_file_refine_job(job_id: str, fields: dict, uploaded: dict):
    tmp_path = None
    try:
        raise_if_cancelled(job_id)
        update_job(job_id, state="running", stage="保存上传文件", progress=5, message=uploaded["filename"])
        suffix = Path(uploaded["filename"]).suffix.lower()
        if suffix not in (".docx", ".md", ".txt"):
            raise ValueError("仅支持 .docx / .md / .txt")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded["content"])
            tmp_path = tmp.name
        config = load_config(str(CONFIG_PATH))
        raise_if_cancelled(job_id)
        update_job(job_id, stage="提取段落", progress=10, message=uploaded["filename"])
        paragraphs = extract_paragraphs(tmp_path, config)
        if not paragraphs:
            raise ValueError("没有提取到有效段落")
        threshold = fields.get("threshold")
        args = argparse.Namespace(engine=None, threshold=float(threshold) if threshold not in (None, "") else None)
        raise_if_cancelled(job_id)
        update_job(job_id, stage="初始化检测引擎", progress=16, message=f"段落数 {len(paragraphs)}")
        engine = build_engine(config, args)
        raise_if_cancelled(job_id)
        update_job(job_id, stage="初检", progress=30, message=f"引擎 {engine.mode}")
        initial_results = engine.detect_batch([p["text"] for p in paragraphs], show_progress=False)
        raise_if_cancelled(job_id)
        initial_summary = summarize(paragraphs, initial_results, engine.get_status())
        refinement_results = asyncio.run(refine_paragraphs_async(paragraphs, initial_results, engine, config, fields, job_id))
        result = {
            "summary": summarize_refinement(paragraphs, refinement_results, initial_summary),
            "paragraphs": serialize_refinement(paragraphs, refinement_results),
            "engine_status": engine.get_status(),
            "filename": uploaded["filename"],
            "mode": "refine",
        }
        update_job(job_id, state="done", stage="完成", progress=100, message="润色完成", result=result)
    except JobCancelled:
        finish_cancelled_job(job_id)
    except Exception as exc:
        update_job(job_id, state="error", stage="失败", progress=100, error=str(exc), message=str(exc))
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def parse_multipart(content_type: str, body: bytes):
    match = re.search(r"boundary=(?P<boundary>[^;]+)", content_type or "")
    if not match:
        raise ValueError("缺少 multipart boundary")
    boundary = match.group("boundary").strip().strip('"').encode()
    fields = {}
    files = {}
    marker = b"--" + boundary
    for part in body.split(marker):
        part = part.strip()
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].rstrip()
        if b"\r\n\r\n" not in part:
            continue
        raw_headers, content = part.split(b"\r\n\r\n", 1)
        headers = raw_headers.decode("utf-8", errors="ignore").split("\r\n")
        disposition = next((h for h in headers if h.lower().startswith("content-disposition:")), "")
        name_match = re.search(r'name="([^"]+)"', disposition)
        if not name_match:
            continue
        name = name_match.group(1)
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        content = content.rstrip(b"\r\n")
        if filename_match and filename_match.group(1):
            files[name] = {"filename": filename_match.group(1), "content": content}
        else:
            fields[name] = content.decode("utf-8", errors="ignore")
    return fields, files


class WebUIHandler(BaseHTTPRequestHandler):
    server_version = "AIGCWebUI/1.0"

    def log_message(self, format, *args):
        return

    def _send_json(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".js":
            ctype = "application/javascript"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if route.startswith("/api/jobs/"):
            job_id = route.rsplit("/", 1)[-1]
            job = snapshot_job(job_id)
            if not job:
                self._send_json({"error": "任务不存在"}, status=404)
                return
            self._send_json(job)
            return
        if route == "/api/config":
            self._send_json(read_web_config())
            return
        if route == "/":
            self._send_file(WEB_ROOT / "index.html")
            return
        target = (WEB_ROOT / route.lstrip("/")).resolve()
        if WEB_ROOT.resolve() not in target.parents and target != WEB_ROOT.resolve():
            self.send_error(403)
            return
        if not target.exists() and not route.startswith("/assets/"):
            self._send_file(WEB_ROOT / "index.html")
            return
        self._send_file(target)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_BODY_BYTES:
                self._send_json({"error": "请求体过大，最大 50MB"}, status=413)
                return
            body = self.rfile.read(length)
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
                parts = parsed.path.strip("/").split("/")
                job_id = parts[2] if len(parts) == 4 else ""
                job = mark_job_cancel_requested(job_id)
                if not job:
                    self._send_json({"error": "任务不存在"}, status=404)
                    return
                self._send_json(job)
                return

            if parsed.path == "/api/detect-text":
                payload = json.loads(body.decode("utf-8"))
                if not payload.get("text", "").strip():
                    self._send_json({"error": "请输入文本"}, status=400)
                    return
                job_id = create_job("文本检测", run_text_job, payload)
                self._send_json({"job_id": job_id})
                return

            if parsed.path == "/api/refine-text":
                payload = json.loads(body.decode("utf-8"))
                if not payload.get("text", "").strip():
                    self._send_json({"error": "请输入文本"}, status=400)
                    return
                job_id = create_job("文本润色", run_text_refine_job, payload)
                self._send_json({"job_id": job_id})
                return

            if parsed.path == "/api/config":
                payload = json.loads(body.decode("utf-8"))
                saved = write_web_config(payload)
                self._send_json({"config": saved})
                return

            if parsed.path == "/api/detect-file":
                fields, files = parse_multipart(self.headers.get("Content-Type", ""), body)
                uploaded = files.get("file")
                if not uploaded:
                    self._send_json({"error": "未上传文件"}, status=400)
                    return
                suffix = Path(uploaded["filename"]).suffix.lower()
                if suffix not in (".docx", ".md", ".txt"):
                    self._send_json({"error": "仅支持 .docx / .md / .txt"}, status=400)
                    return
                job_id = create_job(f"文件检测: {uploaded['filename']}", run_file_job, fields, uploaded)
                self._send_json({"job_id": job_id})
                return

            if parsed.path == "/api/refine-file":
                fields, files = parse_multipart(self.headers.get("Content-Type", ""), body)
                uploaded = files.get("file")
                if not uploaded:
                    self._send_json({"error": "未上传文件"}, status=400)
                    return
                suffix = Path(uploaded["filename"]).suffix.lower()
                if suffix not in (".docx", ".md", ".txt"):
                    self._send_json({"error": "仅支持 .docx / .md / .txt"}, status=400)
                    return
                job_id = create_job(f"文件润色: {uploaded['filename']}", run_file_refine_job, fields, uploaded)
                self._send_json({"job_id": job_id})
                return

            self.send_error(404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)


def main():
    parser = argparse.ArgumentParser(description="AIGC Detector Toolkit WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer((args.host, args.port), WebUIHandler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(f"端口已被占用: {args.host}:{args.port}")
            print(f"查看占用进程: lsof -iTCP:{args.port} -sTCP:LISTEN -n -P")
            print(f"或换一个端口: uv run python webui.py --host {args.host} --port {args.port + 1}")
            return 1
        raise
    print(f"WebUI running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
