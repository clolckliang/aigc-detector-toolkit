"""Optional local causal-LM detector using perplexity and LogRank signals."""
import logging
import math
from typing import Dict, List, Optional

from .progress import progress_iter

logger = logging.getLogger(__name__)


class LocalLogprobDetector:
    """
    Local HuggingFace causal-LM detector.

    This engine is intentionally opt-in: it never downloads or loads a model
    unless a model_name_or_path is provided by config or constructor.
    """

    def __init__(
        self,
        model_name_or_path: str = "",
        device: str = "cpu",
        max_length: int = 512,
        stride: int = 256,
        method: str = "logrank",
        min_text_length: int = 30,
        local_files_only: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_length = max_length
        self.stride = stride
        self.method = method
        self.min_text_length = min_text_length
        self.local_files_only = local_files_only
        self.available = False
        self.tokenizer = None
        self.model = None
        self.error = None

        self._init_detector()

    def _init_detector(self):
        if not self.model_name_or_path:
            self.error = "missing_model_name_or_path"
            logger.info("Local Logprob 未配置模型，跳过本地 logprob 引擎")
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                local_files_only=self.local_files_only,
                trust_remote_code=False,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                local_files_only=self.local_files_only,
                trust_remote_code=False,
            )
            self.model.to(self.device)
            self.model.eval()
            self.available = True
            logger.info("Local Logprob 检测器加载成功: %s", self.model_name_or_path)
        except Exception as e:
            self.error = str(e)
            self.available = False
            logger.warning("Local Logprob 检测器不可用: %s", e)

    def detect(self, text: str) -> Dict:
        if not self.available:
            return self._empty_result()
        text = text.strip()
        if len(text) < self.min_text_length:
            return self._empty_result(method="text_too_short")

        try:
            stats = self._compute_token_stats(text)
            score = self._score_from_stats(stats)
            label = "ai" if score > 0.5 else "human"
            return {
                "engine": "local_logprob",
                "aigc_score": round(score, 4),
                "label": label,
                "confidence": round(max(score, 1 - score), 4),
                "method": self.method,
                "features": {
                    "perplexity": round(stats["perplexity"], 4),
                    "nll": round(stats["nll"], 4),
                    "logrank": round(stats["logrank"], 4) if stats["logrank"] is not None else None,
                    "tokens": stats["tokens"],
                },
                "available": True,
            }
        except Exception as e:
            logger.warning("Local Logprob 检测异常: %s", e)
            return self._empty_result(method="failed")

    def detect_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict]:
        iterable = progress_iter(texts, total=len(texts), desc="Local Logprob") if show_progress else texts
        return [self.detect(text) for text in iterable]

    def _compute_token_stats(self, text: str) -> Dict:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=False)
        all_input_ids = encoded["input_ids"].to(self.device)
        if all_input_ids.shape[1] < 2:
            raise ValueError("tokenized text too short")

        window_stats = []
        step = self.stride if self.stride > 0 else self.max_length
        step = min(step, self.max_length)
        for start in range(0, all_input_ids.shape[1] - 1, step):
            end = min(start + self.max_length, all_input_ids.shape[1])
            window_ids = all_input_ids[:, start:end]
            if window_ids.shape[1] < 2:
                continue
            window_stats.append(self._compute_window_stats(window_ids))
            if end == all_input_ids.shape[1]:
                break

        if not window_stats:
            raise ValueError("no valid token windows")

        total_tokens = sum(item["tokens"] for item in window_stats)
        nll = sum(item["nll"] * item["tokens"] for item in window_stats) / total_tokens
        logrank_values = [item for item in window_stats if item["logrank"] is not None]
        logrank = None
        if logrank_values:
            logrank = sum(item["logrank"] * item["tokens"] for item in logrank_values) / sum(
                item["tokens"] for item in logrank_values
            )

        return {
            "nll": nll,
            "perplexity": math.exp(min(nll, 20)),
            "logrank": logrank,
            "tokens": int(total_tokens),
        }

    def _compute_window_stats(self, input_ids) -> Dict:
        torch = self.torch
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            nll = -token_log_probs.mean().item()

            logrank = None
            if self.method == "logrank":
                target_logits = logits.gather(-1, labels.unsqueeze(-1))
                rank_values = (logits > target_logits).sum(dim=-1).float() + 1.0
                logrank = torch.log(rank_values).mean().item()

        return {
            "nll": nll,
            "logrank": logrank,
            "tokens": int(labels.numel()),
        }

    def _score_from_stats(self, stats: Dict) -> float:
        if self.method == "perplexity":
            # Lower perplexity is more AI-like. Threshold is intentionally conservative.
            value = math.log(stats["perplexity"] + 1)
            score = 1.0 / (1.0 + math.exp(1.2 * (value - 3.0)))
        else:
            # Lower logrank is more AI-like. This mapping should be calibrated with eval.
            value = stats["logrank"] if stats["logrank"] is not None else stats["nll"]
            score = 1.0 / (1.0 + math.exp(1.0 * (value - 3.5)))
        return min(max(score, 0.0), 1.0)

    def _empty_result(self, method: str = "unavailable") -> Dict:
        return {
            "engine": "local_logprob",
            "aigc_score": None,
            "label": "unknown",
            "confidence": 0.0,
            "method": method,
            "features": {},
            "available": False,
        }
