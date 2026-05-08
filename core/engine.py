"""
核心检测引擎 - 多引擎融合（自包含版）
支持 FengCi0 / HC3+m3e / OpenAI 兼容 API / Binoculars / Local Logprob / Lastde
所有引擎代码已内联，无需克隆外部仓库。
"""
import logging
import time
from typing import Dict, List, Optional

from .fengci_adapter import FengCiAdapter
from .hc3_adapter import HC3Adapter
from .openai_adapter import OpenAICompatibleAPIDetector
from .binoculars_adapter import BinocularsAPIDetector
from .local_logprob_adapter import LocalLogprobDetector
from .lastde_adapter import LastdeAdapter
from .progress import progress_iter

logger = logging.getLogger(__name__)


class DetectionEngine:
    """
    AIGC 检测引擎。
    模式: fengci / hc3 / openai / binoculars / local_logprob / lastde / ensemble
    """

    ENGINE_NAMES = ("fengci", "hc3", "openai", "binoculars", "local_logprob", "lastde")

    def __init__(
        self,
        mode: str = "ensemble",
        # 模型路径（自包含，不再需要外部仓库）
        fengci_model: Optional[str] = None,
        hc3_model: Optional[str] = None,
        # 权重
        fengci_weight: float = 0.30,
        hc3_weight: float = 0.20,
        openai_weight: float = 0.25,
        binoculars_weight: float = 0.25,
        local_logprob_weight: float = 0.0,
        lastde_weight: float = 0.0,
        # 阈值
        aigc_threshold: float = 0.5,
        # OpenAI 兼容 API
        openai_api_base: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        openai_strategy: str = "perplexity",
        # Binoculars
        binoculars_api_base: Optional[str] = None,
        binoculars_api_key: Optional[str] = None,
        binoculars_model: str = "gpt-4o-mini",
        # Local Logprob
        local_logprob_model: str = "",
        local_logprob_device: str = "cpu",
        local_logprob_max_length: int = 512,
        local_logprob_stride: int = 256,
        local_logprob_method: str = "logrank",
        local_logprob_local_files_only: bool = False,
        # Lastde
        lastde_embed_size: int = 3,
        lastde_epsilon_factor: float = 10.0,
        lastde_tau_prime: int = 5,
        lastde_agg: str = "std",
        api_concurrency: int = 1,
    ):
        self.mode = mode
        self.weights = {
            "fengci": fengci_weight,
            "hc3": hc3_weight,
            "openai": openai_weight,
            "binoculars": binoculars_weight,
            "local_logprob": local_logprob_weight,
            "lastde": lastde_weight,
        }
        if mode in self.weights and self.weights[mode] <= 0:
            self.weights[mode] = 1.0
        self.aigc_threshold = aigc_threshold
        self.api_concurrency = max(1, int(api_concurrency or 1))

        self.fengci: Optional[FengCiAdapter] = None
        self.hc3: Optional[HC3Adapter] = None
        self.openai: Optional[OpenAICompatibleAPIDetector] = None
        self.binoculars: Optional[BinocularsAPIDetector] = None
        self.local_logprob: Optional[LocalLogprobDetector] = None
        self.lastde: Optional[LastdeAdapter] = None

        # FengCi0
        if mode in ("fengci", "ensemble"):
            self.fengci = FengCiAdapter(model_path=fengci_model)
            if not self.fengci.available:
                logger.warning("FengCi0 引擎不可用")
                self.weights["fengci"] = 0
                if mode == "fengci":
                    raise RuntimeError("FengCi0 引擎不可用")

        # HC3
        if mode in ("hc3", "ensemble"):
            self.hc3 = HC3Adapter(model_path=hc3_model)
            if not self.hc3.available:
                logger.warning("HC3 引擎不可用")
                self.weights["hc3"] = 0
                if mode == "hc3":
                    raise RuntimeError("HC3 引擎不可用")

        # OpenAI 兼容 API
        if mode in ("openai", "ensemble"):
            self.openai = OpenAICompatibleAPIDetector(
                api_base=openai_api_base or "https://api.openai.com/v1",
                api_key=openai_api_key or "",
                model=openai_model,
                strategy=openai_strategy,
                api_concurrency=self.api_concurrency,
            )
            if not self.openai.available:
                logger.warning("OpenAI 兼容 API 引擎不可用")
                self.weights["openai"] = 0
                if mode == "openai":
                    raise RuntimeError("OpenAI 兼容 API 引擎不可用")

        # Binoculars
        if mode in ("binoculars", "ensemble"):
            self.binoculars = BinocularsAPIDetector(
                api_base=binoculars_api_base or openai_api_base or "https://api.openai.com/v1",
                api_key=binoculars_api_key or openai_api_key or "",
                model=binoculars_model or openai_model,
                api_concurrency=self.api_concurrency,
            )
            if not self.binoculars.available:
                logger.warning("Binoculars 引擎不可用")
                self.weights["binoculars"] = 0
                if mode == "binoculars":
                    raise RuntimeError("Binoculars 引擎不可用")

        # Local Logprob
        if mode == "local_logprob" or (mode == "ensemble" and local_logprob_weight > 0):
            self.local_logprob = LocalLogprobDetector(
                model_name_or_path=local_logprob_model,
                device=local_logprob_device,
                max_length=local_logprob_max_length,
                stride=local_logprob_stride,
                method=local_logprob_method,
                local_files_only=local_logprob_local_files_only,
            )
            if not self.local_logprob.available:
                logger.warning("Local Logprob 引擎不可用")
                self.weights["local_logprob"] = 0
                if mode == "local_logprob":
                    raise RuntimeError("Local Logprob 引擎不可用")

        # Lastde (ICLR 2025)
        if mode == "lastde" or (mode == "ensemble" and lastde_weight > 0):
            self.lastde = LastdeAdapter(
                api_base=openai_api_base or "https://api.openai.com/v1",
                api_key=openai_api_key or "",
                model=openai_model,
                embed_size=lastde_embed_size,
                epsilon_factor=lastde_epsilon_factor,
                tau_prime=lastde_tau_prime,
                agg=lastde_agg,
            )
            if not self.lastde.available:
                logger.warning("Lastde 引擎不可用")
                self.weights["lastde"] = 0
                if mode == "lastde":
                    raise RuntimeError("Lastde 引擎不可用")

        if mode == "ensemble":
            if sum(1 for v in self.weights.values() if v > 0) == 0:
                raise RuntimeError("ensemble 模式下没有可用的检测引擎")
            self._normalize_weights()

        self._log_status()

    def _normalize_weights(self):
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] /= total

    def _log_status(self):
        parts = []
        for name in self.ENGINE_NAMES:
            obj = getattr(self, name, None)
            avail = obj.available if obj and hasattr(obj, "available") else False
            w = self.weights.get(name, 0)
            icon = "✓" if avail else "✗"
            parts.append(f"{name}={icon}({w:.0%})")
        logger.info("引擎状态: mode=%s | %s", self.mode, " | ".join(parts))

    def detect_single(self, text: str) -> Dict:
        engine_results = {}
        scores, weights = [], []
        for name in self.ENGINE_NAMES:
            obj = getattr(self, name, None)
            if not obj or not obj.available or self.weights.get(name, 0) <= 0:
                continue
            r = obj.detect(text)
            engine_results[name] = r
            if r.get("aigc_score") is not None:
                scores.append(r["aigc_score"])
                weights.append(self.weights[name])
        return self._fuse(scores, weights, engine_results)

    def detect_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict]:
        total = len(texts)
        start_time = time.time()

        # 先跑支持批量的引擎
        batch_results = {}
        for name in ("hc3", "openai", "binoculars", "local_logprob", "lastde"):
            obj = getattr(self, name, None)
            if not obj or not obj.available or self.weights.get(name, 0) <= 0:
                continue
            if hasattr(obj, "detect_batch"):
                try:
                    batch_results[name] = obj.detect_batch(texts, show_progress=show_progress)
                except TypeError:
                    batch_results[name] = obj.detect_batch(texts)
                logger.info("%s 引擎批量推理完成", name)

        results = []
        iterable = progress_iter(list(enumerate(texts)), total=total, desc="融合结果") if show_progress else enumerate(texts)
        for i, text in iterable:
            engine_results = {}
            scores, weights_list = [], []

            # FengCi0 逐条
            if self.fengci and self.fengci.available and self.weights.get("fengci", 0) > 0:
                r = self.fengci.detect(text)
                engine_results["fengci"] = r
                if r.get("aigc_score") is not None:
                    scores.append(r["aigc_score"])
                    weights_list.append(self.weights["fengci"])

            for name in ("hc3", "openai", "binoculars", "local_logprob", "lastde"):
                if name in batch_results and batch_results[name][i].get("aigc_score") is not None:
                    r = batch_results[name][i]
                    engine_results[name] = r
                    scores.append(r["aigc_score"])
                    weights_list.append(self.weights[name])

            results.append(self._fuse(scores, weights_list, engine_results))

            if show_progress and ((i + 1) % 50 == 0 or i == 0 or i == total - 1):
                elapsed = time.time() - start_time
                eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
                logger.info("  进度: %d/%d (%.0f秒, 剩余%.0f秒)", i + 1, total, elapsed, eta)

        elapsed = time.time() - start_time
        logger.info("批量检测完成: %d 段, %.1f 秒", total, elapsed)
        return results

    def _fuse(self, scores, weights, engine_results) -> Dict:
        if not scores:
            return {
                "aigc_score": None, "label": "unknown",
                "confidence": 0.0, "engine_results": engine_results, "method": self.mode,
            }
        if len(scores) == 1:
            final_score = scores[0]
            method = list(engine_results.keys())[0]
        else:
            final_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            method = "ensemble"

        label = "ai" if final_score > self.aigc_threshold else "human"
        confs = [r.get("confidence", 0) for r in engine_results.values() if r.get("confidence", 0) > 0]
        confidence = sum(confs) / len(confs) if confs else abs(final_score - 0.5) * 2
        return {
            "aigc_score": round(final_score, 4),
            "label": label,
            "confidence": round(confidence, 4),
            "engine_results": engine_results,
            "method": method,
        }

    def get_status(self) -> Dict:
        status = {"mode": self.mode, "threshold": self.aigc_threshold, "api_concurrency": self.api_concurrency}
        for name in self.ENGINE_NAMES:
            obj = getattr(self, name, None)
            status[f"{name}_available"] = obj.available if obj and hasattr(obj, "available") else False
            status[f"{name}_weight"] = self.weights.get(name, 0)
        return status
