"""
核心检测引擎 - 四引擎融合（自包含版）
支持 FengCi0 / HC3+m3e / MiMo API / Binoculars
所有引擎代码已内联，无需克隆外部仓库。
"""
import logging
import time
from typing import Dict, List, Optional

from .fengci_adapter import FengCiAdapter
from .hc3_adapter import HC3Adapter
from .mimo_api_adapter import MiMoAPIDetector
from .binoculars_adapter import BinocularsAPIDetector

logger = logging.getLogger(__name__)


class DetectionEngine:
    """
    AIGC 检测引擎。
    模式: fengci / hc3 / mimo / binoculars / ensemble
    """

    ENGINE_NAMES = ("fengci", "hc3", "mimo", "binoculars")

    def __init__(
        self,
        mode: str = "ensemble",
        # 模型路径（自包含，不再需要外部仓库）
        fengci_model: Optional[str] = None,
        hc3_model: Optional[str] = None,
        # 权重
        fengci_weight: float = 0.30,
        hc3_weight: float = 0.20,
        mimo_weight: float = 0.25,
        binoculars_weight: float = 0.25,
        # 阈值
        aigc_threshold: float = 0.5,
        # MiMo API
        mimo_api_base: Optional[str] = None,
        mimo_api_key: Optional[str] = None,
        mimo_model: str = "mimo-v2.5-pro",
        mimo_strategy: str = "perplexity",
        # Binoculars
        binoculars_api_base: Optional[str] = None,
        binoculars_api_key: Optional[str] = None,
        binoculars_model: str = "mimo-v2.5-pro",
    ):
        self.mode = mode
        self.weights = {
            "fengci": fengci_weight,
            "hc3": hc3_weight,
            "mimo": mimo_weight,
            "binoculars": binoculars_weight,
        }
        self.aigc_threshold = aigc_threshold

        self.fengci: Optional[FengCiAdapter] = None
        self.hc3: Optional[HC3Adapter] = None
        self.mimo: Optional[MiMoAPIDetector] = None
        self.binoculars: Optional[BinocularsAPIDetector] = None

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

        # MiMo API
        if mode in ("mimo", "ensemble"):
            self.mimo = MiMoAPIDetector(
                api_base=mimo_api_base or "https://token-plan-cn.xiaomimimo.com/v1",
                api_key=mimo_api_key or "",
                model=mimo_model,
                strategy=mimo_strategy,
            )
            if not self.mimo.available:
                logger.warning("MiMo API 引擎不可用")
                self.weights["mimo"] = 0
                if mode == "mimo":
                    raise RuntimeError("MiMo API 引擎不可用")

        # Binoculars
        if mode in ("binoculars", "ensemble"):
            self.binoculars = BinocularsAPIDetector(
                api_base=binoculars_api_base or mimo_api_base or "https://token-plan-cn.xiaomimimo.com/v1",
                api_key=binoculars_api_key or mimo_api_key or "",
                model=binoculars_model or mimo_model,
            )
            if not self.binoculars.available:
                logger.warning("Binoculars 引擎不可用")
                self.weights["binoculars"] = 0
                if mode == "binoculars":
                    raise RuntimeError("Binoculars 引擎不可用")

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
        for name in ("hc3", "mimo", "binoculars"):
            obj = getattr(self, name, None)
            if not obj or not obj.available or self.weights.get(name, 0) <= 0:
                continue
            if hasattr(obj, "detect_batch"):
                batch_results[name] = obj.detect_batch(texts)
                logger.info("%s 引擎批量推理完成", name)

        results = []
        for i, text in enumerate(texts):
            engine_results = {}
            scores, weights_list = [], []

            # FengCi0 逐条
            if self.fengci and self.fengci.available and self.weights.get("fengci", 0) > 0:
                r = self.fengci.detect(text)
                engine_results["fengci"] = r
                if r.get("aigc_score") is not None:
                    scores.append(r["aigc_score"])
                    weights_list.append(self.weights["fengci"])

            for name in ("hc3", "mimo", "binoculars"):
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
        status = {"mode": self.mode, "threshold": self.aigc_threshold}
        for name in self.ENGINE_NAMES:
            obj = getattr(self, name, None)
            status[f"{name}_available"] = obj.available if obj and hasattr(obj, "available") else False
            status[f"{name}_weight"] = self.weights.get(name, 0)
        return status
