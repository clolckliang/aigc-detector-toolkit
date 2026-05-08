"""
Lastde Detector 适配器 — ICLR 2025
Training-free LLM-generated Text Detection by Mining Token Probability Sequences

移植自 TrustMedia-zju/Lastde_Detector，适配 OpenAI 兼容 API。

核心算法：
  Lastde = mean(log_likelihood) / MDE
  MDE = std( DE(tau=1), DE(tau=2), ..., DE(tau=tau_prime) )

其中 DE(tau) 是在尺度 tau 下的 Distributional Entropy，
通过对 log-likelihood 序列构建 orbits、计算余弦相似度、离散化后求熵。
"""
import logging
import math
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================
# fastMDE: Fast Multiscale Distributional Entropy
# 移植自 TrustMedia-zju/Lastde_Detector/py_scripts/baselines/scoring_methods/fastMDE.py
# ============================================================

def _histcounts(data, epsilon, min_=-1, max_=1):
    """将数据离散化到 epsilon 个区间，返回频率分布。"""
    try:
        import torch
        hist = torch.histc(data.float(), bins=epsilon, min=min_, max=max_)
        total = torch.sum(hist)
        if total > 0:
            probs = hist / total
        else:
            probs = torch.zeros_like(hist)
        return hist, probs
    except Exception:
        # numpy fallback
        import numpy as np
        counts, _ = np.histogram(data, bins=epsilon, range=(min_, max_))
        total = counts.sum()
        probs = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
        return counts, probs


def _distributional_entropy(probs, epsilon):
    """计算分布熵 DE = -1/ln(ε) * Σ p*log(p)"""
    import math
    de = 0.0
    for p in probs:
        p_val = float(p)
        if p_val > 0:
            de -= p_val * math.log(p_val)
    log_eps = math.log(epsilon)
    return de / log_eps if log_eps > 0 else 0.0


def _calculate_de_numpy(log_likelihoods, embed_size, epsilon):
    """用 numpy 计算单尺度 DE（CPU fallback）。"""
    import numpy as np

    seq = np.array(log_likelihoods, dtype=np.float64)
    if len(seq) < embed_size + 1:
        return 0.0

    # 构建 orbits
    n_orbits = len(seq) - embed_size + 1
    orbits = np.array([seq[i:i + embed_size] for i in range(n_orbits)])

    # 计算相邻 orbits 的余弦相似度
    similarities = []
    for i in range(len(orbits) - 1):
        a, b = orbits[i], orbits[i + 1]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a > 0 and norm_b > 0:
            sim = np.dot(a, b) / (norm_a * norm_b)
        else:
            sim = 0.0
        similarities.append(max(-1.0, min(1.0, sim)))

    if not similarities:
        return 0.0

    # 离散化 + 熵
    sim_tensor = np.array(similarities)
    counts, probs = _histcounts_numpy(sim_tensor, epsilon)
    return _distributional_entropy(probs, epsilon)


def _histcounts_numpy(data, epsilon, min_=-1, max_=1):
    """numpy 版 histcounts。"""
    import numpy as np
    counts, _ = np.histogram(data, bins=epsilon, range=(min_, max_))
    total = counts.sum()
    probs = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
    return counts, probs


def get_tau_scale_de(log_likelihoods, embed_size, epsilon, tau):
    """计算单尺度 tau 下的 DE。"""
    import numpy as np

    seq = np.array(log_likelihoods, dtype=np.float64)
    if len(seq) < tau:
        return 0.0

    # tau 尺度子序列：滑动窗口均值
    if tau > 1:
        windows = np.lib.stride_tricks.sliding_window_view(seq, tau)
        tau_seq = np.mean(windows, axis=-1)
    else:
        tau_seq = seq

    return _calculate_de_numpy(tau_seq, embed_size, epsilon)


def get_tau_multiscale_de(log_likelihoods, embed_size, epsilon, tau_prime):
    """
    计算多尺度 DE 的标准差（Lastde 的核心统计量）。

    Args:
        log_likelihoods: per-token log-likelihood 序列
        embed_size: orbit 嵌入维度 (s)
        epsilon: 区间划分粒度 (ε)
        tau_prime: 最大尺度 (τ')
    Returns:
        MDE = std(DE(1), DE(2), ..., DE(tau_prime))
    """
    import numpy as np

    de_values = []
    for tau in range(1, tau_prime + 1):
        de = get_tau_scale_de(log_likelihoods, embed_size, epsilon, tau)
        de_values.append(de)

    if not de_values:
        return 0.0

    return float(np.std(de_values))


# ============================================================
# Lastde Adapter
# ============================================================

class LastdeAdapter:
    """
    Lastde 检测器适配器。

    通过 OpenAI 兼容 API 获取 per-token logprobs，
    计算 Lastde = mean(log_likelihood) / MDE 统计量。

    参数：
        embed_size (s): orbit 嵌入维度，默认 3
        epsilon_factor: ε = epsilon_factor * n_tokens，默认 10
        tau_prime (τ'): 最大尺度，默认 5
        agg: 聚合函数，可选 std / expstd / 2norm / range / exprange
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        embed_size: int = 3,
        epsilon_factor: float = 10.0,
        tau_prime: int = 5,
        agg: str = "std",
        min_text_length: int = 30,
        temperature: float = 0,
    ):
        self.api_base = (api_base or "").rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("MIMO_API_KEY", "")
        )
        self.model = model
        self.embed_size = embed_size
        self.epsilon_factor = epsilon_factor
        self.tau_prime = tau_prime
        self.agg = agg
        self.min_text_length = min_text_length
        self.temperature = temperature
        self.available = False
        self.client = None

        self._init()

    def _init(self):
        if not self.api_key:
            logger.info("Lastde: 未配置 api_key，跳过")
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=60.0,
            )
            # 测试连通性
            resp = self.client.completions.create(
                model=self.model,
                prompt="测试",
                max_tokens=1,
                logprobs=1,
                temperature=self.temperature,
            )
            data = self._to_dict(resp)
            if data.get("choices") and data["choices"][0].get("logprobs"):
                self.available = True
                logger.info("Lastde 引擎初始化成功 (model=%s)", self.model)
            else:
                logger.warning("Lastde: API 无 logprobs 支持，不可用")
        except Exception as e:
            logger.warning("Lastde 初始化失败: %s", e)

    @staticmethod
    def _to_dict(response) -> Dict:
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump"):
            return response.model_dump(exclude_none=True)
        if hasattr(response, "dict"):
            return response.dict()
        return dict(response)

    def _get_token_logprobs(self, text: str) -> List[float]:
        """通过 completions echo 模式获取 per-token logprobs。"""
        try:
            resp = self.client.completions.create(
                model=self.model,
                prompt=text[:4000],
                max_tokens=0,
                logprobs=1,
                echo=True,
                temperature=self.temperature,
            )
            data = self._to_dict(resp)
            choice = data["choices"][0]
            logprobs_data = choice.get("logprobs") or {}
            token_logprobs = logprobs_data.get("token_logprobs") or []
            return [lp for lp in token_logprobs if lp is not None]
        except Exception as e:
            logger.debug("Lastde echo 模式失败: %s", e)
            return []

    def _compute_lastde(self, token_logprobs: List[float]) -> float:
        """计算 Lastde 统计量。"""
        if len(token_logprobs) < self.embed_size + self.tau_prime:
            return 0.0

        n = len(token_logprobs)
        epsilon = max(int(self.epsilon_factor * n), self.embed_size + 1)

        # mean log-likelihood
        mean_ll = sum(token_logprobs) / n

        # MDE
        mde = get_tau_multiscale_de(
            token_logprobs,
            embed_size=self.embed_size,
            epsilon=epsilon,
            tau_prime=self.tau_prime,
        )

        if mde <= 0:
            return 0.0

        return mean_ll / mde

    def _lastde_to_aigc_score(self, lastde: float) -> float:
        """
        将 Lastde 统计量映射到 AIGC 分数 (0-1)。

        Lastde 值越低（mean_ll 低或 MDE 高）→ 越可能是人类文本
        Lastde 值越高（mean_ll 高且 MDE 低）→ 越可能是 AI 文本

        AI 文本特征：log-likelihood 序列更平坦 → MDE 更低 → Lastde 更高
        """
        # sigmoid 映射：score = 1 / (1 + exp(-k * (lastde - threshold)))
        # Lastde 通常在 -5 到 5 范围内
        threshold = 0.0
        k = 1.0
        score = 1.0 / (1.0 + math.exp(-k * (lastde - threshold)))
        return round(min(max(score, 0.0), 1.0), 4)

    def detect(self, text: str) -> Dict:
        if not self.available:
            return self._empty_result()

        text = text.strip()
        if len(text) < self.min_text_length:
            return self._empty_result()

        try:
            token_logprobs = self._get_token_logprobs(text)
            if len(token_logprobs) < 10:
                return self._empty_result()

            lastde = self._compute_lastde(token_logprobs)
            score = self._lastde_to_aigc_score(lastde)

            return {
                "engine": "lastde",
                "aigc_score": score,
                "lastde_stat": round(lastde, 4),
                "ntokens": len(token_logprobs),
                "label": "ai" if score > 0.5 else "human",
                "confidence": round(max(score, 1 - score), 4),
                "available": True,
            }
        except Exception as e:
            logger.warning("Lastde 检测异常: %s", e)
            return self._empty_result()

    def detect_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict]:
        if not self.available:
            return [self._empty_result() for _ in texts]

        from .progress import progress_iter
        results = []
        iterable = progress_iter(texts, total=len(texts), desc="Lastde") if show_progress else texts
        for i, text in enumerate(iterable):
            results.append(self.detect(text))
        return results

    def _empty_result(self) -> Dict:
        return {
            "engine": "lastde",
            "aigc_score": None,
            "lastde_stat": None,
            "ntokens": 0,
            "label": "unknown",
            "confidence": 0.0,
            "available": False,
        }
