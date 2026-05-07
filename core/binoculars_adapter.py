"""
Binoculars 检测器适配器
基于 ahans30/Binoculars 项目 (ICLR 2024)

原理：Binoculars = perplexity / cross-perplexity
- observer model：计算 perplexity（困惑度）
- performer model：计算 cross-perplexity（交叉困惑度）
- AI 生成文本的 Binoculars 分数更低（对两个模型都"可预测"）

两种使用模式：
1. local：本地加载 falcon-7b / falcon-7b-instruct（需 ~14GB 内存）
2. api：通过 OpenAI 兼容 API 计算近似 perplexity（不需要本地大模型）
"""
import os
import sys
import math
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

from .progress import progress_iter

logger = logging.getLogger(__name__)


# ── Binoculars 原始阈值（基于 Falcon-7B / Falcon-7B-Instruct）──
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527


class BinocularsLocalDetector:
    """
    本地模型版 Binoculars（原项目直调）
    需要加载两个 7B 模型，约 14GB 内存
    """

    def __init__(
        self,
        repo_path: str,
        observer_model: str = "tiiuae/falcon-7b",
        performer_model: str = "tiiuae/falcon-7b-instruct",
        device: str = "cpu",
        use_bfloat16: bool = False,
        max_token_observed: int = 512,
        mode: str = "low-fpr",
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.available = False
        self.detector = None
        self._init_detector(
            observer_model, performer_model, device,
            use_bfloat16, max_token_observed, mode,
        )

    def _init_detector(self, observer, performer, device, bf16, max_tok, mode):
        try:
            scripts_path = os.path.join(self.repo_path, 'binoculars')
            if scripts_path not in sys.path:
                sys.path.insert(0, self.repo_path)

            from binoculars import Binoculars

            self.detector = Binoculars(
                observer_name_or_path=observer,
                performer_name_or_path=performer,
                use_bfloat16=bf16,
                max_token_observed=max_tok,
                mode=mode,
            )
            self.available = True
            logger.info("Binoculars (local) 加载成功: observer=%s, performer=%s",
                        observer, performer)
        except Exception as e:
            logger.error("Binoculars (local) 加载失败: %s", e)
            self.available = False

    def detect(self, text: str, min_length: int = 50) -> Dict:
        if not self.available:
            return self._empty()
        if len(text.strip()) < min_length:
            return self._empty()

        try:
            score = self.detector.compute_score(text)
            # Binoculars 分数越低 → 越可能是 AI
            # 转换为 AIGC 分数 (0-1)：用 sigmoid 映射
            # threshold ≈ 0.85-0.90，低于阈值为 AI
            threshold = self.detector.threshold
            # 转换：score < threshold → 高 AIGC 概率
            aigc_score = 1.0 / (1.0 + math.exp(5.0 * (score - threshold)))
            label = 'ai' if score < threshold else 'human'
            confidence = abs(score - threshold) / max(threshold, 1 - threshold)

            return {
                'engine': 'binoculars',
                'aigc_score': round(aigc_score, 4),
                'raw_score': round(float(score), 6),
                'threshold': round(threshold, 6),
                'label': label,
                'confidence': round(min(confidence, 1.0), 4),
                'mode': 'local',
                'available': True,
            }
        except Exception as e:
            logger.warning("Binoculars 检测异常: %s", e)
            return self._empty()

    def _empty(self):
        return {
            'engine': 'binoculars', 'aigc_score': None, 'raw_score': None,
            'label': 'unknown', 'confidence': 0.0, 'mode': 'local', 'available': False,
        }


class BinocularsAPIDetector:
    """
    API 版 Binoculars
    使用 OpenAI 兼容 API 的 logprobs 近似计算 perplexity / cross-perplexity

    核心思想：用同一模型的两次不同调用来模拟 observer/performer：
    - observer 角色：用"续写"模式获取 token logprobs → perplexity
    - performer 角色：用"翻译/改写"模式获取 token logprobs → cross-perplexity
    - Binoculars ≈ perplexity / cross-perplexity
    """

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        min_text_length: int = 30,
        api_concurrency: int = 1,
    ):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.min_text_length = min_text_length
        self.api_concurrency = max(1, int(api_concurrency or 1))
        self.available = False
        self.client = None
        self._init()

    def _init_client(self, timeout: float = 60.0):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=timeout,
        )
        return self.client

    def _init(self):
        if not self.api_key:
            logger.info("Binoculars (API) 未配置 api_key，跳过 API 引擎")
            self.available = False
            self.has_logprobs = False
            return

        try:
            client = self._init_client(timeout=15.0)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                logprobs=True,
                top_logprobs=5,
                temperature=0,
            )
            self.available = True
            self.has_logprobs = False
            data = self._to_dict(resp)
            choice = data.get("choices", [{}])[0]
            if choice.get("logprobs") or choice.get("message", {}).get("logprobs"):
                self.has_logprobs = True
                logger.info("Binoculars (API) 连通成功，logprobs 可用")
            else:
                logger.info("Binoculars (API) 连通成功，无 logprobs，使用基础模式")
        except Exception as e:
            logger.error("Binoculars (API) 连接失败: %s", e)
            self.available = False
            self.has_logprobs = False

    def _call_api(self, messages: list, max_tokens: int = 200) -> Optional[Dict]:
        try:
            client = self.client or self._init_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=5,
                temperature=0,
            )
            return self._to_dict(resp)
        except Exception as e:
            logger.debug("API 调用失败: %s", e)
        return None

    @staticmethod
    def _to_dict(response) -> Dict:
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump"):
            return response.model_dump(exclude_none=True)
        if hasattr(response, "dict"):
            return response.dict()
        return dict(response)

    def _get_logprobs_from_response(self, result: Dict) -> List[float]:
        """从 API 响应中提取 logprobs"""
        try:
            choice = result["choices"][0]
            # OpenAI 兼容格式
            logprobs_obj = choice.get("logprobs", {})
            if logprobs_obj and logprobs_obj.get("token_logprobs"):
                return [lp for lp in logprobs_obj["token_logprobs"] if lp is not None]
            if logprobs_obj and isinstance(logprobs_obj.get("content"), list):
                return [
                    item["logprob"]
                    for item in logprobs_obj["content"]
                    if isinstance(item, dict) and item.get("logprob") is not None
                ]
            # 某些 API 在 message.logprobs 中
            msg_logprobs = choice.get("message", {}).get("logprobs", {})
            if msg_logprobs and msg_logprobs.get("token_logprobs"):
                return [lp for lp in msg_logprobs["token_logprobs"] if lp is not None]
        except (KeyError, IndexError, TypeError):
            pass
        return []

    def _ppl_from_logprobs(self, logprobs: List[float]) -> float:
        if not logprobs:
            return float('inf')
        avg = sum(logprobs) / len(logprobs)
        return math.pow(2, -avg)

    def detect(self, text: str) -> Dict:
        if not self.available:
            return self._empty()
        text = text.strip()
        if len(text) < self.min_text_length:
            return self._empty()

        try:
            if self.has_logprobs:
                return self._detect_with_logprobs(text)
            else:
                return self._detect_basic(text)
        except Exception as e:
            logger.warning("Binoculars API 检测异常: %s", e)
            return self._empty()

    def _detect_with_logprobs(self, text: str) -> Dict:
        """使用 logprobs 计算 perplexity / cross-perplexity"""
        # 观察者模式：续写
        observer_result = self._call_api(
            [{"role": "user", "content": f"请原样复述以下文本：\n{text[:800]}"}],
            max_tokens=200,
        )
        # 表演者模式：改写/翻译
        performer_result = self._call_api(
            [{"role": "user", "content": f"请用自己的话改写以下文本：\n{text[:800]}"}],
            max_tokens=200,
        )

        observer_lps = self._get_logprobs_from_response(observer_result) if observer_result else []
        performer_lps = self._get_logprobs_from_response(performer_result) if performer_result else []

        if not observer_lps or not performer_lps:
            return self._detect_basic(text)

        ppl = self._ppl_from_logprobs(observer_lps)
        x_ppl = self._ppl_from_logprobs(performer_lps)

        if x_ppl <= 0:
            return self._detect_basic(text)

        binoculars_score = ppl / x_ppl

        # 转换为 AIGC 分数
        threshold = BINOCULARS_FPR_THRESHOLD
        aigc_score = 1.0 / (1.0 + math.exp(5.0 * (binoculars_score - threshold)))
        label = 'ai' if binoculars_score < threshold else 'human'

        return {
            'engine': 'binoculars',
            'aigc_score': round(aigc_score, 4),
            'raw_score': round(binoculars_score, 6),
            'perplexity': round(ppl, 2),
            'cross_perplexity': round(x_ppl, 2),
            'threshold': round(threshold, 6),
            'label': label,
            'confidence': round(min(abs(binoculars_score - threshold) / 0.5, 1.0), 4),
            'mode': 'api_logprobs',
            'available': True,
        }

    def _detect_basic(self, text: str) -> Dict:
        """基础模式：通过文本统计特征估算"""
        import re
        sentences = re.split(r'[。！？；\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return self._empty()

        avg_len = sum(len(s) for s in sentences) / len(sentences)
        if len(sentences) > 1:
            lens = [len(s) for s in sentences]
            mean_l = sum(lens) / len(lens)
            std_l = math.sqrt(sum((l - mean_l)**2 for l in lens) / len(lens))
            cv = std_l / mean_l if mean_l > 0 else 0
        else:
            cv = 0

        # Binoculars 风格的分数估算
        # AI 文本特征：句长均匀（低 CV）→ 更低的 binoculars 分数
        score = 0.9  # 基准分
        if cv < 0.3:
            score -= 0.08
        if avg_len > 80:
            score -= 0.05
        if avg_len > 120:
            score -= 0.05

        threshold = BINOCULARS_FPR_THRESHOLD
        aigc_score = 1.0 / (1.0 + math.exp(5.0 * (score - threshold)))
        label = 'ai' if score < threshold else 'human'

        return {
            'engine': 'binoculars',
            'aigc_score': round(aigc_score, 4),
            'raw_score': round(score, 6),
            'threshold': round(threshold, 6),
            'label': label,
            'confidence': round(min(abs(score - threshold) / 0.3, 1.0), 4),
            'mode': 'api_basic',
            'available': True,
        }

    def detect_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict]:
        if not texts:
            return []

        if self.api_concurrency <= 1:
            results = []
            iterable = progress_iter(texts, total=len(texts), desc="Binoculars") if show_progress else texts
            for i, text in enumerate(iterable):
                results.append(self.detect(text))
                if (i + 1) % 10 == 0:
                    logger.info("  Binoculars API 进度: %d/%d", i + 1, len(texts))
            return results

        workers = min(self.api_concurrency, len(texts))
        logger.info("Binoculars API 并发检测: concurrency=%d, total=%d", workers, len(texts))
        results = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(self.detect, text): i for i, text in enumerate(texts)}
            iterable = as_completed(future_map)
            if show_progress:
                iterable = progress_iter(iterable, total=len(future_map), desc="Binoculars")
            for done, future in enumerate(iterable, start=1):
                i = future_map[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    logger.warning("Binoculars API 并发检测异常: %s", e)
                    results[i] = self._empty()
                if done % 10 == 0 or done == len(texts):
                    logger.info("  Binoculars API 进度: %d/%d", done, len(texts))
        return results

    def _empty(self):
        return {
            'engine': 'binoculars', 'aigc_score': None, 'raw_score': None,
            'label': 'unknown', 'confidence': 0.0, 'mode': 'unavailable', 'available': False,
        }
