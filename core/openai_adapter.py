"""
OpenAI 兼容 API 检测器适配器
使用 OpenAI 兼容接口的 logprobs 功能进行 AIGC 检测
基于 Fast-DetectGPT 的条件概率曲率思想，通过 API 实现

原理：
- 利用 LLM 的 logprobs 计算文本的 token 级别概率
- 通过前缀预测后缀的"惊讶度"来判断文本是否为 AI 生成
- AI 生成的文本对 LLM 而言"更可预测"（更低的困惑度）
"""
import os
import re
import math
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from .progress import progress_iter

logger = logging.getLogger(__name__)


class OpenAICompatibleAPIDetector:
    """
    基于 OpenAI 兼容 API 的 AIGC 检测器

    使用 OpenAI 兼容 API 的 logprobs 功能，
    计算文本的"条件概率曲率"近似值。

    支持两种检测策略：
    1. 完整文本 perplexity：计算整段文本在模型下的困惑度
    2. 前缀-后缀对比：用前缀预测后缀，计算采样偏差
    """

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        min_text_length: int = 30,
        window_size: int = 200,
        strategy: str = "perplexity",
        api_concurrency: int = 1,
    ):
        self.api_base = api_base.rstrip('/')
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("MIMO_API_KEY", "")
        )
        self.model = model
        self.min_text_length = min_text_length
        self.window_size = window_size
        self.strategy = strategy
        self.api_concurrency = max(1, int(api_concurrency or 1))
        self.available = False
        self.client = None
        self.supports_echo = None

        self._init_detector()

    def _init_client(self, timeout: float = 60.0):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=timeout,
        )
        return self.client

    def _init_detector(self):
        """测试 API 连通性"""
        if not self.api_key:
            logger.info("OpenAI 兼容 API 未配置 api_key，跳过 API 引擎")
            self.available = False
            self.use_completions = False
            return

        try:
            client = self._init_client(timeout=30.0)

            # 尝试 completions endpoint with logprobs
            try:
                resp = client.completions.create(
                    model=self.model,
                    prompt="测试",
                    max_tokens=1,
                    logprobs=1,
                    temperature=0,
                )
                data = self._to_dict(resp)
                # 检查是否返回了 logprobs
                if data.get("choices") and data["choices"][0].get("logprobs"):
                    self.available = True
                    self.use_completions = True
                    logger.info("OpenAI 兼容 API (completions+logprobs) 连通成功")
                    return
                else:
                    logger.info("OpenAI 兼容 API completions 可用但无 logprobs，尝试 chat 模式")
            except Exception as e:
                logger.info("OpenAI 兼容 API completions 不可用，尝试 chat 模式: %s", e)

            # 尝试 chat completions with logprobs
            try:
                client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "测试"}],
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=1,
                    temperature=0,
                )
                self.available = True
                self.use_completions = False
                self.supports_echo = False
                logger.info("OpenAI 兼容 API (chat+logprobs) 连通成功")
            except Exception as e:
                # API 可用但不支持 logprobs，退化为基础模式
                logger.warning("OpenAI 兼容 API chat/logprobs 不可用，将使用基础文本分析模式: %s", e)
                self.available = True
                self.use_completions = False
                self.supports_echo = False
                self.strategy = "basic"

        except Exception as e:
            logger.error("OpenAI 兼容 API 连接失败: %s", e)
            self.available = False
            self.use_completions = False

    def _call_api(self, prompt: str, max_tokens: int = 1) -> Optional[Dict]:
        """调用 OpenAI 兼容 API"""
        try:
            client = self.client or self._init_client()
            if self.use_completions:
                resp = client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    logprobs=5,
                    temperature=0,
                )
            else:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": f"请继续以下文本:\n\n{prompt}"}],
                    max_tokens=max_tokens,
                    logprobs=True,
                    top_logprobs=5,
                    temperature=0,
                )
            return self._to_dict(resp)
        except Exception as e:
            logger.debug("API 调用异常: %s", e)
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

    def _compute_perplexity_from_logprobs(self, logprobs_list: List[float]) -> float:
        """从 logprobs 列表计算困惑度"""
        if not logprobs_list:
            return float('inf')
        avg_logprob = sum(logprobs_list) / len(logprobs_list)
        # perplexity = 2^(-avg_logprob)
        return math.pow(2, -avg_logprob)

    def _extract_logprobs(self, choice: Dict) -> List[float]:
        """兼容 completions 与 chat.completions 的 logprobs 响应结构。"""
        logprobs_data = choice.get("logprobs") or {}

        token_logprobs = logprobs_data.get("token_logprobs")
        if token_logprobs:
            return [lp for lp in token_logprobs if lp is not None]

        content_logprobs = logprobs_data.get("content")
        if isinstance(content_logprobs, list):
            values = []
            for item in content_logprobs:
                if isinstance(item, dict) and item.get("logprob") is not None:
                    values.append(item["logprob"])
            return values

        msg_logprobs = choice.get("message", {}).get("logprobs", {})
        token_logprobs = msg_logprobs.get("token_logprobs")
        if token_logprobs:
            return [lp for lp in token_logprobs if lp is not None]

        return []

    def _compute_text_surprise(self, text: str) -> Dict:
        """
        计算文本的"惊讶度"

        策略：将文本分段，用前半段作为 prompt，让模型预测后半段，
        比较模型预测和实际后半段的一致性。

        AI 生成的文本一致性更高（更"不惊讶"）
        """
        # 分割文本
        text = text.strip()
        mid = len(text) // 2
        # 在最近的标点处切分
        cut_pos = mid
        for offset in range(min(50, mid)):
            if mid + offset < len(text) and text[mid + offset] in '。！？；\n':
                cut_pos = mid + offset + 1
                break
            if mid - offset > 0 and text[mid - offset] in '。！？；\n':
                cut_pos = mid - offset + 1
                break

        prefix = text[:cut_pos].strip()
        suffix = text[cut_pos:].strip()

        if len(prefix) < 10 or len(suffix) < 10:
            return {'score': None, 'method': 'text_too_short'}

        # 用 API 获取后半段文本的 logprobs
        result = self._call_api(prefix, max_tokens=min(len(suffix), 200))
        if not result:
            return {'score': None, 'method': 'api_failed'}

        # 从返回结果中提取 logprobs
        try:
            if self.use_completions:
                choice = result["choices"][0]
                token_logprobs = self._extract_logprobs(choice)
            else:
                choice = result["choices"][0]
                token_logprobs = self._extract_logprobs(choice)

            if not token_logprobs:
                return {'score': None, 'method': 'no_logprobs'}

            # 计算困惑度
            valid_logprobs = [lp for lp in token_logprobs if lp is not None]
            if not valid_logprobs:
                return {'score': None, 'method': 'no_valid_logprobs'}

            ppl = self._compute_perplexity_from_logprobs(valid_logprobs)
            avg_logprob = sum(valid_logprobs) / len(valid_logprobs)

            return {
                'score': ppl,
                'avg_logprob': avg_logprob,
                'ntokens': len(valid_logprobs),
                'method': 'logprobs',
            }

        except (KeyError, IndexError, TypeError) as e:
            logger.debug("解析 logprobs 失败: %s", e)
            return {'score': None, 'method': 'parse_failed'}

    def _compute_full_text_perplexity(self, text: str) -> Dict:
        """
        计算完整文本的困惑度（通过 completions echo 模式）

        如果 API 支持 echo=True，可以直接获取整段文本的 logprobs
        """
        if self.use_completions and self.supports_echo is False:
            return self._compute_text_surprise(text)

        try:
            # 尝试 echo 模式（获取 prompt 中每个 token 的 logprobs）
            client = self.client or self._init_client()
            resp = client.completions.create(
                model=self.model,
                prompt=text[:2000],  # 限制长度
                max_tokens=0,
                logprobs=5,
                echo=True,
                temperature=0,
            )
            data = self._to_dict(resp)
            choice = data["choices"][0]
            token_logprobs = self._extract_logprobs(choice)

            if token_logprobs:
                valid_logprobs = [lp for lp in token_logprobs if lp is not None]
                if valid_logprobs:
                    ppl = self._compute_perplexity_from_logprobs(valid_logprobs)
                    avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
                    self.supports_echo = True
                    return {
                        'score': ppl,
                        'avg_logprob': avg_logprob,
                        'ntokens': len(valid_logprobs),
                        'method': 'full_echo',
                    }
            self.supports_echo = False

        except Exception as e:
            logger.debug("echo 模式失败: %s", e)
            self.supports_echo = False

        # 退化到文本分割策略
        return self._compute_text_surprise(text)

    def _text_length_analysis(self, text: str) -> Dict:
        """
        基础文本分析（不依赖 API logprobs）

        使用文本统计特征进行启发式判断
        """
        # 计算基本文本统计特征
        sentences = re.split(r'[。！？；\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {'score': 0.5, 'method': 'basic_no_sentences'}

        # 平均句长
        avg_sent_len = sum(len(s) for s in sentences) / len(sentences)
        # 句长变异系数
        if len(sentences) > 1:
            sent_lens = [len(s) for s in sentences]
            mean_l = sum(sent_lens) / len(sent_lens)
            std_l = math.sqrt(sum((l - mean_l)**2 for l in sent_lens) / len(sent_lens))
            cv = std_l / mean_l if mean_l > 0 else 0
        else:
            cv = 0

        # 词汇多样性
        words = list(text)
        unique_chars = len(set(words))
        total_chars = len(words)
        diversity = unique_chars / total_chars if total_chars > 0 else 0

        # AI 文本特征：句长均匀（低 CV）、句子偏长、词汇多样性中等
        score = 0.5
        if cv < 0.3:
            score += 0.1  # 句长太均匀
        if avg_sent_len > 80:
            score += 0.1  # 句子偏长
        if avg_sent_len > 120:
            score += 0.1  # 句子很长
        if diversity < 0.15:
            score += 0.05  # 用字单一

        score = min(max(score, 0.0), 1.0)
        return {'score': score, 'method': 'basic_stats'}

    def _perplexity_to_aigc_score(self, ppl: float, method: str) -> float:
        """
        将困惑度转换为 AIGC 分数 (0-1)

        较低的困惑度 → 较高的 AIGC 分数
        使用 sigmoid 映射
        """
        if ppl <= 0:
            return 0.5

        log_ppl = math.log(ppl + 1)

        # 基于经验的阈值映射
        # 困惑度 < 5 → 高概率 AI（文本对模型非常"可预测"）
        # 困惑度 5-20 → 中等
        # 困惑度 > 20 → 可能是人工（文本对模型"惊讶"）
        # 使用 sigmoid: score = 1 / (1 + exp(k * (log_ppl - threshold)))
        threshold = 2.5  # log(12) ≈ 2.5
        k = 1.5  # 敏感度

        score = 1.0 / (1.0 + math.exp(k * (log_ppl - threshold)))
        return round(min(max(score, 0.0), 1.0), 4)

    def detect(self, text: str) -> Dict:
        """
        检测单段文本

        Returns:
            {
                'engine': 'openai-api',
                'aigc_score': float,    # 0-1
                'perplexity': float,    # 原始困惑度
                'avg_logprob': float,   # 平均 log 概率
                'label': str,
                'confidence': float,
                'method': str,
                'available': bool
            }
        """
        if not self.available:
            return self._empty_result()

        text = text.strip()
        if len(text) < self.min_text_length:
            return self._empty_result()

        try:
            if self.strategy == "basic":
                # 基础模式：纯文本统计分析
                result = self._text_length_analysis(text)
                score = result['score']
                method = result['method']
                ppl = None
                avg_logprob = None

            elif self.strategy == "perplexity":
                # 尝试 echo 模式获取完整 perplexity
                result = self._compute_full_text_perplexity(text)
                if result.get('score') is not None and result['method'] != 'basic_stats':
                    ppl = result['score']
                    avg_logprob = result.get('avg_logprob')
                    score = self._perplexity_to_aigc_score(ppl, result['method'])
                    method = result['method']
                else:
                    # 退化到文本分析
                    basic = self._text_length_analysis(text)
                    score = basic['score']
                    method = basic['method']
                    ppl = None
                    avg_logprob = None

            else:
                result = self._compute_text_surprise(text)
                if result.get('score') is not None:
                    ppl = result['score']
                    avg_logprob = result.get('avg_logprob')
                    score = self._perplexity_to_aigc_score(ppl, result['method'])
                    method = result['method']
                else:
                    basic = self._text_length_analysis(text)
                    score = basic['score']
                    method = basic['method']
                    ppl = None
                    avg_logprob = None

            label = 'ai' if score > 0.5 else 'human'
            confidence = max(score, 1 - score)

            return {
                'engine': 'openai-api',
                'aigc_score': score,
                'perplexity': round(ppl, 2) if ppl else None,
                'avg_logprob': round(avg_logprob, 4) if avg_logprob else None,
                'label': label,
                'confidence': round(confidence, 4),
                'method': method,
                'available': True,
            }

        except Exception as e:
            logger.warning("OpenAI 兼容 API 检测异常: %s", e)
            return self._empty_result()

    def detect_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict]:
        """
        批量检测
        使用线程池并发调用 API。api_concurrency=1 时保持串行行为。
        """
        if not texts:
            return []

        if self.api_concurrency <= 1:
            results = []
            iterable = progress_iter(texts, total=len(texts), desc="OpenAI API") if show_progress else texts
            for i, text in enumerate(iterable):
                result = self.detect(text)
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info("  OpenAI 兼容 API 进度: %d/%d", i + 1, len(texts))
                    time.sleep(0.5)  # 保守串行模式下避免触发速率限制
            return results

        workers = min(self.api_concurrency, len(texts))
        logger.info("OpenAI 兼容 API 并发检测: concurrency=%d, total=%d", workers, len(texts))
        results = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(self.detect, text): i for i, text in enumerate(texts)}
            iterable = as_completed(future_map)
            if show_progress:
                iterable = progress_iter(iterable, total=len(future_map), desc="OpenAI API")
            for done, future in enumerate(iterable, start=1):
                i = future_map[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    logger.warning("OpenAI 兼容 API 并发检测异常: %s", e)
                    results[i] = self._empty_result()
                if done % 10 == 0 or done == len(texts):
                    logger.info("  OpenAI 兼容 API 进度: %d/%d", done, len(texts))

        return results

    def _empty_result(self) -> Dict:
        return {
            'engine': 'openai-api',
            'aigc_score': None,
            'perplexity': None,
            'avg_logprob': None,
            'label': 'unknown',
            'confidence': 0.0,
            'method': 'unavailable',
            'available': False,
        }
