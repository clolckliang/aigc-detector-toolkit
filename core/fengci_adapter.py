"""
FengCi0/aigc-detector 适配器 - 自包含版
内联了 FeatureExtractor 和 AIGCDetector 的核心逻辑，
无需克隆外部仓库即可使用。
"""
import os
import re
import sys
import math
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 内联：FeatureExtractor（来自 FengCi0/aigc-detector）
# ============================================================

class FeatureExtractor:
    """中文文本特征提取器（轻量、可解释、固定 12 维）。"""

    FEATURE_NAMES = [
        "char_entropy_norm", "avg_sentence_length_norm",
        "sentence_length_cv_norm", "lexical_diversity",
        "hapax_ratio", "repetition_ratio",
        "bigram_repetition_ratio", "function_word_ratio",
        "punctuation_ratio", "long_word_ratio",
        "pos_diversity", "noun_verb_balance",
    ]

    def __init__(self):
        self.feature_names = list(self.FEATURE_NAMES)
        self.function_words = {
            "的", "了", "是", "在", "和", "也", "就", "都", "而", "及",
            "与", "或", "并", "被", "将", "把", "对", "从", "到", "给",
            "于", "以", "为", "这", "那", "这些", "那些", "一个", "我们",
            "你", "我", "他", "她", "它",
        }

    def extract_features(self, text: str) -> List[float]:
        if not text or len(text.strip()) < 10:
            return [0.5] * len(self.feature_names)

        normalized = text.strip()
        sentences = [s.strip() for s in re.split(r"[。！？!?；;\n]+", normalized) if s.strip()]

        try:
            import jieba
            import jieba.posseg as pseg
            words = [w.strip() for w in jieba.cut(normalized) if w.strip()]
            tags = [flag for _, flag in pseg.cut(normalized)]
        except ImportError:
            words = list(normalized)
            tags = []

        chars = [c for c in normalized if not c.isspace()]
        if not words or not chars:
            return [0.5] * len(self.feature_names)

        sentence_lengths = [len(s) for s in sentences] or [len(normalized)]
        word_counter = Counter(words)
        unique_words = len(word_counter)
        total_words = len(words)

        def _safe_div(a, b):
            return a / b if b else 0.0

        features = [
            # 0: char_entropy_norm
            min((self._entropy([c / len(chars) for c in Counter(chars).values()]) if chars else 0.0) / 7.0, 1.0),
            # 1: avg_sentence_length_norm
            min(float(np.mean(sentence_lengths)) / 60.0, 1.0),
            # 2: sentence_length_cv_norm
            min(float(np.std(sentence_lengths) / np.mean(sentence_lengths)) if np.mean(sentence_lengths) > 0 else 0, 2.0),
            # 3: lexical_diversity
            _safe_div(unique_words, total_words),
            # 4: hapax_ratio
            _safe_div(sum(1 for _, c in word_counter.items() if c == 1), total_words),
            # 5: repetition_ratio
            1.0 - _safe_div(unique_words, total_words),
            # 6: bigram_repetition_ratio
            self._bigram_repetition(words),
            # 7: function_word_ratio
            _safe_div(sum(1 for w in words if w in self.function_words), total_words),
            # 8: punctuation_ratio
            min(len(re.findall(r"""[，。！？；：、""''《》【】()\[\],.:;!?"'\-—…]""", normalized)) / max(len(chars), 1), 1.0),
            # 9: long_word_ratio
            _safe_div(sum(1 for w in words if len(w) >= 3), total_words),
            # 10: pos_diversity
            self._pos_diversity(tags),
            # 11: noun_verb_balance
            self._noun_verb_balance(tags),
        ]
        return [float(min(max(v, 0.0), 1.0)) for v in features]

    def get_feature_dict(self, fv):
        return dict(zip(self.feature_names, fv))

    @staticmethod
    def _entropy(probs):
        return -sum(p * math.log2(p) for p in probs if p > 0)

    @staticmethod
    def _bigram_repetition(words):
        if len(words) < 2:
            return 0.0
        bigrams = [f"{words[i]}|{words[i+1]}" for i in range(len(words) - 1)]
        counts = Counter(bigrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        return min(repeated / len(bigrams), 1.0)

    @staticmethod
    def _pos_diversity(tags):
        if not tags:
            return 0.0
        counts = Counter(tags)
        probs = [c / len(tags) for c in counts.values()]
        raw = float(FeatureExtractor._entropy(probs))
        max_ent = np.log2(len(counts) + 1)
        return min(raw / max_ent, 1.0) if max_ent > 0 else 0.0

    @staticmethod
    def _noun_verb_balance(tags):
        nouns = sum(1 for f in tags if f.startswith("n"))
        verbs = sum(1 for f in tags if f.startswith("v"))
        total = nouns + verbs
        return nouns / total if total else 0.5


# ============================================================
# 内联：AIGCDetector（启发式 + 随机森林融合）
# ============================================================

class AIGCDetector:
    """FengCi0 AIGC 文本检测器（自包含版）。"""

    HEURISTIC_WEIGHTS = {
        "char_entropy_norm": -0.22, "sentence_length_cv_norm": -0.18,
        "lexical_diversity": -0.24, "hapax_ratio": -0.16,
        "repetition_ratio": 0.20, "bigram_repetition_ratio": 0.18,
        "function_word_ratio": 0.08, "punctuation_ratio": -0.06,
        "long_word_ratio": -0.06, "pos_diversity": -0.12,
        "noun_verb_balance": 0.04, "avg_sentence_length_norm": 0.02,
    }

    def __init__(self, model_path: str, min_text_length: int = 50):
        self.feature_extractor = FeatureExtractor()
        self.min_text_length = min_text_length
        self.model = None
        self.score_threshold = 0.5

        if model_path and os.path.exists(model_path):
            try:
                import joblib
                bundle = joblib.load(model_path)
                if isinstance(bundle, dict) and "classifier" in bundle:
                    self.model = bundle["classifier"]
                    meta = bundle.get("metadata", {})
                    self.score_threshold = float(meta.get("recommended_threshold", 0.5))
                logger.info("FengCi0 模型加载成功: %s", model_path)
            except Exception as e:
                logger.warning("FengCi0 模型加载失败: %s", e)

    def analyze(self, text: str, include_details: bool = False) -> Dict:
        if not text or len(text.strip()) < self.min_text_length:
            raise ValueError(f"文本长度不足，当前为 {len(text.strip())}，至少需要 {self.min_text_length} 个字符")

        features = self.feature_extractor.extract_features(text)
        feat_dict = self.feature_extractor.get_feature_dict(features)

        heuristic_score = self._heuristic_score(feat_dict)

        if self.model is not None:
            x = np.array(features, dtype=float).reshape(1, -1)
            try:
                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(x)[0]
                    ml_score = float(proba[1]) if len(proba) > 1 else float(proba[0])
                else:
                    ml_score = float(self.model.predict(x)[0])
                # 融合：ML 为主，启发式为辅
                final_score = 0.8 * ml_score + 0.2 * heuristic_score
            except Exception:
                final_score = heuristic_score
        else:
            final_score = heuristic_score

        final_score = max(0.0, min(1.0, final_score))
        label = "ai" if final_score > self.score_threshold else "human"
        confidence = abs(final_score - 0.5) * 2

        result = {
            "aigc_score": round(final_score * 100, 2),  # 百分制
            "label": label,
            "confidence": round(confidence, 2),
            "score_threshold": self.score_threshold,
            "features": {k: round(v, 4) for k, v in feat_dict.items()},
            "model_mode": "ml_plus_heuristic" if self.model else "heuristic_only",
        }
        return result

    @staticmethod
    def _heuristic_score(feat_dict: Dict) -> float:
        weights = AIGCDetector.HEURISTIC_WEIGHTS
        score = 0.5
        for name, w in weights.items():
            value = float(feat_dict.get(name, 0.5))
            score += (value - 0.5) * 2.0 * w
        return max(0.0, min(1.0, score))


# ============================================================
# 适配器接口
# ============================================================

class FengCiAdapter:
    """适配 FengCi0/aigc-detector 检测器（自包含版）"""

    def __init__(self, model_path: Optional[str] = None):
        self.available = False
        self.detector = None

        if model_path is None:
            # 默认查找同级 models/fengci 目录
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models", "fengci", "aigc_detector_model.joblib",
            )

        if not os.path.exists(model_path):
            logger.warning("FengCi0 模型不存在: %s", model_path)
            return

        try:
            self.detector = AIGCDetector(model_path)
            self.available = True
            logger.info("FengCi0 检测器加载成功")
        except Exception as e:
            logger.error("FengCi0 加载失败: %s", e)

    def detect(self, text: str) -> Dict:
        if not self.available or not self.detector:
            return self._empty_result()
        try:
            result = self.detector.analyze(text, include_details=True)
            # score 已是百分制 → 归一化到 0-1
            normalized = result["aigc_score"] / 100.0
            return {
                "engine": "fengci",
                "aigc_score": round(normalized, 4),
                "raw_score": result["aigc_score"],
                "label": result["label"],
                "confidence": result["confidence"],
                "features": result.get("features", {}),
                "available": True,
            }
        except ValueError:
            return self._empty_result()
        except Exception as e:
            logger.warning("FengCi0 检测异常: %s", e)
            return self._empty_result()

    def detect_batch(self, texts: List[str]) -> List[Dict]:
        return [self.detect(t) for t in texts]

    def _empty_result(self):
        return {
            "engine": "fengci", "aigc_score": None, "raw_score": None,
            "label": "unknown", "confidence": 0.0, "features": {},
            "available": False,
        }
