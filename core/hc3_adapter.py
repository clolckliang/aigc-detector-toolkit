"""
HC3 + m3e + CNN 检测器适配器 - 自包含版
内联了 OptimizedCNN 模型定义，无需克隆外部仓库。
"""
import os
import logging
import numpy as np
from typing import Dict, List, Optional

from .progress import progress_iter

logger = logging.getLogger(__name__)


# ============================================================
# 内联：OptimizedCNN 模型架构
# ============================================================

def _build_optimized_cnn(num_classes=2, seq_len=768):
    """构建 OptimizedCNN 模型"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class OptimizedCNN(nn.Module):
        def __init__(self, num_classes, seq_len):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(128)
            self.pool = nn.MaxPool1d(2)
            self.dropout = nn.Dropout(0.5)
            flat_dim = 128 * (seq_len // 4)
            self.fc1 = nn.Linear(flat_dim, 256)
            self.bn_fc1 = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(x.size(0), -1)
            x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
            x = self.fc2(x)
            return x

    return OptimizedCNN(num_classes, seq_len)


# ============================================================
# 适配器
# ============================================================

class HC3Adapter:
    """适配 HC3+m3e+CNN 检测器（自包含版）"""

    def __init__(self, model_path: Optional[str] = None, model_name: str = "moka-ai/m3e-base"):
        self.available = False
        self.model = None
        self.embedding_model = None
        self.embedding_dim = 768
        self.device = None
        self.min_text_length = 30

        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models", "hc3", "OptimizedCNN_aigc_detector.pth",
            )

        self._init_detector(model_path, model_name)

    def _init_detector(self, model_path, model_name):
        try:
            import torch

            if not os.path.exists(model_path):
                logger.warning("HC3 模型不存在: %s", model_path)
                return

            self.device = torch.device("cpu")
            self.model = _build_optimized_cnn(2, self.embedding_dim)
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            self.model.to(self.device)
            self.model.eval()

            logger.info("加载 m3e-base embedding 模型...")
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(model_name)

            self.available = True
            logger.info("HC3 检测器加载成功")
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or str(e)
            logger.warning("HC3 可选依赖未安装 (%s)，跳过 HC3 引擎；可运行 uv sync --extra hc3 安装", missing)
            self.available = False
        except Exception as e:
            logger.error("HC3 检测器加载失败: %s", e)
            self.available = False

    def detect(self, text: str) -> Dict:
        if not self.available:
            return self._empty_result()
        if len(text.strip()) < self.min_text_length:
            return self._empty_result()
        try:
            import torch
            import torch.nn.functional as F

            embedding = self.embedding_model.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )
            input_tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]

            ai_prob = float(probs[1])
            return {
                "engine": "hc3",
                "aigc_score": round(ai_prob, 4),
                "label": "ai" if ai_prob > 0.5 else "human",
                "confidence": round(max(ai_prob, 1 - ai_prob), 4),
                "features": {"ai_prob": round(ai_prob, 4), "human_prob": round(float(probs[0]), 4)},
                "available": True,
            }
        except Exception as e:
            logger.warning("HC3 检测异常: %s", e)
            return self._empty_result()

    def detect_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict]:
        if not self.available:
            return [self._empty_result() for _ in texts]
        valid_indices = [i for i, t in enumerate(texts) if len(t.strip()) >= self.min_text_length]
        if not valid_indices:
            return [self._empty_result() for _ in texts]
        try:
            import torch
            import torch.nn.functional as F

            valid_texts = [texts[i] for i in valid_indices]
            embeddings = self.embedding_model.encode(
                valid_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
            )
            input_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = F.softmax(outputs, dim=1).cpu().numpy()

            results = [self._empty_result() for _ in texts]
            iterator = enumerate(valid_indices)
            if show_progress:
                iterator = progress_iter(list(iterator), total=len(valid_indices), desc="HC3")
            for j, idx in iterator:
                ai_prob = float(probs[j][1])
                results[idx] = {
                    "engine": "hc3",
                    "aigc_score": round(ai_prob, 4),
                    "label": "ai" if ai_prob > 0.5 else "human",
                    "confidence": round(max(ai_prob, 1 - ai_prob), 4),
                    "features": {"ai_prob": round(ai_prob, 4), "human_prob": round(float(probs[j][0]), 4)},
                    "available": True,
                }
            return results
        except Exception as e:
            logger.warning("HC3 批量检测异常: %s", e)
            return [self._empty_result() for _ in texts]

    def _empty_result(self):
        return {
            "engine": "hc3", "aigc_score": None, "label": "unknown",
            "confidence": 0.0, "features": {}, "available": False,
        }
