"""Model training module for FengCi0 RF and HC3 CNN engines."""
import json
import logging
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .evaluation import binary_metrics, roc_auc, threshold_scan

logger = logging.getLogger(__name__)


# ============================================================
# Data loading from directory of .txt files
# ============================================================

def load_text_dir(data_dir: str) -> Tuple[List[str], List[str]]:
    """Load texts from data_dir/ai/ and data_dir/human/ subdirs."""
    base = Path(data_dir)
    texts, labels = [], []
    for label in ("ai", "human"):
        subdir = base / label
        if not subdir.exists():
            raise FileNotFoundError(f"数据目录不存在: {subdir}")
        for f in sorted(subdir.iterdir()):
            if f.suffix.lower() in (".txt", ".md", ".text"):
                text = f.read_text(encoding="utf-8").strip()
                if text:
                    texts.append(text)
                    labels.append(label)
    if not texts:
        raise ValueError(f"未找到任何文本文件: {data_dir}")
    logger.info("加载 %d 条样本 (ai=%d, human=%d)", len(texts),
                labels.count("ai"), labels.count("human"))
    return texts, labels


def load_labeled_jsonl(path: str) -> Tuple[List[str], List[str]]:
    """Load labeled dataset from JSONL (reuses evaluation.load_labeled_dataset)."""
    from .evaluation import load_labeled_dataset
    dataset = load_labeled_dataset(path)
    return [d["text"] for d in dataset], [d["label"] for d in dataset]


# ============================================================
# Data augmentation
# ============================================================

def augment_texts(texts: List[str], labels: List[str],
                  short_ratio: float = 0.3,
                  sentence_drop_prob: float = 0.25) -> Tuple[List[str], List[str]]:
    """Augment training data with sentence dropping and short-text synthesis."""
    aug_texts, aug_labels = list(texts), list(labels)
    n = len(texts)

    # Sentence drop: randomly remove sentences to create harder examples
    drop_count = int(n * sentence_drop_prob)
    indices = random.sample(range(n), min(drop_count, n))
    for idx in indices:
        sentences = re.split(r"([。！？!?；;\n]+)", texts[idx])
        if len(sentences) < 4:
            continue
        kept = []
        for i in range(0, len(sentences), 2):
            if random.random() > 0.3:
                kept.append(sentences[i])
                if i + 1 < len(sentences):
                    kept.append(sentences[i + 1])
        result = "".join(kept).strip()
        if len(result) >= 30:
            aug_texts.append(result)
            aug_labels.append(labels[idx])

    # Short-text synthesis: truncate random segments
    short_count = int(n * short_ratio)
    indices = random.sample(range(n), min(short_count, n))
    for idx in indices:
        text = texts[idx]
        if len(text) < 100:
            continue
        start = random.randint(0, max(0, len(text) - 80))
        length = random.randint(30, min(80, len(text) - start))
        segment = text[start:start + length].strip()
        if segment:
            aug_texts.append(segment)
            aug_labels.append(labels[idx])

    logger.info("数据增强: %d -> %d 条", n, len(aug_texts))
    return aug_texts, aug_labels


def stratified_split(texts: List[str], labels: List[str],
                     test_size: float = 0.2, seed: int = 42
                     ) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Stratified train/test split."""
    rng = random.Random(seed)
    by_class = {"ai": [], "human": []}
    for i, label in enumerate(labels):
        by_class[label].append(i)

    train_idx, test_idx = [], []
    for label, indices in by_class.items():
        rng.shuffle(indices)
        split = max(1, int(len(indices) * test_size))
        test_idx.extend(indices[:split])
        train_idx.extend(indices[split:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return (
        [texts[i] for i in train_idx], [labels[i] for i in train_idx],
        [texts[i] for i in test_idx], [labels[i] for i in test_idx],
    )


# ============================================================
# FengCi0 Trainer
# ============================================================

class FengCiTrainer:
    """Train the FengCi0 Random Forest model."""

    def __init__(self, output_dir: str = "models/fengci"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = None

    def train(self, texts: List[str], labels: List[str],
              eval_texts: Optional[List[str]] = None,
              eval_labels: Optional[List[str]] = None,
              augment: bool = True) -> Dict:
        from .fengci_adapter import FeatureExtractor
        self.feature_extractor = FeatureExtractor()

        # Augment
        if augment:
            texts, labels = augment_texts(texts, labels)

        # Split
        X_texts, y_labels, X_eval_texts, y_eval_labels = \
            stratified_split(texts, labels) if eval_texts is None else \
            (texts, labels, eval_texts, eval_labels)

        logger.info("训练集 %d 条, 验证集 %d 条", len(X_texts), len(X_eval_texts))

        # Extract features
        X_train = self._extract_features(X_texts)
        X_val = self._extract_features(X_eval_texts)
        y_train = np.array([1 if l == "ai" else 0 for l in y_labels])
        y_val = np.array([1 if l == "ai" else 0 for l in y_eval_labels])

        # Train multiple classifiers
        candidates = self._train_candidates(X_train, y_train, X_val, y_val)
        best_name, best_model, best_metrics = self._select_best(candidates)

        # Threshold calibration
        val_proba = best_model.predict_proba(X_val)[:, 1]
        scan = threshold_scan(
            [y_eval_labels[i] for i in range(len(y_eval_labels))],
            [float(p) for p in val_proba],
        )
        best_threshold = scan["best_f1"]["threshold"] if scan["best_f1"] else 0.5

        # Save
        self._save_model(best_model, best_name, best_threshold,
                         len(texts), labels, best_metrics, scan)

        return {
            "best_model": best_name,
            "threshold": best_threshold,
            "metrics": best_metrics,
            "threshold_scan": scan,
        }

    def _extract_features(self, texts: List[str]) -> np.ndarray:
        features = []
        for text in texts:
            fv = self.feature_extractor.extract_features(text)
            features.append(fv)
        return np.array(features, dtype=np.float64)

    def _train_candidates(self, X_train, y_train, X_val, y_val) -> List[Tuple]:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        candidates = []

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        candidates.append(("feature_rf", rf, X_val, y_val))

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        candidates.append(("feature_logreg", lr, X_val, y_val))

        # Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        gb.fit(X_train, y_train)
        candidates.append(("feature_gb", gb, X_val, y_val))

        return candidates

    def _select_best(self, candidates: List[Tuple]) -> Tuple[str, object, Dict]:
        best = None
        for name, model, X_val, y_val in candidates:
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            y_true = y_val.tolist()
            y_scores = y_pred_proba.tolist()

            acc = float(np.mean(y_pred == y_val))
            auc = roc_auc(
                ["ai" if y else "human" for y in y_true],
                y_scores,
            )
            f1 = binary_metrics(
                ["ai" if y else "human" for y in y_true],
                y_scores, 0.5,
            )["f1"]

            metrics = {
                "accuracy": round(acc, 4),
                "f1": f1,
                "roc_auc": round(auc, 4) if auc else None,
            }
            logger.info("  %s: acc=%.3f f1=%.3f auc=%s", name, acc, f1,
                        f"{auc:.3f}" if auc else "N/A")

            if best is None or (f1, acc) > (best[2]["f1"], best[2]["accuracy"]):
                best = (name, model, metrics)

        return best[0], best[1], best[2]

    def _save_model(self, model, model_name: str, threshold: float,
                    total_samples: int, labels: List[str],
                    metrics: Dict, scan: Dict):
        import joblib

        bundle = {
            "classifier": model,
            "metadata": {
                "version": "4.0",
                "model_name": model_name,
                "recommended_threshold": threshold,
            },
        }
        model_path = self.output_dir / "aigc_detector_model.joblib"
        joblib.dump(bundle, model_path)
        logger.info("模型已保存: %s", model_path)

        # Save metadata
        meta = {
            "version": "4.0",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "expected_feature_count": 12,
            "feature_names": list(self.feature_extractor.feature_names),
            "recommended_threshold": threshold,
            "sample_count": total_samples,
            "class_distribution": {
                "ai": labels.count("ai"),
                "human": labels.count("human"),
            },
            "validation_metrics": metrics,
            "threshold_scan": scan,
        }
        meta_path = self.output_dir / "aigc_detector_model.metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("元数据已保存: %s", meta_path)


# ============================================================
# HC3 CNN Trainer
# ============================================================

class HC3Trainer:
    """Train the HC3 m3e+CNN model."""

    def __init__(self, output_dir: str = "models/hc3",
                 embedding_model: str = "moka-ai/m3e-base"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model

    def train(self, texts: List[str], labels: List[str],
              eval_texts: Optional[List[str]] = None,
              eval_labels: Optional[List[str]] = None,
              epochs: int = 30, batch_size: int = 32,
              lr: float = 1e-3, patience: int = 5) -> Dict:
        import torch
        import torch.nn.functional as F

        from .hc3_adapter import _build_optimized_cnn

        # Load embedding model
        logger.info("加载 embedding 模型: %s", self.embedding_model_name)
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(self.embedding_model_name)

        # Split
        if eval_texts is None:
            X_texts, y_labels, X_eval_texts, y_eval_labels = \
                stratified_split(texts, labels)
        else:
            X_texts, y_labels = texts, labels
            X_eval_texts, y_eval_labels = eval_texts, eval_labels

        logger.info("训练集 %d 条, 验证集 %d 条", len(X_texts), len(X_eval_texts))

        # Encode embeddings
        logger.info("编码训练集 embeddings...")
        X_train_emb = st_model.encode(X_texts, normalize_embeddings=True,
                                       show_progress_bar=True, batch_size=32)
        X_val_emb = st_model.encode(X_eval_texts, normalize_embeddings=True,
                                     show_progress_bar=True, batch_size=32)

        y_train = torch.tensor([1 if l == "ai" else 0 for l in y_labels], dtype=torch.long)
        y_val = torch.tensor([1 if l == "ai" else 0 for l in y_eval_labels], dtype=torch.long)
        X_train_t = torch.tensor(X_train_emb, dtype=torch.float32)
        X_val_t = torch.tensor(X_val_emb, dtype=torch.float32)

        # Build model
        model = _build_optimized_cnn(num_classes=2, seq_len=768)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3)

        # Training loop
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(len(X_train_t))
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                xb = X_train_t[idx]
                yb = y_train[idx]
                optimizer.zero_grad()
                out = model(xb)
                loss = F.cross_entropy(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(X_val_t)
                val_loss = F.cross_entropy(val_out, y_val).item()
                val_probs = F.softmax(val_out, dim=1)[:, 1].numpy()
                val_preds = (val_probs > 0.5).astype(int)
                val_acc = float(np.mean(val_preds == y_val.numpy()))

            scheduler.step(val_loss)

            auc = roc_auc(y_eval_labels, val_probs.tolist())
            logger.info("Epoch %d/%d  loss=%.4f  val_loss=%.4f  val_acc=%.3f  auc=%s",
                        epoch, epochs, avg_loss, val_loss, val_acc,
                        f"{auc:.3f}" if auc else "N/A")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Restore best and save
        if best_state:
            model.load_state_dict(best_state)

        save_path = self.output_dir / "OptimizedCNN_aigc_detector.pth"
        torch.save(model.state_dict(), save_path)
        logger.info("模型已保存: %s", save_path)

        # Final metrics
        model.eval()
        with torch.no_grad():
            final_out = model(X_val_t)
            final_probs = F.softmax(final_out, dim=1)[:, 1].numpy()

        metrics = binary_metrics(y_eval_labels, final_probs.tolist(), 0.5)
        scan = threshold_scan(y_eval_labels, final_probs.tolist())

        return {
            "metrics": metrics,
            "threshold_scan": scan,
            "model_path": str(save_path),
        }
