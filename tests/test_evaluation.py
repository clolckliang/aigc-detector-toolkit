import tempfile
import unittest
from pathlib import Path

from core.evaluation import binary_metrics, load_labeled_dataset, threshold_scan


class EvaluationTests(unittest.TestCase):
    def test_binary_metrics(self):
        metrics = binary_metrics(
            ["human", "ai", "ai", "human"],
            [0.1, 0.9, 0.4, 0.8],
            threshold=0.5,
        )

        self.assertEqual(metrics["confusion_matrix"], {"tp": 1, "fp": 1, "tn": 1, "fn": 1})
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_threshold_scan_returns_best_f1(self):
        scan = threshold_scan(
            ["human", "human", "ai", "ai"],
            [0.1, 0.2, 0.8, 0.9],
        )

        self.assertIsNotNone(scan["best_f1"])
        self.assertEqual(scan["best_f1"]["f1"], 1.0)

    def test_load_jsonl_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jsonl"
            path.write_text(
                '{"text":"人工文本样本足够长，用于测试加载逻辑。","label":"human"}\n'
                '{"text":"机器生成文本样本足够长，用于测试加载逻辑。","label":"ai"}\n',
                encoding="utf-8",
            )

            rows = load_labeled_dataset(str(path))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["label"], "human")
        self.assertEqual(rows[1]["label"], "ai")


if __name__ == "__main__":
    unittest.main()
