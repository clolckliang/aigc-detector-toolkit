import unittest

from core.engine import DetectionEngine
from core.local_logprob_adapter import LocalLogprobDetector


class EngineTests(unittest.TestCase):
    def test_fuse_uses_only_valid_scores(self):
        engine = DetectionEngine.__new__(DetectionEngine)
        engine.mode = "ensemble"
        engine.aigc_threshold = 0.5

        result = engine._fuse(
            scores=[0.2, 0.8],
            weights=[0.25, 0.75],
            engine_results={
                "fengci": {"confidence": 0.8},
                "openai": {"confidence": 0.9},
            },
        )

        self.assertEqual(result["aigc_score"], 0.65)
        self.assertEqual(result["label"], "ai")
        self.assertEqual(result["method"], "ensemble")

    def test_local_logprob_missing_model_is_unavailable(self):
        detector = LocalLogprobDetector(model_name_or_path="")

        self.assertFalse(detector.available)
        self.assertEqual(detector.detect("这是一段足够长的测试文本，用于确认未配置模型时不会返回伪分数。")["label"], "unknown")


if __name__ == "__main__":
    unittest.main()
