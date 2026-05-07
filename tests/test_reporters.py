import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from reporters.console_reporter import print_summary
from reporters.html_reporter import generate_html_report


class ReporterTests(unittest.TestCase):
    def test_ensemble_summary_lists_all_engines(self):
        engine_status = {
            "mode": "ensemble",
            "threshold": 0.5,
            "fengci_available": True,
            "fengci_weight": 0.5,
            "hc3_available": False,
            "hc3_weight": 0.0,
            "openai_available": False,
            "openai_weight": 0.0,
            "binoculars_available": False,
            "binoculars_weight": 0.0,
            "local_logprob_available": False,
            "local_logprob_weight": 0.0,
        }
        paragraphs = [{"index": 1, "text": "测试段落", "chapter": 0}]
        results = [{"label": "human", "aigc_score": 0.2}]

        out = io.StringIO()
        with redirect_stdout(out):
            print_summary(paragraphs, results, engine_status)

        text = out.getvalue()
        self.assertIn("OpenAI API", text)
        self.assertIn("Binoculars", text)
        self.assertIn("Local Logprob", text)

    def test_html_report_marks_risky_text_and_guidance(self):
        paragraphs = [
            {"index": 1, "text": "这是一段人工修改后的普通文本<script>alert(1)</script>。", "chapter": 0},
            {"index": 2, "text": "首先，其次，最后，本文提出了系统性的解决方案。", "chapter": 0},
            {"index": 3, "text": "该段文字分数接近阈值，需要人工复核表达是否过于模板化。", "chapter": 0},
        ]
        results = [
            {"label": "human", "aigc_score": 0.2},
            {
                "label": "ai",
                "aigc_score": 0.82,
                "engine_results": {
                    "fengci": {"aigc_score": 0.76, "method": "feature_rf"},
                    "openai": {"aigc_score": 0.88, "method": "basic_stats"},
                },
            },
            {"label": "human", "aigc_score": 0.45},
        ]
        engine_status = {
            "mode": "ensemble",
            "threshold": 0.5,
            "api_concurrency": 8,
            "fengci_available": True,
            "fengci_weight": 0.5,
            "hc3_available": False,
            "hc3_weight": 0.0,
            "openai_available": True,
            "openai_weight": 0.5,
            "binoculars_available": False,
            "binoculars_weight": 0.0,
            "local_logprob_available": False,
            "local_logprob_weight": 0.0,
        }

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "report.html")
            returned = generate_html_report(paragraphs, results, engine_status, path, "sample.docx")
            self.assertEqual(returned, path)
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                html = f.read()

        self.assertIn("risk-ai", html)
        self.assertIn("risk-review", html)
        self.assertIn("原文内容批注", html)
        self.assertIn("高风险索引", html)
        self.assertIn("风险分布", html)
        self.assertIn("检测引擎", html)
        self.assertIn("逐段修改指导", html)
        self.assertIn("FengCi0: 76.0%", html)
        self.assertIn("OpenAI API: 88.0% basic_stats", html)
        self.assertIn("id=\"report-search\"", html)
        self.assertIn("data-filter=\"ai\"", html)
        self.assertIn("copy-btn", html)
        self.assertIn("@media print", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertNotIn("<script>alert(1)</script>", html)


if __name__ == "__main__":
    unittest.main()
