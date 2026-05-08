import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main import load_config


class ConfigTests(unittest.TestCase):
    def test_load_openai_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "openai_api:\n"
                "  api_base: https://example.test/v1\n"
                "  api_key: key-from-file\n"
                "  model: test-model\n",
                encoding="utf-8",
            )

            config = load_config(str(path))

        self.assertEqual(config["openai_api"]["api_base"], "https://example.test/v1")
        self.assertEqual(config["openai_api"]["api_key"], "key-from-file")
        self.assertEqual(config["openai_api"]["model"], "test-model")

    def test_legacy_mimo_config_maps_to_openai(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "mimo_api:\n"
                "  api_base: https://legacy.test/v1\n"
                "  api_key: legacy-key\n"
                "  model: legacy-model\n",
                encoding="utf-8",
            )

            config = load_config(str(path))

        self.assertEqual(config["openai_api"]["api_base"], "https://legacy.test/v1")
        self.assertEqual(config["openai_api"]["api_key"], "legacy-key")
        self.assertEqual(config["openai_api"]["model"], "legacy-model")

    def test_environment_overrides_file_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "openai_api:\n"
                "  api_base: https://example.test/v1\n"
                "  api_key: key-from-file\n"
                "  model: test-model\n",
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {
                    "OPENAI_API_KEY": "key-from-env",
                    "OPENAI_BASE_URL": "https://env.test/v1",
                    "OPENAI_MODEL": "env-model",
                },
                clear=False,
            ):
                config = load_config(str(path))

        self.assertEqual(config["openai_api"]["api_base"], "https://env.test/v1")
        self.assertEqual(config["openai_api"]["api_key"], "key-from-env")
        self.assertEqual(config["openai_api"]["model"], "env-model")

    def test_deepseek_shorthand_model_is_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "openai_api:\n"
                "  model: deepseek\n"
                "binoculars:\n"
                "  model: deepseek\n"
                "refiner_api:\n"
                "  model: deepseek\n",
                encoding="utf-8",
            )

            config = load_config(str(path))

        self.assertEqual(config["openai_api"]["model"], "deepseek-v4-pro")
        self.assertEqual(config["binoculars"]["model"], "deepseek-v4-pro")
        self.assertEqual(config["refiner_api"]["model"], "deepseek-v4-pro")

    def test_deepseek_environment_shorthand_model_is_normalized(self):
        with patch.dict("os.environ", {"OPENAI_MODEL": "deepseek"}, clear=False):
            config = load_config(None)

        self.assertEqual(config["openai_api"]["model"], "deepseek-v4-pro")


if __name__ == "__main__":
    unittest.main()
