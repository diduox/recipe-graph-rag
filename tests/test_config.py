import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recipe_graph_rag.config import load_settings


class SettingsTests(unittest.TestCase):
    def load(self, content, environ=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text(content, encoding="utf-8")
            with patch.dict(os.environ, environ or {}, clear=True):
                return load_settings(path)

    def test_environment_overrides_file(self):
        self.assertEqual(self.load("TOP_K=3", {"TOP_K": "7"}).top_k, 7)

    def test_secrets_are_redacted(self):
        settings = self.load("LLM_API_KEY=test-secret\nNEO4J_PASSWORD=test-password")
        for secret in ("test-secret", "test-password"):
            self.assertNotIn(secret, str(settings.safe_summary()))
            self.assertNotIn(secret, repr(settings))

    def test_invalid_values_fail(self):
        for content in ("MILVUS_PORT=70000", "TOP_K=0", "TOP_K=abc", "NEO4J_URI=wrong", "LLM_BASE_URL=https://user:secret@example.com"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.load(content)

    def test_missing_file_uses_defaults(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            self.assertEqual(load_settings(Path(folder) / "missing.env").top_k, 5)


if __name__ == "__main__":
    unittest.main()
