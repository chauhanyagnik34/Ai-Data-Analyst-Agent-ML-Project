"""
test_llm_client.py
------------------
Unit tests for llm_client.py validating Ollama status detection,
markdown cleaning, and graceful error handling.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm_client import check_api_status, _clean_json_markdown, call_llm


class TestLLMClient(unittest.TestCase):

    def test_status_structure(self):
        """Test status dictionary structure."""
        status = check_api_status()
        self.assertIn("configured", status)
        self.assertIn("provider", status)
        self.assertIn("model", status)
        self.assertIn("message", status)
        self.assertEqual(status["provider"], "Ollama (Local)")

    def test_clean_json_markdown(self):
        """Test stripping of markdown json tags."""
        raw = '```json\n{"task": "mean", "column": "Salary"}\n```'
        cleaned = _clean_json_markdown(raw)
        self.assertEqual(cleaned, '{"task": "mean", "column": "Salary"}')

    def test_call_llm_graceful_error_handling(self):
        """Ensure call_llm returns a safe error tuple if server is unreachable."""
        orig_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        try:
            # Point to a closed port
            os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:59999"
            res, err = call_llm("Ping")
            self.assertIsNone(res)
            self.assertIsNotNone(err)
            self.assertIsInstance(err, str)
        finally:
            os.environ["OLLAMA_BASE_URL"] = orig_url


if __name__ == "__main__":
    unittest.main()
