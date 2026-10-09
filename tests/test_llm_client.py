"""
test_llm_client.py
------------------
Unit tests for llm_client.py validating status detection,
markdown cleaning, and graceful error handling.
"""

import unittest
from llm_client import check_api_status, _clean_json_markdown, call_llm


class TestLLMClient(unittest.TestCase):

    def test_status_structure(self):
        """Test status dictionary structure."""
        status = check_api_status()
        self.assertIn("configured", status)
        self.assertIn("provider", status)
        self.assertIn("model", status)
        self.assertIn("message", status)

    def test_clean_json_markdown(self):
        """Test stripping of markdown json tags."""
        raw = '```json\n{"task": "mean", "column": "Salary"}\n```'
        cleaned = _clean_json_markdown(raw)
        self.assertEqual(cleaned, '{"task": "mean", "column": "Salary"}')

    def test_call_llm_without_key_does_not_crash(self):
        """Ensure call_llm returns a safe error tuple rather than raising unhandled exceptions."""
        res, err = call_llm("Ping")
        # Since API key is placeholder, it must return None for result and descriptive string for error
        self.assertIsNone(res)
        self.assertIsNotNone(err)


if __name__ == "__main__":
    unittest.main()
