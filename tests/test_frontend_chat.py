"""
Tests for Streamlit Frontend Chat Interface.
Verifies scrollable conversation container, fixed composer behavior,
chronological message ordering, and session state synchronization.
"""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "frontend" / "app.py")


class TestFrontendChatLayout(unittest.TestCase):
    def setUp(self):
        def mock_get(url, *args, **kwargs):
            m = MagicMock()
            if "health" in url:
                m.status_code = 200
            elif "documents" in url:
                m.status_code = 200
                m.json.return_value = {"documents": ["industrial_turbine_manual.pdf"]}
            return m

        def mock_post(url, *args, **kwargs):
            m = MagicMock()
            m.status_code = 200
            q = kwargs.get("json", {}).get("question", "Unknown")
            m.json.return_value = {
                "answer": f"Simulated technical answer for '{q}'.",
                "sources": [{"source": "industrial_turbine_manual.pdf", "page": 12}],
            }
            return m

        self.mock_get = mock_get
        self.mock_post = mock_post

    def test_initial_chat_state_empty(self):
        """Verify initial load has 1 chat composer and 0 messages."""
        with patch("requests.get", side_effect=self.mock_get), patch("requests.post", side_effect=self.mock_post):
            at = AppTest.from_file(APP_PATH).run()
            self.assertEqual(len(at.chat_input), 1, "Chat input composer must be present.")
            self.assertEqual(len(at.chat_message), 0, "No messages should be rendered on initial load.")

    def test_single_turn_ordering(self):
        """Verify first query generates user message followed immediately by assistant response."""
        with patch("requests.get", side_effect=self.mock_get), patch("requests.post", side_effect=self.mock_post):
            at = AppTest.from_file(APP_PATH).run()
            at.chat_input[0].set_value("What is the blade tip clearance?").run()

            self.assertEqual(len(at.chat_message), 2, "Should have 1 user message and 1 assistant message.")
            self.assertEqual(at.chat_message[0].avatar, "user")
            self.assertEqual(at.chat_message[0].markdown[0].value, "What is the blade tip clearance?")
            self.assertEqual(at.chat_message[1].avatar, "assistant")
            self.assertIn("Simulated technical answer", at.chat_message[1].markdown[0].value)

    def test_multi_turn_chronological_ordering(self):
        """Verify 5 consecutive queries remain strictly chronological without duplicating."""
        with patch("requests.get", side_effect=self.mock_get), patch("requests.post", side_effect=self.mock_post):
            at = AppTest.from_file(APP_PATH).run()
            queries = [
                "Query 1: Centrifugal pump flow rate",
                "Query 2: Bearing lubrication interval",
                "Query 3: Vibration threshold analysis",
                "Query 4: Emergency shutdown procedure",
                "Query 5: Seal degradation inspection",
            ]

            for idx, q in enumerate(queries, 1):
                at.chat_input[0].set_value(q).run()
                self.assertEqual(len(at.chat_message), idx * 2)

            # Check exact alternating order
            for i, q in enumerate(queries):
                user_msg = at.chat_message[i * 2]
                asst_msg = at.chat_message[i * 2 + 1]
                self.assertEqual(user_msg.avatar, "user")
                self.assertEqual(user_msg.markdown[0].value, q)
                self.assertEqual(asst_msg.avatar, "assistant")
                self.assertEqual(asst_msg.markdown[0].value, f"Simulated technical answer for '{q}'.")

            # Composer must still be present at bottom
            self.assertEqual(len(at.chat_input), 1)


if __name__ == "__main__":
    unittest.main()
