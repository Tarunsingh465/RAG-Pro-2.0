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

    def test_scroll_elements_state_on_empty_and_active_chat(self):
        """Verify bottom anchor and scroll button are absent on empty chat and present with messages."""
        with patch("requests.get", side_effect=self.mock_get), patch("requests.post", side_effect=self.mock_post):
            at = AppTest.from_file(APP_PATH).run()

            # Empty chat: no scroll-to-bottom button
            btn_matches = [m for m in at.markdown if "scroll-to-bottom-btn" in m.value]
            self.assertEqual(len(btn_matches), 0, "Scroll-to-bottom button should not appear on empty chat.")

            # Send a query
            at.chat_input[0].set_value("Test query for scroll").run()

            # Active chat: scroll-to-bottom button and bottom anchor must be rendered
            anchor_matches = [m for m in at.markdown if "chat-bottom-anchor" in m.value]
            self.assertGreaterEqual(len(anchor_matches), 1, "Bottom anchor must be rendered when messages exist.")

            btn_matches_active = [m for m in at.markdown if "scroll-to-bottom-btn" in m.value]
            self.assertGreaterEqual(len(btn_matches_active), 1, "Scroll-to-bottom button must appear when messages exist.")


class TestFrontendSidebarLayout(unittest.TestCase):
    """Verifies that sidebar components render exactly once across all application states."""

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

    def test_sidebar_single_rendering_initial_load(self):
        """Verify each sidebar section renders exactly once on initial load."""
        with patch("requests.get", side_effect=self.mock_get):
            at = AppTest.from_file(APP_PATH).run()

            # 1. File Uploader: exactly 1 instance with key 'sidebar_file_uploader'
            self.assertEqual(len(at.sidebar.file_uploader), 1, "Expected exactly 1 file uploader in sidebar.")
            self.assertEqual(at.sidebar.file_uploader[0].key, "sidebar_file_uploader")

            # 2. Project Branding: exactly 1 title and 1 developer badge
            branding_titles = [m for m in at.sidebar.markdown if "### 🏭 TYRION-2.0" in m.value]
            self.assertEqual(len(branding_titles), 1, "Branding title must render exactly once.")

            dev_cards = [m for m in at.sidebar.markdown if "developer-badge" in m.value]
            self.assertEqual(len(dev_cards), 1, "Developer card must render exactly once.")

            # 3. Section Headers: exactly 1 Upload header and 1 Knowledge Base header
            upload_headers = [m for m in at.sidebar.markdown if "#### 📂 Upload & Ingest" in m.value]
            self.assertEqual(len(upload_headers), 1, "Upload & Ingest header must render exactly once.")

            kb_headers = [m for m in at.sidebar.markdown if "#### 📄 Knowledge Base Documents" in m.value]
            self.assertEqual(len(kb_headers), 1, "Knowledge Base Documents header must render exactly once.")

            # 4. Buttons: exact unique keys for upload, wipe, clear, and item deletion
            button_keys = [b.key for b in at.sidebar.button]
            self.assertEqual(button_keys.count("upload_index_btn"), 1)
            self.assertEqual(button_keys.count("wipe_all_docs_btn"), 1)
            self.assertEqual(button_keys.count("clear_chat_history_btn"), 1)
            self.assertEqual(button_keys.count("del_industrial_turbine_manual.pdf"), 1)

    def test_sidebar_no_duplication_after_multi_turn_reruns(self):
        """Verify sidebar elements do not duplicate across consecutive chat turns."""
        with patch("requests.get", side_effect=self.mock_get), patch("requests.post", side_effect=self.mock_post):
            at = AppTest.from_file(APP_PATH).run()

            for q in ["Query 1: Pump pressure", "Query 2: Bearing clearance", "Query 3: Oil filter"]:
                at.chat_input[0].set_value(q).run()

                # Verify sidebar remains singleton
                self.assertEqual(len(at.sidebar.file_uploader), 1)
                self.assertEqual(
                    len([m for m in at.sidebar.markdown if "### 🏭 TYRION-2.0" in m.value]), 1
                )
                self.assertEqual(
                    len([m for m in at.sidebar.markdown if "#### 📂 Upload & Ingest" in m.value]), 1
                )
                self.assertEqual(
                    len([m for m in at.sidebar.markdown if "#### 📄 Knowledge Base Documents" in m.value]), 1
                )
                button_keys = [b.key for b in at.sidebar.button]
                self.assertEqual(button_keys.count("upload_index_btn"), 1)
                self.assertEqual(button_keys.count("clear_chat_history_btn"), 1)

    def test_sidebar_empty_knowledge_base_state(self):
        """Verify empty documents list renders empty state message without duplicating sections."""
        def mock_get_empty(url, *args, **kwargs):
            m = MagicMock()
            if "health" in url:
                m.status_code = 200
            elif "documents" in url:
                m.status_code = 200
                m.json.return_value = {"documents": []}
            return m

        with patch("requests.get", side_effect=mock_get_empty):
            at = AppTest.from_file(APP_PATH).run()

            self.assertEqual(len(at.sidebar.file_uploader), 1)
            self.assertEqual(
                len([m for m in at.sidebar.markdown if "### 🏭 TYRION-2.0" in m.value]), 1
            )
            self.assertEqual(len(at.sidebar.info), 1)
            self.assertIn("No documents uploaded yet.", at.sidebar.info[0].value)

    def test_sidebar_offline_backend_state(self):
        """Verify backend offline warning does not duplicate sidebar sections."""
        def mock_get_offline(url, *args, **kwargs):
            raise requests.exceptions.ConnectionError("Connection refused")

        with patch("requests.get", side_effect=mock_get_offline):
            at = AppTest.from_file(APP_PATH).run()

            self.assertEqual(len(at.sidebar.file_uploader), 1)
            self.assertEqual(
                len([m for m in at.sidebar.markdown if "### 🏭 TYRION-2.0" in m.value]), 1
            )
            self.assertEqual(
                len([m for m in at.sidebar.markdown if "#### 📂 Upload & Ingest" in m.value]), 1
            )
            self.assertTrue(any("Backend Offline" in w.value for w in at.sidebar.warning))

    def test_sidebar_clear_chat_history_action(self):
        """Verify clicking Clear Chat History clears messages without duplicating sidebar elements."""
        with patch("requests.get", side_effect=self.mock_get), patch("requests.post", side_effect=self.mock_post):
            at = AppTest.from_file(APP_PATH).run()
            at.chat_input[0].set_value("Test query").run()
            self.assertEqual(len(at.chat_message), 2)

            # Click clear chat history
            clear_btn = [b for b in at.sidebar.button if b.key == "clear_chat_history_btn"][0]
            clear_btn.click().run()

            # Messages reset, sidebar remains unique
            self.assertEqual(len(at.chat_message), 0)
            self.assertEqual(len(at.sidebar.file_uploader), 1)
            self.assertEqual(len([m for m in at.sidebar.markdown if "### 🏭 TYRION-2.0" in m.value]), 1)
            self.assertEqual(len([b for b in at.sidebar.button if b.key == "clear_chat_history_btn"]), 1)

    def test_sidebar_document_delete_action(self):
        """Verify clicking delete document keeps sidebar singletons intact."""
        def mock_delete(url, *args, **kwargs):
            m = MagicMock()
            m.status_code = 200
            return m

        with patch("requests.get", side_effect=self.mock_get), patch("requests.delete", side_effect=mock_delete):
            at = AppTest.from_file(APP_PATH).run()
            del_btn = [b for b in at.sidebar.button if b.key == "del_industrial_turbine_manual.pdf"][0]
            del_btn.click().run()

            self.assertEqual(len(at.sidebar.file_uploader), 1)
            self.assertEqual(len([m for m in at.sidebar.markdown if "### 🏭 TYRION-2.0" in m.value]), 1)
            self.assertEqual(len([m for m in at.sidebar.markdown if "#### 📂 Upload & Ingest" in m.value]), 1)

    def test_sidebar_upload_action_no_file(self):
        """Verify clicking Upload & Index without files maintains singleton layout."""
        with patch("requests.get", side_effect=self.mock_get):
            at = AppTest.from_file(APP_PATH).run()
            up_btn = [b for b in at.sidebar.button if b.key == "upload_index_btn"][0]
            up_btn.click().run()

            self.assertEqual(len(at.sidebar.file_uploader), 1)
            self.assertEqual(len(at.sidebar.warning), 1)
            self.assertIn("Please choose one or more documents.", at.sidebar.warning[0].value)
            self.assertEqual(len([m for m in at.sidebar.markdown if "### 🏭 TYRION-2.0" in m.value]), 1)


if __name__ == "__main__":
    unittest.main()
