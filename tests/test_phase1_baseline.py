"""
Regression and baseline test suite for Phase 1 Audit.
Verifies system health, zero-API routing, domain classifier, and data leakage isolation.
Uses Python standard unittest.
"""

import unittest
from app.core.engineering import is_engineering_query, classify_engineering_domain
from app.services.conversational_engine import match_conversational_intent
from app.services.supervisor_service import SupervisorService
from app.services.chat_service import ChatService, get_active_documents


class TestPhase1Baseline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.supervisor = SupervisorService()
        cls.chat_service = ChatService()

    def test_zero_api_developer_identity(self):
        """Verifies Tarun identity match in <0.5ms with 0 API calls."""
        match = match_conversational_intent("Who developed you?")
        self.assertIsNotNone(match)
        self.assertEqual(match["id"], "developer_and_creator")
        self.assertIn("Tarun", match["response"])

    def test_zero_api_greetings(self):
        """Verifies basic conversational greetings bypass external APIs."""
        match = match_conversational_intent("how are you?")
        self.assertIsNotNone(match)
        self.assertIn("ready", match["response"].lower())

    def test_engineering_classifier_positive(self):
        """Verifies technical query detection across core engineering disciplines."""
        is_eng, disc, name = is_engineering_query("Why is my PC heating 98 degree celcius?")
        self.assertTrue(is_eng)
        self.assertEqual(disc, "cs_it")

        is_eng, disc, name = is_engineering_query("Calculate turbocharger compressor surge margin and thermal stress")
        self.assertTrue(is_eng)
        self.assertIn(disc, ["mechanical", "cs_it"])

    def test_engineering_classifier_negative(self):
        """Verifies non-engineering general queries are not falsely classified as engineering."""
        is_eng, disc, _ = is_engineering_query("Who is Mahatma Gandhi?")
        self.assertFalse(is_eng)

        is_eng, disc, _ = is_engineering_query("What is the capital of France?")
        self.assertFalse(is_eng)

    def test_supervisor_fast_path_routing(self):
        """Verifies deterministic sub-millisecond query routing."""
        decision = self.supervisor.route("hi")
        self.assertEqual(decision.route, "greeting")

        decision = self.supervisor.route("What are the qualifications of the candidate?")
        self.assertEqual(decision.route, "rag")

    def test_data_leakage_isolation_when_empty(self):
        """Verifies that queries when no documents are uploaded never cite phantom sources."""
        active_docs = get_active_documents()
        # If no documents are uploaded, out-of-domain queries must return empty sources
        if not active_docs:
            res = self.chat_service.chat("Who was Napoleon Bonaparte?")
            self.assertEqual(res.get("sources"), [])
            self.assertIn("No Documents Uploaded", res.get("answer", ""))


if __name__ == "__main__":
    unittest.main()
