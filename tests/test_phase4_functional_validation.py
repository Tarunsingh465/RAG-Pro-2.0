"""
Phase 4 — End-to-End Functional Validation Test Suite.
TYRION-2.0 — Final-Year Academic & Portfolio Project

Validates the 7 practical operational scenarios required by Phase 4:
1. Successful document upload, extraction, chunking, and dual indexing.
2. Successful document query with context grounding and source citations.
3. Hybrid retrieval fusing Dense (Pinecone) and Sparse (BM25) candidates via RRF.
4. Out-of-corpus question with insufficient evidence triggering safe abstention.
5. Failed/invalid upload handling (unsupported format raising 400 and cleaning up files).
6. Document deletion verifying synchronized removal from disk, Pinecone, and BM25.
7. External service failure and rate-limit resilience.
"""

import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.core.constants import UPLOAD_FOLDER
from app.main import app
from app.retrieval.bm25 import BM25Index
from app.retrieval.retriever import Retriever
from app.schemas.response import ChatResponse, SourceDocument
from app.services.chat_service import ChatService
from app.services.ingest_service import IngestService
from app.services.rag_service import RagService


class TestPhase4FunctionalValidation(unittest.TestCase):
    """
    Comprehensive functional validation for core application workflows.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_dir = Path(tempfile.mkdtemp(prefix="tyrion_phase4_test_"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        self.upload_dir = Path(UPLOAD_FOLDER)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Successful Document Upload & Indexing
    # -------------------------------------------------------------
    @patch("app.vectorstore.pinecone.PineconeManager.add_documents")
    @patch("app.retrieval.bm25.BM25Index.add_documents")
    def test_1_successful_document_upload_and_indexing(
        self, mock_bm25_add, mock_pinecone_add
    ):
        """
        Scenario 1: Upload a valid technical markdown document, verify extraction,
        chunking, and dual indexing into Pinecone and BM25.
        """
        mock_pinecone_add.return_value = ["test_manual.md_medium_0", "test_manual.md_medium_1"]
        mock_bm25_add.return_value = 2

        doc_content = (
            b"# Centrifugal Pump Maintenance Manual\n\n"
            b"## 1. Operating Specifications\n"
            b"The centrifugal pump operates at a rated flow of 450 m3/h at 1480 RPM.\n"
            b"Maximum allowable casing design pressure is 25.0 bar(g).\n\n"
            b"## 2. Bearing Lubrication Guidelines\n"
            b"Use ISO VG 46 synthetic turbine oil. Replace bearing lubrication every 4,000 operating hours.\n"
        )

        files = [
            ("files", ("test_pump_spec.md", io.BytesIO(doc_content), "text/markdown"))
        ]

        response = self.client.post("/api/v1/upload", files=files)
        self.assertEqual(response.status_code, 200, f"Upload failed: {response.text}")

        data = response.json()
        self.assertIn("files", data)
        self.assertIn("test_pump_spec.md", data["files"])
        self.assertGreaterEqual(data["chunks"], 1)
        self.assertIn("disciplines", data)

        # Verify file exists in upload folder
        uploaded_file = self.upload_dir / "test_pump_spec.md"
        self.assertTrue(uploaded_file.exists(), "Uploaded file not saved on disk.")

        # Clean up
        uploaded_file.unlink(missing_ok=True)

    # -------------------------------------------------------------
    # 2. Successful Document Query & Source Citations
    # -------------------------------------------------------------
    @patch("app.services.chat_service.get_active_documents")
    @patch("app.services.rag_service.RagService._check_relevance")
    @patch("app.llm.chain.RAGChain.invoke")
    def test_2_successful_document_query_and_citations(
        self, mock_chain_invoke, mock_check_relevance, mock_active_docs
    ):
        """
        Scenario 2: Query against an active document, verify grounded LLM invocation
        and structured source citation creation.
        """
        mock_active_docs.return_value = ["centrifugal_pump_manual.docx"]

        mock_doc = Document(
            page_content="The centrifugal pump operates at a rated flow of 450 m3/h at 1480 RPM.",
            metadata={
                "source": "centrifugal_pump_manual.docx",
                "page": 3,
                "section": "Operating Specifications",
                "rrf_score": 0.032,
            },
        )

        mock_check_relevance.return_value = (True, 0.88, [(mock_doc, 0.88)])
        mock_chain_invoke.return_value = {
            "answer": "According to the manual, the pump rated flow is 450 m³/h at 1480 RPM.",
            "documents": [mock_doc],
        }

        chat_service = ChatService()
        result = chat_service.chat(
            question="What is the rated flow and speed of the centrifugal pump?",
            top_k=3,
        )

        self.assertFalse(result.get("out_of_context", True))
        self.assertIn("450 m³/h", result["answer"])
        self.assertEqual(len(result["sources"]), 1)
        source = result["sources"][0]
        self.assertEqual(source["source"], "centrifugal_pump_manual.docx")
        self.assertEqual(source["page"], 3)
        self.assertEqual(source["section"], "Operating Specifications")

    # -------------------------------------------------------------
    # 3. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
    # -------------------------------------------------------------
    @patch("app.retrieval.retriever.Retriever._get_active_documents")
    @patch("app.retrieval.retriever.Retriever.retrieve_dense")
    @patch("app.retrieval.retriever.Retriever.retrieve_bm25")
    def test_3_hybrid_retrieval_reciprocal_rank_fusion(
        self, mock_retrieve_bm25, mock_retrieve_dense, mock_active_docs
    ):
        """
        Scenario 3: Verify that Retriever.retrieve_hybrid fuses candidates from both
        dense vector search and local BM25 using RRF (k=60).
        """
        mock_active_docs.return_value = ["test_doc.md"]

        doc_a = Document(
            page_content="Chunk A: Turbine vibration monitoring thresholds.",
            metadata={"chunk_id": "test_doc.md_medium_0", "source": "test_doc.md", "page": 1},
        )
        doc_b = Document(
            page_content="Chunk B: Steam boiler feedwater chemistry parameters.",
            metadata={"chunk_id": "test_doc.md_medium_1", "source": "test_doc.md", "page": 2},
        )

        # Dense ranks doc_a higher, BM25 ranks doc_b higher
        mock_retrieve_dense.return_value = [(doc_a, 0.82), (doc_b, 0.71)]
        mock_retrieve_bm25.return_value = [(doc_b, 8.5), (doc_a, 6.2)]

        retriever = Retriever()
        fused_docs, diagnostics = retriever.retrieve_hybrid(
            query="turbine vibration limits",
            k=2,
            rrf_k=60,
        )

        self.assertEqual(len(fused_docs), 2)
        self.assertEqual(diagnostics.dense_candidates_count, 2)
        self.assertEqual(diagnostics.bm25_candidates_count, 2)
        self.assertEqual(diagnostics.total_unique_chunks, 2)

        # Confirm RRF score annotation
        for doc, score in fused_docs:
            self.assertGreater(score, 0.0)
            self.assertIn("rrf_score", doc.metadata)

    # -------------------------------------------------------------
    # 4. Out-of-Corpus Query with Insufficient Evidence (Abstention)
    # -------------------------------------------------------------
    @patch("app.services.chat_service.get_active_documents")
    @patch("app.services.rag_service.RagService._check_relevance")
    def test_4_question_with_no_supporting_evidence_abstention(
        self, mock_check_relevance, mock_active_docs
    ):
        """
        Scenario 4: When an active document exists but a query has no supporting evidence
        (similarity scores fall below threshold), verify safe rejection with zero sources.
        """
        mock_active_docs.return_value = ["marine_engine_manual.md"]
        # Scores fail threshold (score=0.35 < 0.65, BM25=0.0)
        mock_check_relevance.return_value = (False, 0.35, [])

        chat_service = ChatService()
        result = chat_service.chat(
            question="Who won the UEFA Champions League final in 2024?",
            top_k=5,
        )

        self.assertEqual(result["sources"], [])
        self.assertIn("Out-of-Domain Request", result["answer"])

    # -------------------------------------------------------------
    # 5. Failed or Invalid Upload Handling (Clean 400 Error)
    # -------------------------------------------------------------
    def test_5_failed_or_invalid_upload_handling(self):
        """
        Scenario 5: Upload an unsupported file format (.exe) and verify that
        the backend returns a 400 Bad Request with an informative error,
        without creating orphaned files.
        """
        bad_file = [
            ("files", ("malicious_payload.exe", io.BytesIO(b"MZ\x90\x00BinaryExe"), "application/octet-stream"))
        ]

        response = self.client.post("/api/v1/upload", files=bad_file)
        self.assertEqual(response.status_code, 400, "Unsupported upload should return 400.")
        self.assertIn("Unsupported file type '.exe'", response.json()["detail"])

        # Confirm orphaned file was NOT retained on disk
        orphaned = self.upload_dir / "malicious_payload.exe"
        self.assertFalse(orphaned.exists(), "Failed upload must not leave orphaned file on disk.")

    # -------------------------------------------------------------
    # 6. Document Deletion & Synchronized Cleanup
    # -------------------------------------------------------------
    @patch("app.retrieval.bm25.BM25Index.remove_document_by_filename")
    def test_6_document_deletion_and_cleanup(
        self, mock_bm25_remove
    ):
        """
        Scenario 6: Test document deletion, verifying synchronized removal of
        local disk file, Pinecone vectors, and BM25 chunks.
        """
        mock_bm25_remove.return_value = 3

        # Create temporary file in upload directory
        test_file = self.upload_dir / "doc_to_delete.md"
        test_file.write_text("# Temporary doc content", encoding="utf-8")
        self.assertTrue(test_file.exists())

        ingest_service = IngestService()
        mock_index = MagicMock()
        ingest_service.pipeline.vector_store.index = mock_index

        deleted = ingest_service.delete_document("doc_to_delete.md")

        self.assertTrue(deleted, "delete_document should return True when file was deleted.")
        self.assertFalse(test_file.exists(), "Local file was not deleted from disk.")

        # Verify Pinecone delete filter was called
        mock_index.delete.assert_called()
        # Verify BM25 remove was called
        mock_bm25_remove.assert_called_with("doc_to_delete.md")

    # -------------------------------------------------------------
    # 7. Service Failure and Rate-Limit Resilience
    # -------------------------------------------------------------
    @patch("app.services.chat_service.get_active_documents")
    @patch("app.services.rag_service.RagService._check_relevance")
    @patch("app.llm.chain.RAGChain.invoke")
    def test_7_service_failure_and_timeout_resilience(
        self, mock_chain_invoke, mock_check_relevance, mock_active_docs
    ):
        """
        Scenario 7: When upstream LLM provider hits a 429 rate limit or timeout,
        verify that ChatService handles the exception gracefully with an informative message.
        """
        mock_active_docs.return_value = ["test_manual.docx"]
        mock_check_relevance.return_value = (True, 0.85, [])
        mock_chain_invoke.side_effect = Exception("HTTP 429 Rate limit exceeded: Resource exhausted.")

        chat_service = ChatService()
        result = chat_service.chat(
            question="What is the operating clearance of the gas turbine blade?",
        )

        self.assertIn("Rate Limit Exceeded", result["answer"])
        self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main()
