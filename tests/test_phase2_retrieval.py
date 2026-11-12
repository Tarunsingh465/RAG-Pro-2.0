"""
Comprehensive Phase 2 Test Suite: Document Intelligence and Retrieval Engineering.

Covers:
1. BM25 scoring calculation.
2. BM25 ranking order.
3. BM25 empty corpus handling.
4. BM25 unknown query terms.
5. BM25 duplicate chunk handling.
6. BM25 active-document filtering (anti-leakage).
7. RRF score calculation (k=60).
8. RRF chunk deduplication and fusion boosting.
9. RRF missing chunks handling.
10. DOCX structured extraction (paragraphs, headings, markdown tables).
11. PPTX slide attribution (slide-by-slide pagination, tables).
12. XLSX multi-sheet extraction (sheet names, row metadata, markdown tables).
13. Chunk metadata preservation across small/medium/large strategies.
14. Embedding dimension validation and mismatch error raising.
15. Startup without external embedding calls (caching verification).

All tests use mocks or in-memory fixtures; no external API calls required.
"""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from docx import Document as DocxDoc
from langchain_core.documents import Document
from pptx import Presentation
from pptx.util import Inches

from app.ingestion.extractors.office_extractor import OfficeExtractor
from app.ingestion.extractors.tabular_extractor import TabularExtractor
from app.ingestion.splitter import CHUNK_PRESETS, DocumentSplitter
from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.hybrid import (
    DEFAULT_RRF_K,
    get_chunk_identifier,
    reciprocal_rank_fusion,
)
from app.vectorstore.pinecone import PineconeManager


class TestBM25SparseRetrieval(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_file = Path(self.temp_dir.name) / "test_bm25.json"
        self.bm25 = BM25Index(k1=1.5, b=0.75, storage_path=self.storage_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_bm25_empty_corpus(self):
        """Empty BM25 corpus returns empty list without error."""
        results = self.bm25.search("turbocharger failure", top_k=5)
        self.assertEqual(results, [])

    def test_bm25_unknown_query_terms(self):
        """Query with terms not in corpus returns empty list."""
        doc = Document(
            page_content="Centrifugal pump seal degradation caused cavitation in the impeller.",
            metadata={"chunk_id": "c1", "file_name": "pump_manual.pdf"}
        )
        self.bm25.add_documents([doc])
        results = self.bm25.search("quantum computer superconducting qubit", top_k=5)
        self.assertEqual(results, [])

    def test_bm25_scoring_and_ranking(self):
        """Chunks matching more query terms or with higher TF rank higher."""
        doc1 = Document(
            page_content="High temperature causes bearing vibration and turbocharger compressor surge.",
            metadata={"chunk_id": "c1", "file_name": "turbo.pdf"}
        )
        doc2 = Document(
            page_content="Turbocharger turbine exhaust temperature exceeds limits during prolonged engine load.",
            metadata={"chunk_id": "c2", "file_name": "turbo.pdf"}
        )
        doc3 = Document(
            page_content="Paint finish and cabin upholstery specifications for operator console.",
            metadata={"chunk_id": "c3", "file_name": "interior.pdf"}
        )
        self.bm25.add_documents([doc1, doc2, doc3])

        results = self.bm25.search("turbocharger temperature surge", top_k=3)
        self.assertGreaterEqual(len(results), 2)
        # doc1 matches turbocharger, temperature, surge -> highest score
        self.assertEqual(results[0][0].metadata["chunk_id"], "c1")
        self.assertGreater(results[0][1], results[1][1])

    def test_bm25_duplicate_chunks(self):
        """Adding chunks with identical chunk_id updates cleanly without duplication."""
        doc1 = Document(
            page_content="Initial version of valve specification.",
            metadata={"chunk_id": "c1", "file_name": "valve.pdf"}
        )
        self.bm25.add_documents([doc1])
        self.assertEqual(len(self.bm25.corpus), 1)

        # Update with new content on same chunk_id
        doc1_updated = Document(
            page_content="Revised version of high-pressure safety relief valve specification.",
            metadata={"chunk_id": "c1", "file_name": "valve.pdf"}
        )
        self.bm25.add_documents([doc1_updated])
        self.assertEqual(len(self.bm25.corpus), 1)

        results = self.bm25.search("safety relief valve", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("high-pressure", results[0][0].page_content)

    def test_bm25_active_document_filtering(self):
        """Active document filtering strictly isolates search to allowed filenames."""
        doc1 = Document(
            page_content="Internal confidential turbine test results from plant alpha.",
            metadata={"chunk_id": "c1", "file_name": "confidential_alpha.pdf"}
        )
        doc2 = Document(
            page_content="Public turbine operation guidelines and safety measures.",
            metadata={"chunk_id": "c2", "file_name": "public_guidelines.pdf"}
        )
        self.bm25.add_documents([doc1, doc2])

        # Query allowed only for public_guidelines.pdf
        results = self.bm25.search(
            query="turbine guidelines safety",
            top_k=5,
            active_documents=["public_guidelines.pdf"]
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].metadata["file_name"], "public_guidelines.pdf")


class TestReciprocalRankFusion(unittest.TestCase):

    def test_rrf_score_calculation(self):
        """Verifies exact RRF formula: 1 / (k + rank)."""
        doc_a = Document(page_content="Chunk A", metadata={"chunk_id": "A", "page": 1})
        doc_b = Document(page_content="Chunk B", metadata={"chunk_id": "B", "page": 2})

        dense_results = [(doc_a, 0.90), (doc_b, 0.80)]
        bm25_results = []

        fused, diag = reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=2)
        expected_score_a = 1.0 / (60 + 1)  # 1/61 = 0.01639344...
        expected_score_b = 1.0 / (60 + 2)  # 1/62 = 0.01612903...

        self.assertAlmostEqual(fused[0][1], expected_score_a, places=6)
        self.assertAlmostEqual(fused[1][1], expected_score_b, places=6)
        self.assertEqual(diag.dense_candidates_count, 2)
        self.assertEqual(diag.bm25_candidates_count, 0)

    def test_rrf_deduplication_and_score_boost(self):
        """Chunks appearing in both Dense and BM25 receive boosted reciprocal rank sum."""
        doc_shared = Document(page_content="Shared Chunk", metadata={"chunk_id": "shared"})
        doc_dense_only = Document(page_content="Dense Only", metadata={"chunk_id": "dense_only"})
        doc_sparse_only = Document(page_content="Sparse Only", metadata={"chunk_id": "sparse_only"})

        dense_results = [(doc_shared, 0.88), (doc_dense_only, 0.85)]
        bm25_results = [(doc_shared, 8.2), (doc_sparse_only, 7.5)]

        fused, diag = reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=3)

        # doc_shared is rank 1 in dense (1/61) and rank 1 in bm25 (1/61) -> 2/61 ≈ 0.032786
        expected_shared_score = (1.0 / 61) + (1.0 / 61)
        self.assertEqual(fused[0][0].metadata["chunk_id"], "shared")
        self.assertAlmostEqual(fused[0][1], expected_shared_score, places=6)
        self.assertEqual(diag.total_unique_chunks, 3)

    def test_rrf_missing_documents_in_one_ranking(self):
        """Documents only present in one list are handled gracefully."""
        doc_dense = Document(page_content="Only Dense", metadata={"chunk_id": "d1"})
        doc_sparse = Document(page_content="Only Sparse", metadata={"chunk_id": "s1"})

        dense_results = [(doc_dense, 0.75)]
        bm25_results = [(doc_sparse, 5.0)]

        fused, diag = reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=2)
        self.assertEqual(len(fused), 2)
        ids = [d.metadata["chunk_id"] for d, _ in fused]
        self.assertIn("d1", ids)
        self.assertIn("s1", ids)


class TestStructuredDocumentExtractors(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_docx_table_and_section_extraction(self):
        """DOCX extractor extracts headings, paragraphs, and markdown tables with section context."""
        doc = DocxDoc()
        doc.add_heading("Section 1: Engine Specifications", level=1)
        doc.add_paragraph("The engine operates under standard ambient conditions.")

        # Add table
        table = doc.add_table(rows=3, cols=2)
        table.cell(0, 0).text = "Parameter"
        table.cell(0, 1).text = "Value"
        table.cell(1, 0).text = "Rated Power"
        table.cell(1, 1).text = "1200 kW"
        table.cell(2, 0).text = "Max RPM"
        table.cell(2, 1).text = "1800"

        file_path = self.dir_path / "test_engine.docx"
        doc.save(str(file_path))

        extractor = OfficeExtractor()
        extracted = extractor.extract(file_path)

        self.assertGreaterEqual(len(extracted), 1)
        combined_text = "\n".join(d.page_content for d in extracted)
        self.assertIn("Section 1: Engine Specifications", combined_text)
        self.assertIn("| Parameter | Value |", combined_text)
        self.assertIn("| Rated Power | 1200 kW |", combined_text)
        self.assertTrue(any(d.metadata.get("has_table") for d in extracted))

    def test_pptx_slide_attribution(self):
        """PPTX extractor produces slide-by-slide documents with accurate slide numbers."""
        prs = Presentation()
        # Slide 1
        slide1 = prs.slides.add_slide(prs.slide_layouts[0])
        slide1.shapes.title.text = "Slide 1: Marine Propulsion System"
        # Slide 2
        slide2 = prs.slides.add_slide(prs.slide_layouts[1])
        slide2.shapes.title.text = "Slide 2: Root Cause Analysis"

        file_path = self.dir_path / "test_deck.pptx"
        prs.save(str(file_path))

        extractor = OfficeExtractor()
        extracted = extractor.extract(file_path)

        self.assertEqual(len(extracted), 2)
        self.assertEqual(extracted[0].metadata["page"], 1)
        self.assertEqual(extracted[0].metadata["slide_number"], 1)
        self.assertIn("Marine Propulsion", extracted[0].page_content)
        self.assertEqual(extracted[1].metadata["page"], 2)
        self.assertEqual(extracted[1].metadata["slide_number"], 2)
        self.assertIn("Root Cause Analysis", extracted[1].page_content)

    def test_xlsx_multi_sheet_extraction(self):
        """XLSX extractor extracts all sheets, sheet names, and row batches with repeated headers."""
        file_path = self.dir_path / "test_plant.xlsx"
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df1 = pd.DataFrame({"Pump_ID": ["P-101", "P-102"], "Flow_Rate_GPM": [250, 310]})
            df1.to_excel(writer, sheet_name="Pumps", index=False)

            df2 = pd.DataFrame({"Alarm_Code": ["ALM-01", "ALM-02"], "Severity": ["High", "Critical"]})
            df2.to_excel(writer, sheet_name="Alarms", index=False)

        extractor = TabularExtractor(rows_per_chunk=10)
        extracted = extractor.extract(file_path)

        self.assertEqual(len(extracted), 2)
        sheets_found = [d.metadata["sheet_name"] for d in extracted]
        self.assertIn("Pumps", sheets_found)
        self.assertIn("Alarms", sheets_found)
        self.assertIn("P-101", extracted[0].page_content)
        self.assertIn("ALM-02", extracted[1].page_content)


class TestConfigurableChunking(unittest.TestCase):

    def test_chunking_presets_and_character_units(self):
        """Splitter supports small/medium/large presets and preserves all metadata attributes."""
        sample_doc = Document(
            page_content=(
                "Turbocharger maintenance requires inspecting the compressor wheel, turbine housing, "
                "and floating bearings every 2000 operational hours. Lubrication oil pressure must be "
                "maintained at 2.5 bar to avoid hydrodynamic bearing breakdown. " * 8
            ),
            metadata={"source": "turbo_manual.pdf", "page": 1, "document_id": "doc_123"}
        )

        for preset_name in ["small", "medium", "large"]:
            splitter = DocumentSplitter(strategy=preset_name)
            chunks = splitter.split([sample_doc])
            preset_size = CHUNK_PRESETS[preset_name]["chunk_size"]

            self.assertGreater(len(chunks), 0)
            for idx, chunk in enumerate(chunks):
                self.assertLessEqual(len(chunk.page_content), preset_size + 150)
                self.assertEqual(chunk.metadata["chunk_strategy"], preset_name)
                self.assertEqual(chunk.metadata["chunk_index"], idx)
                self.assertEqual(chunk.metadata["document_id"], "doc_123")
                self.assertIn("chunk_id", chunk.metadata)
                self.assertIn("document_version_id", chunk.metadata)
                self.assertEqual(chunk.metadata["page"], 1)


class TestEmbeddingDimensionAndStartup(unittest.TestCase):

    def test_dimension_mismatch_raises_error(self):
        """Mismatched model and index dimensions raise an informative ValueError."""
        with patch.object(PineconeManager, "__init__", lambda self: None):
            pm = PineconeManager()
            pm.index_name = "rag-pro"
            pm.model_name = "models/test-model"
            pm.dimension = 768
            pm._load_cache = lambda: {}

            mock_client = MagicMock()
            mock_client.list_indexes().names.return_value = ["rag-pro"]
            mock_desc = MagicMock()
            mock_desc.dimension = 3072  # Index expects 3072, but model is 768
            mock_client.describe_index.return_value = mock_desc
            pm.client = mock_client

            with self.assertRaises(ValueError) as ctx:
                pm._ensure_index()
            self.assertIn("Dimension mismatch error", str(ctx.exception))
            self.assertIn("768", str(ctx.exception))
            self.assertIn("3072", str(ctx.exception))

    def test_startup_uses_cache_without_calling_embed_query(self):
        """Cached dimensions bypass external embedding calls completely."""
        with patch.object(PineconeManager, "__init__", lambda self: None):
            pm = PineconeManager()
            pm.model_name = "models/gemini-embedding-001"
            pm.index_name = "rag-pro"
            pm._load_cache = lambda: {
                "model_name": "models/gemini-embedding-001",
                "model_dimension": 3072,
                "index_name": "rag-pro",
                "index_dimension": 3072,
            }
            mock_embedder = MagicMock()
            pm.embedding = mock_embedder

            dim = pm._resolve_model_dimension()
            self.assertEqual(dim, 3072)
            # embed_query must NOT have been called!
            mock_embedder.embed_query.assert_not_called()


if __name__ == "__main__":
    unittest.main()

