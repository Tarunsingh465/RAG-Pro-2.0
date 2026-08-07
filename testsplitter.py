import argparse
import json
import sys
import time
from pathlib import Path

from app.ingestion.mistral_ocr import MistralOCRExtractor
from app.ingestion.loader import DocumentLoader


def test_extractor_direct(pdf_path: Path):
    print(f"\n[1/2] Testing MistralOCRExtractor on: {pdf_path.name}")

    extractor = MistralOCRExtractor(
        save_json=True,
        json_output_dir="data/ocr_output/test_run",
    )

    start = time.perf_counter()

    pages = extractor.extract(pdf_path)

    elapsed = time.perf_counter() - start

    assert isinstance(pages, list)
    assert len(pages) > 0

    total_chars = sum(len(p["text"]) for p in pages)

    print(f"Pages      : {len(pages)}")
    print(f"Characters : {total_chars}")
    print(f"Time       : {elapsed:.2f}s")

    print("\nSample text:\n")
    print(pages[0]["text"][:500])

    json_files = sorted(
        Path("data/ocr_output/test_run").glob(f"{pdf_path.stem}_*.json")
    )

    assert json_files

    print(f"\nSaved JSON : {json_files[-1]}")

    return pages


def test_loader(pdf_path: Path):
    print("\n[2/2] Testing DocumentLoader")

    loader = DocumentLoader()

    docs = loader.load(pdf_path)

    assert len(docs) > 0

    print(f"Documents : {len(docs)}")
    print(docs[0].metadata)
    print(docs[0].page_content[:300])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "pdf_path",
        help="PDF to test",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print("=" * 60)
    print("Mistral OCR Integration Test")
    print("=" * 60)
    print(f"Input PDF : {pdf_path}")
    print()

    test_extractor_direct(pdf_path)

    print()

    test_loader(pdf_path)

    print("\n✅ ALL TESTS PASSED")


if __name__ == "__main__":
    main()