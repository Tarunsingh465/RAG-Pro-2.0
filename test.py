from pathlib import Path

from app.ingestion.loader import DocumentLoader
from app.ingestion.splitter import DocumentSplitter


def main():
    pdf_path = Path("data/uploads/happy.pdf")  # Change this path

    loader = DocumentLoader()
    splitter = DocumentSplitter()

    print("=" * 80)
    print("Loading document...")
    print("=" * 80)

    documents = loader.load(pdf_path)

    print(f"\nPages Loaded : {len(documents)}")

    for i, doc in enumerate(documents):
        print("\n" + "=" * 80)
        print(f"PAGE {i+1}")
        print("=" * 80)
        print(f"Metadata : {doc.metadata}")
        print("\nContent Preview:\n")
        print(doc.page_content[:1000])

    print("\n" + "=" * 80)
    print("Splitting...")
    print("=" * 80)

    chunks = splitter.split(documents)

    print(f"\nTotal Chunks : {len(chunks)}")

    for i, chunk in enumerate(chunks[:5]):
        print("\n" + "=" * 80)
        print(f"CHUNK {i+1}")
        print("=" * 80)
        print(f"Metadata : {chunk.metadata}")
        print("\nChunk Preview:\n")
        print(chunk.page_content[:600])

    print("\n✅ Test Completed Successfully")


if __name__ == "__main__":
    main()