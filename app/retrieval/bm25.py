"""
Lightweight Okapi BM25 Sparse Retrieval Implementation.

Designed for resource-constrained systems (8 GB RAM, CPU-first inference)
without external Elasticsearch or OpenSearch infrastructure.

Mathematical Formulation:
--------------------------
For a query Q with terms q_1, q_2, ..., q_m and document chunk D:

    Score(D, Q) = Σ IDF(q_i) * [ f(q_i, D) * (k1 + 1) ] / [ f(q_i, D) + k1 * (1 - b + b * (|D| / avgdl)) ]

Where:
- f(q_i, D) is the term frequency of token q_i in chunk D.
- |D| is the token count of chunk D.
- avgdl is the average chunk token length across the indexed corpus.
- N is the total number of chunks.
- n(q_i) is the number of chunks containing token q_i.
- IDF(q_i) is calculated with floor smoothing to prevent negative scores:
      IDF(q_i) = ln( (N - n(q_i) + 0.5) / (n(q_i) + 0.5) + 1.0 )
- Default hyperparameters: k1 = 1.5 (term saturation), b = 0.75 (length normalization).
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from app.core.logger import get_logger

logger = get_logger()

# Standard lightweight stopwords set (prevents non-informative term dominance)
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "let's", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your"
}

BM25_STORAGE_FILE = Path("data/.bm25_index.json")


def tokenize(text: str) -> List[str]:
    """
    Case-folds, strips punctuation, and extracts alphanumeric/technical tokens.
    Preserves hyphenated terms, model codes, and numerical identifiers.
    """
    clean = text.lower()
    # Match alphanumeric words, numbers, and technical tokens (e.g. i5-4440, 98c, iso-9001)
    tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", clean)
    return [t for t in tokens if t not in DEFAULT_STOPWORDS and len(t) > 1]


class BM25Index:
    """
    In-memory and file-backed Okapi BM25 Index.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        storage_path: Optional[Path] = None,
    ):
        self.k1 = k1
        self.b = b
        self.storage_path = storage_path or BM25_STORAGE_FILE

        # Inverted index structures:
        # doc_id -> list of tokens
        self.corpus: dict[str, dict[str, Any]] = {}
        # token -> set of chunk_ids
        self.inverted_index: dict[str, set[str]] = defaultdict(set)
        # token -> term frequency in chunk: (chunk_id, token) -> count
        self.tf: dict[Tuple[str, str], int] = Counter()
        # chunk_id -> document length (token count)
        self.doc_lengths: dict[str, int] = {}
        self.avgdl: float = 0.0

        self.load()

    def clear(self) -> None:
        """Wipes the index in memory and on disk."""
        self.corpus.clear()
        self.inverted_index.clear()
        self.tf.clear()
        self.doc_lengths.clear()
        self.avgdl = 0.0
        if self.storage_path.exists():
            try:
                self.storage_path.unlink()
            except Exception as e:
                logger.warning(f"Could not remove BM25 storage file: {e}")

    def add_documents(self, documents: Sequence[Document]) -> int:
        """
        Indexes a collection of LangChain chunks into the BM25 inverted index.
        """
        if not documents:
            return 0

        added_count = 0
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                continue

            tokens = tokenize(doc.page_content)
            if not tokens:
                continue

            self.corpus[chunk_id] = {
                "page_content": doc.page_content,
                "metadata": dict(doc.metadata),
                "tokens": tokens,
            }
            self.doc_lengths[chunk_id] = len(tokens)

            # Update term frequencies and inverted posting list
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                self.inverted_index[token].add(chunk_id)
                self.tf[(chunk_id, token)] = count

            added_count += 1

        self._recompute_stats()
        self.save()
        logger.info(f"BM25: Indexed {added_count} chunk(s). Total corpus size = {len(self.corpus)}.")
        return added_count

    def remove_document_by_filename(self, filename: str) -> int:
        """Removes all chunks belonging to a specific uploaded document."""
        to_remove = [
            cid for cid, item in self.corpus.items()
            if item["metadata"].get("file_name") == filename or item["metadata"].get("source") == filename
        ]
        return self._remove_chunks(to_remove)

    def remove_document_by_id(self, document_id: str) -> int:
        """Removes all chunks belonging to a specific document ID."""
        to_remove = [
            cid for cid, item in self.corpus.items()
            if item["metadata"].get("document_id") == document_id
        ]
        return self._remove_chunks(to_remove)

    def _remove_chunks(self, chunk_ids: List[str]) -> int:
        if not chunk_ids:
            return 0

        for cid in chunk_ids:
            if cid in self.corpus:
                tokens = self.corpus[cid]["tokens"]
                for t in set(tokens):
                    if t in self.inverted_index:
                        self.inverted_index[t].discard(cid)
                        if not self.inverted_index[t]:
                            del self.inverted_index[t]
                    self.tf.pop((cid, t), None)
                del self.corpus[cid]
                self.doc_lengths.pop(cid, None)

        self._recompute_stats()
        self.save()
        return len(chunk_ids)

    def _recompute_stats(self) -> None:
        total_docs = len(self.corpus)
        if total_docs > 0:
            self.avgdl = sum(self.doc_lengths.values()) / total_docs
        else:
            self.avgdl = 0.0

    def _idf(self, token: str, n_docs: int) -> float:
        """
        Calculates smoothed Inverse Document Frequency:
        IDF(q) = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1.0 )
        """
        df = len(self.inverted_index.get(token, set()))
        return math.log(((n_docs - df + 0.5) / (df + 0.5)) + 1.0)

    def search(
        self,
        query: str,
        top_k: int = 5,
        active_documents: Optional[List[str]] = None,
        document_id: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Searches the BM25 index and returns top-K (Document, score) pairs.
        Strictly enforces active-document filtering to prevent data leakage.
        """
        if not self.corpus:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Active document filter
        active_set = set(active_documents) if active_documents is not None else None

        # Filter candidate pool to active documents
        candidate_chunk_ids: set[str] = set()
        for token in query_tokens:
            candidate_chunk_ids.update(self.inverted_index.get(token, set()))

        if not candidate_chunk_ids:
            return []

        n_docs = len(self.corpus)
        scores: dict[str, float] = defaultdict(float)

        for cid in candidate_chunk_ids:
            item = self.corpus[cid]
            meta = item["metadata"]

            # Filter 1: Document ID check
            if document_id and meta.get("document_id") != document_id:
                continue

            # Filter 2: Active document filename check
            if active_set is not None:
                doc_name = meta.get("file_name") or meta.get("source")
                if doc_name not in active_set:
                    continue

            # Compute BM25 score for this chunk
            doc_len = self.doc_lengths.get(cid, 0)
            len_norm = 1.0 - self.b + self.b * (doc_len / self.avgdl if self.avgdl > 0 else 1.0)

            chunk_score = 0.0
            for token in query_tokens:
                tf_val = self.tf.get((cid, token), 0)
                if tf_val > 0:
                    idf_val = self._idf(token, n_docs)
                    numerator = tf_val * (self.k1 + 1.0)
                    denominator = tf_val + self.k1 * len_norm
                    chunk_score += idf_val * (numerator / denominator)

            if chunk_score > 0.0:
                scores[cid] = chunk_score

        if not scores:
            return []

        # Sort descending by score, tie-break by chunk_id for determinism
        sorted_chunks = sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)
        top_results = sorted_chunks[:top_k]

        results: List[Tuple[Document, float]] = []
        for cid, score in top_results:
            item = self.corpus[cid]
            doc = Document(
                page_content=item["page_content"],
                metadata=dict(item["metadata"]),
            )
            results.append((doc, float(score)))

        return results

    def save(self) -> None:
        """Persists the index to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                "k1": self.k1,
                "b": self.b,
                "avgdl": self.avgdl,
                "corpus": self.corpus,
            }
            self.storage_path.write_text(json.dumps(serializable), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not save BM25 index: {e}")

    def load(self) -> None:
        """Loads index from disk if present."""
        if not self.storage_path.exists():
            return

        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self.k1 = data.get("k1", 1.5)
            self.b = data.get("b", 0.75)
            self.avgdl = data.get("avgdl", 0.0)
            self.corpus = data.get("corpus", {})

            # Rebuild inverted index and tf tables from corpus
            self.inverted_index.clear()
            self.tf.clear()
            self.doc_lengths.clear()

            for cid, item in self.corpus.items():
                tokens = item.get("tokens", [])
                self.doc_lengths[cid] = len(tokens)
                counts = Counter(tokens)
                for t, count in counts.items():
                    self.inverted_index[t].add(cid)
                    self.tf[(cid, t)] = count

            self._recompute_stats()
            logger.info(f"Loaded BM25 index with {len(self.corpus)} chunks from '{self.storage_path.name}'.")
        except Exception as e:
            logger.warning(f"Failed to load BM25 index from '{self.storage_path}': {e}")

