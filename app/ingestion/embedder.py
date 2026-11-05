import re
import time
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()


class RateLimitedEmbeddings(Embeddings):
    """
    A robust wrapper around Google embeddings that handles:
    1. Automatic batching into chunks of 15 (preventing burst limit triggers).
    2. Intelligent retry with exponential backoff on 429 RESOURCE_EXHAUSTED errors.
    3. Parsing the exact 'retryDelay' returned by Google's API.
    """

    def __init__(
        self,
        base_embeddings: Embeddings,
        batch_size: int = 15,
        delay_between_batches: float = 0.5,
    ):
        self.base_embeddings = base_embeddings
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total_chunks = len(texts)

        for i in range(0, total_chunks, self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_chunks + self.batch_size - 1) // self.batch_size

            logger.info(
                f"Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)..."
            )

            max_retries = 5
            success = False
            for attempt in range(max_retries):
                try:
                    batch_embeddings = self.base_embeddings.embed_documents(batch)
                    all_embeddings.extend(batch_embeddings)

                    # Polite throttle between batches on free tier
                    if i + self.batch_size < total_chunks:
                        time.sleep(self.delay_between_batches)

                    success = True
                    break
                except Exception as e:
                    err_str = str(e)
                    err_lower = err_str.lower()

                    if (
                        "429" in err_lower
                        or "resource_exhausted" in err_lower
                        or "quota" in err_lower
                        or "rate limit" in err_lower
                    ):
                        if attempt == max_retries - 1:
                            logger.error(
                                f"Failed to embed batch {batch_num} after {max_retries} retries: {e}"
                            )
                            raise e

                        # Extract suggested wait time if available, e.g. "retry in 37.55s" or "'retryDelay': '37s'"
                        wait_time = 25.0 * (attempt + 1)
                        match = re.search(
                            r"(?:retry in|retrydelay['\":\s]+)([\d\.]+)",
                            err_str,
                            re.IGNORECASE,
                        )
                        if match:
                            try:
                                wait_time = float(match.group(1)) + 2.0
                            except ValueError:
                                pass

                        logger.warning(
                            f"Google Embedding rate limit reached (429). Waiting {wait_time:.1f}s "
                            f"(attempt {attempt + 1}/{max_retries}) before retrying..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Non-retryable embedding error: {e}")
                        raise e

            if not success:
                raise RuntimeError(
                    f"Embedding batch {batch_num} failed after {max_retries} attempts."
                )

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        max_retries = 4
        for attempt in range(max_retries):
            try:
                return self.base_embeddings.embed_query(text)
            except Exception as e:
                err_str = str(e)
                err_lower = err_str.lower()
                if (
                    "429" in err_lower
                    or "resource_exhausted" in err_lower
                    or "quota" in err_lower
                    or "rate limit" in err_lower
                ):
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Failed to embed query after {max_retries} retries: {e}"
                        )
                        raise e
                    wait_time = 15.0 * (attempt + 1)
                    match = re.search(
                        r"(?:retry in|retrydelay['\":\s]+)([\d\.]+)",
                        err_str,
                        re.IGNORECASE,
                    )
                    if match:
                        try:
                            wait_time = float(match.group(1)) + 2.0
                        except ValueError:
                            pass
                    logger.warning(
                        f"Google Embedding rate limit on query. Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise e
        raise RuntimeError("Failed to embed query after maximum retries.")


class Embedder:
    """
    Handles embedding generation using Google's embedding model
    wrapped with automatic rate-limiting and backoff protection.
    """

    def __init__(self):
        logger.info("Initializing embedding model...")

        base_model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )

        self.embedding_model = RateLimitedEmbeddings(
            base_embeddings=base_model,
            batch_size=15,
            delay_between_batches=0.6,
        )

        logger.success("Rate-limited Embedding model initialized.")

    def get_embedding_model(self):
        """
        Returns the embedding model instance.
        """
        return self.embedding_model

    def embed_query(self, query: str):
        """
        Generate embedding for a user query.
        """
        return self.embedding_model.embed_query(query)

    def embed_documents(self, texts: list[str]):
        """
        Generate embeddings for multiple text chunks.
        """
        return self.embedding_model.embed_documents(texts)
