from pathlib import Path

from app.core.config import settings
from app.core.constants import UPLOAD_FOLDER
from app.core.engineering import is_engineering_query
from app.core.logger import get_logger
from app.llm.chain import RAGChain
from app.llm.client import LLMClient
from app.vectorstore.pinecone import PineconeManager

logger = get_logger()

EVALUATIVE_KEYWORDS = [
    "hire",
    "hiring",
    "recruit",
    "fit",
    "candidate",
    "recommend",
    "should i",
    "suitability",
    "qualification",
    "shortlist",
]


class RagService:

    def __init__(self):
        self.chain = RAGChain()
        self.vectorstore = PineconeManager()

    def _get_search_query(self, question: str) -> tuple[str, bool]:
        is_evaluative = any(kw in question.lower() for kw in EVALUATIVE_KEYWORDS)
        is_eng, discipline, _ = is_engineering_query(question)
        if is_evaluative:
            return (
                f"{question} candidate qualifications skills experience projects education achievements",
                True,
            )
        if is_eng:
            return (
                f"{question} {discipline} calculation formula equation standard troubleshooting procedure",
                True,
            )
        return question, False

    def _check_relevance(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 5,
    ):
        """
        Run a similarity check before invoking the full chain.
        Returns (is_relevant, top_score, results).
        """
        upload_dir = Path(UPLOAD_FOLDER)
        active_docs = (
            [f.name for f in upload_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
            if upload_dir.exists()
            else []
        )

        if not active_docs:
            logger.info("RagService: No active documents in knowledge base. Bypassing vector search.")
            return False, 0.0, []

        if document_id:
            metadata_filter = {"document_id": document_id}
        elif len(active_docs) == 1:
            metadata_filter = {"file_name": {"$eq": active_docs[0]}}
        else:
            metadata_filter = {"file_name": {"$in": active_docs}}

        search_query, is_evaluative = self._get_search_query(question)

        results = self.vectorstore.similarity_search_with_score(
            search_query, k=top_k, filter=metadata_filter
        )

        if not results:
            logger.warning("No matches found in vectorstore for query.")
            return False, 0.0, []

        top_score = results[0][1]  # (Document, score) tuples

        # Adaptive thresholding: Evaluative & Engineering queries often use cross-domain vocabulary
        # compared to raw technical chunks, so use adaptive threshold 0.50.
        effective_threshold = 0.50 if is_evaluative else settings.RAG_SIMILARITY_THRESHOLD

        logger.info(
            f"Top similarity score: {top_score:.4f} (threshold: {effective_threshold:.2f}, evaluative={is_evaluative})"
        )

        is_relevant = top_score >= effective_threshold
        return is_relevant, top_score, results

    def ask(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 5,
        history: list[dict] | None = None,
    ):
        is_relevant, top_score, results = self._check_relevance(
            question, document_id=document_id, top_k=top_k
        )

        if not is_relevant:
            return {
                "answer": None,
                "sources": [],
                "out_of_context": True,
                "similarity_score": top_score,
            }

        response = self.chain.invoke(
            question=question,
            document_id=document_id,
            top_k=top_k,
            history=history,
        )

        sources = []
        for doc in response["documents"]:
            sources.append(
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                }
            )

        return {
            "answer": response["answer"],
            "sources": sources,
            "out_of_context": False,
            "similarity_score": top_score,
        }

    def answer_engineering_fallback(self, question: str, discipline: str = "general") -> dict:
        """
        Provides a concise, grounded engineering explanation when no documents match in the RAG store.
        Strictly follows user requirements:
        1. Specifies upfront that no documents were uploaded/found, answered from general engineering principles.
        2. Answers in a concise paragraph explaining the problem and key corrective measures.
        3. Asks user to upload relevant docs if they need a detailed/site-specific solution.
        """
        is_issue = any(
            kw in question.lower()
            for kw in [
                "why is", "why does", "heating", "failed", "failure", "crash",
                "error", "deadlock", "troubleshoot", "fix", "resolve", "slow",
                "overheating", "broken", "issue", "problem", "leak"
            ]
        )

        if is_issue:
            prompt = (
                f"You are an engineering specialist. In one short technical paragraph (3-4 sentences), "
                f"explain the engineering issue: '{question}'. "
                f"State the primary root cause and the immediate corrective measures to take. "
                f"Use bold for key terms. Do not include greetings or disclaimers."
            )
        else:
            prompt = (
                f"You are an engineering and technical specialist. In one short technical paragraph (3-4 sentences), "
                f"provide a direct, authoritative technical answer to: '{question}'. "
                f"Define the core concept, its foundational pillars, and primary engineering applications. "
                f"Use bold for key terms. Do not include greetings or disclaimers."
            )

        try:
            llm = LLMClient(temperature=0.2, max_tokens=2048).llm
            response = llm.invoke(prompt)
            content = getattr(response, "content", "")
            if not content:
                content = str(response)

            answer = (
                "ℹ️ **Note:** No relevant documents were found in the uploaded knowledge base for this query. "
                "The following answer is provided from general engineering principles:\n\n"
                f"{content.strip()}\n\n"
                "📂 *For a detailed, component-specific, or manufacturer-compliant solution, "
                "please upload the relevant equipment manual, datasheet, or specification document to the sidebar.*"
            )

            return {
                "answer": answer,
                "sources": [],
                "out_of_context": False,
                "fallback": True,
            }
        except Exception as e:
            logger.error(f"Error in engineering fallback generation: {e}")
            return {
                "answer": (
                    "ℹ️ **Note:** No relevant documents were found in the uploaded knowledge base for this query.\n\n"
                    "For this technical problem, inspect your physical hardware connections, thermal dissipation paths, "
                    "and operational limits. Please upload the relevant equipment manual or datasheet to receive a complete diagnosis."
                ),
                "sources": [],
            }
