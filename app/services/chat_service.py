from pathlib import Path

from app.core.constants import UPLOAD_FOLDER
from app.core.engineering import is_engineering_query
from app.core.logger import get_logger
from app.services.conversational_engine import match_conversational_intent
from app.services.greeting_service import GreetingService
from app.services.rag_service import RagService
from app.services.supervisor_service import SupervisorService

logger = get_logger()


def get_active_documents() -> list[str]:
    """Return list of active document filenames currently stored in the knowledge base."""
    upload_dir = Path(UPLOAD_FOLDER)
    if not upload_dir.exists():
        return []
    return [f.name for f in upload_dir.iterdir() if f.is_file() and not f.name.startswith(".")]


class ChatService:

    def __init__(self):
        self.supervisor = SupervisorService()
        self.greeting_service = GreetingService()
        self.rag_service = RagService()

        logger.info("Chat service initialized.")

    def chat(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 5,
        history: list[dict] | None = None,
        retrieval_mode: str = "hybrid",
    ):
        try:
            # 1. Zero-API conversational interceptor:
            # Intercepts generic small-talk, status checks, bot identity, and capability
            # questions in-memory in < 0.5ms with ZERO external API calls or credit cost.
            intent_match = match_conversational_intent(question)
            if intent_match:
                logger.info(
                    f"Zero-API intent matched ({intent_match['id']}) in <0.5ms. "
                    f"Returning local response with 0 external API calls."
                )
                return {
                    "answer": intent_match["response"],
                    "sources": [],
                }

            decision = self.supervisor.route(question)
            active_docs = get_active_documents()
            has_documents = bool(active_docs)

            logger.info(
                f"Processing route: {decision.route} for query: {question!r} "
                f"(active_documents={len(active_docs)})"
            )

            if decision.route == "greeting":
                return self.greeting_service.reply(question)

            if decision.route == "guardrail":
                return {
                    "answer": (
                        "I can't help with that request. "
                        "I can help with questions related "
                        "to the provided documents."
                    ),
                    "sources": [],
                }

            # If NO documents are currently uploaded in the knowledge base:
            if not has_documents:
                is_eng, disc, _ = is_engineering_query(question)
                if is_eng:
                    logger.info(f"No documents uploaded. Handling engineering query ({disc}) via fallback.")
                    return self.rag_service.answer_engineering_fallback(question, discipline=disc)

                logger.info("No documents uploaded and non-engineering query. Rejecting out-of-domain request with 0 sources.")
                return {
                    "answer": (
                        "⚠️ **No Documents Uploaded:** The knowledge base is currently empty. "
                        "Please upload a document to analyze its content, or ask an engineering, IT, or candidate-evaluation question."
                    ),
                    "sources": [],
                    "out_of_context": True,
                }

            if decision.route == "out_of_scope":
                # Safety check: if vectorstore actually contains matching content from active docs, answer it!
                is_relevant, top_score, _ = self.rag_service._check_relevance(
                    question=question,
                    document_id=document_id,
                    top_k=top_k,
                )
                if is_relevant:
                    logger.info(
                        f"Overriding out_of_scope route to RAG (found relevant content, score={top_score:.4f})"
                    )
                    result = self.rag_service.ask(
                        question=question,
                        document_id=document_id,
                        top_k=top_k,
                        history=history,
                        mode=retrieval_mode,
                    )
                    # Sanitize sources: only allow active documents
                    if result.get("sources"):
                        result["sources"] = [s for s in result["sources"] if s.get("source") in active_docs]
                    return result

                is_eng, disc, _ = is_engineering_query(question)
                if is_eng:
                    logger.info(f"Out-of-scope query recognized as engineering ({disc}). Using engineering fallback.")
                    return self.rag_service.answer_engineering_fallback(question, discipline=disc)

                return {
                    "answer": (
                        "⚠️ **Out-of-Domain Request:** The RAG system is not authorized to answer "
                        "questions unrelated to your uploaded documents or technical engineering domains. "
                        "Please upload a document to analyze or ask an engineering, IT, or candidate-evaluation question."
                    ),
                    "sources": [],
                }

            if decision.route == "rag":
                result = self.rag_service.ask(
                    question=question,
                    document_id=document_id,
                    top_k=top_k,
                    history=history,
                    mode=retrieval_mode,
                )

                if result.get("out_of_context"):
                    is_eng, disc, _ = is_engineering_query(question)
                    if is_eng:
                        logger.info(
                            f"Out-of-context RAG query recognized as engineering ({disc}). Using engineering fallback."
                        )
                        return self.rag_service.answer_engineering_fallback(question, discipline=disc)

                    return {
                        "answer": (
                            "⚠️ **Out-of-Domain Request:** The RAG system is not authorized to answer "
                            "questions unrelated to your uploaded documents or technical engineering domains. "
                            "Please upload a document to analyze or ask an engineering, IT, or candidate-evaluation question."
                        ),
                        "sources": [],
                    }

                # If LLM fell back to context-miss phrase, verify if question is engineering or out-of-domain
                answer_lower = (result.get("answer", "") or "").lower()
                not_found_triggers = [
                    "couldn't find",
                    "could not find",
                    "cannot be found",
                    "can't be found",
                    "not found in",
                    "not mentioned in",
                    "no mention of",
                    "no information about",
                    "not contained in",
                    "does not contain information",
                    "doesn't contain information",
                    "not found in the provided document",
                ]
                if any(trigger in answer_lower for trigger in not_found_triggers):
                    is_eng, disc, _ = is_engineering_query(question)
                    if is_eng:
                        logger.info(
                            f"LLM indicated context-miss for engineering query ({disc}). Using engineering fallback."
                        )
                        return self.rag_service.answer_engineering_fallback(question, discipline=disc)
                    else:
                        logger.info("LLM indicated context-miss for non-engineering query. Returning out-of-domain message without sources.")
                        return {
                            "answer": (
                                "⚠️ **Out-of-Domain Request:** The RAG system is not authorized to answer "
                                "questions unrelated to your uploaded documents or technical engineering domains. "
                                "Please ask a question related to your uploaded documents, or ask an engineering question."
                            ),
                            "sources": [],
                        }

                # Ensure only currently active documents are in sources
                if result.get("sources"):
                    result["sources"] = [s for s in result["sources"] if s.get("source") in active_docs]

                return result

            logger.error(f"Unexpected route: {decision.route}")

            return {
                "answer": "Unable to process the request.",
                "sources": [],
            }
        except Exception as e:
            err_str = str(e)
            if "402" in err_str or "credits" in err_str.lower() or "payment" in err_str.lower():
                logger.error(f"OpenRouter credit limit reached: {e}")
                return {
                    "answer": (
                        "⚠️ **OpenRouter Credit Limit Reached (Error 402)**\n\n"
                        "Your OpenRouter API account has run out of credits to complete this request.\n\n"
                        "**How to resolve this:**\n"
                        "1. Go to [https://openrouter.ai/settings/credits](https://openrouter.ai/settings/credits) and top up your account balance.\n"
                        "2. Or update your `OPENROUTER_API_KEY` in your `.env` file with an active key."
                    ),
                    "sources": [],
                }
            logger.error(f"Unhandled error in chat service: {e}")
            raise e
