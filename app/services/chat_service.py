from app.core.logger import get_logger
from app.services.greeting_service import GreetingService
from app.services.rag_service import RagService
from app.services.supervisor_service import SupervisorService

logger = get_logger()


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
    ):
        try:
            decision = self.supervisor.route(question)

            logger.info(f"Processing route: {decision.route} for query: {question!r}")

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

            if decision.route == "out_of_scope":
                # Safety check: if vectorstore actually contains matching content, answer it!
                is_relevant, top_score, _ = self.rag_service._check_relevance(
                    question=question,
                    document_id=document_id,
                    top_k=top_k,
                )
                if is_relevant:
                    logger.info(
                        f"Overriding out_of_scope route to RAG (found relevant content, score={top_score:.4f})"
                    )
                    return self.rag_service.ask(
                        question=question,
                        document_id=document_id,
                        top_k=top_k,
                        history=history,
                    )

                return {
                    "answer": (
                        "I'm designed to answer questions "
                        "about the documents provided to "
                        "this RAG system."
                    ),
                    "sources": [],
                }

            if decision.route == "rag":
                result = self.rag_service.ask(
                    question=question,
                    document_id=document_id,
                    top_k=top_k,
                    history=history,
                )

                if result.get("out_of_context"):
                    return {
                        "answer": (
                            "I couldn't find anything relevant to that in the "
                            "provided documents. Could you rephrase, or ask "
                            "something related to the document content?"
                        ),
                        "sources": [],
                    }

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
