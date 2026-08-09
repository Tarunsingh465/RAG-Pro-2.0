from app.core.logger import get_logger
from app.services.rag_service import RagService
from app.services.supervisor_service import SupervisorService

logger = get_logger()


class ChatService:

    def __init__(self):

        self.supervisor = SupervisorService()
        self.rag_service = RagService()

        logger.info("Chat service initialized.")

    def chat(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 5,
    ):

        decision = self.supervisor.route(question)

        logger.info(
            f"Processing route: {decision.route}"
        )

        if decision.route == "greeting":

            return {
                "answer": (
                    "Hello! 👋 "
                    "How can I help you with your documents?"
                ),
                "sources": [],
            }

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

            return {
                "answer": (
                    "I'm designed to answer questions "
                    "about the documents provided to "
                    "this RAG system."
                ),
                "sources": [],
            }

        if decision.route == "rag":

            return self.rag_service.ask(
                question=question,
                document_id=document_id,
                top_k=top_k,
            )

        logger.error(
            f"Unexpected route: {decision.route}"
        )

        return {
            "answer": "Unable to process the request.",
            "sources": [],
        }