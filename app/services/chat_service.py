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
    ):

        decision = self.supervisor.route(question)

        logger.info(f"Processing route: {decision.route}")

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