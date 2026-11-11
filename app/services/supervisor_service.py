from app.core.logger import get_logger
from app.core.prompts import SUPERVISOR_PROMPT
from app.llm.client import LLMClient
from app.schemas.routing import RouteDecision

logger = get_logger()


GREETING_PATTERNS = {
    "hi", "hello", "hey", "hola", "greetings", "good morning",
    "good afternoon", "good evening", "how are you", "who are you",
    "what can you do", "help", "bye", "goodbye", "see you"
}


class SupervisorService:
    """
    Determines the appropriate route for a user query.

    The supervisor does not answer the query.
    """

    def __init__(self):

        self.llm = LLMClient(
            temperature=0.0,
            max_tokens=120,
        ).llm

        try:
            self.router = (
                SUPERVISOR_PROMPT
                | self.llm.with_structured_output(RouteDecision)
            )
        except Exception:
            self.router = None

        logger.info("Supervisor service initialized.")

    def route(self, query: str) -> RouteDecision:

        logger.debug("Routing user query.")

        clean_query = query.strip().lower().rstrip("?!.,")

        # Fast rule-based routing for common conversational patterns
        if clean_query in GREETING_PATTERNS:
            logger.info("Supervisor routed to greeting via fast-path rule.")
            return RouteDecision(
                route="greeting",
                confidence=1.0,
                reason="Direct conversational greeting match",
            )

        # LLM classification
        if self.router:
            try:
                decision = self.router.invoke(
                    {
                        "query": query
                    }
                )

                logger.info(
                    f"Supervisor route={decision.route} "
                    f"confidence={decision.confidence:.2f}"
                )

                return decision
            except Exception as e:
                logger.warning(
                    f"Supervisor LLM routing bypassed ({e}). Falling back to 'rag' route."
                )

        return RouteDecision(
            route="rag",
            confidence=1.0,
            reason="Routed to RAG for document processing",
        )