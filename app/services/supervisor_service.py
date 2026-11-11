from app.core.engineering import is_engineering_query
from app.core.logger import get_logger
from app.core.prompts import SUPERVISOR_PROMPT
from app.llm.client import LLMClient
from app.schemas.routing import RouteDecision
from app.services.conversational_engine import match_conversational_intent

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
            max_tokens=1024,
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

        # Fast rule-based routing for common conversational patterns via zero-API engine
        intent_match = match_conversational_intent(query)
        if intent_match:
            logger.info(f"Supervisor routed to greeting via zero-API intent: {intent_match['id']}.")
            return RouteDecision(
                route="greeting",
                confidence=1.0,
                reason=f"Zero-API intent: {intent_match['id']}",
            )

        clean_query = query.strip().lower().rstrip("?!.,")

        if clean_query in GREETING_PATTERNS:
            logger.info("Supervisor routed to greeting via fast-path rule.")
            return RouteDecision(
                route="greeting",
                confidence=1.0,
                reason="Direct conversational greeting match",
            )

        # Fast rule-based routing for engineering queries
        is_eng, disc, disc_name = is_engineering_query(query)
        if is_eng:
            logger.info(f"Supervisor fast-path: recognized engineering query ({disc}) -> 'rag'.")
            return RouteDecision(
                route="rag",
                confidence=1.0,
                reason=f"Engineering discipline: {disc_name}",
            )

        # Fast rule-based routing for candidate / resume evaluation queries
        eval_keywords = ["hire", "hiring", "recruit", "candidate", "resume", "qualification", "cv", "experience"]
        if any(kw in query.lower() for kw in eval_keywords):
            logger.info("Supervisor fast-path: candidate evaluation query -> 'rag'.")
            return RouteDecision(
                route="rag",
                confidence=1.0,
                reason="Candidate evaluation query",
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