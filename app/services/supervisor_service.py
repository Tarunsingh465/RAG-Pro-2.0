from app.core.logger import get_logger
from app.core.prompts import SUPERVISOR_PROMPT
from app.llm.client import LLMClient
from app.schemas.routing import RouteDecision

logger = get_logger()


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

        self.router = (
            SUPERVISOR_PROMPT
            | self.llm.with_structured_output(RouteDecision)
        )

        logger.info("Supervisor service initialized.")

    def route(self, query: str) -> RouteDecision:

        logger.debug("Routing user query.")

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
                f"Supervisor routing failed ({e}). Falling back to 'rag' route."
            )
            return RouteDecision(
                route="rag",
                confidence=1.0,
                reason="Fallback routing due to supervisor error",
            )