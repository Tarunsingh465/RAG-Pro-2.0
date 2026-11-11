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
    Uses ultra-fast deterministic in-memory routing (< 0.5ms) without slow LLM overhead.
    """

    def __init__(self):
        logger.info("Supervisor service initialized with ultra-fast deterministic routing.")

    def route(self, query: str) -> RouteDecision:

        logger.debug("Routing user query.")

        # 1. Fast rule-based routing for conversational patterns via zero-API engine (< 0.5ms)
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

        # 2. Fast rule-based routing for engineering queries (< 0.5ms)
        is_eng, disc, disc_name = is_engineering_query(query)
        if is_eng:
            logger.info(f"Supervisor fast-path: recognized engineering query ({disc}) -> 'rag'.")
            return RouteDecision(
                route="rag",
                confidence=1.0,
                reason=f"Engineering discipline: {disc_name}",
            )

        # 3. Fast rule-based routing for candidate / resume evaluation queries (< 0.5ms)
        eval_keywords = [
            "hire", "hiring", "recruit", "candidate", "resume", "qualification",
            "cv", "experience", "recommend", "recommendation", "future", "suggestion",
            "advice", "suitability", "shortlist", "fit"
        ]
        if any(kw in query.lower() for kw in eval_keywords):
            logger.info("Supervisor fast-path: candidate evaluation query -> 'rag'.")
            return RouteDecision(
                route="rag",
                confidence=1.0,
                reason="Candidate evaluation query",
            )

        # 4. Default: Route directly to RAG (< 0.5ms).
        # RagService._check_relevance will verify against Pinecone in 50ms:
        # - If relevant docs exist: answers via RAG
        # - If no docs exist: evaluates engineering fallback or out-of-domain refusal
        return RouteDecision(
            route="rag",
            confidence=1.0,
            reason="Routed to RAG for document processing",
        )