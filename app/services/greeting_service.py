from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()


class GreetingService:
    """
    Handles casual/greeting queries using a lightweight OpenRouter model,
    fully independent from the supervisor/RAG LLM setup.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    # Small/cheap model — greeting task is trivial.
    MODEL = "meta-llama/llama-3.1-8b-instruct"

    GREETING_PROMPT = (
        "You are a friendly assistant for a document Q&A system.\n"
        "The user has sent a casual greeting or small-talk message, "
        "not a document question.\n"
        "Reply warmly and briefly (1-2 sentences), and invite them to "
        "ask a question about their documents.\n\n"
        "User message: {query}"
    )

    def __init__(self):

        self.llm = ChatOpenAI(
            model=self.MODEL,
            api_key=settings.OPENROUTER_API_KEY,
            base_url=self.OPENROUTER_BASE_URL,
            temperature=0.7,
            max_tokens=256,
            default_headers={
                "HTTP-Referer": f"http://localhost",
                "X-Title": settings.APP_NAME,
            },
        )

        logger.info(f"Greeting service initialized with model={self.MODEL}")

    def reply(self, question: str) -> dict:
        """
        Generate a conversational reply for a greeting-routed query.
        """

        logger.info(f"Handling greeting query: {question}")

        try:
            prompt = self.GREETING_PROMPT.format(query=question)
            response = self.llm.invoke(prompt)

            self._log_response_diagnostics(response)

            answer = self._extract_content(response)

            if not answer:
                logger.warning("Empty greeting response, using fallback.")
                answer = self._fallback_answer()

            return {
                "answer": answer,
                "sources": [],
            }

        except Exception:
            logger.exception("Greeting service failed.")

            return {
                "answer": self._fallback_answer(),
                "sources": [],
            }

    @staticmethod
    def _fallback_answer() -> str:
        return "Hello! 👋 How can I help you with your documents?"

    @staticmethod
    def _log_response_diagnostics(response) -> None:
        """
        Log provider response details needed to diagnose empty completions.
        """

        logger.debug(f"Greeting response object: {response!r}")
        logger.debug(
            f"Greeting response content: {getattr(response, 'content', None)!r}"
        )
        logger.debug(
            "Greeting response metadata: "
            f"{getattr(response, 'response_metadata', None)!r}"
        )
        logger.debug(
            "Greeting usage metadata: "
            f"{getattr(response, 'usage_metadata', None)!r}"
        )

    @staticmethod
    def _extract_content(response) -> str | None:
        """
        Safely extract text from a LangChain AIMessage.
        """

        if response is None:
            return None

        if isinstance(response, str):
            return response

        content = getattr(response, "content", None)

        if isinstance(content, str):
            return content

        if isinstance(content, list):

            parts = []

            for block in content:

                if isinstance(block, dict):

                    text = block.get("text")

                    if text:
                        parts.append(str(text))

                elif isinstance(block, str):

                    parts.append(block)

            return " ".join(parts) or None

        logger.warning(
            f"Unexpected greeting response type: {type(response)!r}"
        )
        return None
