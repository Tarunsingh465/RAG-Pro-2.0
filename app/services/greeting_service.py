from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logger import get_logger
from app.services.conversational_engine import match_conversational_intent

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
        "You are TYRION-2.0, a friendly and intelligent assistant developed by Tarun, "
        "a B.Tech CS (Data Science and AI) specialist and final-year student.\n"
        "Tarun created this project for personal workflows and engineering-related problem-solving.\n"
        "Never say you were created or trained by OpenAI, NVIDIA, or anyone else. Your developer is Tarun.\n"
        "The user has sent a casual greeting or small-talk message, not a document question.\n"
        "Reply warmly and briefly (1-2 sentences), and invite them to ask a question about their documents or engineering problems.\n\n"
        "User message: {query}"
    )

    def __init__(self):
        model_name = settings.LLM_MODEL

        kwargs = {}
        if "nvidia" in settings.active_llm_base_url:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        self.llm = ChatOpenAI(
            model=model_name,
            api_key=settings.active_llm_api_key,
            base_url=settings.active_llm_base_url,
            temperature=0.7,
            max_tokens=256,
            request_timeout=30,
            default_headers={
                "HTTP-Referer": f"http://localhost",
                "X-Title": settings.APP_NAME,
            },
            **kwargs,
        )

        logger.info(f"Greeting service initialized with model={model_name}")

    def reply(self, question: str) -> dict:
        """
        Generate a conversational reply for a greeting-routed query.
        """

        logger.info(f"Handling greeting query: {question}")

        # Zero-API check
        intent_match = match_conversational_intent(question)
        if intent_match:
            logger.info(f"GreetingService returning zero-API reply for '{intent_match['id']}'.")
            return {
                "answer": intent_match["response"],
                "sources": [],
            }

        q_clean = question.strip().lower().rstrip("?!.,")
        instant_greetings = {
            "hi": "Hello! I am your RAG & Industrial Intelligence Assistant. How can I help you analyze your documents or solve an engineering problem today?",
            "hello": "Hi there! I am ready to help. Feel free to ask any question about your uploaded documents or engineering workflows.",
            "hey": "Hey! How can I assist you with your documents or engineering calculations today?",
            "good morning": "Good morning! How can I assist you with your documents or technical tasks today?",
            "good afternoon": "Good afternoon! What documents or engineering topics would you like to explore?",
            "good evening": "Good evening! How can I help you with your documents or technical queries today?",
            "how are you": "I'm doing great, thank you! Ready to help you inspect, analyze, and query your documents. What would you like to know?",
            "who are you": "I am TYRION-2.0, an AI-powered Enterprise Knowledge & Industrial Intelligence Assistant. I can analyze documents, extract technical data, evaluate candidates, and assist with engineering problem-solving.",
            "what can you do": "I can ingest multi-format technical documents (PDFs, Office files, CSVs, Notes), index them with semantic vector search, perform deep technical Q&A across engineering disciplines, troubleshoot faults, and evaluate resumes.",
            "help": "Simply upload your documents in the left sidebar and ask any question! I can explain concepts, calculate engineering formulas, search document sections, or evaluate candidate qualifications.",
        }

        if q_clean in instant_greetings:
            logger.info("Returned instant greeting response (0ms latency, 0 token cost).")
            return {
                "answer": instant_greetings[q_clean],
                "sources": [],
            }

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
