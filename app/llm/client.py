from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()


class LLMClient:
    """
    Creates and returns a LangChain Chat Model.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        logger.info("Initializing OpenRouter LLM...")

        self._llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logger.success("OpenRouter initialized.")

    @property
    def llm(self):
        return self._llm
