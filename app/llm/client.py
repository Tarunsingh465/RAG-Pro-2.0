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
        max_tokens: int = 2048,
    ):
        provider = "NVIDIA NIM" if "nvidia" in settings.active_llm_base_url else "OpenRouter"
        logger.info(f"Initializing {provider} LLM ({settings.LLM_MODEL})...")

        kwargs = {}
        if "nvidia" in settings.active_llm_base_url:
            # Disable recursive reasoning overhead for high-speed deterministic responses (<2s)
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        self._llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.active_llm_api_key,
            base_url=settings.active_llm_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=60,
            **kwargs,
        )

        logger.success(f"{provider} initialized.")

    @property
    def llm(self):
        return self._llm
