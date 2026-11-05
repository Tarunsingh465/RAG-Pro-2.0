from langchain_core.output_parsers import StrOutputParser

from app.core.prompts import RAG_PROMPT
from app.llm.client import LLMClient
from app.retrieval.retriever import Retriever

EVALUATIVE_KEYWORDS = [
    "hire",
    "hiring",
    "recruit",
    "fit",
    "candidate",
    "recommend",
    "should i",
    "suitability",
    "qualification",
    "shortlist",
]


class RAGChain:

    def __init__(self):
        self.retriever = Retriever()
        llm = LLMClient().llm

        self.chain = (
            RAG_PROMPT
            | llm
            | StrOutputParser()
        )

    def invoke(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 5,
        history: list[dict] | None = None,
    ):
        is_evaluative = any(kw in question.lower() for kw in EVALUATIVE_KEYWORDS)
        search_query = (
            f"{question} candidate qualifications skills experience projects education achievements"
            if is_evaluative
            else question
        )

        docs = self.retriever.retrieve(
            query=search_query,
            document_id=document_id,
            k=top_k,
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # Format previous conversation turns if provided
        chat_history = "None."
        if history:
            history_lines = []
            for msg in history[-6:]:
                role = str(msg.get("role", "User")).capitalize()
                content = str(msg.get("content", "")).strip()
                if content:
                    history_lines.append(f"{role}: {content}")
            if history_lines:
                chat_history = "\n".join(history_lines)

        answer = self.chain.invoke(
            {
                "context": context,
                "chat_history": chat_history,
                "question": question,
            }
        )

        return {
            "answer": answer,
            "documents": docs,
        }
