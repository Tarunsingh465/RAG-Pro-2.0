from langchain_core.output_parsers import StrOutputParser

from app.core.engineering import is_engineering_query
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
        top_k: int = 4,
        history: list[dict] | None = None,
        mode: str = "hybrid",
    ):
        is_evaluative = any(kw in question.lower() for kw in EVALUATIVE_KEYWORDS)
        is_eng, discipline, disc_name = is_engineering_query(question)

        if is_evaluative:
            search_query = (
                f"{question} candidate qualifications skills experience projects education achievements"
            )
        elif is_eng:
            search_query = (
                f"{question} {discipline} calculation formula equation standard troubleshooting procedure"
            )
        else:
            search_query = question

        docs = self.retriever.retrieve(
            query=search_query,
            document_id=document_id,
            k=top_k,
            mode=mode,
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # Format previous conversation turns concisely to avoid token bloat and long reasoning delays
        chat_history = "None."
        if history:
            history_lines = []
            for msg in history[-3:]:
                role = str(msg.get("role", "User")).capitalize()
                content = str(msg.get("content", "")).strip()
                if content:
                    compact = content[:350] + "..." if len(content) > 350 else content
                    history_lines.append(f"{role}: {compact}")
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
