from langchain_core.output_parsers import StrOutputParser

from app.core.prompts import RAG_PROMPT
from app.llm.client import LLMClient
from app.retrieval.retriever import Retriever


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
    ):

        docs = self.retriever.retrieve(
            query=question,
            document_id=document_id,
            k=top_k,
        )
        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        answer = self.chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        return {
            "answer": answer,
            "documents": docs,
        }