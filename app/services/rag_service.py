from app.llm.chain import RAGChain


class RagService:

    def __init__(self):
        self.chain = RAGChain()

    def ask(self, question: str, document_id: str | None = None, top_k: int = 5, ):

        response = self.chain.invoke(question=question, document_id=document_id, top_k=top_k,)

        sources = []

        for doc in response["documents"]:

            sources.append(
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                }
            )

        return {
            "answer": response["answer"],
            "sources": sources,
        }