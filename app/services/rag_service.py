from app.llm.chain import RAGChain


class RagService:

    def __init__(self):
        self.chain = RAGChain()

    def ask(self, question: str):

        response = self.chain.invoke(question)

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