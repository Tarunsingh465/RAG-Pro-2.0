class GuardrailService:

    def respond(self, question: str):
        return {
            "answer": (
                "I don't have information about that in the uploaded "
                "document(s). Could you rephrase, or ask something "
                "related to the document content?"
            ),
            "sources": [],
        }