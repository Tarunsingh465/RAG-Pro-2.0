from app.ingestion.embedder import Embedder

embedder = Embedder()

vector = embedder.embed_query("What is Retrieval Augmented Generation?")

print(type(vector))
print(len(vector))
print(vector[:10])