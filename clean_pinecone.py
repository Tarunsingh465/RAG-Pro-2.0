from app.vectorstore.pinecone import PineconeManager
from app.core.logger import get_logger

logger = get_logger()


def clean_database():
    vector_store = PineconeManager()

    logger.info("Deleting all vectors from Pinecone index...")

    vector_store.index.delete(delete_all=True)

    logger.success("Database cleaned successfully!")


if __name__ == "__main__":
    clean_database()