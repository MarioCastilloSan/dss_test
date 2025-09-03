# src/ingestion_agent.py
import logging
from typing import Optional

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.config import DOCS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class IngestionAgent:
    """
    Handles ingestion pipeline: read -> split -> embed -> store in Qdrant.
    """

    def __init__(self, docs_dir: Optional[str] = None):
        self.processor = DocumentProcessor(docs_dir or DOCS_DIR)
        self.vector_store = VectorStore(self.processor.embeddings)

    def ingest(self) -> bool:
        """
        Run the ingestion pipeline: load, split, embed, insert into Qdrant.

        Returns: bool indicating success or failure
        """
        logger.info("Starting ingestion pipeline...")

        chunks, embeddings = self.processor.process_pipeline()
        if not chunks:
            logger.warning("No new documents found for ingestion.")
            return False

        success = self.vector_store.add_documents(chunks)
        if success:
            logger.info("Ingestion completed successfully.")
        else:
            logger.error("Ingestion failed.")
        return success

    def stats(self):
        """
        Returns:
          collection statistics.
        """
        return self.vector_store.get_stats()
